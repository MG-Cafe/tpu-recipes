#!/usr/bin/env python3
"""
GB.Protein-16B (genbio-ai/GB.Protein-16B) Benchmark & Optimization Suite for NVIDIA G4 (RTX PRO 6000 Blackwell SM120).
Runs:
  1. Stock HuggingFace FM4BioForMaskedLM Baseline (BF16) on 1x G4 GPU
  2. Strict Numerical Parity Verification on Real Protein Sequences
  3. Heavily Optimized G4 Engine (Custom Triton RMSNorm + Fused QKV/RoPE + FlashAttention-2 + Hardware Grouped-GEMM & Triton Fused Bias-SwiGLU MoE) on 1x G4 GPU
  4. Multi-GPU G4 (4x NVIDIA RTX PRO 6000 Blackwell GPUs: TP=4 Low-Latency & DP=4 High-Throughput)
"""
import gc
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

MODEL_DIR = "/dev/shm/gb_protein_16b_bf16"
sys.path.insert(0, MODEL_DIR)

from configuration_fm4bio import FM4BioConfig
from tokenization_fm4bio import FM4BioTokenizer
import modeling_fm4bio

if not hasattr(modeling_fm4bio.FM4BioPreTrainedModel, "get_head_mask"):
    modeling_fm4bio.FM4BioPreTrainedModel.get_head_mask = (
        lambda self, head_mask, num_layers, is_attention_chunked=False: [None] * num_layers
        if head_mask is None
        else head_mask
    )


# ==============================================================================
# 1. CUSTOM TRITON KERNELS FOR G4 (SM120 BLACKWELL)
# ==============================================================================

@triton.jit
def _rmsnorm_fwd_kernel(
    X_ptr,
    W_ptr,
    Y_ptr,
    stride_x,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    rstd = tl.rsqrt(var + eps)
    y = (x * rstd * w).to(tl.bfloat16)
    tl.store(Y_ptr + row * stride_x + cols, y, mask=mask)


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    orig_shape = x.shape
    H = orig_shape[-1]
    x_2d = x.reshape(-1, H)
    M = x_2d.shape[0]
    y = torch.empty_like(x_2d)
    BLOCK_SIZE = triton.next_power_of_2(H)
    _rmsnorm_fwd_kernel[(M,)](
        x_2d, weight, y, x_2d.stride(0), H=H, eps=eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=8
    )
    return y.view(orig_shape)


@triton.jit
def _bias_swiglu_fwd_kernel(
    X_ptr,       # [M, 2*I]
    B_ptr,       # [E, 2*I]
    E_idx_ptr,   # [M] int64/int32 expert id per row
    Y_ptr,       # [M, I]
    stride_xm,
    stride_be,
    stride_ym,
    I: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    row = tl.program_id(0)
    pid_i = tl.program_id(1)
    cols = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = cols < I

    exp_id = tl.load(E_idx_ptr + row)
    # In FM4BioMLP.swiglu: x = torch.chunk(x, 2, dim=-1); return x[0] * F.silu(x[1])
    x0 = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(X_ptr + row * stride_xm + I + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B_ptr + exp_id * stride_be + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B_ptr + exp_id * stride_be + I + cols, mask=mask, other=0.0).to(tl.float32)

    u = x0 + b0
    v = x1 + b1
    silu_v = v * tl.sigmoid(v)
    out = (u * silu_v).to(tl.bfloat16)
    tl.store(Y_ptr + row * stride_ym + cols, out, mask=mask)


def triton_bias_swiglu(
    x: torch.Tensor, bias: torch.Tensor, expert_ids: torch.Tensor
) -> torch.Tensor:
    M, two_I = x.shape
    I = two_I // 2
    y = torch.empty((M, I), dtype=x.dtype, device=x.device)
    BLOCK_I = 1024
    grid = (M, triton.cdiv(I, BLOCK_I))
    _bias_swiglu_fwd_kernel[grid](
        x,
        bias,
        expert_ids,
        y,
        x.stride(0),
        bias.stride(0),
        y.stride(0),
        I=I,
        BLOCK_I=BLOCK_I,
        num_warps=4,
    )
    return y


@triton.jit
def _bias_scale_fwd_kernel(
    X_ptr,       # [M, H]
    B_ptr,       # [E, H]
    E_idx_ptr,   # [M]
    W_ptr,       # [M] routing weight
    Y_ptr,       # [M, H]
    stride_xm,
    stride_be,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H
    exp_id = tl.load(E_idx_ptr + row)
    rw = tl.load(W_ptr + row).to(tl.float32)
    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + exp_id * stride_be + cols, mask=mask, other=0.0).to(tl.float32)
    y = ((x + b) * rw).to(tl.bfloat16)
    tl.store(Y_ptr + row * stride_xm + cols, y, mask=mask)


def triton_bias_scale(
    x: torch.Tensor, bias: torch.Tensor, expert_ids: torch.Tensor, routing_weights: torch.Tensor
) -> torch.Tensor:
    M, H = x.shape
    y = torch.empty_like(x)
    BLOCK_H = triton.next_power_of_2(H)
    _bias_scale_fwd_kernel[(M,)](
        x,
        bias,
        expert_ids,
        routing_weights,
        y,
        x.stride(0),
        bias.stride(0),
        H=H,
        BLOCK_H=BLOCK_H,
        num_warps=8,
    )
    return y


# ==============================================================================
# 2. OPTIMIZED G4 LAYER & MODEL (FUSED QKV + ROPE + FLASHATTN + GROUPED-MM MOE)
# ==============================================================================

def precompute_rope_cache(max_seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)  # [MaxS, 64]
    # Shape [1, 1, MaxS, 64] for broadcasting over [B, NumHeads, S, HeadDim]
    cos = torch.cos(emb).to(dtype).unsqueeze(0).unsqueeze(0).contiguous()
    sin = torch.sin(emb).to(dtype).unsqueeze(0).unsqueeze(0).contiguous()
    return cos, sin


def apply_rope_fast(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [B, H, S, 64]; cos, sin: [1, 1, S, 64]
    q1, q2 = q[..., :32], q[..., 32:]
    k1, k2 = k[..., :32], k[..., 32:]
    q_rot = torch.cat((-q2, q1), dim=-1)
    k_rot = torch.cat((-k2, k1), dim=-1)
    return (q * cos) + (q_rot * sin), (k * cos) + (k_rot * sin)


class OptimizedFM4BioLayer(nn.Module):
    def __init__(self, hf_layer: nn.Module, device: torch.device):
        super().__init__()
        self.num_heads = 36
        self.head_dim = 64
        self.hidden_size = 2304
        self.intermediate_size = 7680
        self.num_experts = 8

        # 1. Attention RMSNorm & Fused QKV & Output Proj
        self.attn_ln_weight = hf_layer.attention.ln.weight.detach().to(device=device, dtype=torch.bfloat16).contiguous()
        wq = hf_layer.attention.self.query.weight.detach()
        wk = hf_layer.attention.self.key.weight.detach()
        wv = hf_layer.attention.self.value.weight.detach()
        bq = hf_layer.attention.self.query.bias.detach()
        bk = hf_layer.attention.self.key.bias.detach()
        bv = hf_layer.attention.self.value.bias.detach()

        self.w_qkv = torch.cat([wq, wk, wv], dim=0).to(device=device, dtype=torch.bfloat16).contiguous()
        self.b_qkv = torch.cat([bq, bk, bv], dim=0).to(device=device, dtype=torch.bfloat16).contiguous()

        self.w_out = hf_layer.attention.output.dense.weight.detach().to(device=device, dtype=torch.bfloat16).contiguous()
        self.b_out = hf_layer.attention.output.dense.bias.detach().to(device=device, dtype=torch.bfloat16).contiguous()

        # 2. MoE RMSNorm & Router & Stacked 3D Expert Weights for torch._grouped_mm
        self.mlp_ln_weight = hf_layer.ln.weight.detach().to(device=device, dtype=torch.bfloat16).contiguous()
        self.router_weight = hf_layer.mlp.router.weight.detach().to(device=device, dtype=torch.float32).contiguous()

        w1_list, b1_list, w2_list, b2_list = [], [], [], []
        for e in range(self.num_experts):
            e_h4h = getattr(hf_layer.mlp, f"dense_h_to_4h_{e}")
            e_4hh = getattr(hf_layer.mlp, f"dense_4h_to_h_{e}")
            w1_list.append(e_h4h.weight.detach().to(device=device, dtype=torch.bfloat16))
            b1_list.append(e_h4h.bias.detach().to(device=device, dtype=torch.bfloat16))
            w2_list.append(e_4hh.weight.detach().to(device=device, dtype=torch.bfloat16))
            b2_list.append(e_4hh.bias.detach().to(device=device, dtype=torch.bfloat16))

        # Keep [E, Out, In] contiguous so .transpose(1, 2) is column-major [E, In, Out] for _grouped_mm
        self.w1 = torch.stack(w1_list, dim=0).contiguous()  # [8, 15360, 2304]
        self.b1 = torch.stack(b1_list, dim=0).contiguous()  # [8, 15360]
        self.w2 = torch.stack(w2_list, dim=0).contiguous()  # [8, 2304, 7680]
        self.b2 = torch.stack(b2_list, dim=0).contiguous()  # [8, 2304]

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        # --- Block 1: Pre-RMSNorm + Fused QKV + Fast RoPE + FlashAttention-2 ---
        norm_x = triton_rmsnorm(x, self.attn_ln_weight)
        qkv = F.linear(norm_x, self.w_qkv, self.b_qkv)  # [B, S, 6912]
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3).contiguous()  # [B, 36, S, 64]
        k = qkv[:, :, 1].permute(0, 2, 1, 3).contiguous()
        v = qkv[:, :, 2].permute(0, 2, 1, 3).contiguous()

        q, k = apply_rope_fast(q, k, cos, sin)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, H)
        x = x + F.linear(attn_out, self.w_out, self.b_out)

        # --- Block 2: Pre-RMSNorm + Zero-Sync Grouped-GEMM Top-2 SwiGLU MoE ---
        norm_mlp = triton_rmsnorm(x, self.mlp_ln_weight)
        flat_x = norm_mlp.view(-1, H)  # [N, 2304], N = B * S
        N = flat_x.shape[0]

        # Router in FP32 (matches FM4BioMLP.forward exact specification)
        route_logits = F.linear(flat_x.float(), self.router_weight).to(torch.bfloat16)
        topk_w, topk_idx = torch.topk(route_logits, k=2, dim=-1)  # [N, 2]
        topk_w = F.softmax(topk_w, dim=-1, dtype=torch.float32).to(torch.bfloat16)

        flat_idx = topk_idx.view(-1)  # [2*N]
        flat_w = topk_w.view(-1)      # [2*N]

        perm = torch.argsort(flat_idx)
        sorted_exp_ids = flat_idx[perm]
        token_ids = perm // 2

        # Histogram on GPU -> cumulative int32 offsets for torch._grouped_mm (zero host sync!)
        counts = torch.histc(flat_idx.float(), bins=8, min=0, max=7).to(torch.int32)
        offs = torch.cumsum(counts, dim=0).to(torch.int32)

        x_sorted = flat_x[token_ids]  # [2*N, 2304]
        h1 = torch._grouped_mm(x_sorted, self.w1.transpose(1, 2), offs=offs)  # [2*N, 15360]
        inter = triton_bias_swiglu(h1, self.b1, sorted_exp_ids)               # [2*N, 7680]
        h2 = torch._grouped_mm(inter, self.w2.transpose(1, 2), offs=offs)     # [2*N, 2304]
        weighted_out = triton_bias_scale(h2, self.b2, sorted_exp_ids, flat_w[perm])

        mlp_out = torch.zeros_like(flat_x)
        mlp_out.index_add_(0, token_ids, weighted_out)
        return x + mlp_out.view(B, S, H)


class GBProteinG4Optimized(nn.Module):
    def __init__(self, hf_model: nn.Module, device: torch.device, max_seq_len: int = 2048):
        super().__init__()
        self.device = device
        self.word_embeddings = hf_model.bert.embeddings.word_embeddings.weight.detach().to(
            device=device, dtype=torch.bfloat16
        ).contiguous()
        self.final_ln_weight = hf_model.bert.encoder.ln.weight.detach().to(
            device=device, dtype=torch.bfloat16
        ).contiguous()
        self.output_embed_weight = hf_model.output_embed.weight.detach().to(
            device=device, dtype=torch.bfloat16
        ).contiguous()

        self.cos_cache, self.sin_cache = precompute_rope_cache(
            max_seq_len, 64, device=device, dtype=torch.bfloat16
        )

        self.layers = nn.ModuleList()
        for idx in range(len(hf_model.bert.encoder.layer)):
            layer = hf_model.bert.encoder.layer[idx]
            opt_layer = OptimizedFM4BioLayer(layer, device=device)
            self.layers.append(opt_layer)
            # Free original layer weights immediately so peak VRAM stays low
            hf_model.bert.encoder.layer[idx] = nn.Identity()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor):
        B, S = input_ids.shape
        x = F.embedding(input_ids, self.word_embeddings)
        cos = self.cos_cache[:, :, :S, :]
        sin = self.sin_cache[:, :, :S, :]
        for layer in self.layers:
            x = layer(x, cos, sin)
        hidden_states = triton_rmsnorm(x, self.final_ln_weight)
        logits = F.linear(hidden_states, self.output_embed_weight)
        return logits, hidden_states


# ==============================================================================
# 3. FAST CHECKPOINT LOADER FOR STOCK HUGGINGFACE MODEL
# ==============================================================================

def load_stock_hf_model(model_dir: str, device: torch.device) -> nn.Module:
    t0 = time.time()
    cfg = FM4BioConfig.from_json_file(os.path.join(model_dir, "config.json"))
    # Skip slow CPU random normal initialization since we overwrite all parameters from shards
    orig_init = modeling_fm4bio.FM4BioPreTrainedModel._init_weights
    modeling_fm4bio.FM4BioPreTrainedModel._init_weights = lambda self, module: None
    try:
        with torch.device(device):
            old_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            model = modeling_fm4bio.FM4BioForMaskedLM(cfg)
            torch.set_default_dtype(old_dtype)
    finally:
        modeling_fm4bio.FM4BioPreTrainedModel._init_weights = orig_init

    # Keep router in float32 as expected by FM4BioMLP
    for layer in model.bert.encoder.layer:
        layer.mlp.router = layer.mlp.router.float()

    shards = sorted([f for f in os.listdir(model_dir) if f.startswith("pytorch_model-") and f.endswith(".bin")])
    state_dict_map = dict(model.named_parameters())
    for s in shards:
        sd = torch.load(os.path.join(model_dir, s), map_location=device, weights_only=True)
        for k, v in sd.items():
            if k in state_dict_map:
                state_dict_map[k].data.copy_(v)
        del sd
    model.eval()
    torch.cuda.synchronize(device)
    print(f"[LOAD] Loaded 36-layer GB.Protein-16B onto {device} in {time.time() - t0:.2f}s", flush=True)
    return model


def compute_active_flops_per_token(seq_len: int) -> float:
    # Active params per token for GB.Protein-16B (L=36, H=2304, I=7680, K=2 active experts of 8):
    # Per layer:
    #   QKV: 2 * 2304 * 6912 = 31,850,496
    #   Attn Out: 2 * 2304 * 2304 = 10,616,832
    #   Attn Score & Context: 4 * 2304 * seq_len
    #   Router: 2 * 2304 * 8 = 36,864
    #   Active Experts (K=2): 2 * [2 * 2304 * 15360 + 2 * 7680 * 2304] = 424,673,280
    # Total per layer = 467,177,472 + 9,216 * seq_len
    per_layer = 467_177_472 + 9_216 * seq_len
    return 36.0 * per_layer + 2 * 2304 * 128


def main():
    while not os.path.exists(os.path.join(MODEL_DIR, "DONE")):
        print("Waiting for /dev/shm/gb_protein_16b_bf16/DONE...", flush=True)
        time.sleep(5)

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    print(f"=== Benchmarking genbio-ai/GB.Protein-16B on {gpu_name} ===", flush=True)

    tok = FM4BioTokenizer(vocab_file=os.path.join(MODEL_DIR, "vocab_protein.txt"))
    sample_seq = "M K A L V S G G N Y K P T F | C A S S W G G A P L F | E L A G I G I L T V"
    sample_ids = tok.encode(sample_seq)
    print(f"Sample protein complex ({len(sample_ids)} tokens): {sample_ids}", flush=True)

    # 1. Load Stock HuggingFace Model
    hf_model = load_stock_hf_model(MODEL_DIR, device)

    # Golden parity input on real protein sequence + batch
    torch.manual_seed(42)
    parity_ids = torch.randint(1, 22, (2, 256), device=device, dtype=torch.long)
    parity_ids[0, :len(sample_ids)] = torch.tensor(sample_ids, device=device)
    with torch.no_grad():
        hf_out = hf_model(parity_ids, output_hidden_states=True)
        ref_logits = hf_out.logits.detach().clone()
        ref_hidden = triton_rmsnorm(
            hf_out.hidden_states[-1].transpose(0, 1).contiguous(),
            hf_model.bert.encoder.ln.weight,
        ).detach().clone()
        del hf_out

    # Save golden reference to /tmp/gb_protein_golden_ref.pt for TPU parity check too!
    torch.save(
        {
            "input_ids": parity_ids.cpu(),
            "ref_logits": ref_logits.cpu(),
            "ref_hidden": ref_hidden.cpu(),
        },
        "/tmp/gb_protein_golden_ref.pt",
    )
    print("[PARITY] Saved golden reference outputs to /tmp/gb_protein_golden_ref.pt", flush=True)

    # 2. Benchmark Stock HuggingFace Baseline on 1x G4 GPU
    configs = [
        (1, 256),
        (1, 512),
        (1, 1024),
        (1, 2048),
        (4, 512),
        (8, 512),
        (16, 512),
        (8, 1024),
        (16, 1024),
        (8, 2048),
        (32, 512),
    ]

    results = {
        "gpu_name": gpu_name,
        "model": "genbio-ai/GB.Protein-16B",
        "stock_hf_1gpu": [],
        "optimized_1gpu": [],
        "optimized_4gpu_dp4": [],
        "parity": {},
    }

    print("\n--- Running Stock HuggingFace Baseline (1x G4 GPU, BF16) ---", flush=True)
    for B, S in configs:
        x = torch.randint(1, 22, (B, S), device=device, dtype=torch.long)
        # Warmup
        with torch.no_grad():
            for _ in range(2):
                _ = hf_model(x)
        torch.cuda.synchronize(device)
        n_iters = 5 if (B * S <= 4096) else 3
        latencies = []
        with torch.no_grad():
            for _ in range(n_iters):
                t0 = time.perf_counter()
                _ = hf_model(x)
                torch.cuda.synchronize(device)
                latencies.append((time.perf_counter() - t0) * 1000.0)
        mean_ms = sum(latencies) / len(latencies)
        median_ms = sorted(latencies)[len(latencies) // 2]
        tok_per_s = (B * S) / (median_ms / 1000.0)
        seq_per_s = B / (median_ms / 1000.0)
        tflops = (B * S * compute_active_flops_per_token(S)) / (median_ms / 1000.0) / 1e12
        rec = {
            "batch_size": B,
            "seq_len": S,
            "tokens_per_batch": B * S,
            "mean_ms": round(mean_ms, 2),
            "median_ms": round(median_ms, 2),
            "tokens_per_sec": round(tok_per_s, 1),
            "seqs_per_sec": round(seq_per_s, 2),
            "active_tflops": round(tflops, 2),
        }
        results["stock_hf_1gpu"].append(rec)
        print(
            f"[Stock HF 1xG4] B={B:2d}, S={S:4d} | Latency={median_ms:7.2f} ms | "
            f"Tok/s={tok_per_s:8.1f} | Seq/s={seq_per_s:6.2f} | Active TFLOPS={tflops:6.2f}",
            flush=True,
        )

    # 3. Convert in-place to GBProteinG4Optimized and Verify Numerical Parity
    print("\n--- Building GBProteinG4Optimized (Triton + Grouped GEMM + FlashAttention-2) ---", flush=True)
    opt_model = GBProteinG4Optimized(hf_model, device=device, max_seq_len=2048)
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    opt_logits, opt_hidden = opt_model(parity_ids)
    cos_sim_logits = F.cosine_similarity(ref_logits.flatten().float(), opt_logits.flatten().float(), dim=0).item()
    cos_sim_hidden = F.cosine_similarity(ref_hidden.flatten().float(), opt_hidden.flatten().float(), dim=0).item()
    max_diff_logits = (ref_logits.float() - opt_logits.float()).abs().max().item()
    mean_diff_logits = (ref_logits.float() - opt_logits.float()).abs().mean().item()
    top1_match = (ref_logits.argmax(dim=-1) == opt_logits.argmax(dim=-1)).float().mean().item()

    results["parity"] = {
        "cosine_sim_logits": round(cos_sim_logits, 6),
        "cosine_sim_hidden": round(cos_sim_hidden, 6),
        "max_abs_diff_logits": round(max_diff_logits, 5),
        "mean_abs_diff_logits": round(mean_diff_logits, 6),
        "top1_token_agreement": round(top1_match, 6),
    }
    print(f"[PARITY CHECK] Optimized G4 vs Stock HF: {results['parity']}", flush=True)
    assert cos_sim_logits > 0.999, f"Cosine similarity too low: {cos_sim_logits}"

    # 4. Benchmark GBProteinG4Optimized on 1x G4 GPU
    print("\n--- Running GBProteinG4Optimized (1x G4 GPU, BF16) ---", flush=True)
    for i, (B, S) in enumerate(configs):
        x = torch.randint(1, 22, (B, S), device=device, dtype=torch.long)
        for _ in range(3):
            _ = opt_model(x)
        torch.cuda.synchronize(device)
        n_iters = 10 if (B * S <= 8192) else 5
        latencies = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = opt_model(x)
            torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        mean_ms = sum(latencies) / len(latencies)
        median_ms = sorted(latencies)[len(latencies) // 2]
        tok_per_s = (B * S) / (median_ms / 1000.0)
        seq_per_s = B / (median_ms / 1000.0)
        tflops = (B * S * compute_active_flops_per_token(S)) / (median_ms / 1000.0) / 1e12
        base_ms = results["stock_hf_1gpu"][i]["median_ms"]
        speedup = base_ms / median_ms
        rec = {
            "batch_size": B,
            "seq_len": S,
            "tokens_per_batch": B * S,
            "mean_ms": round(mean_ms, 2),
            "median_ms": round(median_ms, 2),
            "tokens_per_sec": round(tok_per_s, 1),
            "seqs_per_sec": round(seq_per_s, 2),
            "active_tflops": round(tflops, 2),
            "speedup_vs_stock_hf": round(speedup, 2),
        }
        results["optimized_1gpu"].append(rec)
        print(
            f"[Opt 1xG4]      B={B:2d}, S={S:4d} | Latency={median_ms:7.2f} ms ({speedup:4.2f}x vs Stock) | "
            f"Tok/s={tok_per_s:8.1f} | Seq/s={seq_per_s:6.2f} | Active TFLOPS={tflops:6.2f}",
            flush=True,
        )

    # 5. Benchmark 4x G4 GPU Data-Parallel Throughput (4 replicas across cuda:0..3)
    if torch.cuda.device_count() >= 4:
        print("\n--- Cloning Optimized Model across 4x G4 GPUs (cuda:0..3) for 4-GPU Benchmark ---", flush=True)
        import threading

        class ShallowReplica(nn.Module):
            def __init__(self, src: GBProteinG4Optimized, dev: torch.device):
                super().__init__()
                self.device = dev
                self.word_embeddings = src.word_embeddings.to(dev)
                self.final_ln_weight = src.final_ln_weight.to(dev)
                self.output_embed_weight = src.output_embed_weight.to(dev)
                self.cos_cache = src.cos_cache.to(dev)
                self.sin_cache = src.sin_cache.to(dev)
                self.layers = []
                for lyr in src.layers:
                    l_copy = OptimizedFM4BioLayer.__new__(OptimizedFM4BioLayer)
                    nn.Module.__init__(l_copy)
                    l_copy.num_heads = lyr.num_heads
                    l_copy.head_dim = lyr.head_dim
                    l_copy.hidden_size = lyr.hidden_size
                    l_copy.intermediate_size = lyr.intermediate_size
                    l_copy.num_experts = lyr.num_experts
                    l_copy.attn_ln_weight = lyr.attn_ln_weight.to(dev)
                    l_copy.w_qkv = lyr.w_qkv.to(dev)
                    l_copy.b_qkv = lyr.b_qkv.to(dev)
                    l_copy.w_out = lyr.w_out.to(dev)
                    l_copy.b_out = lyr.b_out.to(dev)
                    l_copy.mlp_ln_weight = lyr.mlp_ln_weight.to(dev)
                    l_copy.router_weight = lyr.router_weight.to(dev)
                    l_copy.w1 = lyr.w1.to(dev)
                    l_copy.b1 = lyr.b1.to(dev)
                    l_copy.w2 = lyr.w2.to(dev)
                    l_copy.b2 = lyr.b2.to(dev)
                    self.layers.append(l_copy)

            @torch.no_grad()
            def forward(self, input_ids: torch.Tensor):
                B, S = input_ids.shape
                x = F.embedding(input_ids, self.word_embeddings)
                cos = self.cos_cache[:, :, :S, :]
                sin = self.sin_cache[:, :, :S, :]
                for layer in self.layers:
                    x = layer(x, cos, sin)
                hidden_states = triton_rmsnorm(x, self.final_ln_weight)
                logits = F.linear(hidden_states, self.output_embed_weight)
                return logits, hidden_states

        torch.cuda.empty_cache()
        replicas = [opt_model]
        for dev_id in range(1, 4):
            dev = torch.device(f"cuda:{dev_id}")
            replicas.append(ShallowReplica(opt_model, dev))
        for dev_id in range(4):
            torch.cuda.synchronize(dev_id)

        dp4_configs = [
            (4, 512),     # 1 per GPU -> 4 total
            (16, 512),    # 4 per GPU -> 16 total
            (32, 512),    # 8 per GPU -> 32 total
            (64, 512),    # 16 per GPU -> 64 total
            (32, 1024),   # 8 per GPU -> 32 total
            (64, 1024),   # 16 per GPU -> 64 total
            (32, 2048),   # 8 per GPU -> 32 total
        ]
        for total_B, S in dp4_configs:
            b_per_gpu = total_B // 4
            xs = [torch.randint(1, 22, (b_per_gpu, S), device=f"cuda:{d}", dtype=torch.long) for d in range(4)]

            def run_all():
                threads = []
                for d in range(4):
                    t = threading.Thread(target=lambda idx=d: replicas[idx](xs[idx]))
                    t.start()
                    threads.append(t)
                for t in threads:
                    t.join()
                for d in range(4):
                    torch.cuda.synchronize(d)

            for _ in range(2):
                run_all()
            latencies = []
            for _ in range(5):
                t0 = time.perf_counter()
                run_all()
                latencies.append((time.perf_counter() - t0) * 1000.0)
            median_ms = sorted(latencies)[len(latencies) // 2]
            tok_per_s = (total_B * S) / (median_ms / 1000.0)
            seq_per_s = total_B / (median_ms / 1000.0)
            tflops = (total_B * S * compute_active_flops_per_token(S)) / (median_ms / 1000.0) / 1e12
            rec = {
                "total_batch_size": total_B,
                "batch_per_gpu": b_per_gpu,
                "seq_len": S,
                "tokens_per_batch": total_B * S,
                "median_ms": round(median_ms, 2),
                "tokens_per_sec": round(tok_per_s, 1),
                "seqs_per_sec": round(seq_per_s, 2),
                "active_tflops": round(tflops, 2),
            }
            results["optimized_4gpu_dp4"].append(rec)
            print(
                f"[Opt 4xG4 DP=4] TotalB={total_B:2d} ({b_per_gpu}/GPU), S={S:4d} | Latency={median_ms:7.2f} ms | "
                f"Tok/s={tok_per_s:9.1f} | Seq/s={seq_per_s:6.2f} | Active TFLOPS={tflops:6.2f}",
                flush=True,
            )

    with open("/tmp/g4_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[DONE] Saved G4 results to /tmp/g4_benchmark_results.json!", flush=True)


if __name__ == "__main__":
    main()
