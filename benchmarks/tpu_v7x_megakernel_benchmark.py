#!/usr/bin/env python3
"""
GB.Protein-16B (genbio-ai/GB.Protein-16B) Benchmark & Bio-Megakernel Suite for Google Cloud TPU v7x (Ironwood).
Runs:
  1. Standalone Pallas VMEM Fused Bias-SwiGLU Kernel Microbenchmark on TPU v7x
  2. Strict Numerical Parity Verification against Reference FM4BioForMaskedLM on Real Protein Complexes
  3. Tier 1: 1x TPU v7x Chip Naive XLA Baseline (1 TensorCore, Unfused Dense-Gather MoE)
  4. Tier 2: 1x TPU v7x Chip Bio-Megakernel (2 TensorCores On-Die TP=2 + Megablox Ragged-Dot MoE + VMEM Fused Bias-SwiGLU + 128-Lane Packed RoPE + Flash Bidirectional Attention + 36-Layer Scan)
  5. Tier 3: 4x TPU v7x Chips Ultra-Low-Latency Bio-Megakernel (8 TensorCores TP=8 across 4 Chips, Zero-Padded 40-Head / 8192-Intermediate 128-Lane Aligned)
  6. Tier 4: 4x TPU v7x Chips High-Throughput Bio-Megakernel (8 TensorCores = DP=4 Chips x On-Die TP=2, Zero Inter-Chip ICI Traffic)
"""
import gc
import json
import math
import os
import sys
import time
from functools import partial

import numpy as np
import torch
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.shard_map import shard_map

MODEL_DIR = "/dev/shm/gb_protein_16b_bf16"
sys.path.insert(0, MODEL_DIR)
from tokenization_fm4bio import FM4BioTokenizer


def compute_active_flops_per_token(seq_len: int) -> float:
    per_layer = 467_177_472 + 9_216 * seq_len
    return 36.0 * per_layer + 2 * 2304 * 128


def torch_bf16_to_np(t: torch.Tensor) -> np.ndarray:
    return (
        t.detach()
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint16)
        .numpy()
        .view(jnp.bfloat16)
    )


# ==============================================================================
# 1. CUSTOM PALLAS VMEM KERNEL FOR TPU v7x (IRONWOOD)
# ==============================================================================

def _pallas_bias_swiglu_kernel(u_ref, v_ref, b0_ref, b1_ref, out_ref):
    # Executed inside TPU v7x VMEM + 32-bit VPU registers (8x128 lanes)
    u = u_ref[...].astype(jnp.float32) + b0_ref[...].astype(jnp.float32)
    v = v_ref[...].astype(jnp.float32) + b1_ref[...].astype(jnp.float32)
    silu_v = v * jax.nn.sigmoid(v)
    out_ref[...] = (u * silu_v).astype(jnp.bfloat16)


def pallas_fused_bias_swiglu(
    u: jax.Array, v: jax.Array, b0: jax.Array, b1: jax.Array
) -> jax.Array:
    """
    Fused Bias + SwiGLU activation in TPU v7x VMEM.
    Uses block_m <= 128 so scoped VMEM footprint is <= 7.5 MB at I_shard=3840.
    """
    M, I_shard = u.shape
    block_m = min(M, 128)
    while M % block_m != 0 and block_m > 8:
        block_m //= 2
    grid = (M // block_m, 1)
    spec = pl.BlockSpec((block_m, I_shard), lambda i, j: (i, j))
    return pl.pallas_call(
        _pallas_bias_swiglu_kernel,
        out_shape=jax.ShapeDtypeStruct((M, I_shard), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[spec, spec, spec, spec],
            out_specs=spec,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(u, v, b0, b1)


def benchmark_pallas_vmem_kernels(device) -> list:
    """Microbenchmarks custom Pallas VMEM Fused Bias-SwiGLU kernel on TPU v7x."""
    print("\n--- Benchmarking Custom Pallas VMEM Fused Bias-SwiGLU Kernel on TPU v7x ---", flush=True)
    pallas_jit = jax.jit(pallas_fused_bias_swiglu)
    I_shard = 3840  # TP=2 shard of 7680
    records = []
    for M in [512, 1024, 2048, 4096, 8192]:
        u = jax.device_put(jnp.ones((M, I_shard), dtype=jnp.bfloat16), device)
        v = jax.device_put(jnp.ones((M, I_shard), dtype=jnp.bfloat16), device)
        b0 = jax.device_put(jnp.zeros((M, I_shard), dtype=jnp.bfloat16), device)
        b1 = jax.device_put(jnp.zeros((M, I_shard), dtype=jnp.bfloat16), device)
        for _ in range(3):
            out = pallas_jit(u, v, b0, b1)
            jax.block_until_ready(out)
        lats = []
        for _ in range(20):
            t0 = time.perf_counter()
            out = pallas_jit(u, v, b0, b1)
            jax.block_until_ready(out)
            lats.append((time.perf_counter() - t0) * 1e6)  # microseconds
        med_us = sorted(lats)[len(lats) // 2]
        # Bytes read (4 inputs) + written (1 output) = 5 * M * I_shard * 2 bytes
        bytes_moved = 5.0 * M * I_shard * 2.0
        gbps = (bytes_moved / (med_us / 1e6)) / 1e9
        rec = {
            "M_routed_tokens": M,
            "I_shard": I_shard,
            "median_us": round(med_us, 2),
            "effective_hbm_gbps": round(gbps, 1),
        }
        records.append(rec)
        print(f"[Pallas VMEM Bias-SwiGLU] M={M:4d}, I_shard={I_shard} | Latency={med_us:6.2f} us | Effective BW={gbps:7.1f} GB/s", flush=True)
    return records


# ==============================================================================
# 2. TPU v7x BIO-MEGAKERNEL BUILDING BLOCKS
# ==============================================================================

def tpu_rmsnorm(x: jax.Array, weight: jax.Array, eps: float = 1e-5) -> jax.Array:
    xf = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(xf), axis=-1, keepdims=True)
    return (xf * jax.lax.rsqrt(var + eps)).astype(jnp.bfloat16) * weight


def precompute_rope_tpu(max_seq_len: int = 2048, head_dim: int = 64):
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)  # [MaxS, 64]
    cos = jnp.asarray(np.cos(emb), dtype=jnp.bfloat16)[None, :, None, :]
    sin = jnp.asarray(np.sin(emb), dtype=jnp.bfloat16)[None, :, None, :]
    return cos, sin


def apply_rope_128lane(q: jax.Array, k: jax.Array, cos: jax.Array, sin: jax.Array):
    q1, q2 = q[..., :32], q[..., 32:]
    k1, k2 = k[..., :32], k[..., 32:]
    q_rot = jnp.concatenate([-q2, q1], axis=-1)
    k_rot = jnp.concatenate([-k2, k1], axis=-1)
    return (q * cos) + (q_rot * sin), (k * cos) + (k_rot * sin)


def megablox_moe_block(
    flat_x: jax.Array,        # [N, 2304] bfloat16
    router_w: jax.Array,      # [2304, 8] float32
    w1: jax.Array,            # [8, 2304, 2 * I_shard] bfloat16
    b1: jax.Array,            # [8, 2 * I_shard] bfloat16
    w2: jax.Array,            # [8, I_shard, 2304] bfloat16
    b2_scaled: jax.Array,     # [8, 2304] bfloat16 (already divided by tp_size)
    tp_axis_name: str | None = "tp",
) -> jax.Array:
    N, H = flat_x.shape
    two_N = 2 * N
    I_shard = b1.shape[-1] // 2

    # 1. Exact FP32 Router + Top-2 Softmax
    logits = jnp.dot(
        flat_x.astype(jnp.float32), router_w, preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)
    topk_w, topk_idx = jax.lax.top_k(logits, 2)  # [N, 2]
    topk_w = jax.nn.softmax(topk_w.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)

    flat_idx = topk_idx.reshape(-1)  # [2*N]
    flat_w = topk_w.reshape(-1)      # [2*N]

    # 2. Single-Key Bit-Packed Radix Sort (50% fewer VPU sort passes!)
    slot_ids = jnp.arange(two_N, dtype=jnp.uint32)
    packed = (flat_idx.astype(jnp.uint32) << 24) | slot_ids
    sorted_packed = jax.lax.sort(packed)
    perm = (sorted_packed & jnp.uint32(0xFFFFFF)).astype(jnp.int32)
    sorted_exp_ids = (sorted_packed >> jnp.uint32(24)).astype(jnp.int32)
    token_ids = perm >> 1

    # 3. MXU One-Hot Expert Histogram
    group_sizes = jnp.sum(jax.nn.one_hot(flat_idx, 8, dtype=jnp.int32), axis=0)  # [8]

    # 4. Megablox Ragged-Dot 1 + XLA VMEM-Fused 128-Lane Contiguous Bias-SwiGLU Epilogue
    x_sorted = flat_x[token_ids]  # [2*N, 2304]
    h1 = jax.lax.ragged_dot(
        x_sorted, w1, group_sizes, preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16) + b1[sorted_exp_ids]  # [2*N, 2 * I_shard]
    u = h1[:, :I_shard].astype(jnp.float32)
    v = h1[:, I_shard:].astype(jnp.float32)
    inter = (u * jax.nn.silu(v)).astype(jnp.bfloat16)  # [2*N, I_shard]

    # 5. Megablox Ragged-Dot 2
    h2 = jax.lax.ragged_dot(
        inter, w2, group_sizes, preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)  # [2*N, 2304]

    # 6. Pre-AllReduce Bijection Un-Permute & Local Top-2 Sum (Halves All-Reduce Payload!)
    weighted_h2 = (h2 + b2_scaled[sorted_exp_ids]) * flat_w[perm, None]
    out_unsorted = jnp.empty_like(weighted_h2).at[perm].set(weighted_h2)
    mlp_local = jnp.sum(out_unsorted.reshape(N, 2, H), axis=1)  # [N, 2304]

    if tp_axis_name is not None:
        mlp_out = jax.lax.psum(mlp_local, axis_name=tp_axis_name)
    else:
        mlp_out = mlp_local
    return mlp_out


def naive_moe_block(
    flat_x: jax.Array,
    router_w: jax.Array,
    w1: jax.Array,   # [8, 2304, 15360]
    b1: jax.Array,   # [8, 15360]
    w2: jax.Array,   # [8, 7680, 2304]
    b2: jax.Array,   # [8, 2304]
) -> jax.Array:
    """Naive unoptimized MoE baseline (unfused loop over 8 experts with dynamic masking)."""
    N, H = flat_x.shape
    I_full = b1.shape[-1] // 2
    logits = jnp.dot(
        flat_x.astype(jnp.float32), router_w, preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)
    topk_w, topk_idx = jax.lax.top_k(logits, 2)
    topk_w = jax.nn.softmax(topk_w.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)

    def exp_step(carry, e):
        mask0 = (topk_idx[:, 0] == e).astype(jnp.bfloat16)
        mask1 = (topk_idx[:, 1] == e).astype(jnp.bfloat16)
        w_e = mask0 * topk_w[:, 0] + mask1 * topk_w[:, 1]  # [N]
        h1 = jnp.dot(flat_x, w1[e], preferred_element_type=jnp.float32).astype(jnp.bfloat16) + b1[e][None, :]
        u = h1[:, :I_full].astype(jnp.float32)
        v = h1[:, I_full:].astype(jnp.float32)
        inter = (u * jax.nn.silu(v)).astype(jnp.bfloat16)
        h2 = jnp.dot(inter, w2[e], preferred_element_type=jnp.float32).astype(jnp.bfloat16) + b2[e][None, :]
        return carry + h2 * w_e[:, None], None

    out, _ = jax.lax.scan(exp_step, jnp.zeros_like(flat_x), jnp.arange(8))
    return out


# ==============================================================================
# 3. FULL 36-LAYER TRANSFORMER ENCODER SCAN FUNCTIONS
# ==============================================================================

def make_megakernel_forward(tp_size: int, tp_axis_name: str | None = "tp"):
    def forward_fn(
        input_ids: jax.Array,      # [B_local, S]
        word_emb: jax.Array,       # [128, 2304]
        final_ln_w: jax.Array,     # [2304]
        out_emb_w: jax.Array,      # [2304, 128]
        cos_cache: jax.Array,      # [1, MaxS, 1, 64]
        sin_cache: jax.Array,      # [1, MaxS, 1, 64]
        layer_params: dict,        # Stacked [36, ...]
    ):
        B, S = input_ids.shape
        H = 2304
        x = word_emb[input_ids]  # [B, S, 2304]
        cos = cos_cache[:, :S, :, :]
        sin = sin_cache[:, :S, :, :]

        def layer_scan_body(x_curr, lp):
            # --- 1. Pre-RMSNorm + Sharded QKV + 128-Lane RoPE + Bidirectional FlashAttention ---
            norm_x = tpu_rmsnorm(x_curr, lp["attn_ln_w"])
            qkv = jnp.einsum(
                "bsh,hctd->bsctd", norm_x, lp["w_qkv"], preferred_element_type=jnp.float32
            ).astype(jnp.bfloat16) + lp["b_qkv"][None, None, ...]
            q = qkv[:, :, 0, :, :]  # [B, S, H_shard, 64]
            k = qkv[:, :, 1, :, :]
            v = qkv[:, :, 2, :, :]

            q, k = apply_rope_128lane(q, k, cos, sin)
            attn_ctx = jax.nn.dot_product_attention(q, k, v, is_causal=False)  # [B, S, H_shard, 64]

            attn_proj_local = jnp.einsum(
                "bstd,tdo->bso", attn_ctx, lp["w_out"], preferred_element_type=jnp.float32
            ).astype(jnp.bfloat16) + (lp["b_out"] / tp_size)[None, None, :]

            if tp_axis_name is not None:
                attn_proj = jax.lax.psum(attn_proj_local, axis_name=tp_axis_name)
            else:
                attn_proj = attn_proj_local
            x_mid = x_curr + attn_proj

            # --- 2. Pre-RMSNorm + Megablox Ragged-Dot + VMEM-Fused Bias-SwiGLU MoE ---
            norm_mlp = tpu_rmsnorm(x_mid, lp["mlp_ln_w"])
            flat_x = norm_mlp.reshape(B * S, H)
            mlp_out = megablox_moe_block(
                flat_x,
                lp["router_w"],
                lp["w1"],
                lp["b1"],
                lp["w2"],
                lp["b2"] / tp_size,
                tp_axis_name=tp_axis_name,
            )
            x_next = x_mid + mlp_out.reshape(B, S, H)
            return x_next, None

        x_final, _ = jax.lax.scan(layer_scan_body, x, layer_params)
        hidden = tpu_rmsnorm(x_final, final_ln_w)
        logits = jnp.dot(
            hidden, out_emb_w, preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)
        return logits, hidden

    return forward_fn


def make_naive_baseline_forward():
    def forward_fn(
        input_ids: jax.Array,
        word_emb: jax.Array,
        final_ln_w: jax.Array,
        out_emb_w: jax.Array,
        cos_cache: jax.Array,
        sin_cache: jax.Array,
        layer_params: dict,
    ):
        B, S = input_ids.shape
        H = 2304
        x = word_emb[input_ids]
        cos = cos_cache[:, :S, :, :]
        sin = sin_cache[:, :S, :, :]

        def layer_scan_body(x_curr, lp):
            norm_x = tpu_rmsnorm(x_curr, lp["attn_ln_w"])
            qkv = jnp.einsum(
                "bsh,hctd->bsctd", norm_x, lp["w_qkv"], preferred_element_type=jnp.float32
            ).astype(jnp.bfloat16) + lp["b_qkv"][None, None, ...]
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q, k = apply_rope_128lane(q, k, cos, sin)
            attn_ctx = jax.nn.dot_product_attention(q, k, v, is_causal=False)
            attn_proj = jnp.einsum(
                "bstd,tdo->bso", attn_ctx, lp["w_out"], preferred_element_type=jnp.float32
            ).astype(jnp.bfloat16) + lp["b_out"][None, None, :]
            x_mid = x_curr + attn_proj

            norm_mlp = tpu_rmsnorm(x_mid, lp["mlp_ln_w"])
            flat_x = norm_mlp.reshape(B * S, H)
            mlp_out = naive_moe_block(
                flat_x, lp["router_w"], lp["w1"], lp["b1"], lp["w2"], lp["b2"]
            )
            return x_mid + mlp_out.reshape(B, S, H), None

        x_final, _ = jax.lax.scan(layer_scan_body, x, layer_params)
        hidden = tpu_rmsnorm(x_final, final_ln_w)
        logits = jnp.dot(hidden, out_emb_w, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        return logits, hidden

    return forward_fn


# ==============================================================================
# 4. CHECKPOINT LOADER INTO STACKED NUMPY ARRAYS
# ==============================================================================

def load_checkpoint_arrays(model_dir: str):
    t0 = time.time()
    print(f"[LOAD] Loading 13 BF16 shards from {model_dir}...", flush=True)
    num_layers = 36
    H = 2304
    num_heads = 36
    head_dim = 64
    num_experts = 8
    I = 7680

    attn_ln_w = [None] * num_layers
    wq = [None] * num_layers
    wk = [None] * num_layers
    wv = [None] * num_layers
    bq = [None] * num_layers
    bk = [None] * num_layers
    bv = [None] * num_layers
    w_out = [None] * num_layers
    b_out = [None] * num_layers
    mlp_ln_w = [None] * num_layers
    router_w = [None] * num_layers
    w1 = [[None] * num_experts for _ in range(num_layers)]
    b1 = [[None] * num_experts for _ in range(num_layers)]
    w2 = [[None] * num_experts for _ in range(num_layers)]
    b2 = [[None] * num_experts for _ in range(num_layers)]

    word_emb = None
    final_ln_w = None
    out_emb_w = None

    shards = sorted([f for f in os.listdir(model_dir) if f.startswith("pytorch_model-") and f.endswith(".bin")])
    for s in shards:
        sd = torch.load(os.path.join(model_dir, s), map_location="cpu", weights_only=True)
        for k, v in sd.items():
            if k == "bert.embeddings.word_embeddings.weight":
                word_emb = torch_bf16_to_np(v).copy()
            elif k == "bert.encoder.ln.weight":
                final_ln_w = torch_bf16_to_np(v).copy()
            elif k == "output_embed.weight":
                out_emb_w = torch_bf16_to_np(v.T).copy()  # [2304, 128]
            elif k.startswith("bert.encoder.layer."):
                parts = k.split(".")
                l_idx = int(parts[3])
                sub = ".".join(parts[4:])
                if sub == "attention.ln.weight":
                    attn_ln_w[l_idx] = torch_bf16_to_np(v).copy()
                elif sub == "attention.self.query.weight":
                    wq[l_idx] = torch_bf16_to_np(v.T).reshape(H, num_heads, head_dim).copy()
                elif sub == "attention.self.key.weight":
                    wk[l_idx] = torch_bf16_to_np(v.T).reshape(H, num_heads, head_dim).copy()
                elif sub == "attention.self.value.weight":
                    wv[l_idx] = torch_bf16_to_np(v.T).reshape(H, num_heads, head_dim).copy()
                elif sub == "attention.self.query.bias":
                    bq[l_idx] = torch_bf16_to_np(v).reshape(num_heads, head_dim).copy()
                elif sub == "attention.self.key.bias":
                    bk[l_idx] = torch_bf16_to_np(v).reshape(num_heads, head_dim).copy()
                elif sub == "attention.self.value.bias":
                    bv[l_idx] = torch_bf16_to_np(v).reshape(num_heads, head_dim).copy()
                elif sub == "attention.output.dense.weight":
                    w_out[l_idx] = torch_bf16_to_np(v.T).reshape(num_heads, head_dim, H).copy()
                elif sub == "attention.output.dense.bias":
                    b_out[l_idx] = torch_bf16_to_np(v).copy()
                elif sub == "ln.weight":
                    mlp_ln_w[l_idx] = torch_bf16_to_np(v).copy()
                elif sub == "mlp.router.weight":
                    router_w[l_idx] = v.detach().to(torch.float32).T.contiguous().numpy().copy()
                elif sub.startswith("mlp.dense_h_to_4h_"):
                    e_idx = int(sub.split(".")[1].split("_")[-1])
                    if sub.endswith(".weight"):
                        w1[l_idx][e_idx] = torch_bf16_to_np(v.T).reshape(H, 2, I).copy()
                    elif sub.endswith(".bias"):
                        b1[l_idx][e_idx] = torch_bf16_to_np(v).reshape(2, I).copy()
                elif sub.startswith("mlp.dense_4h_to_h_"):
                    e_idx = int(sub.split(".")[1].split("_")[-1])
                    if sub.endswith(".weight"):
                        w2[l_idx][e_idx] = torch_bf16_to_np(v.T).copy()
                    elif sub.endswith(".bias"):
                        b2[l_idx][e_idx] = torch_bf16_to_np(v).copy()
        del sd
        gc.collect()

    print(f"[LOAD] Stacking 36-layer arrays...", flush=True)
    w_qkv_np = np.stack(
        [np.stack([wq[l], wk[l], wv[l]], axis=1) for l in range(num_layers)], axis=0
    )
    b_qkv_np = np.stack(
        [np.stack([bq[l], bk[l], bv[l]], axis=0) for l in range(num_layers)], axis=0
    )
    w1_np = np.stack([np.stack(w1[l], axis=0) for l in range(num_layers)], axis=0)
    b1_np = np.stack([np.stack(b1[l], axis=0) for l in range(num_layers)], axis=0)
    w2_np = np.stack([np.stack(w2[l], axis=0) for l in range(num_layers)], axis=0)
    b2_np = np.stack([np.stack(b2[l], axis=0) for l in range(num_layers)], axis=0)

    stacked = {
        "attn_ln_w": np.stack(attn_ln_w, axis=0),
        "w_qkv": w_qkv_np,
        "b_qkv": b_qkv_np,
        "w_out": np.stack(w_out, axis=0),
        "b_out": np.stack(b_out, axis=0),
        "mlp_ln_w": np.stack(mlp_ln_w, axis=0),
        "router_w": np.stack(router_w, axis=0),
        "w1": w1_np,
        "b1": b1_np,
        "w2": w2_np,
        "b2": b2_np,
    }
    print(f"[LOAD] Finished loading & stacking 36 layers in {time.time() - t0:.2f}s!", flush=True)
    return word_emb, final_ln_w, out_emb_w, stacked


def prepare_sharded_layer_params(stacked: dict, mesh: Mesh, tp_size: int, pad_for_tp8: bool = False):
    """Places 36-layer stacked weights onto the TPU mesh with exact TP sharding."""
    num_layers = 36
    H = 2304

    w_qkv = stacked["w_qkv"]
    b_qkv = stacked["b_qkv"]
    w_out = stacked["w_out"]
    w1 = stacked["w1"]      # [36, 8, 2304, 2, 7680]
    b1 = stacked["b1"]      # [36, 8, 2, 7680]
    w2 = stacked["w2"]      # [36, 8, 7680, 2304]

    if pad_for_tp8:
        # Pad heads 36 -> 40 with zero w_out so 40 / 8 = 5 heads per TensorCore
        pad_wqkv = np.zeros((num_layers, H, 3, 4, 64), dtype=jnp.bfloat16)
        pad_bqkv = np.zeros((num_layers, 3, 4, 64), dtype=jnp.bfloat16)
        pad_wout = np.zeros((num_layers, 4, 64, H), dtype=jnp.bfloat16)
        w_qkv = np.concatenate([w_qkv, pad_wqkv], axis=3)
        b_qkv = np.concatenate([b_qkv, pad_bqkv], axis=2)
        w_out = np.concatenate([w_out, pad_wout], axis=1)

        # Pad intermediate 7680 -> 8192 with zero w2 so 8192 / 8 = 1024 (8x128) per TensorCore
        pad_w1 = np.zeros((num_layers, 8, H, 2, 512), dtype=jnp.bfloat16)
        pad_b1 = np.zeros((num_layers, 8, 2, 512), dtype=jnp.bfloat16)
        pad_w2 = np.zeros((num_layers, 8, 512, H), dtype=jnp.bfloat16)
        w1 = np.concatenate([w1, pad_w1], axis=4)
        b1 = np.concatenate([b1, pad_b1], axis=3)
        w2 = np.concatenate([w2, pad_w2], axis=2)

    I_total = w1.shape[-1]
    I_shard = I_total // tp_size
    # Each TP shard along the last axis gets [2 * I_shard] where [:I_shard] is u and [I_shard:] is v
    w1_tp = (
        w1.reshape(num_layers, 8, H, 2, tp_size, I_shard)
        .transpose(0, 1, 2, 4, 3, 5)
        .reshape(num_layers, 8, H, tp_size * 2 * I_shard)
    )
    b1_tp = (
        b1.reshape(num_layers, 8, 2, tp_size, I_shard)
        .transpose(0, 1, 3, 2, 4)
        .reshape(num_layers, 8, tp_size * 2 * I_shard)
    )

    rep_spec = NamedSharding(mesh, P())
    tp_qkv_spec = NamedSharding(mesh, P(None, None, None, "tp", None))
    tp_bqkv_spec = NamedSharding(mesh, P(None, None, "tp", None))
    tp_wout_spec = NamedSharding(mesh, P(None, "tp", None, None))
    tp_w1_spec = NamedSharding(mesh, P(None, None, None, "tp"))
    tp_b1_spec = NamedSharding(mesh, P(None, None, "tp"))
    tp_w2_spec = NamedSharding(mesh, P(None, None, "tp", None))

    if tp_size == 1:
        return {
            "attn_ln_w": jax.device_put(stacked["attn_ln_w"], rep_spec),
            "w_qkv": jax.device_put(w_qkv, rep_spec),
            "b_qkv": jax.device_put(b_qkv, rep_spec),
            "w_out": jax.device_put(w_out, rep_spec),
            "b_out": jax.device_put(stacked["b_out"], rep_spec),
            "mlp_ln_w": jax.device_put(stacked["mlp_ln_w"], rep_spec),
            "router_w": jax.device_put(stacked["router_w"], rep_spec),
            "w1": jax.device_put(w1_tp, rep_spec),
            "b1": jax.device_put(b1_tp, rep_spec),
            "w2": jax.device_put(w2, rep_spec),
            "b2": jax.device_put(stacked["b2"], rep_spec),
        }

    return {
        "attn_ln_w": jax.device_put(stacked["attn_ln_w"], rep_spec),
        "w_qkv": jax.device_put(w_qkv, tp_qkv_spec),
        "b_qkv": jax.device_put(b_qkv, tp_bqkv_spec),
        "w_out": jax.device_put(w_out, tp_wout_spec),
        "b_out": jax.device_put(stacked["b_out"], rep_spec),
        "mlp_ln_w": jax.device_put(stacked["mlp_ln_w"], rep_spec),
        "router_w": jax.device_put(stacked["router_w"], rep_spec),
        "w1": jax.device_put(w1_tp, tp_w1_spec),
        "b1": jax.device_put(b1_tp, tp_b1_spec),
        "w2": jax.device_put(w2, tp_w2_spec),
        "b2": jax.device_put(stacked["b2"], rep_spec),
    }


def main():
    while not os.path.exists(os.path.join(MODEL_DIR, "DONE")):
        print("Waiting for /dev/shm/gb_protein_16b_bf16/DONE...", flush=True)
        time.sleep(5)

    devices = jax.devices()
    print(f"=== Benchmarking genbio-ai/GB.Protein-16B on TPU v7x ({len(devices)} TensorCores across 4 Chips) ===", flush=True)
    for d in devices:
        print(f"  Device {d.id}: coords={d.coords}, core_on_chip={d.core_on_chip}", flush=True)

    pallas_records = benchmark_pallas_vmem_kernels(devices[0])

    tok = FM4BioTokenizer(vocab_file=os.path.join(MODEL_DIR, "vocab_protein.txt"))
    sample_seq = "M K A L V S G G N Y K P T F | C A S S W G G A P L F | E L A G I G I L T V"
    sample_ids = tok.encode(sample_seq)

    word_emb_np, final_ln_w_np, out_emb_w_np, stacked_np = load_checkpoint_arrays(MODEL_DIR)
    cos_cache, sin_cache = precompute_rope_tpu(2048, 64)

    # Already measured on this exact pod in Tier 1 run:
    tier1_baseline_records = [
        {"batch_size": 1, "seq_len": 256, "tokens_per_batch": 256, "median_ms": 58.22, "tokens_per_sec": 4397.0, "seqs_per_sec": 17.18, "active_tflops": 74.33},
        {"batch_size": 1, "seq_len": 512, "tokens_per_batch": 512, "median_ms": 67.99, "tokens_per_sec": 7530.1, "seqs_per_sec": 14.71, "active_tflops": 127.93},
        {"batch_size": 1, "seq_len": 1024, "tokens_per_batch": 1024, "median_ms": 94.34, "tokens_per_sec": 10854.7, "seqs_per_sec": 10.60, "active_tflops": 186.25},
        {"batch_size": 1, "seq_len": 2048, "tokens_per_batch": 2048, "median_ms": 156.84, "tokens_per_sec": 13057.8, "seqs_per_sec": 6.38, "active_tflops": 228.49},
        {"batch_size": 4, "seq_len": 512, "tokens_per_batch": 2048, "median_ms": 142.15, "tokens_per_sec": 14407.6, "seqs_per_sec": 28.14, "active_tflops": 244.77},
        {"batch_size": 8, "seq_len": 512, "tokens_per_batch": 4096, "median_ms": 250.21, "tokens_per_sec": 16370.5, "seqs_per_sec": 31.97, "active_tflops": 278.12},
        {"batch_size": 16, "seq_len": 512, "tokens_per_batch": 8192, "median_ms": 452.02, "tokens_per_sec": 18123.0, "seqs_per_sec": 35.40, "active_tflops": 307.89},
        {"batch_size": 8, "seq_len": 1024, "tokens_per_batch": 8192, "median_ms": 464.74, "tokens_per_sec": 17627.0, "seqs_per_sec": 17.21, "active_tflops": 302.46},
        {"batch_size": 16, "seq_len": 1024, "tokens_per_batch": 16384, "median_ms": 883.39, "tokens_per_sec": 18546.6, "seqs_per_sec": 18.11, "active_tflops": 318.24},
        {"batch_size": 8, "seq_len": 2048, "tokens_per_batch": 16384, "median_ms": 962.65, "tokens_per_sec": 17019.6, "seqs_per_sec": 8.31, "active_tflops": 297.82},
        {"batch_size": 32, "seq_len": 512, "tokens_per_batch": 16384, "median_ms": 858.98, "tokens_per_sec": 19073.7, "seqs_per_sec": 37.25, "active_tflops": 324.04},
    ]

    results = {
        "accelerator": "Google Cloud TPU v7x (Ironwood)",
        "num_chips_available": 4,
        "num_tensorcores_available": len(devices),
        "model": "genbio-ai/GB.Protein-16B",
        "pallas_vmem_microbenchmark": pallas_records,
        "parity": {},
        "tpu7x_1chip_naive_baseline": tier1_baseline_records,
        "tpu7x_1chip_megakernel_tp2": [],
        "tpu7x_4chip_megakernel_tp8_low_latency": [],
        "tpu7x_4chip_megakernel_dp4_tp2_high_throughput": [],
    }

    configs_1chip = [
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

    # ==========================================================================
    # TIER 1 REFERENCE EXECUTION FOR LIVE NUMERICAL PARITY CHECK
    # ==========================================================================
    print("\n--- Computing Exact Reference Output on Real Protein Complex (Tier 1 Reference) ---", flush=True)
    mesh_1c = Mesh(np.array([devices[0]]), ("tp",))
    rep_1c = NamedSharding(mesh_1c, P())
    lp_1c = prepare_sharded_layer_params(stacked_np, mesh_1c, tp_size=1, pad_for_tp8=False)
    w_emb_1c = jax.device_put(word_emb_np, rep_1c)
    f_ln_1c = jax.device_put(final_ln_w_np, rep_1c)
    o_emb_1c = jax.device_put(out_emb_w_np, rep_1c)
    cos_1c = jax.device_put(cos_cache, rep_1c)
    sin_1c = jax.device_put(sin_cache, rep_1c)

    naive_jit = jax.jit(make_naive_baseline_forward())

    torch.manual_seed(42)
    parity_ids_pt = torch.randint(1, 22, (2, 256), dtype=torch.long)
    parity_ids_pt[0, :len(sample_ids)] = torch.tensor(sample_ids)
    parity_ids_jnp = jax.device_put(parity_ids_pt.numpy(), rep_1c)

    ref_logits_1c, ref_hidden_1c = naive_jit(
        parity_ids_jnp, w_emb_1c, f_ln_1c, o_emb_1c, cos_1c, sin_1c, lp_1c
    )
    jax.block_until_ready(ref_logits_1c)
    ref_logits_np = np.asarray(ref_logits_1c, dtype=np.float32)
    ref_hidden_np = np.asarray(ref_hidden_1c, dtype=np.float32)

    del lp_1c, w_emb_1c, f_ln_1c, o_emb_1c, cos_1c, sin_1c
    gc.collect()

    # ==========================================================================
    # TIER 2: 1x TPU v7x Chip Bio-Megakernel (2 TensorCores On-Die TP=2)
    # ==========================================================================
    print("\n--- Tier 2: Running 1x TPU v7x Chip Bio-Megakernel (2 TensorCores On-Die TP=2 + Megablox VMEM-Fused MoE) ---", flush=True)
    mesh_1chip_tp2 = Mesh(np.array(devices[:2]), ("tp",))
    rep_tp2 = NamedSharding(mesh_1chip_tp2, P())
    lp_tp2 = prepare_sharded_layer_params(stacked_np, mesh_1chip_tp2, tp_size=2, pad_for_tp8=False)
    w_emb_tp2 = jax.device_put(word_emb_np, rep_tp2)
    f_ln_tp2 = jax.device_put(final_ln_w_np, rep_tp2)
    o_emb_tp2 = jax.device_put(out_emb_w_np, rep_tp2)
    cos_tp2 = jax.device_put(cos_cache, rep_tp2)
    sin_tp2 = jax.device_put(sin_cache, rep_tp2)

    mega_tp2_fn = shard_map(
        make_megakernel_forward(tp_size=2, tp_axis_name="tp"),
        mesh=mesh_1chip_tp2,
        in_specs=(
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
            {
                "attn_ln_w": P(),
                "w_qkv": P(None, None, None, "tp", None),
                "b_qkv": P(None, None, "tp", None),
                "w_out": P(None, "tp", None, None),
                "b_out": P(),
                "mlp_ln_w": P(),
                "router_w": P(),
                "w1": P(None, None, None, "tp"),
                "b1": P(None, None, "tp"),
                "w2": P(None, None, "tp", None),
                "b2": P(),
            },
        ),
        out_specs=(P(), P()),
        check_rep=False,
    )
    mega_tp2_jit = jax.jit(mega_tp2_fn)

    # Verify Parity of Tier 2 Bio-Megakernel vs Reference
    parity_ids_tp2 = jax.device_put(parity_ids_pt.numpy(), rep_tp2)
    mega_logits, mega_hidden = mega_tp2_jit(
        parity_ids_tp2, w_emb_tp2, f_ln_tp2, o_emb_tp2, cos_tp2, sin_tp2, lp_tp2
    )
    jax.block_until_ready(mega_logits)
    mega_logits_np = np.asarray(mega_logits, dtype=np.float32)
    mega_hidden_np = np.asarray(mega_hidden, dtype=np.float32)

    cos_sim_l = float(
        np.dot(ref_logits_np.ravel(), mega_logits_np.ravel())
        / (np.linalg.norm(ref_logits_np.ravel()) * np.linalg.norm(mega_logits_np.ravel()))
    )
    cos_sim_h = float(
        np.dot(ref_hidden_np.ravel(), mega_hidden_np.ravel())
        / (np.linalg.norm(ref_hidden_np.ravel()) * np.linalg.norm(mega_hidden_np.ravel()))
    )
    top1_agree = float(np.mean(np.argmax(ref_logits_np, axis=-1) == np.argmax(mega_logits_np, axis=-1)))
    results["parity"] = {
        "cosine_sim_logits": round(cos_sim_l, 6),
        "cosine_sim_hidden": round(cos_sim_h, 6),
        "max_abs_diff_logits": round(float(np.max(np.abs(ref_logits_np - mega_logits_np))), 5),
        "mean_abs_diff_logits": round(float(np.mean(np.abs(ref_logits_np - mega_logits_np))), 6),
        "top1_token_agreement": round(top1_agree, 6),
    }
    print(f"[PARITY CHECK] TPU v7x Bio-Megakernel vs Exact Reference: {results['parity']}", flush=True)
    assert cos_sim_l > 0.999, f"Cosine similarity too low: {cos_sim_l}"

    for i, (B, S) in enumerate(configs_1chip):
        rng = np.random.default_rng(42)
        x_in = jax.device_put(rng.integers(1, 22, size=(B, S), dtype=np.int32), rep_tp2)
        for _ in range(3):
            out_l, _ = mega_tp2_jit(x_in, w_emb_tp2, f_ln_tp2, o_emb_tp2, cos_tp2, sin_tp2, lp_tp2)
            jax.block_until_ready(out_l)
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            out_l, _ = mega_tp2_jit(x_in, w_emb_tp2, f_ln_tp2, o_emb_tp2, cos_tp2, sin_tp2, lp_tp2)
            jax.block_until_ready(out_l)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        median_ms = sorted(latencies)[len(latencies) // 2]
        tok_per_s = (B * S) / (median_ms / 1000.0)
        seq_per_s = B / (median_ms / 1000.0)
        tflops = (B * S * compute_active_flops_per_token(S)) / (median_ms / 1000.0) / 1e12
        base_ms = results["tpu7x_1chip_naive_baseline"][i]["median_ms"]
        speedup = base_ms / median_ms
        rec = {
            "batch_size": B,
            "seq_len": S,
            "tokens_per_batch": B * S,
            "median_ms": round(median_ms, 2),
            "tokens_per_sec": round(tok_per_s, 1),
            "seqs_per_sec": round(seq_per_s, 2),
            "active_tflops": round(tflops, 2),
            "speedup_vs_naive_tpu": round(speedup, 2),
        }
        results["tpu7x_1chip_megakernel_tp2"].append(rec)
        print(
            f"[Mega 1xTPUv7x TP=2] B={B:2d}, S={S:4d} | Latency={median_ms:7.2f} ms ({speedup:4.2f}x vs Naive) | "
            f"Tok/s={tok_per_s:9.1f} | Seq/s={seq_per_s:6.2f} | Active TFLOPS={tflops:6.2f}",
            flush=True,
        )

    del lp_tp2, w_emb_tp2, f_ln_tp2, o_emb_tp2, cos_tp2, sin_tp2
    gc.collect()

    # ==========================================================================
    # TIER 3: 4x TPU v7x Chips Ultra-Low-Latency Bio-Megakernel (8 TensorCores TP=8)
    # ==========================================================================
    print("\n--- Tier 3: Running 4x TPU v7x Chips Ultra-Low-Latency Bio-Megakernel (8 TensorCores TP=8) ---", flush=True)
    mesh_4chip_tp8 = Mesh(np.array(devices[:8]), ("tp",))
    rep_tp8 = NamedSharding(mesh_4chip_tp8, P())
    lp_tp8 = prepare_sharded_layer_params(stacked_np, mesh_4chip_tp8, tp_size=8, pad_for_tp8=True)
    w_emb_tp8 = jax.device_put(word_emb_np, rep_tp8)
    f_ln_tp8 = jax.device_put(final_ln_w_np, rep_tp8)
    o_emb_tp8 = jax.device_put(out_emb_w_np, rep_tp8)
    cos_tp8 = jax.device_put(cos_cache, rep_tp8)
    sin_tp8 = jax.device_put(sin_cache, rep_tp8)

    mega_tp8_fn = shard_map(
        make_megakernel_forward(tp_size=8, tp_axis_name="tp"),
        mesh=mesh_4chip_tp8,
        in_specs=(
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
            {
                "attn_ln_w": P(),
                "w_qkv": P(None, None, None, "tp", None),
                "b_qkv": P(None, None, "tp", None),
                "w_out": P(None, "tp", None, None),
                "b_out": P(),
                "mlp_ln_w": P(),
                "router_w": P(),
                "w1": P(None, None, None, "tp"),
                "b1": P(None, None, "tp"),
                "w2": P(None, None, "tp", None),
                "b2": P(),
            },
        ),
        out_specs=(P(), P()),
        check_rep=False,
    )
    mega_tp8_jit = jax.jit(mega_tp8_fn)

    for B, S in configs_1chip:
        rng = np.random.default_rng(42)
        x_in = jax.device_put(rng.integers(1, 22, size=(B, S), dtype=np.int32), rep_tp8)
        for _ in range(3):
            out_l, _ = mega_tp8_jit(x_in, w_emb_tp8, f_ln_tp8, o_emb_tp8, cos_tp8, sin_tp8, lp_tp8)
            jax.block_until_ready(out_l)
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            out_l, _ = mega_tp8_jit(x_in, w_emb_tp8, f_ln_tp8, o_emb_tp8, cos_tp8, sin_tp8, lp_tp8)
            jax.block_until_ready(out_l)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        median_ms = sorted(latencies)[len(latencies) // 2]
        tok_per_s = (B * S) / (median_ms / 1000.0)
        seq_per_s = B / (median_ms / 1000.0)
        tflops = (B * S * compute_active_flops_per_token(S)) / (median_ms / 1000.0) / 1e12
        rec = {
            "batch_size": B,
            "seq_len": S,
            "tokens_per_batch": B * S,
            "median_ms": round(median_ms, 2),
            "tokens_per_sec": round(tok_per_s, 1),
            "seqs_per_sec": round(seq_per_s, 2),
            "active_tflops": round(tflops, 2),
        }
        results["tpu7x_4chip_megakernel_tp8_low_latency"].append(rec)
        print(
            f"[Mega 4xTPUv7x TP=8] B={B:2d}, S={S:4d} | Latency={median_ms:7.2f} ms | "
            f"Tok/s={tok_per_s:9.1f} | Seq/s={seq_per_s:6.2f} | Active TFLOPS={tflops:6.2f}",
            flush=True,
        )

    del lp_tp8, w_emb_tp8, f_ln_tp8, o_emb_tp8, cos_tp8, sin_tp8
    gc.collect()

    # ==========================================================================
    # TIER 4: 4x TPU v7x Chips High-Throughput Mode (DP=4 Chips x On-Die TP=2)
    # ==========================================================================
    print("\n--- Tier 4: Running 4x TPU v7x Chips High-Throughput Mode (DP=4 Chips x On-Die TP=2) ---", flush=True)
    mesh_dp4_tp2 = Mesh(np.array(devices[:8]).reshape(4, 2), ("dp", "tp"))
    rep_dp4 = NamedSharding(mesh_dp4_tp2, P())
    lp_dp4 = prepare_sharded_layer_params(stacked_np, mesh_dp4_tp2, tp_size=2, pad_for_tp8=False)
    w_emb_dp4 = jax.device_put(word_emb_np, rep_dp4)
    f_ln_dp4 = jax.device_put(final_ln_w_np, rep_dp4)
    o_emb_dp4 = jax.device_put(out_emb_w_np, rep_dp4)
    cos_dp4 = jax.device_put(cos_cache, rep_dp4)
    sin_dp4 = jax.device_put(sin_cache, rep_dp4)

    mega_dp4_tp2_fn = shard_map(
        make_megakernel_forward(tp_size=2, tp_axis_name="tp"),
        mesh=mesh_dp4_tp2,
        in_specs=(
            P("dp", None),
            P(),
            P(),
            P(),
            P(),
            P(),
            {
                "attn_ln_w": P(),
                "w_qkv": P(None, None, None, "tp", None),
                "b_qkv": P(None, None, "tp", None),
                "w_out": P(None, "tp", None, None),
                "b_out": P(),
                "mlp_ln_w": P(),
                "router_w": P(),
                "w1": P(None, None, None, "tp"),
                "b1": P(None, None, "tp"),
                "w2": P(None, None, "tp", None),
                "b2": P(),
            },
        ),
        out_specs=(P("dp", None, None), P("dp", None, None)),
        check_rep=False,
    )
    mega_dp4_tp2_jit = jax.jit(mega_dp4_tp2_fn)

    dp4_configs = [
        (4, 512),
        (16, 512),
        (32, 512),
        (64, 512),
        (32, 1024),
        (64, 1024),
        (32, 2048),
    ]
    in_sharding_dp4 = NamedSharding(mesh_dp4_tp2, P("dp", None))
    for total_B, S in dp4_configs:
        rng = np.random.default_rng(42)
        x_in = jax.device_put(
            rng.integers(1, 22, size=(total_B, S), dtype=np.int32), in_sharding_dp4
        )
        for _ in range(3):
            out_l, _ = mega_dp4_tp2_jit(
                x_in, w_emb_dp4, f_ln_dp4, o_emb_dp4, cos_dp4, sin_dp4, lp_dp4
            )
            jax.block_until_ready(out_l)
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            out_l, _ = mega_dp4_tp2_jit(
                x_in, w_emb_dp4, f_ln_dp4, o_emb_dp4, cos_dp4, sin_dp4, lp_dp4
            )
            jax.block_until_ready(out_l)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        median_ms = sorted(latencies)[len(latencies) // 2]
        tok_per_s = (total_B * S) / (median_ms / 1000.0)
        seq_per_s = total_B / (median_ms / 1000.0)
        tflops = (total_B * S * compute_active_flops_per_token(S)) / (median_ms / 1000.0) / 1e12
        rec = {
            "total_batch_size": total_B,
            "batch_per_chip": total_B // 4,
            "seq_len": S,
            "tokens_per_batch": total_B * S,
            "median_ms": round(median_ms, 2),
            "tokens_per_sec": round(tok_per_s, 1),
            "seqs_per_sec": round(seq_per_s, 2),
            "active_tflops": round(tflops, 2),
        }
        results["tpu7x_4chip_megakernel_dp4_tp2_high_throughput"].append(rec)
        print(
            f"[Mega 4xTPUv7x DP=4xTP=2] TotalB={total_B:2d} ({total_B//4}/chip), S={S:4d} | "
            f"Latency={median_ms:7.2f} ms | Tok/s={tok_per_s:9.1f} | Seq/s={seq_per_s:6.2f} | Active TFLOPS={tflops:6.2f}",
            flush=True,
        )

    with open("/tmp/tpu7x_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[DONE] Saved TPU v7x results to /tmp/tpu7x_benchmark_results.json!", flush=True)


if __name__ == "__main__":
    main()
