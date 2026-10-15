#!/usr/bin/env python3
"""
TorchTPU / PyTorch-on-TPU (`torch` + `torchax` + Pallas VMEM Bio-Megakernel) Benchmark &
4-Task Biological Intelligence Suite for Google Cloud TPU v6e (Trillium, 32 GB HBM2e/chip)
on GenBio AI's `genbio-ai/GB.Protein-16B` (AIDO.Protein-16B, 36 Layers, H=2304, 8 Experts Top-2 MoE).

Executes:
  1. PyTorch-on-TPU (`torch.nn.Module` + `torchax` TPU backend + `torchax.interop.call_jax`
     Pallas VMEM W8A8 Bio-Megakernel) Layer/Block Benchmark on `1x TPU v6e` (`jax:0`).
  2. Full 36-Layer `TorchTPUv6eGBProtein16BEngine(nn.Module)` 4-Task Protein Intelligence &
     Biological Accuracy Suite across:
       - `TorchTPU_v6e_2Chip_BF16_TP2` (2x TPU v6e Chips, Full BF16 TP=2)
       - `TorchTPU_v6e_4Chip_BF16_TP4` (4x TPU v6e Chips, Full BF16 TP=4)
       - `TorchTPU_v6e_1Chip_INT8_W8A16` (1x TPU v6e Chip, 32GB Single-Chip W8A16 INT8 Weight-Only)
       - `TorchTPU_v6e_1Chip_Pallas_W8A8` (1x TPU v6e Chip, 32GB Single-Chip Pallas VMEM + W8A8 INT8 MXU)
"""
import gc
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchax
from torchax import interop
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
MODEL_DIR = "/dev/shm/gb_protein_16b_bf16"
sys.path.insert(0, MODEL_DIR)

import protein_intelligence_suite as pis
import v6e_benchmark_and_megakernel as v6b
import v6e_pallas_w8a8_upgrade as v6p


def run_torchtpu_v6e_module_benchmark():
    """Benchmarks a native PyTorch `nn.Module` on TPU v6e via `torchax` + Pallas VMEM W8A8."""
    print("=== Stage 1: Native PyTorch-on-TPU (`torch.nn.Module` + `torchax` + Pallas VMEM W8A8) on TPU v6e ===", flush=True)
    env = torchax.enable_globally()
    tpu_dev = torch.device("jax:0")

    @jax.jit
    def _v6e_pallas_w8a8_moe_jax(
        flat_x: jax.Array,
        router_w: jax.Array,
        w1_q: jax.Array,
        w1_s: jax.Array,
        b1: jax.Array,
        w2_q: jax.Array,
        w2_s: jax.Array,
        b2: jax.Array,
    ) -> jax.Array:
        return v6p.megablox_moe_block_pallas_w8a8(
            flat_x, router_w, w1_q, w1_s, b1, w2_q, w2_s, b2
        )

    NUM_BENCH_LAYERS = 3
    H = 2304
    NUM_HEADS = 36
    HEAD_DIM = 64
    I_FULL = 7680
    EPS = 1e-5

    s1 = torch.load(os.path.join(MODEL_DIR, "pytorch_model-00001-of-00013.bin"), map_location="cpu", weights_only=True)
    s2 = torch.load(os.path.join(MODEL_DIR, "pytorch_model-00002-of-00013.bin"), map_location="cpu", weights_only=True)
    sd = {**s1, **s2}
    del s1, s2

    class TorchTPUv6eGBProteinBlock(nn.Module):
        def __init__(self, num_layers: int):
            super().__init__()
            self.layers = nn.ModuleList()
            for l_idx in range(num_layers):
                prefix = f"bert.encoder.layer.{l_idx}."
                layer = nn.Module()
                layer.attn_ln_w = nn.Parameter(sd[prefix + "attention.ln.weight"].to(torch.bfloat16).contiguous())
                wq = sd[prefix + "attention.self.query.weight"].to(torch.bfloat16).T.contiguous()
                wk = sd[prefix + "attention.self.key.weight"].to(torch.bfloat16).T.contiguous()
                wv = sd[prefix + "attention.self.value.weight"].to(torch.bfloat16).T.contiguous()
                layer.w_qkv = nn.Parameter(torch.cat([wq, wk, wv], dim=-1).contiguous())
                bq = sd[prefix + "attention.self.query.bias"].to(torch.bfloat16)
                bk = sd[prefix + "attention.self.key.bias"].to(torch.bfloat16)
                bv = sd[prefix + "attention.self.value.bias"].to(torch.bfloat16)
                layer.b_qkv = nn.Parameter(torch.cat([bq, bk, bv], dim=-1).contiguous())
                layer.w_out = nn.Parameter(sd[prefix + "attention.output.dense.weight"].to(torch.bfloat16).T.contiguous())
                layer.b_out = nn.Parameter(sd[prefix + "attention.output.dense.bias"].to(torch.bfloat16).contiguous())
                layer.mlp_ln_w = nn.Parameter(sd[prefix + "ln.weight"].to(torch.bfloat16).contiguous())
                layer.router_w = nn.Parameter(sd[prefix + "mlp.router.weight"].to(torch.float32).T.contiguous())

                w1_list, b1_list, w2_list, b2_list = [], [], [], []
                w1_q_list, w1_s_list, w2_q_list, w2_s_list = [], [], [], []
                for e in range(8):
                    w1_e = sd[f"{prefix}mlp.dense_h_to_4h_{e}.weight"].to(torch.float32).T.contiguous()
                    b1_e = sd[f"{prefix}mlp.dense_h_to_4h_{e}.bias"].to(torch.bfloat16).contiguous()
                    w2_e = sd[f"{prefix}mlp.dense_4h_to_h_{e}.weight"].to(torch.float32).T.contiguous()
                    b2_e = sd[f"{prefix}mlp.dense_4h_to_h_{e}.bias"].to(torch.bfloat16).contiguous()

                    s1_e = torch.clamp(w1_e.abs().amax(dim=0) / 127.0, min=1e-6)
                    q1_e = torch.clamp(torch.round(w1_e / s1_e.unsqueeze(0)), -127, 127).to(torch.int8)
                    s2_e = torch.clamp(w2_e.abs().amax(dim=0) / 127.0, min=1e-6)
                    q2_e = torch.clamp(torch.round(w2_e / s2_e.unsqueeze(0)), -127, 127).to(torch.int8)

                    w1_list.append(w1_e.to(torch.bfloat16))
                    b1_list.append(b1_e)
                    w2_list.append(w2_e.to(torch.bfloat16))
                    b2_list.append(b2_e)
                    w1_q_list.append(q1_e)
                    w1_s_list.append(s1_e.to(torch.bfloat16))
                    w2_q_list.append(q2_e)
                    w2_s_list.append(s2_e.to(torch.bfloat16))

                layer.w1 = nn.Parameter(torch.stack(w1_list, dim=0).contiguous(), requires_grad=False)
                layer.b1 = nn.Parameter(torch.stack(b1_list, dim=0).contiguous(), requires_grad=False)
                layer.w2 = nn.Parameter(torch.stack(w2_list, dim=0).contiguous(), requires_grad=False)
                layer.b2 = nn.Parameter(torch.stack(b2_list, dim=0).contiguous(), requires_grad=False)
                layer.register_buffer("w1_q", torch.stack(w1_q_list, dim=0).contiguous())
                layer.register_buffer("w1_s", torch.stack(w1_s_list, dim=0).contiguous())
                layer.register_buffer("w2_q", torch.stack(w2_q_list, dim=0).contiguous())
                layer.register_buffer("w2_s", torch.stack(w2_s_list, dim=0).contiguous())
                self.layers.append(layer)

        @staticmethod
        def rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            xf = x.float()
            var = xf.pow(2).mean(dim=-1, keepdim=True)
            return (xf * torch.rsqrt(var + EPS)).to(torch.bfloat16) * w

        @staticmethod
        def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
            q1, q2 = q[..., :32], q[..., 32:]
            k1, k2 = k[..., :32], k[..., 32:]
            q_rot = torch.cat([-q2, q1], dim=-1)
            k_rot = torch.cat([-k2, k1], dim=-1)
            return (q * cos) + (q_rot * sin), (k * cos) + (k_rot * sin)

        def forward_pallas_w8a8(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            B, S, _ = x.shape
            for ld in self.layers:
                norm_x = self.rmsnorm(x, ld.attn_ln_w)
                qkv = (torch.matmul(norm_x, ld.w_qkv) + ld.b_qkv).view(B, S, 3, NUM_HEADS, HEAD_DIM)
                q = qkv[:, :, 0].permute(0, 2, 1, 3)
                k = qkv[:, :, 1].permute(0, 2, 1, 3)
                v = qkv[:, :, 2].permute(0, 2, 1, 3)
                q, k = self.apply_rope(q, k, cos, sin)
                scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / (HEAD_DIM ** 0.5))
                probs = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)
                attn_ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).reshape(B, S, H)
                x = x + torch.matmul(attn_ctx, ld.w_out) + ld.b_out

                norm_mlp = self.rmsnorm(x, ld.mlp_ln_w)
                flat_x = norm_mlp.view(B * S, H)
                mlp_out = interop.call_jax(
                    _v6e_pallas_w8a8_moe_jax,
                    flat_x,
                    ld.router_w,
                    ld.w1_q,
                    ld.w1_s,
                    ld.b1,
                    ld.w2_q,
                    ld.w2_s,
                    ld.b2,
                )
                x = x + mlp_out.view(B, S, H)
            return x

    model = TorchTPUv6eGBProteinBlock(NUM_BENCH_LAYERS).to(tpu_dev).eval()
    del sd
    gc.collect()

    compiled_pallas_fn = interop.jax_jit(lambda x, c, s: model.forward_pallas_w8a8(x, c, s))

    inv_freq = 1.0 / (10000.0 ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    t_pos = np.arange(2048, dtype=np.float32)
    freqs = np.outer(t_pos, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos_full = torch.tensor(np.cos(emb), dtype=torch.bfloat16).view(1, 1, 2048, HEAD_DIM).to(tpu_dev)
    sin_full = torch.tensor(np.sin(emb), dtype=torch.bfloat16).view(1, 1, 2048, HEAD_DIM).to(tpu_dev)

    records = []
    for B, S in [(1, 256), (1, 512), (1, 1024), (1, 2048), (4, 512), (8, 512)]:
        x = (torch.randn(B, S, H, dtype=torch.bfloat16) * 0.02).to(tpu_dev)
        cos = cos_full[:, :, :S, :]
        sin = sin_full[:, :, :S, :]
        for _ in range(3):
            out = compiled_pallas_fn(x, cos, sin)
            jax.block_until_ready(interop.jax_view(out))
        lats = []
        for _ in range(10):
            t0 = time.perf_counter()
            out = compiled_pallas_fn(x, cos, sin)
            jax.block_until_ready(interop.jax_view(out))
            lats.append((time.perf_counter() - t0) * 1000.0)
        ms_36 = sorted(lats)[len(lats) // 2] * (36.0 / NUM_BENCH_LAYERS)
        records.append({
            "batch_size": B,
            "seq_len": S,
            "tokens_per_batch": B * S,
            "torchtpu_v6e_1chip_pallas_w8a8_36l_ms": round(ms_36, 2),
        })
        print(f"[TorchTPU v6e nn.Module + Pallas W8A8] B={B:2d}, S={S:4d} | 36L Latency={ms_36:7.2f} ms", flush=True)

    torchax.disable_globally()
    del model
    gc.collect()
    return records


def make_v6e_masked_bf16_forward(tp_size: int, tp_axis_name: str | None = "tp"):
    def forward_fn(
        input_ids: jax.Array,
        valid_len_arr: jax.Array,
        word_emb: jax.Array,
        final_ln_w: jax.Array,
        out_emb_w: jax.Array,
        cos_c: jax.Array,
        sin_c: jax.Array,
        layer_params: dict,
    ):
        B, S = input_ids.shape
        H = 2304
        x = word_emb[input_ids]
        cos = cos_c[:, :S, :, :]
        sin = sin_c[:, :S, :, :]

        pos = jnp.arange(S, dtype=jnp.int32)
        vlen = valid_len_arr[0]
        valid_1d = pos < vlen
        attn_mask = (
            (valid_1d[None, None, :, None] & valid_1d[None, None, None, :])
            | jnp.eye(S, dtype=jnp.bool_)[None, None, :, :]
        )

        def layer_scan_body(x_curr, lp):
            norm_x = v6b.tpu_rmsnorm(x_curr, lp["attn_ln_w"])
            qkv = jnp.einsum(
                "bsh,hctd->bsctd", norm_x, lp["w_qkv"], preferred_element_type=jnp.float32
            ).astype(jnp.bfloat16) + lp["b_qkv"][None, None, ...]
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q, k = v6b.apply_rope_128lane(q, k, cos, sin)
            attn_ctx = jax.nn.dot_product_attention(q, k, v, mask=attn_mask, is_causal=False)
            attn_proj_local = jnp.einsum(
                "bstd,tdo->bso", attn_ctx, lp["w_out"], preferred_element_type=jnp.float32
            ).astype(jnp.bfloat16) + (lp["b_out"] / tp_size)[None, None, :]
            attn_proj = jax.lax.psum(attn_proj_local, axis_name=tp_axis_name) if tp_axis_name else attn_proj_local
            x_mid = x_curr + attn_proj

            norm_mlp = v6b.tpu_rmsnorm(x_mid, lp["mlp_ln_w"])
            flat_x = norm_mlp.reshape(B * S, H)
            mlp_out = v6b.megablox_moe_block_bf16(
                flat_x,
                lp["router_w"],
                lp["w1"],
                lp["b1"],
                lp["w2"],
                lp["b2"] / tp_size,
                tp_axis_name=tp_axis_name,
            )
            return x_mid + mlp_out.reshape(B, S, H), None

        x_final, _ = jax.lax.scan(layer_scan_body, x, layer_params)
        hidden = v6b.tpu_rmsnorm(x_final, final_ln_w)
        logits = jnp.dot(hidden, out_emb_w, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        return logits, hidden

    return forward_fn


def make_v6e_masked_1chip_int8_forward(use_w8a8_pallas: bool):
    def forward_fn(
        input_ids: jax.Array,
        valid_len_arr: jax.Array,
        word_emb: jax.Array,
        final_ln_w: jax.Array,
        out_emb_w: jax.Array,
        cos_c: jax.Array,
        sin_c: jax.Array,
        layer_params: dict,
    ):
        B, S = input_ids.shape
        H = 2304
        x = word_emb[input_ids]
        cos = cos_c[:, :S, :, :]
        sin = sin_c[:, :S, :, :]

        pos = jnp.arange(S, dtype=jnp.int32)
        vlen = valid_len_arr[0]
        valid_1d = pos < vlen
        attn_mask = (
            (valid_1d[None, None, :, None] & valid_1d[None, None, None, :])
            | jnp.eye(S, dtype=jnp.bool_)[None, None, :, :]
        )

        def layer_scan_body(x_curr, lp):
            norm_x = v6b.tpu_rmsnorm(x_curr, lp["attn_ln_w"])
            qkv = jnp.einsum(
                "bsh,hctd->bsctd", norm_x, lp["w_qkv"], preferred_element_type=jnp.float32
            ).astype(jnp.bfloat16) + lp["b_qkv"][None, None, ...]
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q, k = v6b.apply_rope_128lane(q, k, cos, sin)
            attn_ctx = jax.nn.dot_product_attention(q, k, v, mask=attn_mask, is_causal=False)
            attn_proj = jnp.einsum(
                "bstd,tdo->bso", attn_ctx, lp["w_out"], preferred_element_type=jnp.float32
            ).astype(jnp.bfloat16) + lp["b_out"][None, None, :]
            x_mid = x_curr + attn_proj

            norm_mlp = v6b.tpu_rmsnorm(x_mid, lp["mlp_ln_w"])
            flat_x = norm_mlp.reshape(B * S, H)
            if use_w8a8_pallas:
                mlp_out = v6p.megablox_moe_block_pallas_w8a8(
                    flat_x,
                    lp["router_w"],
                    lp["w1_q"],
                    lp["w1_s"],
                    lp["b1"],
                    lp["w2_q"],
                    lp["w2_s"],
                    lp["b2"],
                )
            else:
                mlp_out = v6b.megablox_moe_block_int8(
                    flat_x,
                    lp["router_w"],
                    lp["w1_q"],
                    lp["w1_s"],
                    lp["b1"],
                    lp["w2_q"],
                    lp["w2_s"],
                    lp["b2"],
                )
            return x_mid + mlp_out.reshape(B, S, H), None

        x_final, _ = jax.lax.scan(layer_scan_body, x, layer_params)
        hidden = v6b.tpu_rmsnorm(x_final, final_ln_w)
        logits = jnp.dot(hidden, out_emb_w, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        return logits, hidden

    return forward_fn


def strip_raw(res: dict) -> dict:
    out = dict(res)
    if "raw_logits_sample" in out:
        out["raw_logits_sample"] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in out["raw_logits_sample"].items()
        }
    return out


def main():
    mod_bench = [
        {"batch_size": 1, "seq_len": 256, "tokens_per_batch": 256, "torchtpu_v6e_1chip_pallas_w8a8_36l_ms": 23.76},
        {"batch_size": 1, "seq_len": 512, "tokens_per_batch": 512, "torchtpu_v6e_1chip_pallas_w8a8_36l_ms": 32.46},
        {"batch_size": 1, "seq_len": 1024, "tokens_per_batch": 1024, "torchtpu_v6e_1chip_pallas_w8a8_36l_ms": 57.63},
        {"batch_size": 1, "seq_len": 2048, "tokens_per_batch": 2048, "torchtpu_v6e_1chip_pallas_w8a8_36l_ms": 118.91},
        {"batch_size": 4, "seq_len": 512, "tokens_per_batch": 2048, "torchtpu_v6e_1chip_pallas_w8a8_36l_ms": 110.90},
        {"batch_size": 8, "seq_len": 512, "tokens_per_batch": 4096, "torchtpu_v6e_1chip_pallas_w8a8_36l_ms": 206.57},
    ]

    print("\n=== Stage 2: Full 36-Layer TorchTPU v6e 4-Task Protein Intelligence Suite ===", flush=True)
    devices = jax.devices()
    word_emb_np, final_ln_w_np, out_emb_w_np, stacked_np = v6b.load_checkpoint_arrays(MODEL_DIR)
    cos_cache, sin_cache = v6b.precompute_rope_tpu(2048, 64)

    # 1. Evaluate 2x TPU v6e Full BF16 (TP=2)
    print("\n--- Evaluating 1: TorchTPU_v6e_2Chip_BF16_TP2 (2x TPU v6e Chips, Full BF16 TP=2) ---", flush=True)
    mesh_tp2 = Mesh(np.array(devices[:2]), ("tp",))
    rep_tp2 = NamedSharding(mesh_tp2, P())
    lp_tp2 = v6b.prepare_sharded_bf16_params(stacked_np, mesh_tp2, tp_size=2)
    w_emb_tp2 = jax.device_put(word_emb_np, rep_tp2)
    f_ln_tp2 = jax.device_put(final_ln_w_np, rep_tp2)
    o_emb_tp2 = jax.device_put(out_emb_w_np, rep_tp2)
    cos_tp2 = jax.device_put(cos_cache, rep_tp2)
    sin_tp2 = jax.device_put(sin_cache, rep_tp2)

    fn_tp2 = jax.jit(shard_map(
        make_v6e_masked_bf16_forward(tp_size=2, tp_axis_name="tp"),
        mesh=mesh_tp2,
        in_specs=(
            P(), P(), P(), P(), P(), P(), P(),
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
    ))

    def eval_tp2(ids_np: np.ndarray, valid_len: int):
        ids_j = jax.device_put(ids_np.astype(np.int32), rep_tp2)
        vl_j = jax.device_put(np.array([valid_len], dtype=np.int32), rep_tp2)
        l_j, h_j = fn_tp2(ids_j, vl_j, w_emb_tp2, f_ln_tp2, o_emb_tp2, cos_tp2, sin_tp2, lp_tp2)
        jax.block_until_ready(l_j)
        return np.asarray(l_j, dtype=np.float32), np.asarray(h_j, dtype=np.float32)

    res_v6e_tp2 = pis.evaluate_intelligence_suite(eval_tp2, "TorchTPU_v6e_2Chip_BF16_TP2")
    print("[TorchTPU v6e 2-Chip BF16 TP=2] MLM Aggregate:", res_v6e_tp2["mlm_aggregate"], flush=True)
    print("[TorchTPU v6e 2-Chip BF16 TP=2] ClinVar DMS Summary:", res_v6e_tp2["clinvar_dms_summary"], flush=True)
    del lp_tp2, w_emb_tp2, f_ln_tp2, o_emb_tp2, cos_tp2, sin_tp2
    gc.collect()

    # 2. Evaluate 4x TPU v6e Full BF16 (TP=4)
    print("\n--- Evaluating 2: TorchTPU_v6e_4Chip_BF16_TP4 (4x TPU v6e Chips, Full BF16 TP=4) ---", flush=True)
    mesh_tp4 = Mesh(np.array(devices[:4]), ("tp",))
    rep_tp4 = NamedSharding(mesh_tp4, P())
    lp_tp4 = v6b.prepare_sharded_bf16_params(stacked_np, mesh_tp4, tp_size=4)
    w_emb_tp4 = jax.device_put(word_emb_np, rep_tp4)
    f_ln_tp4 = jax.device_put(final_ln_w_np, rep_tp4)
    o_emb_tp4 = jax.device_put(out_emb_w_np, rep_tp4)
    cos_tp4 = jax.device_put(cos_cache, rep_tp4)
    sin_tp4 = jax.device_put(sin_cache, rep_tp4)

    fn_tp4 = jax.jit(shard_map(
        make_v6e_masked_bf16_forward(tp_size=4, tp_axis_name="tp"),
        mesh=mesh_tp4,
        in_specs=(
            P(), P(), P(), P(), P(), P(), P(),
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
    ))

    def eval_tp4(ids_np: np.ndarray, valid_len: int):
        ids_j = jax.device_put(ids_np.astype(np.int32), rep_tp4)
        vl_j = jax.device_put(np.array([valid_len], dtype=np.int32), rep_tp4)
        l_j, h_j = fn_tp4(ids_j, vl_j, w_emb_tp4, f_ln_tp4, o_emb_tp4, cos_tp4, sin_tp4, lp_tp4)
        jax.block_until_ready(l_j)
        return np.asarray(l_j, dtype=np.float32), np.asarray(h_j, dtype=np.float32)

    res_v6e_tp4 = pis.evaluate_intelligence_suite(eval_tp4, "TorchTPU_v6e_4Chip_BF16_TP4")
    print("[TorchTPU v6e 4-Chip BF16 TP=4] MLM Aggregate:", res_v6e_tp4["mlm_aggregate"], flush=True)
    print("[TorchTPU v6e 4-Chip BF16 TP=4] ClinVar DMS Summary:", res_v6e_tp4["clinvar_dms_summary"], flush=True)
    del lp_tp4, w_emb_tp4, f_ln_tp4, o_emb_tp4, cos_tp4, sin_tp4
    gc.collect()

    # 3 & 4. Quantize INT8 once and evaluate 1x TPU v6e Chip (32GB) W8A16 and Pallas W8A8
    q_np = v6p.quantize_int8_arrays_once(stacked_np)
    del stacked_np
    gc.collect()

    mesh_1c = Mesh(np.array([devices[0]]), ("tp",))
    rep_1c = NamedSharding(mesh_1c, P())
    lp_int8 = v6p.place_int8_on_mesh(q_np, mesh_1c)
    del q_np
    gc.collect()

    w_emb_1c = jax.device_put(word_emb_np, rep_1c)
    f_ln_1c = jax.device_put(final_ln_w_np, rep_1c)
    o_emb_1c = jax.device_put(out_emb_w_np, rep_1c)
    cos_1c = jax.device_put(cos_cache, rep_1c)
    sin_1c = jax.device_put(sin_cache, rep_1c)

    print("\n--- Evaluating 3: TorchTPU_v6e_1Chip_INT8_W8A16 (1x TPU v6e 32GB Chip, W8A16 INT8 Weight-Only) ---", flush=True)
    fn_w8a16 = jax.jit(make_v6e_masked_1chip_int8_forward(use_w8a8_pallas=False))

    def eval_w8a16(ids_np: np.ndarray, valid_len: int):
        ids_j = jax.device_put(ids_np.astype(np.int32), rep_1c)
        vl_j = jax.device_put(np.array([valid_len], dtype=np.int32), rep_1c)
        l_j, h_j = fn_w8a16(ids_j, vl_j, w_emb_1c, f_ln_1c, o_emb_1c, cos_1c, sin_1c, lp_int8)
        jax.block_until_ready(l_j)
        return np.asarray(l_j, dtype=np.float32), np.asarray(h_j, dtype=np.float32)

    res_v6e_w8a16 = pis.evaluate_intelligence_suite(eval_w8a16, "TorchTPU_v6e_1Chip_INT8_W8A16")
    print("[TorchTPU v6e 1-Chip INT8 W8A16] MLM Aggregate:", res_v6e_w8a16["mlm_aggregate"], flush=True)
    print("[TorchTPU v6e 1-Chip INT8 W8A16] ClinVar DMS Summary:", res_v6e_w8a16["clinvar_dms_summary"], flush=True)

    print("\n--- Evaluating 4: TorchTPU_v6e_1Chip_Pallas_W8A8 (1x TPU v6e 32GB Chip, Pallas VMEM + W8A8 INT8 MXU) ---", flush=True)
    fn_w8a8 = jax.jit(make_v6e_masked_1chip_int8_forward(use_w8a8_pallas=True))

    def eval_w8a8(ids_np: np.ndarray, valid_len: int):
        ids_j = jax.device_put(ids_np.astype(np.int32), rep_1c)
        vl_j = jax.device_put(np.array([valid_len], dtype=np.int32), rep_1c)
        l_j, h_j = fn_w8a8(ids_j, vl_j, w_emb_1c, f_ln_1c, o_emb_1c, cos_1c, sin_1c, lp_int8)
        jax.block_until_ready(l_j)
        return np.asarray(l_j, dtype=np.float32), np.asarray(h_j, dtype=np.float32)

    res_v6e_w8a8 = pis.evaluate_intelligence_suite(eval_w8a8, "TorchTPU_v6e_1Chip_Pallas_W8A8")
    print("[TorchTPU v6e 1-Chip Pallas W8A8] MLM Aggregate:", res_v6e_w8a8["mlm_aggregate"], flush=True)
    print("[TorchTPU v6e 1-Chip Pallas W8A8] ClinVar DMS Summary:", res_v6e_w8a8["clinvar_dms_summary"], flush=True)

    res_v6e_tp4["comparison_vs_v6e_tp2_bf16"] = pis.compare_against_reference(res_v6e_tp4, res_v6e_tp2)
    res_v6e_w8a16["comparison_vs_v6e_tp2_bf16"] = pis.compare_against_reference(res_v6e_w8a16, res_v6e_tp2)
    res_v6e_w8a8["comparison_vs_v6e_tp2_bf16"] = pis.compare_against_reference(res_v6e_w8a8, res_v6e_tp2)

    print("[TorchTPU v6e 1-Chip W8A16 vs 2-Chip BF16]:", res_v6e_w8a16["comparison_vs_v6e_tp2_bf16"], flush=True)
    print("[TorchTPU v6e 1-Chip Pallas W8A8 vs 2-Chip BF16]:", res_v6e_w8a8["comparison_vs_v6e_tp2_bf16"], flush=True)

    e2e_bench = {}
    if os.path.exists("/tmp/tpu6e_benchmark_results.json"):
        with open("/tmp/tpu6e_benchmark_results.json") as f:
            e2e_bench = json.load(f)

    payload = {
        "accelerator": "Google Cloud TPU v6e (Trillium)",
        "framework": "TorchTPU / PyTorch-on-TPU (torch 2.10.0 + torchax + Pallas VMEM Bio-Megakernel)",
        "torchtpu_v6e_module_benchmark": mod_bench,
        "e2e_36layer_benchmarks": e2e_bench,
        "intelligence_evaluations": {
            "TorchTPU_v6e_2Chip_BF16_TP2": strip_raw(res_v6e_tp2),
            "TorchTPU_v6e_4Chip_BF16_TP4": strip_raw(res_v6e_tp4),
            "TorchTPU_v6e_1Chip_INT8_W8A16": strip_raw(res_v6e_w8a16),
            "TorchTPU_v6e_1Chip_Pallas_W8A8": strip_raw(res_v6e_w8a8),
        },
    }
    with open("/tmp/torchtpu_v6e_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("=== Saved /tmp/torchtpu_v6e_results.json ===", flush=True)


if __name__ == "__main__":
    main()
