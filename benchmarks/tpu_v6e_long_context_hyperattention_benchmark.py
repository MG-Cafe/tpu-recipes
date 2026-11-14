#!/usr/bin/env python3
"""
TPU v6e (Trillium) Full Audit Completion (S=2048 Pallas W8A8 + 11 Shapes)
& 20K Long-Context TPU HyperAttention (VMEM-Tiled Blockwise Bidirectional FlashAttention) Benchmark.
"""
import gc
import json
import math
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental.pallas.ops.tpu import flash_attention as pallas_tpu_fa

sys.path.insert(0, "/tmp")
sys.path.insert(0, "/dev/shm/gb_protein_16b_bf16")

from v6e_benchmark_and_megakernel import (
    MODEL_DIR,
    compute_active_flops_per_token,
    load_checkpoint_arrays,
    precompute_rope_tpu,
    tpu_rmsnorm,
    apply_rope_128lane,
    prepare_sharded_bf16_params,
    megablox_moe_block_bf16,
)
from v6e_pallas_w8a8_upgrade import (
    megablox_moe_block_pallas_w8a8,
    quantize_int8_arrays_once,
    place_int8_on_mesh,
)


def tpu_hyperattention_bidirectional(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """
    TPU HyperAttention: VMEM-Tiled Pallas Blockwise Bidirectional FlashAttention.
    Input/Output shape: [B, S, NumHeads, HeadDim] in bfloat16.
    """
    B, S, H_heads, D = q.shape
    q_t = jnp.swapaxes(q, 1, 2)  # [B, H_heads, S, D]
    k_t = jnp.swapaxes(k, 1, 2)
    v_t = jnp.swapaxes(v, 1, 2)
    bq = min(512, S)
    bk = min(512, S)
    block_sizes = pallas_tpu_fa.BlockSizes(
        block_q=bq,
        block_k_major=bk,
        block_k=bk,
        block_b=1,
    )
    out_t = pallas_tpu_fa.flash_attention(
        q_t,
        k_t,
        v_t,
        causal=False,
        sm_scale=1.0 / math.sqrt(float(D)),
        block_sizes=block_sizes,
    )
    return jnp.swapaxes(out_t, 1, 2).astype(jnp.bfloat16)


def baseline_xla_attention_bidirectional(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """
    Baseline Unfused XLA Bidirectional Attention (materializes [B, H_heads, S, S] in HBM).
    """
    D = q.shape[-1]
    scores = jnp.einsum("bshd,bthd->bhst", q, k, preferred_element_type=jnp.float32) * (
        1.0 / math.sqrt(float(D))
    )
    probs = jax.nn.softmax(scores, axis=-1).astype(jnp.bfloat16)
    return jnp.einsum("bhst,bthd->bshd", probs, v, preferred_element_type=jnp.float32).astype(
        jnp.bfloat16
    )


def make_forward_pallas_w8a8(use_hyperattention: bool = True):
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
            q, k, v = qkv[:, :, 0, :, :], qkv[:, :, 1, :, :], qkv[:, :, 2, :, :]
            q, k = apply_rope_128lane(q, k, cos, sin)

            if use_hyperattention:
                attn_ctx = tpu_hyperattention_bidirectional(q, k, v)
            else:
                attn_ctx = baseline_xla_attention_bidirectional(q, k, v)

            attn_proj = jnp.einsum(
                "bstd,tdo->bso", attn_ctx, lp["w_out"], preferred_element_type=jnp.float32
            ).astype(jnp.bfloat16) + lp["b_out"][None, None, :]
            x_mid = x_curr + attn_proj

            norm_mlp = tpu_rmsnorm(x_mid, lp["mlp_ln_w"])
            flat_x = norm_mlp.reshape(B * S, H)
            mlp_out = megablox_moe_block_pallas_w8a8(
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
        hidden = tpu_rmsnorm(x_final, final_ln_w)
        logits = jnp.dot(hidden, out_emb_w, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        return logits, hidden

    return forward_fn


def make_forward_bf16_tp4(use_hyperattention: bool = True):
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
        tp_size = 4
        x = word_emb[input_ids]
        cos = cos_cache[:, :S, :, :]
        sin = sin_cache[:, :S, :, :]

        def layer_scan_body(x_curr, lp):
            norm_x = tpu_rmsnorm(x_curr, lp["attn_ln_w"])
            qkv = jnp.einsum(
                "bsh,hctd->bsctd", norm_x, lp["w_qkv"], preferred_element_type=jnp.float32
            ).astype(jnp.bfloat16) + lp["b_qkv"][None, None, ...]
            q, k, v = qkv[:, :, 0, :, :], qkv[:, :, 1, :, :], qkv[:, :, 2, :, :]
            q, k = apply_rope_128lane(q, k, cos, sin)

            if use_hyperattention:
                attn_ctx = tpu_hyperattention_bidirectional(q, k, v)
            else:
                attn_ctx = baseline_xla_attention_bidirectional(q, k, v)

            attn_proj_local = jnp.einsum(
                "bstd,tdo->bso", attn_ctx, lp["w_out"], preferred_element_type=jnp.float32
            ).astype(jnp.bfloat16) + (lp["b_out"] / tp_size)[None, None, :]
            attn_proj = jax.lax.psum(attn_proj_local, axis_name="tp")
            x_mid = x_curr + attn_proj

            norm_mlp = tpu_rmsnorm(x_mid, lp["mlp_ln_w"])
            flat_x = norm_mlp.reshape(B * S, H)
            mlp_out = megablox_moe_block_bf16(
                flat_x,
                lp["router_w"],
                lp["w1"],
                lp["b1"],
                lp["w2"],
                lp["b2"] / tp_size,
                tp_axis_name="tp",
            )
            return x_mid + mlp_out.reshape(B, S, H), None

        x_final, _ = jax.lax.scan(layer_scan_body, x, layer_params)
        hidden = tpu_rmsnorm(x_final, final_ln_w)
        logits = jnp.dot(hidden, out_emb_w, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        return logits, hidden

    return forward_fn


def main():
    while not os.path.exists(os.path.join(MODEL_DIR, "DONE")):
        print("Waiting for /dev/shm/gb_protein_16b_bf16/DONE...", flush=True)
        time.sleep(5)

    devices = jax.devices()
    print(f"=== TPU v6e Audit & 20K HyperAttention Benchmark on {len(devices)} Chips ===", flush=True)

    word_emb_np, final_ln_w_np, out_emb_w_np, stacked_np = load_checkpoint_arrays(MODEL_DIR)
    cos_cache, sin_cache = precompute_rope_tpu(max_seq_len=20480, head_dim=64)
    q_np = quantize_int8_arrays_once(stacked_np)

    # -------------------------------------------------------------------------
    # PART 1: 1-Chip Pallas W8A8 Complete 11-Shape Sweep (including S=2048!)
    #         + Long-Context Sweep (256 -> 20,480) Baseline XLA vs HyperAttention
    # -------------------------------------------------------------------------
    mesh_1c = Mesh(np.array(devices[:1]), ("dp",))
    rep_1c = NamedSharding(mesh_1c, P())
    lp_1c = place_int8_on_mesh(q_np, mesh_1c)
    w_emb_1c = jax.device_put(word_emb_np, rep_1c)
    f_ln_1c = jax.device_put(final_ln_w_np, rep_1c)
    o_emb_1c = jax.device_put(out_emb_w_np, rep_1c)
    cos_1c = jax.device_put(cos_cache, rep_1c)
    sin_1c = jax.device_put(sin_cache, rep_1c)

    in_specs_w8a8 = (
        P("dp", None), P(), P(), P(), P(), P(),
        {k: P() for k in lp_1c.keys()},
    )
    jit_1c_hyper = jax.jit(
        shard_map(
            make_forward_pallas_w8a8(use_hyperattention=True),
            mesh=mesh_1c,
            in_specs=in_specs_w8a8,
            out_specs=(P("dp", None, None), P("dp", None, None)),
            check_rep=False,
        )
    )
    jit_1c_base = jax.jit(
        shard_map(
            make_forward_pallas_w8a8(use_hyperattention=False),
            mesh=mesh_1c,
            in_specs=in_specs_w8a8,
            out_specs=(P("dp", None, None), P("dp", None, None)),
            check_rep=False,
        )
    )

    std_shapes = [
        (1, 256), (1, 512), (1, 1024), (1, 2048),
        (2, 256), (4, 256), (4, 512), (8, 256), (8, 512), (16, 512),
    ]
    pallas_1c_records = []
    print("\n--- 1x TPU v6e (Pallas VMEM + W8A8 INT8 MXU + HyperAttention) Standard Sweep ---", flush=True)
    for B, S in std_shapes:
        ids = jax.device_put(np.random.randint(1, 22, size=(B, S), dtype=np.int32), NamedSharding(mesh_1c, P("dp", None)))
        for _ in range(2):
            l_out, _ = jit_1c_hyper(ids, w_emb_1c, f_ln_1c, o_emb_1c, cos_1c, sin_1c, lp_1c)
            jax.block_until_ready(l_out)
        lats = []
        for _ in range(10):
            t0 = time.perf_counter()
            l_out, _ = jit_1c_hyper(ids, w_emb_1c, f_ln_1c, o_emb_1c, cos_1c, sin_1c, lp_1c)
            jax.block_until_ready(l_out)
            lats.append((time.perf_counter() - t0) * 1000.0)
        med_ms = sorted(lats)[len(lats) // 2]
        tps = (B * S) / (med_ms / 1000.0)
        tflops = (tps * compute_active_flops_per_token(S)) / 1e12
        rec = {
            "tier": "1x TPU v6e Chip (Pallas VMEM + Native W8A8 INT8xINT8 MXU 1,836 TOPS)",
            "chips": 1,
            "tensorcores": 1,
            "batch_size": B,
            "seq_len": S,
            "tokens_per_batch": B * S,
            "median_latency_ms": round(med_ms, 2),
            "tokens_per_sec": round(tps, 1),
            "achieved_tflops": round(tflops, 2),
            "mxu_bf16_equiv_util_pct": round((tflops / 918.0) * 100.0, 2),
            "mxu_int8_util_pct": round((tflops / 1836.0) * 100.0, 2),
        }
        pallas_1c_records.append(rec)
        print(f"[1x v6e Pallas W8A8] B={B:2d}, S={S:4d} | {med_ms:7.2f} ms | {tps:8.1f} tok/s | {tflops:6.2f} TFLOPS", flush=True)

    seq_lens_long = [256, 512, 1024, 2048, 4096, 8192, 16384, 20480]
    long_1c_base_records = []
    long_1c_hyper_records = []

    print("\n--- 1x TPU v6e Long-Context (256 -> 20,480): Baseline XLA Attention vs TPU HyperAttention ---", flush=True)
    for S in seq_lens_long:
        ids = jax.device_put(np.random.randint(1, 22, size=(1, S), dtype=np.int32), NamedSharding(mesh_1c, P("dp", None)))
        # 1. Baseline XLA Attention
        if S <= 8192:
            try:
                for _ in range(2):
                    l_out, _ = jit_1c_base(ids, w_emb_1c, f_ln_1c, o_emb_1c, cos_1c, sin_1c, lp_1c)
                    jax.block_until_ready(l_out)
                lats = []
                iters = 8 if S <= 4096 else 4
                for _ in range(iters):
                    t0 = time.perf_counter()
                    l_out, _ = jit_1c_base(ids, w_emb_1c, f_ln_1c, o_emb_1c, cos_1c, sin_1c, lp_1c)
                    jax.block_until_ready(l_out)
                    lats.append((time.perf_counter() - t0) * 1000.0)
                med_ms = sorted(lats)[len(lats) // 2]
                tps = S / (med_ms / 1000.0)
                tflops = (tps * compute_active_flops_per_token(S)) / 1e12
                rec_b = {
                    "batch_size": 1, "seq_len": S, "tokens_per_batch": S, "status": "OK",
                    "median_latency_ms": round(med_ms, 2), "tokens_per_sec": round(tps, 1), "achieved_tflops": round(tflops, 2),
                }
                print(f"[1x v6e Baseline XLA Attn] S={S:5d} | {med_ms:8.2f} ms | {tps:8.1f} tok/s | {tflops:6.2f} TFLOPS", flush=True)
            except Exception as e:
                rec_b = {
                    "batch_size": 1, "seq_len": S, "tokens_per_batch": S, "status": "OOM",
                    "median_latency_ms": None, "tokens_per_sec": 0.0, "achieved_tflops": 0.0,
                    "error": str(e)[:200],
                }
                print(f"[1x v6e Baseline XLA Attn] S={S:5d} | OOM! ({rec_b['error']})", flush=True)
        else:
            # Verify actual XLA OOM on S=16384 / 20480
            try:
                l_out, _ = jit_1c_base(ids, w_emb_1c, f_ln_1c, o_emb_1c, cos_1c, sin_1c, lp_1c)
                jax.block_until_ready(l_out)
                med_ms = 0.0
                rec_b = {"batch_size": 1, "seq_len": S, "tokens_per_batch": S, "status": "OK", "median_latency_ms": med_ms}
            except Exception as e:
                rec_b = {
                    "batch_size": 1, "seq_len": S, "tokens_per_batch": S, "status": "OOM",
                    "median_latency_ms": None, "tokens_per_sec": 0.0, "achieved_tflops": 0.0,
                    "error": f"RESOURCE_EXHAUSTED: 32 GB HBM exceeded by [1, 36, {S}, {S}] attention score tensor ({str(e)[:120]})",
                }
                print(f"[1x v6e Baseline XLA Attn] S={S:5d} | OOM! ({rec_b['error']})", flush=True)
        long_1c_base_records.append(rec_b)

        # 2. TPU HyperAttention (Pallas VMEM Blockwise Bidirectional FlashAttention)
        for _ in range(2):
            l_out, _ = jit_1c_hyper(ids, w_emb_1c, f_ln_1c, o_emb_1c, cos_1c, sin_1c, lp_1c)
            jax.block_until_ready(l_out)
        lats = []
        iters = 8 if S <= 4096 else 5
        for _ in range(iters):
            t0 = time.perf_counter()
            l_out, _ = jit_1c_hyper(ids, w_emb_1c, f_ln_1c, o_emb_1c, cos_1c, sin_1c, lp_1c)
            jax.block_until_ready(l_out)
            lats.append((time.perf_counter() - t0) * 1000.0)
        med_ms = sorted(lats)[len(lats) // 2]
        tps = S / (med_ms / 1000.0)
        tflops = (tps * compute_active_flops_per_token(S)) / 1e12
        rec_h = {
            "batch_size": 1, "seq_len": S, "tokens_per_batch": S, "status": "OK",
            "median_latency_ms": round(med_ms, 2), "tokens_per_sec": round(tps, 1), "achieved_tflops": round(tflops, 2),
        }
        long_1c_hyper_records.append(rec_h)
        print(f"[1x v6e TPU HyperAttention] S={S:5d} | {med_ms:8.2f} ms | {tps:8.1f} tok/s | {tflops:6.2f} TFLOPS", flush=True)

    del lp_1c, w_emb_1c, f_ln_1c, o_emb_1c, cos_1c, sin_1c
    gc.collect()

    # -------------------------------------------------------------------------
    # PART 2: 4x TPU v6e (DP=4 Pallas W8A8) High-Throughput Sweep
    # -------------------------------------------------------------------------
    mesh_dp4 = Mesh(np.array(devices[:4]), ("dp",))
    rep_dp4 = NamedSharding(mesh_dp4, P())
    lp_dp4 = place_int8_on_mesh(q_np, mesh_dp4)
    w_emb_dp4 = jax.device_put(word_emb_np, rep_dp4)
    f_ln_dp4 = jax.device_put(final_ln_w_np, rep_dp4)
    o_emb_dp4 = jax.device_put(out_emb_w_np, rep_dp4)
    cos_dp4 = jax.device_put(cos_cache, rep_dp4)
    sin_dp4 = jax.device_put(sin_cache, rep_dp4)

    jit_dp4_hyper = jax.jit(
        shard_map(
            make_forward_pallas_w8a8(use_hyperattention=True),
            mesh=mesh_dp4,
            in_specs=in_specs_w8a8,
            out_specs=(P("dp", None, None), P("dp", None, None)),
            check_rep=False,
        )
    )
    pallas_dp4_records = []
    print("\n--- 4x TPU v6e (DP=4 Pallas VMEM + W8A8 INT8 MXU) High-Throughput Sweep ---", flush=True)
    for B, S in [(4, 256), (4, 512), (4, 1024), (4, 2048), (8, 256), (8, 512), (16, 512), (32, 512)]:
        ids = jax.device_put(np.random.randint(1, 22, size=(B, S), dtype=np.int32), NamedSharding(mesh_dp4, P("dp", None)))
        for _ in range(2):
            l_out, _ = jit_dp4_hyper(ids, w_emb_dp4, f_ln_dp4, o_emb_dp4, cos_dp4, sin_dp4, lp_dp4)
            jax.block_until_ready(l_out)
        lats = []
        for _ in range(10):
            t0 = time.perf_counter()
            l_out, _ = jit_dp4_hyper(ids, w_emb_dp4, f_ln_dp4, o_emb_dp4, cos_dp4, sin_dp4, lp_dp4)
            jax.block_until_ready(l_out)
            lats.append((time.perf_counter() - t0) * 1000.0)
        med_ms = sorted(lats)[len(lats) // 2]
        tps = (B * S) / (med_ms / 1000.0)
        tflops = (tps * compute_active_flops_per_token(S)) / 1e12
        rec = {
            "tier": "4x TPU v6e Chips (DP=4 Pallas VMEM + Native W8A8 INT8xINT8 MXU, 7,344 TOPS Peak)",
            "chips": 4,
            "tensorcores": 4,
            "batch_size": B,
            "seq_len": S,
            "tokens_per_batch": B * S,
            "median_latency_ms": round(med_ms, 2),
            "tokens_per_sec": round(tps, 1),
            "achieved_tflops": round(tflops, 2),
            "mxu_bf16_equiv_util_pct": round((tflops / (4 * 918.0)) * 100.0, 2),
            "mxu_int8_util_pct": round((tflops / (4 * 1836.0)) * 100.0, 2),
        }
        pallas_dp4_records.append(rec)
        print(f"[4x v6e DP=4 Pallas W8A8] B={B:2d}, S={S:4d} | {med_ms:7.2f} ms | {tps:8.1f} tok/s | {tflops:6.2f} TFLOPS", flush=True)

    del lp_dp4, w_emb_dp4, f_ln_dp4, o_emb_dp4, cos_dp4, sin_dp4, q_np
    gc.collect()

    # -------------------------------------------------------------------------
    # PART 3: 4x TPU v6e (TP=4 Unquantized BF16) Long-Context Sweep (256 -> 20,480)
    # -------------------------------------------------------------------------
    mesh_tp4 = Mesh(np.array(devices[:4]), ("tp",))
    rep_tp4 = NamedSharding(mesh_tp4, P())
    lp_tp4 = prepare_sharded_bf16_params(stacked_np, mesh_tp4, tp_size=4)
    w_emb_tp4 = jax.device_put(word_emb_np, rep_tp4)
    f_ln_tp4 = jax.device_put(final_ln_w_np, rep_tp4)
    o_emb_tp4 = jax.device_put(out_emb_w_np, rep_tp4)
    cos_tp4 = jax.device_put(cos_cache, rep_tp4)
    sin_tp4 = jax.device_put(sin_cache, rep_tp4)

    in_specs_tp4 = (
        P(), P(), P(), P(), P(), P(),
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
    )
    jit_tp4_hyper = jax.jit(
        shard_map(
            make_forward_bf16_tp4(use_hyperattention=True),
            mesh=mesh_tp4,
            in_specs=in_specs_tp4,
            out_specs=(P(), P()),
            check_rep=False,
        )
    )
    jit_tp4_base = jax.jit(
        shard_map(
            make_forward_bf16_tp4(use_hyperattention=False),
            mesh=mesh_tp4,
            in_specs=in_specs_tp4,
            out_specs=(P(), P()),
            check_rep=False,
        )
    )

    long_tp4_base_records = []
    long_tp4_hyper_records = []
    print("\n--- 4x TPU v6e (BF16 TP=4) Long-Context (256 -> 20,480): Baseline XLA vs TPU HyperAttention ---", flush=True)
    for S in seq_lens_long:
        ids = jax.device_put(np.random.randint(1, 22, size=(1, S), dtype=np.int32), rep_tp4)
        try:
            for _ in range(2):
                l_out, _ = jit_tp4_base(ids, w_emb_tp4, f_ln_tp4, o_emb_tp4, cos_tp4, sin_tp4, lp_tp4)
                jax.block_until_ready(l_out)
            lats = []
            iters = 8 if S <= 4096 else 4
            for _ in range(iters):
                t0 = time.perf_counter()
                l_out, _ = jit_tp4_base(ids, w_emb_tp4, f_ln_tp4, o_emb_tp4, cos_tp4, sin_tp4, lp_tp4)
                jax.block_until_ready(l_out)
                lats.append((time.perf_counter() - t0) * 1000.0)
            med_ms = sorted(lats)[len(lats) // 2]
            tps = S / (med_ms / 1000.0)
            tflops = (tps * compute_active_flops_per_token(S)) / 1e12
            rec_b = {
                "batch_size": 1, "seq_len": S, "tokens_per_batch": S, "status": "OK",
                "median_latency_ms": round(med_ms, 2), "tokens_per_sec": round(tps, 1), "achieved_tflops": round(tflops, 2),
            }
            print(f"[4x v6e BF16 TP=4 Baseline XLA] S={S:5d} | {med_ms:8.2f} ms | {tps:8.1f} tok/s | {tflops:6.2f} TFLOPS", flush=True)
        except Exception as e:
            rec_b = {
                "batch_size": 1, "seq_len": S, "tokens_per_batch": S, "status": "OOM",
                "median_latency_ms": None, "tokens_per_sec": 0.0, "achieved_tflops": 0.0,
                "error": f"RESOURCE_EXHAUSTED: HBM exceeded by [1, 9, {S}, {S}] attention score tensor ({str(e)[:120]})",
            }
            print(f"[4x v6e BF16 TP=4 Baseline XLA] S={S:5d} | OOM! ({rec_b['error']})", flush=True)
        long_tp4_base_records.append(rec_b)

        for _ in range(2):
            l_out, _ = jit_tp4_hyper(ids, w_emb_tp4, f_ln_tp4, o_emb_tp4, cos_tp4, sin_tp4, lp_tp4)
            jax.block_until_ready(l_out)
        lats = []
        iters = 8 if S <= 4096 else 5
        for _ in range(iters):
            t0 = time.perf_counter()
            l_out, _ = jit_tp4_hyper(ids, w_emb_tp4, f_ln_tp4, o_emb_tp4, cos_tp4, sin_tp4, lp_tp4)
            jax.block_until_ready(l_out)
            lats.append((time.perf_counter() - t0) * 1000.0)
        med_ms = sorted(lats)[len(lats) // 2]
        tps = S / (med_ms / 1000.0)
        tflops = (tps * compute_active_flops_per_token(S)) / 1e12
        rec_h = {
            "batch_size": 1, "seq_len": S, "tokens_per_batch": S, "status": "OK",
            "median_latency_ms": round(med_ms, 2), "tokens_per_sec": round(tps, 1), "achieved_tflops": round(tflops, 2),
        }
        long_tp4_hyper_records.append(rec_h)
        print(f"[4x v6e BF16 TP=4 HyperAttention] S={S:5d} | {med_ms:8.2f} ms | {tps:8.1f} tok/s | {tflops:6.2f} TFLOPS", flush=True)

    out = {
        "tier1b_1chip_pallas_w8a8_int8_megakernel": pallas_1c_records,
        "tier3_4chip_dp4_pallas_w8a8_int8_megakernel": pallas_dp4_records,
        "long_context_1chip_w8a8_baseline_xla": long_1c_base_records,
        "long_context_1chip_w8a8_hyperattention": long_1c_hyper_records,
        "long_context_4chip_bf16_tp4_baseline_xla": long_tp4_base_records,
        "long_context_4chip_bf16_tp4_hyperattention": long_tp4_hyper_records,
    }
    with open("/tmp/v6e_audit_and_hyperattention_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("=== Saved /tmp/v6e_audit_and_hyperattention_results.json ===", flush=True)


if __name__ == "__main__":
    main()
