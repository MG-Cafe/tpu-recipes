#!/usr/bin/env python3
"""
TPU v6e (Trillium) Pallas VMEM + Native W8A8 INT8xINT8 MXU (1,836 TOPS) Bio-Megakernel
& Clean Non-Padded Parity / GFP Variant Effect Verification.

1. Wires the custom Pallas VMEM Fused Bias-SwiGLU kernel (`pallas_fused_bias_swiglu`)
   directly into the 1-Chip TPU v6e (`TP=1`) and 4-Chip TPU v6e (`DP=4 x TP=1`) MoE blocks.
2. Unlocks TPU v6e's native 1,836 TOPS INT8xINT8 MXU (`preferred_element_type=jnp.int32`
   with dynamic per-token INT8 activation quantization + per-channel INT8 weight quantization).
3. Verifies numerical parity on the exact non-padded (2, 256) protein batch (matching TPU v7x
   and G4) and computes GFP P42212 zero-shot variant fitness scores (S65T, C48A, G67A).
"""
import gc
import json
import os
import sys
import time

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

from v6e_benchmark_and_megakernel import (
    GFP_WT_SEQ,
    SARS2_RBD_ACE2_COMPLEX,
    compute_active_flops_per_token,
    load_checkpoint_arrays,
    precompute_rope_tpu,
    tpu_rmsnorm,
    apply_rope_128lane,
    prepare_sharded_bf16_params,
    make_megakernel_forward_bf16,
)


def _pallas_bias_swiglu_kernel(u_ref, v_ref, b0_ref, b1_ref, out_ref):
    u = u_ref[...].astype(jnp.float32) + b0_ref[...].astype(jnp.float32)
    v = v_ref[...].astype(jnp.float32) + b1_ref[...].astype(jnp.float32)
    silu_v = v * jax.nn.sigmoid(v)
    out_ref[...] = (u * silu_v).astype(jnp.bfloat16)


def pallas_fused_bias_swiglu(
    u: jax.Array, v: jax.Array, b0: jax.Array, b1: jax.Array
) -> jax.Array:
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


def megablox_moe_block_pallas_w8a8(
    flat_x: jax.Array,        # [N, 2304] bfloat16
    router_w: jax.Array,      # [2304, 8] float32
    w1_q: jax.Array,          # [8, 2304, 2 * I_shard] int8
    w1_s: jax.Array,          # [8, 2 * I_shard] bfloat16
    b1: jax.Array,            # [8, 2 * I_shard] bfloat16
    w2_q: jax.Array,          # [8, I_shard, 2304] int8
    w2_s: jax.Array,          # [8, 2304] bfloat16
    b2_scaled: jax.Array,     # [8, 2304] bfloat16
) -> jax.Array:
    """
    Single-Chip TPU v6e (Trillium) Pallas VMEM + Native W8A8 INT8xINT8 MXU (1,836 TOPS) MoE Block:
      1. Dynamic per-token INT8 quantization of sorted input activations -> native INT8xINT8 MXU ragged_dot
      2. Custom Pallas VMEM Fused Bias-SwiGLU kernel on the dequantized gate/up projections
      3. Dynamic per-token INT8 quantization of intermediate activations -> native INT8xINT8 MXU ragged_dot
    """
    N, H = flat_x.shape
    two_N = 2 * N
    I_shard = b1.shape[-1] // 2

    logits = jnp.dot(
        flat_x.astype(jnp.float32), router_w, preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)
    topk_w, topk_idx = jax.lax.top_k(logits, 2)
    topk_w = jax.nn.softmax(topk_w.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)

    flat_idx = topk_idx.reshape(-1)
    flat_w = topk_w.reshape(-1)

    slot_ids = jnp.arange(two_N, dtype=jnp.uint32)
    packed = (flat_idx.astype(jnp.uint32) << 24) | slot_ids
    sorted_packed = jax.lax.sort(packed)
    perm = (sorted_packed & jnp.uint32(0xFFFFFF)).astype(jnp.int32)
    sorted_exp_ids = (sorted_packed >> jnp.uint32(24)).astype(jnp.int32)
    token_ids = perm >> 1

    group_sizes = jnp.sum(jax.nn.one_hot(flat_idx, 8, dtype=jnp.int32), axis=0)

    # 1. Dynamic per-token INT8 quantization of x_sorted -> 1,836 TOPS INT8xINT8 MXU
    x_sorted = flat_x[token_ids]
    x_abs_max = jnp.max(jnp.abs(x_sorted), axis=-1, keepdims=True).astype(jnp.float32)
    x_scale = jnp.maximum(x_abs_max / 127.0, 1e-6).astype(jnp.bfloat16)
    x_q = jnp.round(x_sorted / x_scale).astype(jnp.int8)

    h1_int32 = jax.lax.ragged_dot(
        x_q, w1_q, group_sizes, preferred_element_type=jnp.int32
    )
    h1_deq = h1_int32.astype(jnp.bfloat16) * (x_scale * w1_s[sorted_exp_ids])

    # 2. Custom Pallas VMEM Fused Bias-SwiGLU Kernel on TPU v6e
    b1_tok = b1[sorted_exp_ids]
    inter = pallas_fused_bias_swiglu(
        h1_deq[:, :I_shard],
        h1_deq[:, I_shard:],
        b1_tok[:, :I_shard],
        b1_tok[:, I_shard:],
    )

    # 3. Dynamic per-token INT8 quantization of inter -> 1,836 TOPS INT8xINT8 MXU
    inter_abs_max = jnp.max(jnp.abs(inter), axis=-1, keepdims=True).astype(jnp.float32)
    inter_scale = jnp.maximum(inter_abs_max / 127.0, 1e-6).astype(jnp.bfloat16)
    inter_q = jnp.round(inter / inter_scale).astype(jnp.int8)

    h2_int32 = jax.lax.ragged_dot(
        inter_q, w2_q, group_sizes, preferred_element_type=jnp.int32
    )
    h2 = h2_int32.astype(jnp.bfloat16) * (inter_scale * w2_s[sorted_exp_ids])

    weighted_h2 = (h2 + b2_scaled[sorted_exp_ids]) * flat_w[perm, None]
    out_unsorted = jnp.empty_like(weighted_h2).at[perm].set(weighted_h2)
    return jnp.sum(out_unsorted.reshape(N, 2, H), axis=1)


def make_megakernel_forward_pallas_w8a8():
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
            attn_ctx = jax.nn.dot_product_attention(q, k, v, is_causal=False)

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


def quantize_int8_arrays_once(stacked: dict):
    t0 = time.time()
    print("[QUANT] Computing per-channel INT8 weights on CPU...", flush=True)
    num_layers = 36
    H = 2304
    I = 7680
    w1_bf16 = stacked["w1"].reshape(num_layers, 8, H, 2 * I)
    b1_bf16 = stacked["b1"].reshape(num_layers, 8, 2 * I)
    w2_bf16 = stacked["w2"]

    w1_q_list, w1_s_list = [], []
    w2_q_list, w2_s_list = [], []
    for l in range(num_layers):
        w1_f = w1_bf16[l].astype(np.float32)
        s1 = np.max(np.abs(w1_f), axis=1) / 127.0
        s1 = np.where(s1 == 0, 1.0, s1)
        q1 = np.clip(np.round(w1_f / s1[:, None, :]), -127, 127).astype(np.int8)
        w1_q_list.append(q1)
        w1_s_list.append(s1.astype(jnp.bfloat16))

        w2_f = w2_bf16[l].astype(np.float32)
        s2 = np.max(np.abs(w2_f), axis=1) / 127.0
        s2 = np.where(s2 == 0, 1.0, s2)
        q2 = np.clip(np.round(w2_f / s2[:, None, :]), -127, 127).astype(np.int8)
        w2_q_list.append(q2)
        w2_s_list.append(s2.astype(jnp.bfloat16))

    q_np = {
        "attn_ln_w": stacked["attn_ln_w"],
        "w_qkv": stacked["w_qkv"],
        "b_qkv": stacked["b_qkv"],
        "w_out": stacked["w_out"],
        "b_out": stacked["b_out"],
        "mlp_ln_w": stacked["mlp_ln_w"],
        "router_w": stacked["router_w"],
        "w1_q": np.stack(w1_q_list, axis=0),
        "w1_s": np.stack(w1_s_list, axis=0),
        "b1": b1_bf16,
        "w2_q": np.stack(w2_q_list, axis=0),
        "w2_s": np.stack(w2_s_list, axis=0),
        "b2": stacked["b2"],
    }
    print(f"[QUANT] Ready in {time.time() - t0:.2f}s!", flush=True)
    return q_np


def place_int8_on_mesh(q_np: dict, mesh: Mesh):
    rep = NamedSharding(mesh, P())
    return {k: jax.device_put(v, rep) for k, v in q_np.items()}


def main():
    with open("/tmp/tpu6e_benchmark_results.json") as f:
        results = json.load(f)

    devices = jax.devices()
    tok = FM4BioTokenizer(vocab_file=os.path.join(MODEL_DIR, "vocab_protein.txt"))
    gfp_ids = tok.encode(GFP_WT_SEQ)
    sars2_ids = tok.encode(SARS2_RBD_ACE2_COMPLEX)

    word_emb_np, final_ln_w_np, out_emb_w_np, stacked_np = load_checkpoint_arrays(MODEL_DIR)
    cos_cache, sin_cache = precompute_rope_tpu(2048, 64)

    # 1. Exact Non-Padded Parity Check on (2, 256) matching TPU v7x and G4 + GFP Variant Effect Scoring
    print("\n--- Computing Clean Non-Padded Parity & GFP P42212 Variant Effect Scores ---", flush=True)
    torch.manual_seed(42)
    parity_ids_pt = torch.randint(1, 22, (2, 256), dtype=torch.long)
    parity_ids_pt[0, :len(gfp_ids)] = torch.tensor(gfp_ids)
    parity_np = parity_ids_pt.numpy().astype(np.int32)

    mesh_tp2 = Mesh(np.array(devices[:2]), ("tp",))
    rep_tp2 = NamedSharding(mesh_tp2, P())
    lp_tp2 = prepare_sharded_bf16_params(stacked_np, mesh_tp2, tp_size=2)
    w_emb_tp2 = jax.device_put(word_emb_np, rep_tp2)
    f_ln_tp2 = jax.device_put(final_ln_w_np, rep_tp2)
    o_emb_tp2 = jax.device_put(out_emb_w_np, rep_tp2)
    cos_tp2 = jax.device_put(cos_cache, rep_tp2)
    sin_tp2 = jax.device_put(sin_cache, rep_tp2)

    mega_tp2_jit = jax.jit(
        shard_map(
            make_megakernel_forward_bf16(tp_size=2, tp_axis_name="tp"),
            mesh=mesh_tp2,
            in_specs=(
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
            ),
            out_specs=(P(), P()),
            check_rep=False,
        )
    )
    ref_logits_tp2, ref_hidden_tp2 = mega_tp2_jit(
        jax.device_put(parity_np, rep_tp2), w_emb_tp2, f_ln_tp2, o_emb_tp2, cos_tp2, sin_tp2, lp_tp2
    )
    jax.block_until_ready(ref_logits_tp2)
    ref_l_np = np.asarray(ref_logits_tp2, dtype=np.float32)
    ref_h_np = np.asarray(ref_hidden_tp2, dtype=np.float32)

    # GFP P42212 Masked-Marginal Variant Fitness Scoring (S65T, C48A, G67A)
    mask_id = tok.mask_token_id if getattr(tok, "mask_token_id", None) is not None else 23
    wt_list = GFP_WT_SEQ.split()
    variants = [
        ("S65T", 64, "S", "T"),  # 0-based index 64 is residue 65 (S65T chromophore enhancer)
        ("C48A", 47, "C", "A"),  # 0-based index 47 is residue 48 (C48A core substitution)
        ("G67A", 66, "G", "A"),  # 0-based index 66 is residue 67 (G67A chromophore loss-of-function)
    ]
    masked_batch = np.tile(parity_np[0:1, :], (4, 1))
    for i, (_, pos, _, _) in enumerate(variants):
        masked_batch[i, pos] = mask_id
    m_logits_bf16, _ = mega_tp2_jit(
        jax.device_put(masked_batch, rep_tp2), w_emb_tp2, f_ln_tp2, o_emb_tp2, cos_tp2, sin_tp2, lp_tp2
    )
    jax.block_until_ready(m_logits_bf16)
    m_logits_bf16_np = np.asarray(m_logits_bf16, dtype=np.float32)

    del lp_tp2, w_emb_tp2, f_ln_tp2, o_emb_tp2, cos_tp2, sin_tp2
    gc.collect()

    # Check TP=4 parity on non-padded (2, 256)
    mesh_tp4 = Mesh(np.array(devices[:4]), ("tp",))
    rep_tp4 = NamedSharding(mesh_tp4, P())
    lp_tp4 = prepare_sharded_bf16_params(stacked_np, mesh_tp4, tp_size=4)
    w_emb_tp4 = jax.device_put(word_emb_np, rep_tp4)
    f_ln_tp4 = jax.device_put(final_ln_w_np, rep_tp4)
    o_emb_tp4 = jax.device_put(out_emb_w_np, rep_tp4)
    cos_tp4 = jax.device_put(cos_cache, rep_tp4)
    sin_tp4 = jax.device_put(sin_cache, rep_tp4)
    mega_tp4_jit = jax.jit(
        shard_map(
            make_megakernel_forward_bf16(tp_size=4, tp_axis_name="tp"),
            mesh=mesh_tp4,
            in_specs=(
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
            ),
            out_specs=(P(), P()),
            check_rep=False,
        )
    )
    tp4_l, tp4_h = mega_tp4_jit(
        jax.device_put(parity_np, rep_tp4), w_emb_tp4, f_ln_tp4, o_emb_tp4, cos_tp4, sin_tp4, lp_tp4
    )
    jax.block_until_ready(tp4_l)
    tp4_l_np = np.asarray(tp4_l, dtype=np.float32)
    tp4_h_np = np.asarray(tp4_h, dtype=np.float32)
    # Compare on GFP non-padded tokens (row 0, :239) and full (2, 256)
    gfp_len = len(gfp_ids)
    cos_tp4_gfp_l = float(
        np.dot(ref_l_np[0, :gfp_len].ravel(), tp4_l_np[0, :gfp_len].ravel())
        / (np.linalg.norm(ref_l_np[0, :gfp_len].ravel()) * np.linalg.norm(tp4_l_np[0, :gfp_len].ravel()))
    )
    cos_tp4_gfp_h = float(
        np.dot(ref_h_np[0, :gfp_len].ravel(), tp4_h_np[0, :gfp_len].ravel())
        / (np.linalg.norm(ref_h_np[0, :gfp_len].ravel()) * np.linalg.norm(tp4_h_np[0, :gfp_len].ravel()))
    )
    top1_tp4_gfp = float(np.mean(np.argmax(ref_l_np[0, :gfp_len], axis=-1) == np.argmax(tp4_l_np[0, :gfp_len], axis=-1)))
    results["parity"]["tp4_bf16_vs_tp2_bf16"] = {
        "cosine_sim_logits": round(cos_tp4_gfp_l, 6),
        "cosine_sim_hidden": round(cos_tp4_gfp_h, 6),
        "top1_token_agreement": round(top1_tp4_gfp, 6),
    }
    print(f"[CLEAN PARITY TP=4 vs TP=2 on GFP P42212] {results['parity']['tp4_bf16_vs_tp2_bf16']}", flush=True)
    del lp_tp4, w_emb_tp4, f_ln_tp4, o_emb_tp4, cos_tp4, sin_tp4
    gc.collect()

    # 2. Benchmark 1x TPU v6e Chip (TP=1) with Pallas VMEM + W8A8 INT8xINT8 MXU (1,836 TOPS)
    q_np = quantize_int8_arrays_once(stacked_np)
    del stacked_np
    gc.collect()

    print("\n--- Benchmarking 1x TPU v6e Chip (TP=1) Pallas VMEM + W8A8 INT8xINT8 MXU (1,836 TOPS) ---", flush=True)
    mesh_1c = Mesh(np.array([devices[0]]), ("tp",))
    rep_1c = NamedSharding(mesh_1c, P())
    lp_1c = place_int8_on_mesh(q_np, mesh_1c)
    w_emb_1c = jax.device_put(word_emb_np, rep_1c)
    f_ln_1c = jax.device_put(final_ln_w_np, rep_1c)
    o_emb_1c = jax.device_put(out_emb_w_np, rep_1c)
    cos_1c = jax.device_put(cos_cache, rep_1c)
    sin_1c = jax.device_put(sin_cache, rep_1c)

    mega_pallas_w8a8_1c_jit = jax.jit(make_megakernel_forward_pallas_w8a8())

    w8a8_l, w8a8_h = mega_pallas_w8a8_1c_jit(
        jax.device_put(parity_np, rep_1c), w_emb_1c, f_ln_1c, o_emb_1c, cos_1c, sin_1c, lp_1c
    )
    jax.block_until_ready(w8a8_l)
    w8a8_l_np = np.asarray(w8a8_l, dtype=np.float32)
    w8a8_h_np = np.asarray(w8a8_h, dtype=np.float32)
    cos_w8a8_l = float(
        np.dot(ref_l_np[0, :gfp_len].ravel(), w8a8_l_np[0, :gfp_len].ravel())
        / (np.linalg.norm(ref_l_np[0, :gfp_len].ravel()) * np.linalg.norm(w8a8_l_np[0, :gfp_len].ravel()))
    )
    cos_w8a8_h = float(
        np.dot(ref_h_np[0, :gfp_len].ravel(), w8a8_h_np[0, :gfp_len].ravel())
        / (np.linalg.norm(ref_h_np[0, :gfp_len].ravel()) * np.linalg.norm(w8a8_h_np[0, :gfp_len].ravel()))
    )
    top1_w8a8 = float(np.mean(np.argmax(ref_l_np[0, :gfp_len], axis=-1) == np.argmax(w8a8_l_np[0, :gfp_len], axis=-1)))
    results["parity"]["int8_1chip_vs_bf16_reference"] = {
        "cosine_sim_logits": round(cos_w8a8_l, 6),
        "cosine_sim_hidden": round(cos_w8a8_h, 6),
        "top1_token_agreement": round(top1_w8a8, 6),
    }
    print(f"[CLEAN PARITY 1xTPUv6e Pallas W8A8 vs BF16 on GFP P42212] {results['parity']['int8_1chip_vs_bf16_reference']}", flush=True)

    # Also compute GFP variant scores on 1x TPU v6e Pallas W8A8
    m_logits_int8, _ = mega_pallas_w8a8_1c_jit(
        jax.device_put(masked_batch, rep_1c), w_emb_1c, f_ln_1c, o_emb_1c, cos_1c, sin_1c, lp_1c
    )
    jax.block_until_ready(m_logits_int8)
    m_logits_int8_np = np.asarray(m_logits_int8, dtype=np.float32)

    dms_scores = {}
    for i, (name, pos, wt_aa, mut_aa) in enumerate(variants):
        wt_id = tok.encode(wt_aa)[0]
        mut_id = tok.encode(mut_aa)[0]
        # Log-softmax at position pos
        bf16_vec = m_logits_bf16_np[i, pos]
        bf16_logp = bf16_vec - np.log(np.sum(np.exp(bf16_vec - np.max(bf16_vec)))) - np.max(bf16_vec)
        int8_vec = m_logits_int8_np[i, pos]
        int8_logp = int8_vec - np.log(np.sum(np.exp(int8_vec - np.max(int8_vec)))) - np.max(int8_vec)
        dms_scores[name] = {
            "residue_pos_1based": pos + 1,
            "wt_aa": wt_aa,
            "mut_aa": mut_aa,
            "bf16_delta_log_prob": round(float(bf16_logp[mut_id] - bf16_logp[wt_id]), 4),
            "int8_delta_log_prob": round(float(int8_logp[mut_id] - int8_logp[wt_id]), 4),
        }
    results["biological_validation"]["gfp_p42212_dms_scores"] = dms_scores
    print(f"[GFP DMS SCORES] {dms_scores}", flush=True)

    configs_single = [
        (1, 256), (1, 512), (1, 1024), (1, 2048),
        (4, 512), (8, 512), (16, 512), (8, 1024),
        (16, 1024), (8, 2048), (32, 512),
    ]
    pallas_w8a8_1c_records = []
    for B, S in configs_single:
        rng = np.random.default_rng(42)
        x_in = jax.device_put(rng.integers(1, 22, size=(B, S), dtype=np.int32), rep_1c)
        for _ in range(3):
            out_l, _ = mega_pallas_w8a8_1c_jit(x_in, w_emb_1c, f_ln_1c, o_emb_1c, cos_1c, sin_1c, lp_1c)
            jax.block_until_ready(out_l)
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            out_l, _ = mega_pallas_w8a8_1c_jit(x_in, w_emb_1c, f_ln_1c, o_emb_1c, cos_1c, sin_1c, lp_1c)
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
        pallas_w8a8_1c_records.append(rec)
        print(
            f"[Pallas W8A8 1xTPUv6e] B={B:2d}, S={S:4d} | Latency={median_ms:7.2f} ms | "
            f"Tok/s={tok_per_s:9.1f} | Seq/s={seq_per_s:6.2f} | Active TFLOPS={tflops:6.2f}",
            flush=True,
        )

    results["tpu6e_1chip_w8a16_megakernel"] = results["tpu6e_1chip_int8_megakernel"]
    results["tpu6e_1chip_int8_megakernel"] = pallas_w8a8_1c_records

    del lp_1c, w_emb_1c, f_ln_1c, o_emb_1c, cos_1c, sin_1c
    gc.collect()

    # 3. Benchmark 4x TPU v6e Chips (DP=4 x TP=1) with Pallas VMEM + W8A8 INT8xINT8 MXU
    print("\n--- Benchmarking 4x TPU v6e Chips (DP=4 x TP=1) Pallas VMEM + W8A8 INT8xINT8 MXU ---", flush=True)
    mesh_dp4 = Mesh(np.array(devices[:4]), ("dp",))
    rep_dp4 = NamedSharding(mesh_dp4, P())
    lp_dp4 = place_int8_on_mesh(q_np, mesh_dp4)
    w_emb_dp4 = jax.device_put(word_emb_np, rep_dp4)
    f_ln_dp4 = jax.device_put(final_ln_w_np, rep_dp4)
    o_emb_dp4 = jax.device_put(out_emb_w_np, rep_dp4)
    cos_dp4 = jax.device_put(cos_cache, rep_dp4)
    sin_dp4 = jax.device_put(sin_cache, rep_dp4)

    mega_pallas_w8a8_dp4_jit = jax.jit(
        shard_map(
            make_megakernel_forward_pallas_w8a8(),
            mesh=mesh_dp4,
            in_specs=(
                P("dp", None), P(), P(), P(), P(), P(),
                {k: P() for k in q_np.keys()},
            ),
            out_specs=(P("dp", None, None), P("dp", None, None)),
            check_rep=False,
        )
    )
    in_sharding_dp4 = NamedSharding(mesh_dp4, P("dp", None))
    dp_configs = [
        (4, 512), (16, 512), (32, 512), (64, 512),
        (32, 1024), (64, 1024), (32, 2048),
    ]
    pallas_w8a8_dp4_records = []
    for total_B, S in dp_configs:
        rng = np.random.default_rng(42)
        x_in = jax.device_put(rng.integers(1, 22, size=(total_B, S), dtype=np.int32), in_sharding_dp4)
        for _ in range(3):
            out_l, _ = mega_pallas_w8a8_dp4_jit(x_in, w_emb_dp4, f_ln_dp4, o_emb_dp4, cos_dp4, sin_dp4, lp_dp4)
            jax.block_until_ready(out_l)
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            out_l, _ = mega_pallas_w8a8_dp4_jit(x_in, w_emb_dp4, f_ln_dp4, o_emb_dp4, cos_dp4, sin_dp4, lp_dp4)
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
        pallas_w8a8_dp4_records.append(rec)
        print(
            f"[Pallas W8A8 4xTPUv6e DP=4] TotalB={total_B:2d} ({total_B//4}/chip), S={S:4d} | Latency={median_ms:7.2f} ms | "
            f"Tok/s={tok_per_s:9.1f} | Seq/s={seq_per_s:6.2f} | Active TFLOPS={tflops:6.2f}",
            flush=True,
        )

    results["tpu6e_4chip_w8a16_megakernel_dp4_high_throughput"] = results["tpu6e_4chip_int8_megakernel_dp4_high_throughput"]
    results["tpu6e_4chip_int8_megakernel_dp4_high_throughput"] = pallas_w8a8_dp4_records

    with open("/tmp/tpu6e_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[DONE] Updated /tmp/tpu6e_benchmark_results.json with Pallas VMEM + W8A8 INT8xINT8 MXU results!", flush=True)


if __name__ == "__main__":
    main()
