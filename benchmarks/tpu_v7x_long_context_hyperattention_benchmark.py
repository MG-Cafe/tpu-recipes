#!/usr/bin/env python3
"""
TPU v7x (Ironwood) Full Unstripped Intelligence Evaluation (`raw_logits_sample` preserved)
& 20K Long-Context TPU HyperAttention (VMEM-Tiled Blockwise Bidirectional FlashAttention) Benchmark.
"""
import base64
import gc
import gzip
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

import protein_intelligence_suite as pis
import tpu_benchmark_and_megakernel as tbm


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


def make_masked_megakernel_forward(
    tp_size: int,
    tp_axis_name: str | None = "tp",
    use_mask: bool = True,
    bf16_router: bool = False,
):
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

        if use_mask:
            pos = jnp.arange(S, dtype=jnp.int32)
            vlen = valid_len_arr[0]
            valid_1d = pos < vlen
            attn_mask = (
                (valid_1d[None, None, :, None] & valid_1d[None, None, None, :])
                | jnp.eye(S, dtype=jnp.bool_)[None, None, :, :]
            )
        else:
            attn_mask = None

        def layer_scan_body(x_curr, lp):
            norm_x = tbm.tpu_rmsnorm(x_curr, lp["attn_ln_w"])
            qkv = jnp.einsum(
                "bsh,hctd->bsctd", norm_x, lp["w_qkv"], preferred_element_type=jnp.float32
            ).astype(jnp.bfloat16) + lp["b_qkv"][None, None, ...]
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q, k = tbm.apply_rope_128lane(q, k, cos, sin)
            attn_ctx = jax.nn.dot_product_attention(q, k, v, mask=attn_mask, is_causal=False)
            attn_proj_local = jnp.einsum(
                "bstd,tdo->bso", attn_ctx, lp["w_out"], preferred_element_type=jnp.float32
            ).astype(jnp.bfloat16) + (lp["b_out"] / tp_size)[None, None, :]
            attn_proj = jax.lax.psum(attn_proj_local, axis_name=tp_axis_name) if tp_axis_name else attn_proj_local
            x_mid = x_curr + attn_proj

            norm_mlp = tbm.tpu_rmsnorm(x_mid, lp["mlp_ln_w"])
            flat_x = norm_mlp.reshape(B * S, H)
            if bf16_router:
                rw = lp["router_w"].astype(jnp.bfloat16)
                N = flat_x.shape[0]
                two_N = 2 * N
                I_shard = lp["b1"].shape[-1] // 2
                logits = jnp.dot(flat_x, rw, preferred_element_type=jnp.bfloat16)
                topk_w, topk_idx = jax.lax.top_k(logits, 2)
                topk_w = jax.nn.softmax(topk_w, axis=-1).astype(jnp.bfloat16)
                flat_idx = topk_idx.reshape(-1)
                flat_w = topk_w.reshape(-1)
                slot_ids = jnp.arange(two_N, dtype=jnp.uint32)
                packed = (flat_idx.astype(jnp.uint32) << 24) | slot_ids
                sorted_packed = jax.lax.sort(packed)
                perm = (sorted_packed & jnp.uint32(0xFFFFFF)).astype(jnp.int32)
                sorted_exp_ids = (sorted_packed >> jnp.uint32(24)).astype(jnp.int32)
                token_ids = perm >> 1
                group_sizes = jnp.sum(jax.nn.one_hot(flat_idx, 8, dtype=jnp.int32), axis=0)
                x_sorted = flat_x[token_ids]
                h1 = jax.lax.ragged_dot(x_sorted, lp["w1"], group_sizes).astype(jnp.bfloat16) + lp["b1"][sorted_exp_ids]
                u = h1[:, :I_shard]
                v_act = h1[:, I_shard:]
                inter = (u * jax.nn.silu(v_act)).astype(jnp.bfloat16)
                h2 = jax.lax.ragged_dot(inter, lp["w2"], group_sizes).astype(jnp.bfloat16)
                weighted_h2 = (h2 + (lp["b2"] / tp_size)[sorted_exp_ids]) * flat_w[perm, None]
                out_unsorted = jnp.empty_like(weighted_h2).at[perm].set(weighted_h2)
                mlp_local = jnp.sum(out_unsorted.reshape(N, 2, H), axis=1)
                mlp_out = jax.lax.psum(mlp_local, axis_name=tp_axis_name) if tp_axis_name else mlp_local
            else:
                mlp_out = tbm.megablox_moe_block(
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
        hidden = tbm.tpu_rmsnorm(x_final, final_ln_w)
        logits = jnp.dot(hidden, out_emb_w, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        return logits, hidden

    return forward_fn


def make_long_ctx_forward(tp_size: int, use_hyperattention: bool = True):
    def forward_fn(
        input_ids: jax.Array,
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

        def layer_scan_body(x_curr, lp):
            norm_x = tbm.tpu_rmsnorm(x_curr, lp["attn_ln_w"])
            qkv = jnp.einsum(
                "bsh,hctd->bsctd", norm_x, lp["w_qkv"], preferred_element_type=jnp.float32
            ).astype(jnp.bfloat16) + lp["b_qkv"][None, None, ...]
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q, k = tbm.apply_rope_128lane(q, k, cos, sin)

            if use_hyperattention:
                attn_ctx = tpu_hyperattention_bidirectional(q, k, v)
            else:
                attn_ctx = baseline_xla_attention_bidirectional(q, k, v)

            attn_proj_local = jnp.einsum(
                "bstd,tdo->bso", attn_ctx, lp["w_out"], preferred_element_type=jnp.float32
            ).astype(jnp.bfloat16) + (lp["b_out"] / tp_size)[None, None, :]
            attn_proj = jax.lax.psum(attn_proj_local, axis_name="tp")
            x_mid = x_curr + attn_proj

            norm_mlp = tbm.tpu_rmsnorm(x_mid, lp["mlp_ln_w"])
            flat_x = norm_mlp.reshape(B * S, H)
            mlp_out = tbm.megablox_moe_block(
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
        hidden = tbm.tpu_rmsnorm(x_final, final_ln_w)
        logits = jnp.dot(hidden, out_emb_w, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        return logits, hidden

    return forward_fn


def serialize_raw_arrays(res: dict) -> dict:
    out = dict(res)
    if "raw_logits_sample" in out:
        raw = out["raw_logits_sample"]
        out["raw_logits_sample"] = {
            k: np.round(v, 6).tolist() if isinstance(v, np.ndarray) else v
            for k, v in raw.items()
        }
    return out


def main():
    while not os.path.exists(os.path.join(tbm.MODEL_DIR, "DONE")):
        print("Waiting for /dev/shm/gb_protein_16b_bf16/DONE...", flush=True)
        time.sleep(5)

    devices = jax.devices()
    print(f"=== TPU v7x Unstripped Intelligence & 20K HyperAttention Benchmark on {len(devices)} TensorCores ===", flush=True)

    word_emb_np, final_ln_w_np, out_emb_w_np, stacked_np = tbm.load_checkpoint_arrays(tbm.MODEL_DIR)
    cos_cache, sin_cache = tbm.precompute_rope_tpu(max_seq_len=20480, head_dim=64)

    lp_specs = {
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
    }

    def run_intel_config(mesh, tp_size, pad_tp8, use_mask, bf16_router, label):
        rep = NamedSharding(mesh, P())
        lp = tbm.prepare_sharded_layer_params(stacked_np, mesh, tp_size=tp_size, pad_for_tp8=pad_tp8)
        w_emb = jax.device_put(word_emb_np, rep)
        f_ln = jax.device_put(final_ln_w_np, rep)
        o_emb = jax.device_put(out_emb_w_np, rep)
        cos = jax.device_put(cos_cache, rep)
        sin = jax.device_put(sin_cache, rep)
        fn = jax.jit(
            shard_map(
                make_masked_megakernel_forward(tp_size=tp_size, tp_axis_name="tp", use_mask=use_mask, bf16_router=bf16_router),
                mesh=mesh,
                in_specs=(P(), P(), P(), P(), P(), P(), P(), lp_specs),
                out_specs=(P(), P()),
                check_rep=False,
            )
        )
        def eval_fn(ids_np: np.ndarray, valid_len: int):
            ids_j = jax.device_put(ids_np.astype(np.int32), rep)
            vlen_j = jax.device_put(np.array([valid_len], dtype=np.int32), rep)
            l_j, h_j = fn(ids_j, vlen_j, w_emb, f_ln, o_emb, cos, sin, lp)
            jax.block_until_ready(l_j)
            return np.asarray(l_j, dtype=np.float32), np.asarray(h_j, dtype=np.float32)

        res = pis.evaluate_intelligence_suite(eval_fn, label)
        print(f"[{label}] MLM={res['mlm_aggregate']} | ClinVar={res['clinvar_dms_summary']}", flush=True)
        del lp, w_emb, f_ln, o_emb, cos, sin, fn
        gc.collect()
        return res

    mesh_tp2 = Mesh(np.array(devices[:2]), ("tp",))
    mesh_tp8 = Mesh(np.array(devices[:8]), ("tp",))

    res_naive = run_intel_config(mesh_tp2, 2, False, False, True, "TorchTPU_v7x_Naive_Port_Unmasked_BF16Router")
    res_tp2 = run_intel_config(mesh_tp2, 2, False, True, False, "TorchTPU_v7x_1Chip_BF16_TP2")
    res_tp8 = run_intel_config(mesh_tp8, 8, True, True, False, "TorchTPU_v7x_4Chip_BF16_TP8")

    res_naive["comparison_vs_tpu7x_tp2"] = pis.compare_against_reference(res_naive, res_tp2)
    res_tp8["comparison_vs_tpu7x_tp2"] = pis.compare_against_reference(res_tp8, res_tp2)

    # -------------------------------------------------------------------------
    # Long-Context Sweep (256 -> 20,480): 1x TPU v7x (TP=2) & 4x TPU v7x (TP=8)
    # -------------------------------------------------------------------------
    seq_lens_long = [256, 512, 1024, 2048, 4096, 8192, 16384, 20480]

    def run_long_ctx_sweep(mesh, tp_size, pad_tp8, tag):
        rep = NamedSharding(mesh, P())
        lp = tbm.prepare_sharded_layer_params(stacked_np, mesh, tp_size=tp_size, pad_for_tp8=pad_tp8)
        w_emb = jax.device_put(word_emb_np, rep)
        f_ln = jax.device_put(final_ln_w_np, rep)
        o_emb = jax.device_put(out_emb_w_np, rep)
        cos = jax.device_put(cos_cache, rep)
        sin = jax.device_put(sin_cache, rep)

        in_s = (P(), P(), P(), P(), P(), P(), lp_specs)
        jit_base = jax.jit(shard_map(make_long_ctx_forward(tp_size, False), mesh=mesh, in_specs=in_s, out_specs=(P(), P()), check_rep=False))
        jit_hyper = jax.jit(shard_map(make_long_ctx_forward(tp_size, True), mesh=mesh, in_specs=in_s, out_specs=(P(), P()), check_rep=False))

        base_recs, hyper_recs = [], []
        for S in seq_lens_long:
            ids = jax.device_put(np.random.randint(1, 22, size=(1, S), dtype=np.int32), rep)
            # 1. Baseline XLA Attention
            try:
                for _ in range(2):
                    l_out, _ = jit_base(ids, w_emb, f_ln, o_emb, cos, sin, lp)
                    jax.block_until_ready(l_out)
                lats = []
                iters = 8 if S <= 4096 else 4
                for _ in range(iters):
                    t0 = time.perf_counter()
                    l_out, _ = jit_base(ids, w_emb, f_ln, o_emb, cos, sin, lp)
                    jax.block_until_ready(l_out)
                    lats.append((time.perf_counter() - t0) * 1000.0)
                med_ms = sorted(lats)[len(lats) // 2]
                tps = S / (med_ms / 1000.0)
                tflops = (tps * tbm.compute_active_flops_per_token(S)) / 1e12
                rec_b = {
                    "batch_size": 1, "seq_len": S, "tokens_per_batch": S, "status": "OK",
                    "median_latency_ms": round(med_ms, 2), "tokens_per_sec": round(tps, 1), "achieved_tflops": round(tflops, 2),
                }
                print(f"[{tag} Baseline XLA] S={S:5d} | {med_ms:8.2f} ms | {tps:8.1f} tok/s | {tflops:6.2f} TFLOPS", flush=True)
            except Exception as e:
                rec_b = {
                    "batch_size": 1, "seq_len": S, "tokens_per_batch": S, "status": "OOM",
                    "median_latency_ms": None, "tokens_per_sec": 0.0, "achieved_tflops": 0.0,
                    "error": str(e)[:200],
                }
                print(f"[{tag} Baseline XLA] S={S:5d} | OOM! ({rec_b['error']})", flush=True)
            base_recs.append(rec_b)

            # 2. TPU HyperAttention
            for _ in range(2):
                l_out, _ = jit_hyper(ids, w_emb, f_ln, o_emb, cos, sin, lp)
                jax.block_until_ready(l_out)
            lats = []
            iters = 10 if S <= 4096 else 5
            for _ in range(iters):
                t0 = time.perf_counter()
                l_out, _ = jit_hyper(ids, w_emb, f_ln, o_emb, cos, sin, lp)
                jax.block_until_ready(l_out)
                lats.append((time.perf_counter() - t0) * 1000.0)
            med_ms = sorted(lats)[len(lats) // 2]
            tps = S / (med_ms / 1000.0)
            tflops = (tps * tbm.compute_active_flops_per_token(S)) / 1e12
            rec_h = {
                "batch_size": 1, "seq_len": S, "tokens_per_batch": S, "status": "OK",
                "median_latency_ms": round(med_ms, 2), "tokens_per_sec": round(tps, 1), "achieved_tflops": round(tflops, 2),
            }
            hyper_recs.append(rec_h)
            print(f"[{tag} TPU HyperAttention] S={S:5d} | {med_ms:8.2f} ms | {tps:8.1f} tok/s | {tflops:6.2f} TFLOPS", flush=True)

        del lp, w_emb, f_ln, o_emb, cos, sin, jit_base, jit_hyper
        gc.collect()
        return base_recs, hyper_recs

    long_1c_base, long_1c_hyper = run_long_ctx_sweep(mesh_tp2, 2, False, "1x v7x TP=2")
    long_4c_base, long_4c_hyper = run_long_ctx_sweep(mesh_tp8, 8, True, "4x v7x TP=8")

    payload = {
        "intelligence_evaluations_with_raw_logits": {
            "TorchTPU_v7x_Naive_Port_Unmasked_BF16Router": serialize_raw_arrays(res_naive),
            "TorchTPU_v7x_1Chip_BF16_TP2": serialize_raw_arrays(res_tp2),
            "TorchTPU_v7x_4Chip_BF16_TP8": serialize_raw_arrays(res_tp8),
        },
        "long_context_1chip_bf16_tp2_baseline_xla": long_1c_base,
        "long_context_1chip_bf16_tp2_hyperattention": long_1c_hyper,
        "long_context_4chip_bf16_tp8_baseline_xla": long_4c_base,
        "long_context_4chip_bf16_tp8_hyperattention": long_4c_hyper,
    }
    raw_bytes = json.dumps(payload).encode("utf-8")
    with open("/tmp/v7x_audit_and_hyperattention_results.json", "wb") as f:
        f.write(raw_bytes)

    b64 = base64.b64encode(gzip.compress(raw_bytes)).decode("ascii")
    chunk_sz = 24000
    chunks = [b64[i : i + chunk_sz] for i in range(0, len(b64), chunk_sz)]
    for idx, ch in enumerate(chunks):
        print(f"V7X_GZIP_B64_CHUNK_{idx+1}_OF_{len(chunks)}:{ch}", flush=True)
    print("=== Saved /tmp/v7x_audit_and_hyperattention_results.json ===", flush=True)


if __name__ == "__main__":
    main()
