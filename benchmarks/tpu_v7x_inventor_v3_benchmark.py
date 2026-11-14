#!/usr/bin/env python3
"""
TPU v7x (Ironwood) Inventor Bio-Megakernel v3 + TPU HyperAttention v2 (SplashAttention Megacore)
Complete End-to-End Benchmark Suite (1x TPU v7x = 2 TensorCores, 4x TPU v7x = 8 TensorCores).

Benchmarks:
  1. 1x TPU v7x (BF16 TP=2 + Bio-Megakernel v3 + HyperAttention v2): Latency, Batch Throughput & 20K Long-Context
  2. 1x TPU v7x (INT8 TP=2 + Bio-Megakernel v3 + HyperAttention v2): Ultra-Low Latency & 20K Long-Context
  3. 1x TPU v7x (INT8 DP=2 = 2xTP1 across 2 on-die 96GB TensorCores): Peak Single-Chip Throughput
  4. 4x TPU v7x (BF16 TP=8 & INT8 TP=8 + HyperAttention v2): Ultra-Low Latency & 20K Long-Context
  5. 4x TPU v7x (BF16 4xTP2 & INT8 8xTP1 across 8 96GB TensorCores): Peak 4-Chip Cluster Throughput
  6. Numerical Parity against BF16 Reference on Real Biological Sequences (GFP P42212 + SARS-CoV-2 RBD-ACE2)
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
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", "/dev/shm/gb_protein_16b_bf16")
OUT_JSON = os.environ.get("OUT_JSON", os.path.join(os.path.dirname(_HERE), "results", "v7x_inventor_v3_results.json"))
sys.path.insert(0, _HERE)
sys.path.insert(0, MODEL_DIR)

try:
    import tpu_v7x_megakernel_benchmark as tbm
except ImportError:
    sys.path.insert(0, "/tmp")
    import tpu_benchmark_and_megakernel as tbm
from tokenization_fm4bio import FM4BioTokenizer

GFP_WT_SEQ = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
    "VTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLV"
    "NRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLAD"
    "HYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)
SARS2_RBD_ACE2_SEQ = (
    "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFK"
    "CYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNS"
    "NNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQ"
    "PTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF | "
    "STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQST"
    "LAQMYPLQEIQNLTVKLQLQALQQNGSSVLSEDKSKRLNTILNTMSTIYSTGKVCNPDNP"
    "QECLLLEPGLNEIMANSLDYNERLWAWESWRSEVGKQLRPLYEEYVVLKNEMARANHYED"
    "YGDYWRGDYEVNGVDGYDYSRGQLIEDVEHTFEEIKPLYEHLHAYVRAKLMNAYPSYISP"
    "IGCLPAHLLGDMWGRFWTNLYSLTVPFGQKPNIDVTDAMVDQAWDAQRIFKEAEKFFVSV"
    "GLPNMTQGFWENSMLTDPGNVQKAVCHPTAWDLGKGDFRILMCTKVTMDDFLTAHHEMGHI"
    "QYDMAYAAQPFLLRNGANEGFHEAVGEIMSLSAATPKHLKSIGLLSPDFQEDNETEINFLL"
    "KQALTIVGTLPFTYMLEKWRWMVFKGEIPKDQWMKKWWEMKREIVGVVEPVPHDETYCDPA"
    "SLFHVSNDYSFIRYYTRTLYQFQFQEALCQAAKHEGPLHKCDISNSTEAGQKLFNMLRLGK"
    "SEPWTLALENVVGAKNMNVRPLLNYFEPLFTWLKDQNKNSFVGWSTDWSPYAD"
)


def precompute_rope_v3(max_seq_len: int = 20480, head_dim: int = 64, theta: float = 10000.0):
    half = head_dim // 2
    freq_seq = jnp.arange(0, half, dtype=jnp.float32)
    inv_freq = 1.0 / (theta ** (freq_seq / float(half)))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    cos_c = jnp.cos(emb)[None, None, :, :].astype(jnp.bfloat16)
    sin_c = jnp.sin(emb)[None, None, :, :].astype(jnp.bfloat16)
    return np.asarray(cos_c), np.asarray(sin_c)


def apply_rope_native_bhsd(q: jax.Array, k: jax.Array, cos: jax.Array, sin: jax.Array):
    q1, q2 = q[..., :32], q[..., 32:]
    k1, k2 = k[..., :32], k[..., 32:]
    q_rot = jnp.concatenate([-q2, q1], axis=-1)
    k_rot = jnp.concatenate([-k2, k1], axis=-1)
    return (q * cos) + (q_rot * sin), (k * cos) + (k_rot * sin)


@jax.jit
def _quant_layer_v3(w1_l, b1_l, w2_l):
    w1_f = w1_l.astype(jnp.float32)
    b1_f = b1_l.astype(jnp.float32)
    w2_f = w2_l.astype(jnp.float32)

    s1 = jnp.maximum(jnp.max(jnp.abs(w1_f), axis=1, keepdims=True) / 127.0, 1e-6)
    q1 = jnp.clip(jnp.round(w1_f / s1), -127, 127).astype(jnp.int8)
    s1_sq = s1.squeeze(1)

    s2 = jnp.maximum(jnp.max(jnp.abs(w2_f), axis=2, keepdims=True) / 127.0, 1e-6)
    q2 = jnp.clip(jnp.round(w2_f / s2), -127, 127).astype(jnp.int8)
    s2_in = s2.squeeze(2)

    s1_folded = jnp.stack([s1_sq[:, 0, :] * s2_in, s1_sq[:, 1, :]], axis=1).astype(jnp.bfloat16)
    b1_folded = jnp.stack([b1_f[:, 0, :] * s2_in, b1_f[:, 1, :]], axis=1).astype(jnp.bfloat16)
    return q1, s1_folded, b1_folded, q2


def prepare_all_v3_weights(stacked_np: dict):
    t0 = time.time()
    dev0 = jax.devices()[0]
    q1_list, s1_list, b1_list, q2_list = [], [], [], []
    for l in range(36):
        w1_d = jax.device_put(stacked_np["w1"][l], dev0)
        b1_d = jax.device_put(stacked_np["b1"][l], dev0)
        w2_d = jax.device_put(stacked_np["w2"][l], dev0)
        q1, s1_f, b1_f, q2 = _quant_layer_v3(w1_d, b1_d, w2_d)
        q1_list.append(np.asarray(q1))
        s1_list.append(np.asarray(s1_f))
        b1_list.append(np.asarray(b1_f))
        q2_list.append(np.asarray(q2))

    w_qkv_v3 = np.transpose(stacked_np["w_qkv"], (0, 2, 1, 3, 4))  # [36, 3, H, 36, 64]
    b_qkv_v3 = stacked_np["b_qkv"][:, :, :, None, :]               # [36, 3, 36, 1, 64]
    out = {
        "attn_ln_w": stacked_np["attn_ln_w"],
        "w_qkv": w_qkv_v3,
        "b_qkv": b_qkv_v3,
        "w_out": stacked_np["w_out"],
        "b_out": stacked_np["b_out"],
        "mlp_ln_w": stacked_np["mlp_ln_w"],
        "router_w": stacked_np["router_w"],  # float32 for exact routing parity
        "w1_bf16_5d": stacked_np["w1"],
        "b1_bf16_4d": stacked_np["b1"],
        "w2_bf16": stacked_np["w2"],
        "w1_q_5d": np.stack(q1_list, axis=0),
        "w1_s_4d": np.stack(s1_list, axis=0),
        "b1_4d": np.stack(b1_list, axis=0),
        "w2_q": np.stack(q2_list, axis=0),
        "b2": stacked_np["b2"],
    }
    print(f"[QUANT v3] Prepared BF16 & INT8 Folded weights in {time.time() - t0:.2f}s!", flush=True)
    return out


def place_v3_params(v3_weights: dict, mesh: Mesh, tp_size: int, tp_axis_name: str | None, quant_mode: str = "int8", pad_for_tp8: bool = False):
    w_qkv = v3_weights["w_qkv"]
    b_qkv = v3_weights["b_qkv"]
    w_out = v3_weights["w_out"]

    if pad_for_tp8:
        # Pad 36 heads -> 40 heads (5 per TensorCore) and 7680 -> 8192 intermediate (1024 = 8x128 per TensorCore)
        w_qkv = np.pad(w_qkv, ((0, 0), (0, 0), (0, 0), (0, 4), (0, 0)))
        b_qkv = np.pad(b_qkv, ((0, 0), (0, 0), (0, 4), (0, 0), (0, 0)))
        w_out = np.pad(w_out, ((0, 0), (0, 4), (0, 0), (0, 0)))
        I_total = 8192
    else:
        I_total = 7680

    I_shard = I_total // tp_size

    if quant_mode == "bf16":
        w1_5d = v3_weights["w1_bf16_5d"]
        b1_4d = v3_weights["b1_bf16_4d"]
        w2_4d = v3_weights["w2_bf16"]
        if pad_for_tp8:
            w1_5d = np.pad(w1_5d, ((0, 0), (0, 0), (0, 0), (0, 0), (0, 512)))
            b1_4d = np.pad(b1_4d, ((0, 0), (0, 0), (0, 0), (0, 512)))
            w2_4d = np.pad(w2_4d, ((0, 0), (0, 0), (0, 512), (0, 0)))
        w1_flat = np.swapaxes(w1_5d.reshape(36, 8, 2304, 2, tp_size, I_shard), 3, 4).reshape(36, 8, 2304, 2 * I_total)
        b1_flat = np.swapaxes(b1_4d.reshape(36, 8, 2, tp_size, I_shard), 2, 3).reshape(36, 8, 2 * I_total)
        specs = {
            "attn_ln_w": P(),
            "w_qkv": P(None, None, None, tp_axis_name, None),
            "b_qkv": P(None, None, tp_axis_name, None, None),
            "w_out": P(None, tp_axis_name, None, None),
            "b_out": P(),
            "mlp_ln_w": P(),
            "router_w": P(),
            "w1": P(None, None, None, tp_axis_name),
            "b1": P(None, None, tp_axis_name),
            "w2": P(None, None, tp_axis_name, None),
            "b2": P(),
        }
        raw = {
            "attn_ln_w": v3_weights["attn_ln_w"],
            "w_qkv": w_qkv,
            "b_qkv": b_qkv,
            "w_out": w_out,
            "b_out": v3_weights["b_out"],
            "mlp_ln_w": v3_weights["mlp_ln_w"],
            "router_w": v3_weights["router_w"],
            "w1": w1_flat,
            "b1": b1_flat,
            "w2": w2_4d,
            "b2": v3_weights["b2"],
        }
    else:
        w1_q_5d = v3_weights["w1_q_5d"]
        w1_s_4d = v3_weights["w1_s_4d"]
        b1_4d = v3_weights["b1_4d"]
        w2_q_4d = v3_weights["w2_q"]
        if pad_for_tp8:
            w1_q_5d = np.pad(w1_q_5d, ((0, 0), (0, 0), (0, 0), (0, 0), (0, 512)))
            w1_s_4d = np.pad(w1_s_4d, ((0, 0), (0, 0), (0, 0), (0, 512)))
            b1_4d = np.pad(b1_4d, ((0, 0), (0, 0), (0, 0), (0, 512)))
            w2_q_4d = np.pad(w2_q_4d, ((0, 0), (0, 0), (0, 512), (0, 0)))
        w1_q_flat = np.swapaxes(w1_q_5d.reshape(36, 8, 2304, 2, tp_size, I_shard), 3, 4).reshape(36, 8, 2304, 2 * I_total)
        w1_s_flat = np.swapaxes(w1_s_4d.reshape(36, 8, 2, tp_size, I_shard), 2, 3).reshape(36, 8, 2 * I_total)
        b1_flat = np.swapaxes(b1_4d.reshape(36, 8, 2, tp_size, I_shard), 2, 3).reshape(36, 8, 2 * I_total)
        specs = {
            "attn_ln_w": P(),
            "w_qkv": P(None, None, None, tp_axis_name, None),
            "b_qkv": P(None, None, tp_axis_name, None, None),
            "w_out": P(None, tp_axis_name, None, None),
            "b_out": P(),
            "mlp_ln_w": P(),
            "router_w": P(),
            "w1_q": P(None, None, None, tp_axis_name),
            "w1_s": P(None, None, tp_axis_name),
            "b1": P(None, None, tp_axis_name),
            "w2_q": P(None, None, tp_axis_name, None),
            "b2": P(),
        }
        raw = {
            "attn_ln_w": v3_weights["attn_ln_w"],
            "w_qkv": w_qkv,
            "b_qkv": b_qkv,
            "w_out": w_out,
            "b_out": v3_weights["b_out"],
            "mlp_ln_w": v3_weights["mlp_ln_w"],
            "router_w": v3_weights["router_w"],
            "w1_q": w1_q_flat,
            "w1_s": w1_s_flat,
            "b1": b1_flat,
            "w2_q": w2_q_4d,
            "b2": v3_weights["b2"],
        }

    placed = {k: jax.device_put(v, NamedSharding(mesh, specs[k])) for k, v in raw.items()}
    return placed, specs


def make_forward_inventor_v3(
    tp_size: int = 1,
    tp_axis_name: str | None = None,
    quant_mode: str = "int8",
    use_hyperattention: bool = True,
):
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
        H_heads_local = layer_params["w_qkv"].shape[3]
        D = 64
        sm_scale = 1.0 / math.sqrt(float(D))

        x = word_emb[input_ids]
        cos = cos_c[:, :, :S, :]
        sin = sin_c[:, :, :S, :]

        if not use_hyperattention:
            attn_mode = "baseline_xla"
        elif S >= 2048:
            attn_mode = "splash"
            bq = min(1024, S)
            bkv = min(4096, S)
            bkv_c = min(2048, S)
            mask = splash_attention_mask.MultiHeadMask(
                [splash_attention_mask.FullMask((S, S)) for _ in range(H_heads_local)]
            )
            splash_bs = splash_attention_kernel.BlockSizes(
                block_q=bq, block_kv=bkv, block_kv_compute=bkv_c
            )
            splash_fn = splash_attention_kernel.make_splash_mha(
                mask, head_shards=1, q_seq_shards=1, block_sizes=splash_bs
            )
        elif S >= 1024:
            attn_mode = "fa"
            bq = min(512, S)
            fa_bs = pallas_tpu_fa.BlockSizes(block_q=bq, block_k_major=bq, block_k=bq, block_b=1)
        else:
            attn_mode = "vmem_einsum"

        def layer_scan_body(x_curr, lp):
            norm_x = tbm.tpu_rmsnorm(x_curr, lp["attn_ln_w"])
            qkv = (
                jnp.einsum("bsh,chtd->cbtsd", norm_x, lp["w_qkv"], preferred_element_type=jnp.bfloat16)
                + lp["b_qkv"][:, None, :, :, :]
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            q, k = apply_rope_native_bhsd(q, k, cos, sin)

            if attn_mode == "splash":
                attn_out = jax.vmap(splash_fn)(q * sm_scale, k, v).astype(jnp.bfloat16)
            elif attn_mode == "fa":
                attn_out = pallas_tpu_fa.flash_attention(
                    q, k, v, causal=False, sm_scale=sm_scale, block_sizes=fa_bs
                ).astype(jnp.bfloat16)
            else:
                scores = jnp.einsum("bhsd,bhtd->bhst", q, k, preferred_element_type=jnp.float32) * sm_scale
                probs = jax.nn.softmax(scores, axis=-1).astype(jnp.bfloat16)
                attn_out = jnp.einsum("bhst,bhtd->bhsd", probs, v, preferred_element_type=jnp.bfloat16)

            attn_proj_local = (
                jnp.einsum("btsd,tdo->bso", attn_out, lp["w_out"], preferred_element_type=jnp.bfloat16)
                + (lp["b_out"] / tp_size)[None, None, :]
            )
            attn_proj = (
                jax.lax.psum(attn_proj_local, axis_name=tp_axis_name)
                if tp_axis_name is not None
                else attn_proj_local
            )
            x_mid = x_curr + attn_proj

            norm_mlp = tbm.tpu_rmsnorm(x_mid, lp["mlp_ln_w"])
            flat_x = norm_mlp.reshape(B * S, H)
            N = flat_x.shape[0]
            two_N = 2 * N
            I_shard = lp["b1"].shape[-1] // 2

            logits = jnp.dot(flat_x, lp["router_w"], preferred_element_type=jnp.float32)
            topk_w, topk_idx = jax.lax.top_k(logits, 2)
            topk_w = jax.nn.softmax(topk_w, axis=-1).astype(jnp.bfloat16)
            flat_idx = topk_idx.reshape(-1)
            slot_ids = jnp.arange(two_N, dtype=jnp.uint32)
            packed = (flat_idx.astype(jnp.uint32) << 24) | slot_ids
            sorted_packed = jax.lax.sort(packed)
            perm = (sorted_packed & jnp.uint32(0xFFFFFF)).astype(jnp.int32)
            sorted_exp_ids = (sorted_packed >> jnp.uint32(24)).astype(jnp.int32)
            token_ids = perm >> 1

            oh_2n = jax.nn.one_hot(topk_idx, 8, dtype=jnp.bfloat16)
            probs_sparse = jnp.sum(oh_2n * topk_w[:, :, None], axis=1)
            group_sizes = jnp.sum(oh_2n, axis=(0, 1)).astype(jnp.int32)
            oh_sorted = jax.nn.one_hot(sorted_exp_ids, 8, dtype=jnp.bfloat16)

            x_sorted = flat_x[token_ids]
            if quant_mode == "bf16":
                h1 = jax.lax.ragged_dot(x_sorted, lp["w1"], group_sizes, preferred_element_type=jnp.bfloat16) + jnp.dot(oh_sorted, lp["b1"])
                u, v_act = h1[:, :I_shard], h1[:, I_shard:]
                inter = (u * jax.nn.silu(v_act)).astype(jnp.bfloat16)
                h2 = jax.lax.ragged_dot(inter, lp["w2"], group_sizes, preferred_element_type=jnp.bfloat16)
            else:
                h1_raw = jax.lax.ragged_dot(x_sorted, lp["w1_q"], group_sizes, preferred_element_type=jnp.bfloat16)
                h1 = h1_raw * jnp.dot(oh_sorted, lp["w1_s"]) + jnp.dot(oh_sorted, lp["b1"])
                u, v_act = h1[:, :I_shard], h1[:, I_shard:]
                inter = (u * jax.nn.silu(v_act)).astype(jnp.bfloat16)
                h2 = jax.lax.ragged_dot(inter, lp["w2_q"], group_sizes, preferred_element_type=jnp.bfloat16)

            if N <= 512:
                flat_w = topk_w.reshape(-1)
                weighted_h2 = h2 * flat_w[perm, None]
                out_unsorted = jnp.empty_like(weighted_h2).at[perm].set(weighted_h2)
                mlp_local = jnp.sum(out_unsorted.reshape(N, 2, H), axis=1) + jnp.dot(probs_sparse, lp["b2"] / tp_size)
            else:
                inv2 = jnp.empty_like(perm).at[perm].set(jnp.arange(two_N, dtype=jnp.int32)).reshape(N, 2)
                mlp_local = (
                    h2[inv2[:, 0]] * topk_w[:, 0:1]
                    + h2[inv2[:, 1]] * topk_w[:, 1:2]
                    + jnp.dot(probs_sparse, lp["b2"] / tp_size)
                )

            mlp_out = (
                jax.lax.psum(mlp_local, axis_name=tp_axis_name)
                if tp_axis_name is not None
                else mlp_local
            )
            return x_mid + mlp_out.reshape(B, S, H), None

        x_final, _ = jax.lax.scan(layer_scan_body, x, layer_params)
        hidden = tbm.tpu_rmsnorm(x_final, final_ln_w)
        logits = jnp.dot(hidden, out_emb_w, preferred_element_type=jnp.bfloat16)
        return logits, hidden

    return forward_fn


def run_shape_sweep(jit_fn, w_emb, f_ln, o_emb, cos, sin, lp, id_sharding, shapes, tier_name, chips, tensorcores, quant_mode, cost_od):
    records = []
    for B, S in shapes:
        ids = jax.device_put(np.random.randint(1, 22, size=(B, S), dtype=np.int32), id_sharding)
        for _ in range(2):
            l_out, _ = jit_fn(ids, w_emb, f_ln, o_emb, cos, sin, lp)
            jax.block_until_ready(l_out)
        lats = []
        iters = 10 if S <= 2048 else 5
        for _ in range(iters):
            t0 = time.perf_counter()
            l_out, _ = jit_fn(ids, w_emb, f_ln, o_emb, cos, sin, lp)
            jax.block_until_ready(l_out)
            lats.append((time.perf_counter() - t0) * 1000.0)
        med_ms = sorted(lats)[len(lats) // 2]
        tps = (B * S) / (med_ms / 1000.0)
        tflops = (tps * tbm.compute_active_flops_per_token(S)) / 1e12
        rec = {
            "tier": tier_name,
            "chips": chips,
            "tensorcores": tensorcores,
            "quant_mode": quant_mode,
            "batch_size": B,
            "seq_len": S,
            "tokens_per_batch": B * S,
            "median_latency_ms": round(med_ms, 2),
            "tokens_per_sec": round(tps, 1),
            "tok_per_s_per_dollar_ondemand": round(tps / cost_od, 1),
            "tok_per_s_per_dollar_dws_flex": round(tps / (cost_od * 0.5), 1),
            "achieved_tflops": round(tflops, 2),
            "mxu_util_pct": round((tflops / (chips * 2307.0)) * 100.0, 2),
        }
        records.append(rec)
        print(
            f"[{tier_name}] B={B:3d}, S={S:5d} | {med_ms:7.2f} ms | {tps:9.1f} tok/s | {tps/cost_od:7.1f} tok/s/$(OD) | {tflops:6.2f} TFLOPS",
            flush=True,
        )
    return records


def main():
    while not os.path.exists(os.path.join(MODEL_DIR, "DONE")):
        print("Waiting for /dev/shm/gb_protein_16b_bf16/DONE...", flush=True)
        time.sleep(5)

    devices = jax.devices()
    print(f"=== TPU v7x Inventor v3 Full Benchmark on {len(devices)} TensorCores ===", flush=True)

    tokenizer = FM4BioTokenizer(vocab_file=os.path.join(MODEL_DIR, "vocab_protein.txt"))
    word_emb_np, final_ln_w_np, out_emb_w_np, stacked_np = tbm.load_checkpoint_arrays(MODEL_DIR)
    cos_c, sin_c = precompute_rope_v3(max_seq_len=20480, head_dim=64)
    v3_weights = prepare_all_v3_weights(stacked_np)

    ids_gfp = tokenizer.encode(" ".join(list(GFP_WT_SEQ)))
    ids_sars = tokenizer.encode(" ".join(list(SARS2_RBD_ACE2_SEQ[:240])))
    pad_len = 256
    batch_bio = np.zeros((2, pad_len), dtype=np.int32)
    batch_bio[0, : min(len(ids_gfp), pad_len)] = ids_gfp[:pad_len]
    batch_bio[1, : min(len(ids_sars), pad_len)] = ids_sars[:pad_len]

    std_shapes = [
        (1, 256), (1, 512), (1, 1024), (1, 2048),
        (2, 256), (4, 256), (4, 512), (8, 256), (8, 512), (16, 512), (32, 512),
    ]
    seq_lens_long = [256, 512, 1024, 2048, 4096, 8192, 16384, 20480]

    # -------------------------------------------------------------------------
    # PART 1: Tier 1a — 1x TPU v7x (BF16 TP=2 + Bio-Megakernel v3 + HyperAttention v2)
    # -------------------------------------------------------------------------
    mesh_tp2 = Mesh(np.array(devices[:2]), ("tp",))
    rep_tp2 = NamedSharding(mesh_tp2, P())
    lp_bf16_tp2, specs_bf16_tp2 = place_v3_params(v3_weights, mesh_tp2, tp_size=2, tp_axis_name="tp", quant_mode="bf16")
    w_emb_tp2 = jax.device_put(word_emb_np, rep_tp2)
    f_ln_tp2 = jax.device_put(final_ln_w_np, rep_tp2)
    o_emb_tp2 = jax.device_put(out_emb_w_np, rep_tp2)
    cos_tp2 = jax.device_put(cos_c, rep_tp2)
    sin_tp2 = jax.device_put(sin_c, rep_tp2)
    jit_bf16_tp2 = jax.jit(
        shard_map(
            make_forward_inventor_v3(tp_size=2, tp_axis_name="tp", quant_mode="bf16", use_hyperattention=True),
            mesh=mesh_tp2,
            in_specs=(P(), P(), P(), P(), P(), P(), specs_bf16_tp2),
            out_specs=(P(), P()),
            check_rep=False,
        )
    )
    bio_tp2 = jax.device_put(batch_bio, rep_tp2)
    ref_logits_j, ref_hidden_j = jit_bf16_tp2(bio_tp2, w_emb_tp2, f_ln_tp2, o_emb_tp2, cos_tp2, sin_tp2, lp_bf16_tp2)
    jax.block_until_ready(ref_logits_j)
    ref_logits = np.asarray(ref_logits_j, dtype=np.float32)
    ref_hidden = np.asarray(ref_hidden_j, dtype=np.float32)

    tier1a_bf16_tp2 = run_shape_sweep(
        jit_bf16_tp2, w_emb_tp2, f_ln_tp2, o_emb_tp2, cos_tp2, sin_tp2, lp_bf16_tp2, rep_tp2,
        std_shapes,
        "1x TPU v7x (BF16 TP=2 + Bio-Megakernel v3 + HyperAttention v2)", 1, 2, "bf16", 12.00
    )
    long_1c_bf16_tp2 = run_shape_sweep(
        jit_bf16_tp2, w_emb_tp2, f_ln_tp2, o_emb_tp2, cos_tp2, sin_tp2, lp_bf16_tp2, rep_tp2,
        [(1, s) for s in seq_lens_long],
        "1x TPU v7x (BF16 TP=2 + Bio-Megakernel v3 + HyperAttention v2)", 1, 2, "bf16", 12.00
    )
    del lp_bf16_tp2, jit_bf16_tp2
    gc.collect()

    # -------------------------------------------------------------------------
    # PART 2: Tier 1b — 1x TPU v7x (INT8 Folded TP=2 + Bio-Megakernel v3 + HyperAttention v2)
    # -------------------------------------------------------------------------
    lp_int8_tp2, specs_int8_tp2 = place_v3_params(v3_weights, mesh_tp2, tp_size=2, tp_axis_name="tp", quant_mode="int8")
    jit_int8_tp2 = jax.jit(
        shard_map(
            make_forward_inventor_v3(tp_size=2, tp_axis_name="tp", quant_mode="int8", use_hyperattention=True),
            mesh=mesh_tp2,
            in_specs=(P(), P(), P(), P(), P(), P(), specs_int8_tp2),
            out_specs=(P(), P()),
            check_rep=False,
        )
    )
    v3_logits_j, v3_hidden_j = jit_int8_tp2(bio_tp2, w_emb_tp2, f_ln_tp2, o_emb_tp2, cos_tp2, sin_tp2, lp_int8_tp2)
    jax.block_until_ready(v3_logits_j)
    v3_logits = np.asarray(v3_logits_j, dtype=np.float32)
    v3_hidden = np.asarray(v3_hidden_j, dtype=np.float32)

    cos_logits = float(
        np.sum(ref_logits * v3_logits) / (np.linalg.norm(ref_logits) * np.linalg.norm(v3_logits) + 1e-12)
    )
    cos_hidden = float(
        np.sum(ref_hidden * v3_hidden) / (np.linalg.norm(ref_hidden) * np.linalg.norm(v3_hidden) + 1e-12)
    )
    top1_agree = float(np.mean(np.argmax(ref_logits[:, :, 4:24], axis=-1) == np.argmax(v3_logits[:, :, 4:24], axis=-1)))
    parity_summary = {
        "cosine_sim_logits": round(cos_logits, 6),
        "cosine_sim_hidden": round(cos_hidden, 6),
        "aa_top1_agreement": round(top1_agree, 4),
    }
    print(f"[PARITY v7x] Bio-Megakernel v3 INT8 Folded vs BF16 Ref: {parity_summary}", flush=True)

    tier1b_int8_tp2 = run_shape_sweep(
        jit_int8_tp2, w_emb_tp2, f_ln_tp2, o_emb_tp2, cos_tp2, sin_tp2, lp_int8_tp2, rep_tp2,
        std_shapes,
        "1x TPU v7x (INT8 TP=2 + Bio-Megakernel v3 + HyperAttention v2)", 1, 2, "int8", 12.00
    )
    long_1c_int8_tp2 = run_shape_sweep(
        jit_int8_tp2, w_emb_tp2, f_ln_tp2, o_emb_tp2, cos_tp2, sin_tp2, lp_int8_tp2, rep_tp2,
        [(1, s) for s in seq_lens_long],
        "1x TPU v7x (INT8 TP=2 + Bio-Megakernel v3 + HyperAttention v2)", 1, 2, "int8", 12.00
    )
    del lp_int8_tp2, jit_int8_tp2, w_emb_tp2, f_ln_tp2, o_emb_tp2, cos_tp2, sin_tp2
    gc.collect()

    # -------------------------------------------------------------------------
    # PART 3: Tier 1c — 1x TPU v7x (INT8 DP=2 = 2xTP1 across 2 on-die 96GB TensorCores)
    # -------------------------------------------------------------------------
    mesh_dp2 = Mesh(np.array(devices[:2]), ("dp",))
    rep_dp2 = NamedSharding(mesh_dp2, P())
    dp_dp2 = NamedSharding(mesh_dp2, P("dp", None))
    lp_dp2, specs_dp2 = place_v3_params(v3_weights, mesh_dp2, tp_size=1, tp_axis_name=None, quant_mode="int8")
    w_emb_dp2 = jax.device_put(word_emb_np, rep_dp2)
    f_ln_dp2 = jax.device_put(final_ln_w_np, rep_dp2)
    o_emb_dp2 = jax.device_put(out_emb_w_np, rep_dp2)
    cos_dp2 = jax.device_put(cos_c, rep_dp2)
    sin_dp2 = jax.device_put(sin_c, rep_dp2)
    jit_dp2 = jax.jit(
        shard_map(
            make_forward_inventor_v3(tp_size=1, tp_axis_name=None, quant_mode="int8", use_hyperattention=True),
            mesh=mesh_dp2,
            in_specs=(P("dp", None), P(), P(), P(), P(), P(), specs_dp2),
            out_specs=(P("dp", None, None), P("dp", None, None)),
            check_rep=False,
        )
    )
    dp2_shapes = [(2, 256), (4, 256), (4, 512), (8, 256), (8, 512), (16, 512), (32, 512), (64, 512)]
    tier1c_int8_dp2 = run_shape_sweep(
        jit_dp2, w_emb_dp2, f_ln_dp2, o_emb_dp2, cos_dp2, sin_dp2, lp_dp2, dp_dp2,
        dp2_shapes,
        "1x TPU v7x (INT8 DP=2 = 2xTP1 across 2 on-die 96GB TensorCores)", 1, 2, "int8", 12.00
    )
    del lp_dp2, jit_dp2, w_emb_dp2, f_ln_dp2, o_emb_dp2, cos_dp2, sin_dp2
    gc.collect()

    # -------------------------------------------------------------------------
    # PART 4: Tier 2a & 2b — 4x TPU v7x (BF16 TP=8 & INT8 TP=8 + HyperAttention v2)
    # -------------------------------------------------------------------------
    mesh_tp8 = Mesh(np.array(devices[:8]), ("tp",))
    rep_tp8 = NamedSharding(mesh_tp8, P())
    w_emb_tp8 = jax.device_put(word_emb_np, rep_tp8)
    f_ln_tp8 = jax.device_put(final_ln_w_np, rep_tp8)
    o_emb_tp8 = jax.device_put(out_emb_w_np, rep_tp8)
    cos_tp8 = jax.device_put(cos_c, rep_tp8)
    sin_tp8 = jax.device_put(sin_c, rep_tp8)

    lp_bf16_tp8, specs_bf16_tp8 = place_v3_params(v3_weights, mesh_tp8, tp_size=8, tp_axis_name="tp", quant_mode="bf16", pad_for_tp8=True)
    jit_bf16_tp8 = jax.jit(
        shard_map(
            make_forward_inventor_v3(tp_size=8, tp_axis_name="tp", quant_mode="bf16", use_hyperattention=True),
            mesh=mesh_tp8,
            in_specs=(P(), P(), P(), P(), P(), P(), specs_bf16_tp8),
            out_specs=(P(), P()),
            check_rep=False,
        )
    )
    tier2a_bf16_tp8 = run_shape_sweep(
        jit_bf16_tp8, w_emb_tp8, f_ln_tp8, o_emb_tp8, cos_tp8, sin_tp8, lp_bf16_tp8, rep_tp8,
        std_shapes,
        "4x TPU v7x (BF16 TP=8 + Bio-Megakernel v3 + HyperAttention v2)", 4, 8, "bf16", 48.00
    )
    long_4c_bf16_tp8 = run_shape_sweep(
        jit_bf16_tp8, w_emb_tp8, f_ln_tp8, o_emb_tp8, cos_tp8, sin_tp8, lp_bf16_tp8, rep_tp8,
        [(1, s) for s in seq_lens_long],
        "4x TPU v7x (BF16 TP=8 + Bio-Megakernel v3 + HyperAttention v2)", 4, 8, "bf16", 48.00
    )
    del lp_bf16_tp8, jit_bf16_tp8
    gc.collect()

    lp_int8_tp8, specs_int8_tp8 = place_v3_params(v3_weights, mesh_tp8, tp_size=8, tp_axis_name="tp", quant_mode="int8", pad_for_tp8=True)
    jit_int8_tp8 = jax.jit(
        shard_map(
            make_forward_inventor_v3(tp_size=8, tp_axis_name="tp", quant_mode="int8", use_hyperattention=True),
            mesh=mesh_tp8,
            in_specs=(P(), P(), P(), P(), P(), P(), specs_int8_tp8),
            out_specs=(P(), P()),
            check_rep=False,
        )
    )
    tier2b_int8_tp8 = run_shape_sweep(
        jit_int8_tp8, w_emb_tp8, f_ln_tp8, o_emb_tp8, cos_tp8, sin_tp8, lp_int8_tp8, rep_tp8,
        std_shapes,
        "4x TPU v7x (INT8 TP=8 + Bio-Megakernel v3 + HyperAttention v2)", 4, 8, "int8", 48.00
    )
    long_4c_int8_tp8 = run_shape_sweep(
        jit_int8_tp8, w_emb_tp8, f_ln_tp8, o_emb_tp8, cos_tp8, sin_tp8, lp_int8_tp8, rep_tp8,
        [(1, s) for s in seq_lens_long],
        "4x TPU v7x (INT8 TP=8 + Bio-Megakernel v3 + HyperAttention v2)", 4, 8, "int8", 48.00
    )
    del lp_int8_tp8, jit_int8_tp8, w_emb_tp8, f_ln_tp8, o_emb_tp8, cos_tp8, sin_tp8
    gc.collect()

    # -------------------------------------------------------------------------
    # PART 5: Tier 3a & 3b — 4x TPU v7x Peak Cluster Throughput (BF16 4xTP2 & INT8 8xTP1)
    # -------------------------------------------------------------------------
    mesh_4xtp2 = Mesh(np.array(devices[:8]).reshape(4, 2), ("dp", "tp"))
    rep_4xtp2 = NamedSharding(mesh_4xtp2, P())
    dp_4xtp2 = NamedSharding(mesh_4xtp2, P("dp", None))
    lp_bf16_4xtp2, specs_bf16_4xtp2 = place_v3_params(v3_weights, mesh_4xtp2, tp_size=2, tp_axis_name="tp", quant_mode="bf16")
    w_emb_4xtp2 = jax.device_put(word_emb_np, rep_4xtp2)
    f_ln_4xtp2 = jax.device_put(final_ln_w_np, rep_4xtp2)
    o_emb_4xtp2 = jax.device_put(out_emb_w_np, rep_4xtp2)
    cos_4xtp2 = jax.device_put(cos_c, rep_4xtp2)
    sin_4xtp2 = jax.device_put(sin_c, rep_4xtp2)
    jit_bf16_4xtp2 = jax.jit(
        shard_map(
            make_forward_inventor_v3(tp_size=2, tp_axis_name="tp", quant_mode="bf16", use_hyperattention=True),
            mesh=mesh_4xtp2,
            in_specs=(P("dp", None), P(), P(), P(), P(), P(), specs_bf16_4xtp2),
            out_specs=(P("dp", None, None), P("dp", None, None)),
            check_rep=False,
        )
    )
    shapes_4xtp2 = [(4, 256), (8, 256), (8, 512), (16, 256), (16, 512), (32, 256), (32, 512), (64, 512), (128, 512)]
    tier3a_bf16_4xtp2 = run_shape_sweep(
        jit_bf16_4xtp2, w_emb_4xtp2, f_ln_4xtp2, o_emb_4xtp2, cos_4xtp2, sin_4xtp2, lp_bf16_4xtp2, dp_4xtp2,
        shapes_4xtp2,
        "4x TPU v7x (BF16 4xTP2 + Bio-Megakernel v3 + HyperAttention v2)", 4, 8, "bf16", 48.00
    )
    del lp_bf16_4xtp2, jit_bf16_4xtp2, w_emb_4xtp2, f_ln_4xtp2, o_emb_4xtp2, cos_4xtp2, sin_4xtp2
    gc.collect()

    mesh_dp8 = Mesh(np.array(devices[:8]), ("dp",))
    rep_dp8 = NamedSharding(mesh_dp8, P())
    dp_dp8 = NamedSharding(mesh_dp8, P("dp", None))
    lp_dp8, specs_dp8 = place_v3_params(v3_weights, mesh_dp8, tp_size=1, tp_axis_name=None, quant_mode="int8")
    w_emb_dp8 = jax.device_put(word_emb_np, rep_dp8)
    f_ln_dp8 = jax.device_put(final_ln_w_np, rep_dp8)
    o_emb_dp8 = jax.device_put(out_emb_w_np, rep_dp8)
    cos_dp8 = jax.device_put(cos_c, rep_dp8)
    sin_dp8 = jax.device_put(sin_c, rep_dp8)
    jit_dp8 = jax.jit(
        shard_map(
            make_forward_inventor_v3(tp_size=1, tp_axis_name=None, quant_mode="int8", use_hyperattention=True),
            mesh=mesh_dp8,
            in_specs=(P("dp", None), P(), P(), P(), P(), P(), specs_dp8),
            out_specs=(P("dp", None, None), P("dp", None, None)),
            check_rep=False,
        )
    )
    shapes_dp8 = [(8, 256), (8, 512), (16, 256), (16, 512), (32, 256), (32, 512), (64, 512), (128, 512), (256, 512)]
    tier3b_int8_dp8 = run_shape_sweep(
        jit_dp8, w_emb_dp8, f_ln_dp8, o_emb_dp8, cos_dp8, sin_dp8, lp_dp8, dp_dp8,
        shapes_dp8,
        "4x TPU v7x (INT8 DP=8 = 8xTP1 across 8 96GB TensorCores)", 4, 8, "int8", 48.00
    )

    out = {
        "numerical_parity_vs_bf16": parity_summary,
        "tier1a_1chip_v7x_bf16_tp2_v3": tier1a_bf16_tp2,
        "tier1b_1chip_v7x_int8_tp2_v3": tier1b_int8_tp2,
        "tier1c_1chip_v7x_int8_dp2_v3": tier1c_int8_dp2,
        "tier2a_4chip_v7x_bf16_tp8_v3": tier2a_bf16_tp8,
        "tier2b_4chip_v7x_int8_tp8_v3": tier2b_int8_tp8,
        "tier3a_4chip_v7x_bf16_4xtp2_v3": tier3a_bf16_4xtp2,
        "tier3b_4chip_v7x_int8_dp8_v3": tier3b_int8_dp8,
        "long_context_1chip_v7x_bf16_tp2_v3": long_1c_bf16_tp2,
        "long_context_1chip_v7x_int8_tp2_v3": long_1c_int8_tp2,
        "long_context_4chip_v7x_bf16_tp8_v3": long_4c_bf16_tp8,
        "long_context_4chip_v7x_int8_tp8_v3": long_4c_int8_tp8,
    }
    raw_bytes = json.dumps(out, indent=2).encode("utf-8")
    os.makedirs(os.path.dirname(os.path.abspath(OUT_JSON)), exist_ok=True)
    with open(OUT_JSON, "wb") as f:
        f.write(raw_bytes)

    b64 = base64.b64encode(gzip.compress(raw_bytes)).decode("ascii")
    chunk_sz = 24000
    chunks = [b64[i : i + chunk_sz] for i in range(0, len(b64), chunk_sz)]
    for idx, ch in enumerate(chunks):
        print(f"V7X_V3_GZIP_B64_CHUNK_{idx+1}_OF_{len(chunks)}:{ch}", flush=True)
    print(f"=== Saved {OUT_JSON} ===", flush=True)


if __name__ == "__main__":
    main()
