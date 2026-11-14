#!/usr/bin/env python3
"""
Standalone Cloud TPU Inference & Zero-Shot Variant Scoring Demo for GenBio AI GB.Protein-16B
(`genbio-ai/AIDO.Protein-16B`, 16.05B total parameters, 4.21B active per token, 36 layers, 8 experts Top-2).

Supports:
  - 1x Cloud TPU v6e (32 GB HBM, Bio-Megakernel v3 INT8 Cross-SwiGLU Folded + HyperAttention v2)
  - 2x / 4x Cloud TPU v6e (TP=2 / TP=4 INT8 or BF16)
  - 1x Cloud TPU v7x (192 GB HBM, TP=2 BF16 or INT8 across 2 on-die TensorCores)
  - 4x Cloud TPU v7x (768 GB HBM, TP=8 BF16 or INT8 across 8 TensorCores)
  - Sequence lengths from S=256 up to S=20,480 amino acids (via Pallas SplashAttention Megacore)

Usage:
  python3 benchmarks/run_tpu_inference_demo.py --sequence "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL..." --quant int8
"""
import argparse
import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from tpu_v6e_inventor_v3_benchmark import (
    MODEL_DIR,
    GFP_WT_SEQ,
    precompute_rope_v3,
    prepare_all_v3_weights,
    place_v3_params,
    make_forward_inventor_v3,
)
from tpu_v6e_megakernel_benchmark import load_checkpoint_arrays
from tokenization_fm4bio import FM4BioTokenizer


def main():
    parser = argparse.ArgumentParser(description="Run GB.Protein-16B Inference on Cloud TPU (v6e / v7x)")
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR, help="Path to pre-stacked BF16 checkpoint")
    parser.add_argument("--sequence", type=str, default=GFP_WT_SEQ, help="Amino acid sequence to embed & score")
    parser.add_argument("--quant", type=str, choices=["int8", "bf16"], default="int8", help="Precision mode (int8 or bf16)")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor Parallelism degree (1, 2, 4, or 8)")
    parser.add_argument("--bucket-len", type=int, default=256, help="Sequence length bucket (256..20480)")
    args = parser.parse_args()

    sys.path.insert(0, args.model_dir)
    devices = jax.devices()
    assert len(devices) >= args.tp_size, f"Requested --tp-size={args.tp_size}, but only {len(devices)} TPU cores visible."

    print(f"[TPU Engine] Initializing GB.Protein-16B on {args.tp_size} TPU core(s) (mode={args.quant})...")
    tokenizer = FM4BioTokenizer(vocab_file=os.path.join(args.model_dir, "vocab_protein.txt"))
    word_emb_np, final_ln_w_np, out_emb_w_np, stacked_np = load_checkpoint_arrays(args.model_dir)
    cos_c, sin_c = precompute_rope_v3(max_seq_len=max(20480, args.bucket_len), head_dim=64)
    v3_weights = prepare_all_v3_weights(stacked_np)

    tp_axis = "tp" if args.tp_size > 1 else None
    mesh_axis = ("tp",) if args.tp_size > 1 else ("dp",)
    mesh = Mesh(np.array(devices[: args.tp_size]), mesh_axis)
    rep = NamedSharding(mesh, P())
    lp, specs = place_v3_params(v3_weights, mesh, tp_size=args.tp_size, tp_axis_name=tp_axis, quant_mode=args.quant)

    w_emb = jax.device_put(word_emb_np, rep)
    f_ln = jax.device_put(final_ln_w_np, rep)
    o_emb = jax.device_put(out_emb_w_np, rep)
    cos = jax.device_put(cos_c, rep)
    sin = jax.device_put(sin_c, rep)

    in_id_spec = P() if args.tp_size > 1 else P("dp", None)
    out_spec = P() if args.tp_size > 1 else P("dp", None, None)

    jit_fwd = jax.jit(
        shard_map(
            make_forward_inventor_v3(tp_size=args.tp_size, tp_axis_name=tp_axis, quant_mode=args.quant, use_hyperattention=True),
            mesh=mesh,
            in_specs=(in_id_spec, P(), P(), P(), P(), P(), specs),
            out_specs=(out_spec, out_spec),
            check_rep=False,
        )
    )

    spaced_seq = " ".join(list(args.sequence.strip()))
    token_ids = tokenizer.encode(spaced_seq)
    S = max(args.bucket_len, int(2 ** np.ceil(np.log2(max(256, len(token_ids))))))
    batch = np.zeros((1, S), dtype=np.int32)
    valid_len = min(len(token_ids), S)
    batch[0, :valid_len] = token_ids[:valid_len]
    ids_dev = jax.device_put(batch, NamedSharding(mesh, in_id_spec))

    # Warmup XLA compilation
    logits, hidden = jit_fwd(ids_dev, w_emb, f_ln, o_emb, cos, sin, lp)
    jax.block_until_ready(logits)

    # Timed inference
    t0 = time.perf_counter()
    logits, hidden = jit_fwd(ids_dev, w_emb, f_ln, o_emb, cos, sin, lp)
    jax.block_until_ready(logits)
    lat_ms = (time.perf_counter() - t0) * 1000.0

    logits_np = np.asarray(logits[0, :valid_len], dtype=np.float32)
    hidden_np = np.asarray(hidden[0, :valid_len], dtype=np.float32)
    mean_embedding = hidden_np.mean(axis=0)

    print(f"\n=== Inference Complete ===")
    print(f"Sequence Length : {valid_len} tokens (bucket S={S})")
    print(f"Latency         : {lat_ms:.2f} ms ({S / (lat_ms / 1000.0):,.1f} tokens/sec)")
    print(f"Logits Shape    : {logits_np.shape}")
    print(f"Hidden Shape    : {hidden_np.shape} (L2 norm of mean embedding = {np.linalg.norm(mean_embedding):.4f})")


if __name__ == "__main__":
    main()
