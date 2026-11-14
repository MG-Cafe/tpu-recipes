#!/usr/bin/env python3
"""Long-Context (256 -> 20,480 tokens) Benchmark on NVIDIA G4 (RTX PRO 6000 Blackwell SM120)."""
import gc
import json
import os
import sys
import time
import torch

sys.path.insert(0, "/tmp")
sys.path.insert(0, "/dev/shm/gb_protein_16b_bf16")

import modeling_fm4bio
from configuration_fm4bio import FM4BioConfig
from g4_benchmark_and_optimize import (
    MODEL_DIR,
    GBProteinG4Optimized,
    compute_active_flops_per_token,
)

if not hasattr(modeling_fm4bio.FM4BioPreTrainedModel, "get_head_mask"):
    modeling_fm4bio.FM4BioPreTrainedModel.get_head_mask = (
        lambda self, head_mask, num_layers, is_attention_chunked=False: [None] * num_layers
        if head_mask is None
        else head_mask
    )


def load_stock_hf_long_ctx(model_dir: str, device: torch.device):
    t0 = time.time()
    cfg = FM4BioConfig.from_json_file(os.path.join(model_dir, "config.json"))
    cfg.max_position_embeddings = 32768
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

    for layer in model.bert.encoder.layer:
        layer.mlp.router = layer.mlp.router.float()

    shards = sorted([f for f in os.listdir(model_dir) if f.startswith("pytorch_model-") and f.endswith(".bin")])
    sd_map = dict(model.named_parameters())
    for s in shards:
        sd = torch.load(os.path.join(model_dir, s), map_location=device, weights_only=True)
        for k, v in sd.items():
            if k in sd_map:
                sd_map[k].data.copy_(v)
        del sd
    model.eval()
    torch.cuda.synchronize(device)
    print(f"[G4-LONG] Loaded Stock HF model (max_pos=32768) in {time.time() - t0:.2f}s", flush=True)
    return model


def main():
    while not os.path.exists(os.path.join(MODEL_DIR, "DONE")):
        print("Waiting for /dev/shm/gb_protein_16b_bf16/DONE...", flush=True)
        time.sleep(5)

    device = torch.device("cuda:0")
    seq_lens = [256, 512, 1024, 2048, 4096, 8192, 16384, 20480]

    # 1. Stock HF Eager BF16 Long-Context Sweep
    hf_model = load_stock_hf_long_ctx(MODEL_DIR, device)
    stock_records = []
    for S in seq_lens:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            ids = torch.randint(1, 22, (1, S), device=device, dtype=torch.long)
            with torch.no_grad():
                _ = hf_model(ids)
                torch.cuda.synchronize(device)
                lats = []
                iters = 5 if S <= 4096 else 3
                for _ in range(iters):
                    t0 = time.perf_counter()
                    _ = hf_model(ids)
                    torch.cuda.synchronize(device)
                    lats.append((time.perf_counter() - t0) * 1000.0)
            med_ms = sorted(lats)[len(lats) // 2]
            tps = S / (med_ms / 1000.0)
            tflops = (tps * compute_active_flops_per_token(S)) / 1e12
            peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
            rec = {
                "batch_size": 1,
                "seq_len": S,
                "tokens_per_batch": S,
                "status": "OK",
                "median_latency_ms": round(med_ms, 2),
                "tokens_per_sec": round(tps, 1),
                "achieved_tflops": round(tflops, 2),
                "peak_vram_gb": round(peak_gb, 2),
            }
            print(f"[G4 Stock HF Eager] S={S:5d} | {med_ms:8.2f} ms | {tps:8.1f} tok/s | {tflops:6.2f} TFLOPS | Peak VRAM={peak_gb:.2f} GB", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            rec = {
                "batch_size": 1,
                "seq_len": S,
                "tokens_per_batch": S,
                "status": "OOM",
                "median_latency_ms": None,
                "tokens_per_sec": 0.0,
                "achieved_tflops": 0.0,
                "error": f"CUDA OOM (96 GB GDDR7 exceeded by O(S^2) [1, 36, {S}, {S}] Eager attention matrix)",
            }
            print(f"[G4 Stock HF Eager] S={S:5d} | OOM! ({rec['error']})", flush=True)
        stock_records.append(rec)

    # 2. Convert in-place to GBProteinG4Optimized with max_seq_len=20480
    print("\n[G4-LONG] Converting to GBProteinG4Optimized (max_seq_len=20480)...", flush=True)
    opt_model = GBProteinG4Optimized(hf_model, device=device, max_seq_len=20480)
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    opt_records = []
    for S in seq_lens:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        ids = torch.randint(1, 22, (1, S), device=device, dtype=torch.long)
        with torch.no_grad():
            for _ in range(2):
                _ = opt_model(ids)
            torch.cuda.synchronize(device)
            lats = []
            iters = 10 if S <= 4096 else 5
            for _ in range(iters):
                t0 = time.perf_counter()
                _ = opt_model(ids)
                torch.cuda.synchronize(device)
                lats.append((time.perf_counter() - t0) * 1000.0)
        med_ms = sorted(lats)[len(lats) // 2]
        tps = S / (med_ms / 1000.0)
        tflops = (tps * compute_active_flops_per_token(S)) / 1e12
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        rec = {
            "batch_size": 1,
            "seq_len": S,
            "tokens_per_batch": S,
            "status": "OK",
            "median_latency_ms": round(med_ms, 2),
            "tokens_per_sec": round(tps, 1),
            "achieved_tflops": round(tflops, 2),
            "peak_vram_gb": round(peak_gb, 2),
        }
        opt_records.append(rec)
        print(f"[G4 Best-Opt SDPA + GroupedMM] S={S:5d} | {med_ms:8.2f} ms | {tps:8.1f} tok/s | {tflops:6.2f} TFLOPS | Peak VRAM={peak_gb:.2f} GB", flush=True)

    out = {
        "gpu_model": torch.cuda.get_device_name(device),
        "long_context_1gpu_stock_hf_eager": stock_records,
        "long_context_1gpu_best_opt_sdpa": opt_records,
    }
    with open("/tmp/g4_long_context_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("=== Saved /tmp/g4_long_context_results.json ===", flush=True)


if __name__ == "__main__":
    main()
