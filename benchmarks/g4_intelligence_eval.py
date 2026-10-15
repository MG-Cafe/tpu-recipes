#!/usr/bin/env python3
"""
Evaluates the 4-Task Protein Intelligence & Biological Accuracy Benchmark Suite
on NVIDIA G4 (RTX PRO 6000 Blackwell, 96 GB GDDR7) across:
  1. G4 Stock HuggingFace FP32 Reference (IEEE FP32 mathematical gold standard)
  2. G4 Stock HuggingFace BF16
  3. G4 Best-Optimized BF16 (Custom Triton + Grouped GEMM + SDPA)
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
MODEL_DIR = "/dev/shm/gb_protein_16b_bf16"
sys.path.insert(0, MODEL_DIR)

import protein_intelligence_suite as pis
import g4_benchmark_and_optimize as g4b
from configuration_fm4bio import FM4BioConfig
import modeling_fm4bio


def load_hf_model_with_dtype(model_dir: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
    t0 = time.time()
    cfg = FM4BioConfig.from_json_file(os.path.join(model_dir, "config.json"))
    cfg.torch_dtype = dtype
    orig_init = modeling_fm4bio.FM4BioPreTrainedModel._init_weights
    modeling_fm4bio.FM4BioPreTrainedModel._init_weights = lambda self, module: None
    try:
        with torch.device(device):
            old_dtype = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            model = modeling_fm4bio.FM4BioForMaskedLM(cfg)
            torch.set_default_dtype(old_dtype)
    finally:
        modeling_fm4bio.FM4BioPreTrainedModel._init_weights = orig_init

    model = model.to(dtype=dtype)
    for layer in model.bert.encoder.layer:
        layer.mlp.router = layer.mlp.router.float()

    shards = sorted([f for f in os.listdir(model_dir) if f.startswith("pytorch_model-") and f.endswith(".bin")])
    state_dict_map = dict(model.named_parameters())
    for s in shards:
        sd = torch.load(os.path.join(model_dir, s), map_location=device, weights_only=True)
        for k, v in sd.items():
            if k in state_dict_map:
                state_dict_map[k].data.copy_(v.to(dtype=state_dict_map[k].dtype))
        del sd
    model.eval()
    torch.cuda.synchronize(device)
    print(f"[G4-INTEL] Loaded GB.Protein-16B ({dtype}) onto {device} in {time.time() - t0:.2f}s", flush=True)
    return model


def make_hf_forward_fn(hf_model: nn.Module, device: torch.device):
    @torch.no_grad()
    def forward_fn(input_ids_np: np.ndarray, valid_len: int):
        padded_len = input_ids_np.shape[1]
        ids_valid = torch.tensor(input_ids_np[:, :valid_len], device=device, dtype=torch.long)
        out = hf_model(ids_valid, output_hidden_states=True)
        logits_valid = out.logits.float().cpu().numpy()
        # out.hidden_states[-1] is already [B, S, H] post-final-RMSNorm (sequence_output)
        norm_h = out.hidden_states[-1].float().cpu().numpy()

        logits_pad = np.zeros((1, padded_len, logits_valid.shape[-1]), dtype=np.float32)
        hidden_pad = np.zeros((1, padded_len, norm_h.shape[-1]), dtype=np.float32)
        logits_pad[:, :valid_len, :] = logits_valid
        hidden_pad[:, :valid_len, :] = norm_h
        return logits_pad, hidden_pad

    return forward_fn


def make_opt_forward_fn(opt_model: g4b.GBProteinG4Optimized, device: torch.device):
    @torch.no_grad()
    def forward_fn(input_ids_np: np.ndarray, valid_len: int):
        padded_len = input_ids_np.shape[1]
        ids_valid = torch.tensor(input_ids_np[:, :valid_len], device=device, dtype=torch.long)
        logits_valid, hidden_valid = opt_model(ids_valid)
        logits_np = logits_valid.float().cpu().numpy()
        hidden_np = hidden_valid.float().cpu().numpy()

        logits_pad = np.zeros((1, padded_len, logits_np.shape[-1]), dtype=np.float32)
        hidden_pad = np.zeros((1, padded_len, hidden_np.shape[-1]), dtype=np.float32)
        logits_pad[:, :valid_len, :] = logits_np
        hidden_pad[:, :valid_len, :] = hidden_np
        return logits_pad, hidden_pad

    return forward_fn


def strip_raw_arrays(res: dict) -> dict:
    out = dict(res)
    if "raw_logits_sample" in out:
        raw = out["raw_logits_sample"]
        out["raw_logits_sample"] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in raw.items()
        }
    return out


def main():
    device = torch.device("cuda:0")
    print("=== Running GB.Protein-16B Intelligence Benchmark on NVIDIA G4 ===", flush=True)

    # 1. Evaluate FP32 Exact Reference
    hf_fp32 = load_hf_model_with_dtype(MODEL_DIR, device, torch.float32)
    res_fp32 = pis.evaluate_intelligence_suite(make_hf_forward_fn(hf_fp32, device), "G4_Stock_FP32_Reference")
    print("[G4 FP32 Reference] MLM Aggregate:", res_fp32["mlm_aggregate"], flush=True)
    print("[G4 FP32 Reference] ClinVar DMS Summary:", res_fp32["clinvar_dms_summary"], flush=True)
    del hf_fp32
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Evaluate Stock HF BF16 & Best-Optimized G4 BF16
    hf_bf16 = load_hf_model_with_dtype(MODEL_DIR, device, torch.bfloat16)
    res_stock_bf16 = pis.evaluate_intelligence_suite(make_hf_forward_fn(hf_bf16, device), "G4_Stock_BF16")
    comp_stock_bf16 = pis.compare_against_reference(res_stock_bf16, res_fp32)
    res_stock_bf16["comparison_vs_fp32_ref"] = comp_stock_bf16
    print("[G4 Stock BF16] Comparison vs FP32:", comp_stock_bf16, flush=True)

    opt_bf16 = g4b.GBProteinG4Optimized(hf_bf16, device=device, max_seq_len=2048)
    del hf_bf16
    gc.collect()
    torch.cuda.empty_cache()

    res_opt_bf16 = pis.evaluate_intelligence_suite(make_opt_forward_fn(opt_bf16, device), "G4_Best_Optimized_BF16")
    comp_opt_bf16 = pis.compare_against_reference(res_opt_bf16, res_fp32)
    res_opt_bf16["comparison_vs_fp32_ref"] = comp_opt_bf16
    res_opt_bf16["comparison_vs_g4_opt_bf16"] = pis.compare_against_reference(res_opt_bf16, res_opt_bf16)
    print("[G4 Best-Opt BF16] MLM Aggregate:", res_opt_bf16["mlm_aggregate"], flush=True)
    print("[G4 Best-Opt BF16] ClinVar DMS Summary:", res_opt_bf16["clinvar_dms_summary"], flush=True)
    print("[G4 Best-Opt BF16] Comparison vs FP32:", comp_opt_bf16, flush=True)

    payload = {
        "G4_Stock_FP32_Reference": strip_raw_arrays(res_fp32),
        "G4_Stock_BF16": strip_raw_arrays(res_stock_bf16),
        "G4_Best_Optimized_BF16": strip_raw_arrays(res_opt_bf16),
    }
    with open("/tmp/g4_intelligence_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("=== Saved /tmp/g4_intelligence_results.json ===", flush=True)


if __name__ == "__main__":
    main()
