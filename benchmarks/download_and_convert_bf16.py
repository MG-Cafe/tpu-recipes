#!/usr/bin/env python3
"""Fast, resilient download & FP32->BF16 conversion of genbio-ai/GB.Protein-16B."""
import concurrent.futures
import gc
import json
import os
import shutil
import subprocess
import time
import urllib.request
import torch

HF_CODE_DIR = "/dev/shm/hf_code"
OUT_DIR = "/dev/shm/gb_protein_16b_bf16"
TMP_RAW_DIR = "/dev/shm/gb_protein_raw_shards"
BASE_URL = "https://huggingface.co/genbio-ai/GB.Protein-16B/resolve/main"


def download_and_convert_one(shard_name: str) -> str:
    os.makedirs(TMP_RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, shard_name)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 100_000_000:
        return out_path

    raw_path = os.path.join(TMP_RAW_DIR, shard_name)
    url = f"{BASE_URL}/{shard_name}"

    for attempt in range(5):
        try:
            if os.path.exists(raw_path):
                os.remove(raw_path)
            t0 = time.time()
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as resp, open(raw_path, "wb") as f:
                expected_bytes = int(resp.headers.get("Content-Length", "0"))
                shutil.copyfileobj(resp, f, length=16 * 1024 * 1024)
            actual_bytes = os.path.getsize(raw_path)
            if expected_bytes > 0 and actual_bytes != expected_bytes:
                raise IOError(f"Incomplete download for {shard_name}: got {actual_bytes} vs {expected_bytes}")
            sz_gb = actual_bytes / 1e9
            print(f"[DL] Downloaded {shard_name} ({sz_gb:.2f} GB) in {time.time() - t0:.1f}s", flush=True)

            t_c = time.time()
            sd = torch.load(raw_path, map_location="cpu", weights_only=True)
            sd_bf16 = {}
            for k, v in sd.items():
                if "router.weight" in k:
                    sd_bf16[k] = v.to(torch.float32).contiguous()
                elif v.is_floating_point():
                    sd_bf16[k] = v.to(torch.bfloat16).contiguous()
                else:
                    sd_bf16[k] = v
            tmp_out = out_path + ".tmp"
            torch.save(sd_bf16, tmp_out)
            os.replace(tmp_out, out_path)
            os.remove(raw_path)
            del sd, sd_bf16
            gc.collect()
            out_gb = os.path.getsize(out_path) / 1e9
            print(
                f"[CONV] Converted {shard_name} -> BF16 ({out_gb:.2f} GB) in {time.time() - t_c:.1f}s",
                flush=True,
            )
            return out_path
        except Exception as e:
            if os.path.exists(raw_path):
                try:
                    os.remove(raw_path)
                except OSError:
                    pass
            if attempt == 4:
                raise
            print(f"[RETRY] Attempt {attempt+1}/5 for {shard_name} failed: {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    return out_path


def main():
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_RAW_DIR, exist_ok=True)
    done_flag = os.path.join(OUT_DIR, "DONE")
    if os.path.exists(done_flag):
        print("[PREP] /dev/shm/gb_protein_16b_bf16/DONE already exists!", flush=True)
        return

    # Clean up any leftover partial raw shards in /tmp or /dev/shm
    shutil.rmtree("/tmp/gb_protein_raw_shards", ignore_errors=True)
    shutil.rmtree(TMP_RAW_DIR, ignore_errors=True)
    os.makedirs(TMP_RAW_DIR, exist_ok=True)

    print("[PREP] Starting download & BF16 conversion of genbio-ai/GB.Protein-16B...", flush=True)
    if os.path.exists(HF_CODE_DIR):
        for fn in os.listdir(HF_CODE_DIR):
            src = os.path.join(HF_CODE_DIR, fn)
            dst = os.path.join(OUT_DIR, fn)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

    # Ensure required tokenizer, config, and modeling files are downloaded if not present
    for meta_fn in ["config.json", "tokenization_fm4bio.py", "vocab_protein.txt", "modeling_fm4bio.py", "configuration_fm4bio.py"]:
        dst = os.path.join(OUT_DIR, meta_fn)
        if not os.path.exists(dst):
            meta_url = f"{BASE_URL}/{meta_fn}"
            req = urllib.request.Request(meta_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(dst, "wb") as f:
                shutil.copyfileobj(resp, f)
            print(f"[DL] Downloaded {meta_fn}", flush=True)

    cfg_path = os.path.join(OUT_DIR, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        cfg["torch_dtype"] = "bfloat16"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

    shards = [f"pytorch_model-{i:05d}-of-00013.bin" for i in range(1, 14)]

    batch_size = 2
    for b_idx in range(0, len(shards), batch_size):
        batch = [
            s for s in shards[b_idx : b_idx + batch_size]
            if not (os.path.exists(os.path.join(OUT_DIR, s)) and os.path.getsize(os.path.join(OUT_DIR, s)) > 100_000_000)
        ]
        if not batch:
            continue
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as ex:
            futs = [ex.submit(download_and_convert_one, s) for s in batch]
            for fut in concurrent.futures.as_completed(futs):
                fut.result()

    shutil.rmtree(TMP_RAW_DIR, ignore_errors=True)
    with open(done_flag, "w") as f:
        f.write(f"DONE at {time.time()}\n")
    print(f"[PREP] All 13 shards ready in {OUT_DIR} in {time.time() - t_start:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
