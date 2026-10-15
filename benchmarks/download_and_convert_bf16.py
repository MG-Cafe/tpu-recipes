#!/usr/bin/env python3
"""Fast parallel download & FP32->BF16 conversion of genbio-ai/GB.Protein-16B."""
import concurrent.futures
import gc
import json
import os
import shutil
import time
import urllib.request
import torch

HF_CODE_DIR = "/dev/shm/hf_code"
OUT_DIR = "/dev/shm/gb_protein_16b_bf16"
TMP_RAW_DIR = "/tmp/gb_protein_raw_shards"
BASE_URL = "https://huggingface.co/genbio-ai/GB.Protein-16B/resolve/main"


def download_shard(shard_name: str) -> str:
    os.makedirs(TMP_RAW_DIR, exist_ok=True)
    raw_path = os.path.join(TMP_RAW_DIR, shard_name)
    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 100_000_000:
        return raw_path
    url = f"{BASE_URL}/{shard_name}"
    t0 = time.time()
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(raw_path, "wb") as f:
                shutil.copyfileobj(resp, f, length=16 * 1024 * 1024)
            break
        except Exception as e:
            if attempt == 2:
                raise
            print(f"[DL] Retry {attempt+1} for {shard_name}: {e}", flush=True)
            time.sleep(2)
    sz_gb = os.path.getsize(raw_path) / 1e9
    print(f"[DL] Downloaded {shard_name} ({sz_gb:.2f} GB) in {time.time() - t0:.1f}s", flush=True)
    return raw_path


def main():
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    done_flag = os.path.join(OUT_DIR, "DONE")
    if os.path.exists(done_flag):
        print("[PREP] /dev/shm/gb_protein_16b_bf16/DONE already exists!", flush=True)
        return

    print("[PREP] Starting download & BF16 conversion of genbio-ai/GB.Protein-16B...", flush=True)
    for fn in os.listdir(HF_CODE_DIR):
        src = os.path.join(HF_CODE_DIR, fn)
        dst = os.path.join(OUT_DIR, fn)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    cfg_path = os.path.join(OUT_DIR, "config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    cfg["torch_dtype"] = "bfloat16"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    shards = [f"pytorch_model-{i:05d}-of-00013.bin" for i in range(1, 14)]

    batch_size = 4
    for b_idx in range(0, len(shards), batch_size):
        batch = [
            s for s in shards[b_idx : b_idx + batch_size]
            if not os.path.exists(os.path.join(OUT_DIR, s))
        ]
        if not batch:
            continue
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as ex:
            fut_map = {ex.submit(download_shard, s): s for s in batch}
            for fut in concurrent.futures.as_completed(fut_map):
                s = fut_map[fut]
                raw_path = fut.result()
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
                out_path = os.path.join(OUT_DIR, s)
                torch.save(sd_bf16, out_path)
                os.remove(raw_path)
                del sd, sd_bf16
                gc.collect()
                out_gb = os.path.getsize(out_path) / 1e9
                print(
                    f"[CONV] Converted {s} -> BF16 ({out_gb:.2f} GB) in {time.time() - t_c:.1f}s",
                    flush=True,
                )

    with open(done_flag, "w") as f:
        f.write(f"DONE at {time.time()}\n")
    print(f"[PREP] All 13 shards ready in {OUT_DIR} in {time.time() - t_start:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
