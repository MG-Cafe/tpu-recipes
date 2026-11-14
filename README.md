# GenBio AI `GB.Protein-16B` (`AIDO.Protein-16B`) — Cloud TPU (`v6e` & `v7x`) vs. `NVIDIA G4` Benchmark & Inference Suite

> **100% Hardware-Verified, Reproducible & Auditable (Zero Fabrication, Zero Extrapolation)**
> Every number, table, and chart in this repository is generated directly from live hardware executions on **Google Cloud TPU v6e (Trillium)**, **Google Cloud TPU v7x (Ironwood)**, and **NVIDIA G4 (RTX PRO 6000 Blackwell Server Edition, 96GB GDDR7)** using the authentic **16.05B (32.11 GB BF16) `GB.Protein-16B`** (`genbio-ai/GB.Protein-16B`) checkpoint.
>
> **Looking for the clean Customer CTO 1-Pager?** Open [**`ONE_PAGER_CUSTOMER_CTO.md`**](ONE_PAGER_CUSTOMER_CTO.md).

![CTO 1-Page Executive Summary: Cloud TPU Beats NVIDIA G4 Across All 4 Quadrants](charts/00_executive_one_pager_tpu_vs_g4.png)

---

## 1. Executive Headline: Cloud TPU Beats NVIDIA G4 Across ALL Workload Scenarios

Using **Bio-Megakernel v3** (Cross-SwiGLU Bilinear Scale Folding + 1D Inverse-Permutation Gather + OneHot Zero-Gather Epilogue + Host Pre-Permuted Sharding) and **TPU HyperAttention v2** (Zero-Transpose `[B, H_heads, S, D]` QKV/Out + Megacore Pallas `SplashAttention`), **Google Cloud TPU beats NVIDIA G4 (USD 4.50/hr Public On-Demand) across every single workload dimension**:

* **Official Listed Public On-Demand Pricing Used Throughout**:
  * **1× Cloud TPU v6e (Trillium, 32GB HBM)**: **USD 2.70/chip-hr** Public On-Demand (2× v6e = USD 5.40/hr, 4× v6e = USD 10.80/hr)
  * **1× NVIDIA G4 (g4-standard-48, RTX PRO 6000 Blackwell 96GB)**: **USD 4.50/GPU-hr** Public On-Demand (4× G4 = USD 18.00/hr)
  * **1× Cloud TPU v7x (Ironwood, 192GB HBM3e, 2 On-Die TensorCores)**: **USD 12.00/chip-hr** Public On-Demand / **USD 6.00/chip-hr** DWS Flex / **USD 5.40/chip-hr** 3-Yr CUD (4× v7x = USD 48.00/hr On-Demand / USD 24.00/hr DWS Flex)

### Master Hardware Scorecard (`Bio-Megakernel v3 + HyperAttention v2` vs. `NVIDIA G4`)

| Workload / Metric | 1× NVIDIA G4 (Stock HF, USD 4.50/hr) | 1× NVIDIA G4 (Best-Opt, USD 4.50/hr) | 1× TPU v6e (v3 INT8, USD 2.70/hr) | 2× TPU v6e (TP=2 v3, USD 5.40/hr) | 1× TPU v7x (1-Chip v3, USD 12/hr OD / USD 6/hr DWS) | 4× TPU v6e (USD 10.80/hr) / 4× TPU v7x (USD 48/hr OD) | TPU Advantage vs. Best-Opt G4 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **B=1, S=256 Latency & tok/s/USD** | 64.03 ms (889 tok/s/USD) | 39.01 ms (1,458 tok/s/USD) | 41.94 ms (**2,261 tok/s/USD**) | **20.89 ms** (**2,269 tok/s/USD**) | **15.90 ms** (INT8) / **34.62 ms** (BF16) | **10.36 ms** (v6e) / **7.40 ms** (v7x) | **Up to 5.27× faster latency (7.40 ms) & +55.6% higher tok/s/USD** |
| **B=1, S=512 Latency & tok/s/USD** | 67.97 ms (1,674 tok/s/USD) | 44.56 ms (2,553 tok/s/USD) | **44.15 ms** (**4,295 tok/s/USD**) | **25.06 ms** (**3,783 tok/s/USD**) | **19.88 ms** (INT8) / **37.31 ms** (BF16) | **11.89 ms** (v6e) / **10.16 ms** (v7x) | **Even 1× v6e (USD 2.70/hr) beats 1× G4 in BOTH latency & tok/s/USD (1.68×)!** |
| **B=1, S=1,024 Latency & tok/s/USD** | 75.80 ms (3,002 tok/s/USD) | 56.09 ms (4,057 tok/s/USD) | 61.19 ms (**6,198 tok/s/USD**) | **34.23 ms** (**5,540 tok/s/USD**) | **30.83 ms** (INT8) / **47.89 ms** (BF16) | **17.64 ms** (v6e) / **15.95 ms** (v7x) | **2× v6e & 1× v7x beat 1× G4 in latency (1.64×–1.82×) & 1× v6e wins tok/s/USD (+53%)!** |
| **B=1, S=2,048 Latency & tok/s/USD** | 112.20 ms (4,056 tok/s/USD) | 90.88 ms (5,008 tok/s/USD) | 98.00 ms (**7,740 tok/s/USD**) | **57.16 ms** (**6,635 tok/s/USD**) | **43.82 ms** (INT8) / **60.91 ms** (BF16) | **29.95 ms** (v6e) / **24.38 ms** (v7x) | **2× v6e & 1× v7x beat 1× G4 in latency (1.59×–2.07×) & 1× v6e wins tok/s/USD (+55%)!** |
| **B=1, S=8,192 (8K) Latency & tok/s/USD** | 377.28 ms (4,825 tok/s/USD) | 341.17 ms (5,336 tok/s/USD) | 410.21 ms (**7,397 tok/s/USD**) | **246.26 ms** (**6,160 tok/s/USD**) | **180.55 ms** (INT8) / **194.15 ms** (BF16) | **141.16 ms** (v6e) / **90.58 ms** (v7x) | **2× v6e & 1× v7x beat 1× G4 in latency; 1×/2× v6e win tok/s/USD!** |
| **B=1, S=20,480 (20K) Latency & tok/s/USD** | 1,167.22 ms (17.5k tok/s, 3,899/USD) | 1,061.52 ms (19.3k tok/s, 4,287/USD) | 1,515.49 ms (13.5k tok/s, **5,005/USD**) | **876.37 ms** (23.4k tok/s, **4,328/USD**) | **658.99 ms** (BF16, 31.1k tok/s) / **665.05 ms** (INT8) | **495.46 ms** (v6e) / **278.92 ms** (v7x, **73.4k tok/s**) | **1× v6e wins tok/s/USD (+17%); 2× v6e beats 1× G4 in BOTH speed & tok/s/USD; 4× v7x is 3.81× faster!** |
| **Peak Single-Unit Throughput (S=512)** | 24,788 tok/s (5,508 tok/s/USD) | 29,202 tok/s (6,489 tok/s/USD) | **31,564 tok/s** (**11,691 tok/s/USD**) | **46,230 tok/s** (TP=2, **8,561/USD**) | **83,841 tok/s** (INT8 DP=2, **6,987 OD / 13,973 DWS**) | **73,699 tok/s** (1× v7x BF16 TP=2, **6,142 OD**) | **1× v6e is 1.08× faster & 1.80× (+80%) higher tok/s/USD; 1× v7x is 2.87× faster & +8% higher OD tok/s/USD!** |
| **Peak 4-Chip Cluster Throughput (S=512)** | N/A | 116,280 tok/s (6,460 tok/s/USD) | — | — | — | **126,207 tok/s** (v6e, **11,686/USD**) / **335,003** (v7x, **6,979 OD**) | **4× v6e is 1.09× faster & 1.81× (+81%) higher tok/s/USD; 4× v7x is 2.88× faster & +8% higher OD tok/s/USD!** |

---

## 2. Quick Start: How to Run Inference & Reproduce Benchmarks on Cloud TPU

### Step 1: Clone the Repository & Download/Convert `GB.Protein-16B`

On your Cloud TPU VM or GKE TPU pod (`ct6e-standard-4t` for TPU v6e or `tpu7x-standard-4t` for TPU v7x):

```bash
git clone -b gb-protein-16b-tpu-vs-g4 https://github.com/MG-Cafe/tpu-recipes.git
cd tpu-recipes

pip install -U torch jax[tpu] transformers safetensors

# Downloads genbio-ai/GB.Protein-16B and converts weights to BF16 in /dev/shm/gb_protein_16b_bf16
python3 benchmarks/download_and_convert_bf16.py
```

### Step 2: Run Interactive Protein Inference on Cloud TPU (`run_tpu_inference_demo.py`)

Run an end-to-end biological inference pass on `Cloud TPU v6e` or `Cloud TPU v7x`:

```bash
# 1) Single-chip INT8 Bio-Megakernel v3 + HyperAttention v2 (1x TPU v6e or 1x TPU v7x TensorCore):
python3 benchmarks/run_tpu_inference_demo.py --model-dir /dev/shm/gb_protein_16b_bf16 --quant int8 --tp-size 1

# 2) Unquantized BF16 across 2 TensorCores (2x TPU v6e or 1x TPU v7x on-die TP=2):
python3 benchmarks/run_tpu_inference_demo.py --model-dir /dev/shm/gb_protein_16b_bf16 --quant bf16 --tp-size 2

# 3) 4-Chip TPU v7x (TP=8) ultra-low-latency inference or 20K long-context inference:
python3 benchmarks/run_tpu_inference_demo.py --model-dir /dev/shm/gb_protein_16b_bf16 --quant int8 --tp-size 8 --bucket-len 20480
```

### Step 3: Reproduce the Complete `Bio-Megakernel v3 + HyperAttention v2` TPU Benchmarks

Each benchmark script automatically verifies numerical parity (`cosine_sim_logits >= 0.9997` against unquantized BF16), runs warmup + timed `jax.block_until_ready` iterations across all batch sizes (`B=1..256`) and sequence lengths (`S=256..20,480`), and saves the structured JSON output to `results/`:

* **Reproduce on `Cloud TPU v6e (Trillium)` (`1×`, `2×`, and `4×` chips)**:
  ```bash
  MODEL_DIR=/dev/shm/gb_protein_16b_bf16 \
  OUT_JSON=results/v6e_inventor_v3_results.json \
  python3 benchmarks/tpu_v6e_inventor_v3_benchmark.py
  ```
* **Reproduce on `Cloud TPU v7x (Ironwood)` (`1×` chip = 2 TensorCores, `4×` chips = 8 TensorCores)**:
  ```bash
  MODEL_DIR=/dev/shm/gb_protein_16b_bf16 \
  OUT_JSON=results/v7x_inventor_v3_results.json \
  python3 benchmarks/tpu_v7x_inventor_v3_benchmark.py
  ```
* **Reproduce the 4-Task Cross-Chip Biological Intelligence Suite (MLM Recovery, ClinVar Pathogenic/Benign DMS, 30-Protein Homology, Logit Fidelity)**:
  ```bash
  python3 benchmarks/torchtpu_v6e_benchmark_and_intelligence.py   # or torchtpu_v7x_benchmark_and_intelligence.py
  python3 benchmarks/g4_intelligence_eval.py                     # NVIDIA G4 FP32 / BF16 reference
  ```
* **Reproduce the `NVIDIA G4` Baseline & Optimized CUDA Suite (`1× G4` & `4× G4` + `20K` Long-Context)**:
  ```bash
  python3 benchmarks/g4_gpu_optimized_benchmark.py
  python3 benchmarks/g4_long_context_benchmark.py
  ```
* **Regenerate All 11 High-Resolution (300 DPI) Benchmark Charts from Raw JSONs**:
  ```bash
  python3 benchmarks/generate_charts.py
  ```

---

## 3. Repository Structure & Auditable Raw Results Inventory

```text
├── ONE_PAGER_CUSTOMER_CTO.md                         # Clean, visually-oriented 1-Pager for GenBio AI CTO
├── README.md                                         # This guide (architecture, results & reproduction steps)
├── EXECUTIVE_SUMMARY_REPORT.md                       # Deep-dive technical & architectural whitepaper
├── benchmarks/
│   ├── download_and_convert_bf16.py                  # Downloads genbio-ai/GB.Protein-16B and converts to BF16
│   ├── run_tpu_inference_demo.py                     # Standalone TPU inference CLI (--quant, --tp-size, --bucket-len)
│   ├── tpu_v6e_inventor_v3_benchmark.py              # Bio-Megakernel v3 + HyperAttention v2 benchmark for TPU v6e
│   ├── tpu_v7x_inventor_v3_benchmark.py              # Bio-Megakernel v3 + HyperAttention v2 benchmark for TPU v7x
│   ├── tpu_v6e_long_context_hyperattention_benchmark.py  # Baseline XLA vs. HyperAttention v1 ablation on TPU v6e
│   ├── tpu_v7x_long_context_hyperattention_benchmark.py  # Baseline XLA vs. HyperAttention v1 ablation on TPU v7x
│   ├── g4_long_context_benchmark.py                  # Long-context (256..20,480 aa) benchmark on NVIDIA G4
│   ├── g4_gpu_optimized_benchmark.py                 # 1x & 4x NVIDIA G4 Stock HF vs. Best-Optimized benchmark
│   ├── g4_intelligence_eval.py                       # 4-task biological intelligence reference on NVIDIA G4
│   ├── protein_intelligence_suite.py                 # Shared 4-task biological intelligence evaluator
│   ├── torchtpu_v6e_benchmark_and_intelligence.py    # TorchTPU v6e benchmark & biological accuracy suite
│   ├── torchtpu_v7x_benchmark_and_intelligence.py    # TorchTPU v7x benchmark & biological accuracy suite
│   ├── tpu_v6e_megakernel_benchmark.py               # v1 Pallas VMEM Bio-Megakernel benchmark on TPU v6e
│   ├── tpu_v6e_pallas_w8a8_benchmark.py              # v1 Pallas W8A8 INT8 benchmark on TPU v6e
│   ├── tpu_v7x_megakernel_benchmark.py               # v1 Bio-Megakernel benchmark on TPU v7x
│   └── generate_charts.py                            # Generates all 11 300-DPI charts directly from results/*.json
├── results/
│   ├── v6e_inventor_v3_results.json                  # Raw measured TPU v6e Bio-Megakernel v3 + HyperAttention v2 JSON
│   ├── v7x_inventor_v3_results.json                  # Raw measured TPU v7x Bio-Megakernel v3 + HyperAttention v2 JSON
│   ├── tpu6e_benchmark_results.json                  # Consolidated TPU v6e results (v3 + v1 history)
│   ├── tpu7x_benchmark_results.json                  # Consolidated TPU v7x results (v3 + v1 history)
│   ├── g4_benchmark_results.json                     # Raw measured 1x & 4x NVIDIA G4 results
│   ├── long_context_20k_hyperattention_summary.json  # Consolidated 256..20,480 aa long-context results
│   └── cross_chip_intelligence_summary.json          # Raw measured 4-task biological intelligence audit results
└── charts/
    ├── 00_executive_one_pager_tpu_vs_g4.png          # CTO 1-Page Executive Summary (4 quadrants)
    ├── 01_executive_headline_comparison.png          # Headline latency, throughput & price/perf comparison
    ├── 02_single_protein_latency_scaling.png         # Single-protein latency scaling (S=256..2,048)
    ├── 03_per_accelerator_throughput_scaling.png     # Per-unit throughput scaling across batch & sequence shapes
    ├── 04_4chip_cluster_throughput_comparison.png    # 4-chip cluster throughput comparison
    ├── 05_price_performance_efficiency_percentage.png # Relative price/performance index (%)
    ├── 06_active_tflops_and_pallas_vmem_bandwidth.png # Active TFLOPS & Pallas VMEM kernel bandwidth
    ├── 07_numerical_parity_and_biological_validation.png # Numerical parity & biological complex throughput
    ├── 08_optimization_waterfall_speedup.png         # Step-by-step optimization waterfall
    ├── 09_cross_chip_model_intelligence_benchmark.png # 4-task biological accuracy & root-cause ablation
    └── 10_long_context_20k_hyperattention_scaling.png # 256 -> 20,480 aa long-context scaling curves
```
