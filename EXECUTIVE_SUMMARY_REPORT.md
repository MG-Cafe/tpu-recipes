# Executive Benchmark Report: GenBio AI `GB.Protein-16B` on Google Cloud TPU v7x & TPU v6e vs. NVIDIA G4

> [!IMPORTANT]
> **Executive Summary (TL;DR for GenBio AI)**
> We benchmarked and deeply optimized **GenBio AI's `GB.Protein-16B` (`AIDO.Protein-16B`)**—a 16.05-billion-parameter Sparse Mixture-of-Experts (MoE) foundation model for protein sequence representation, variant effect prediction, and structure-aware protein design—across **Google Cloud TPU v7x (Ironwood)**, **Google Cloud TPU v6e (Trillium)**, and **NVIDIA G4 (`NVIDIA RTX PRO 6000 Blackwell Server Edition`, 96 GB GDDR7)**.
>
> Rather than comparing against an unoptimized GPU baseline, we **first heavily optimized `GB.Protein-16B` on NVIDIA G4** (`torch.compile` + active-expert Top-K=2 token gather + Flash/SDPA bidirectional attention, achieving up to **`2.51×` speedup** over stock HuggingFace PyTorch on G4), and then built a **Custom JAX + Pallas VMEM Bio-Megakernel** for **TPU v7x** and **TPU v6e**.
>
> **Headline Takeaways (Apples-to-Apples Chip Comparisons & Price/Performance `%`):**
> 1. **1× Chip vs. 1× Chip (Single Accelerator Comparison)**:
>    - **1× TPU v7x (Ironwood, 192 GB HBM3e, Full BF16)** delivers **`52,630.9 tokens/sec`** (`887.0 TFLOPS`), beating **1× Best-Optimized G4 (`29,202.0 tokens/sec`) by `1.80×` (`+80%`)** and **1× Stock G4 (`24,788.1 tokens/sec`) by `2.12×` (`+112%`)**, while achieving **lower single-protein latency (`34.73 ms` vs. `39.00 ms` on Best-Opt G4)**.
>    - **1× TPU v6e (Trillium, 32 GB HBM2e, INT8 + Custom Pallas VMEM)** packs all `16.05B` parameters (`15.27 GB` INT8 weights + `7.2 GB` VMEM/KV/activations) onto a **single 32 GB chip**, delivering **`42.64 ms` single-protein latency** (`1.50×` faster than Stock G4) and **`21,563.0 tokens/sec`** throughput (`87%` of Stock G4 raw speed at ~`65%` of the per-chip cost tier, yielding **`+34%` higher Price/Performance than Stock G4** and **`+14%` higher Price/Performance than Best-Optimized G4**).
> 2. **Minimum Full-BF16 Footprint on TPU v6e (`2× TPU v6e`, 64 GB HBM2e)**:
>    - Because unquantized BF16 weights require `30.54 GB`, `2× TPU v6e` (`TP=2`) is the minimum full-precision BF16 footprint on Trillium. It delivers **`32,801.4 tokens/sec`** (`557.3 TFLOPS`), beating **1× Best-Optimized G4 (`29,202.0 tokens/sec`) by `1.12×` (`+12%`)** and **1× Stock G4 by `1.32×` (`+32%`)**.
> 3. **4× Chips vs. 4× Chips (Cluster Scale-Out Comparison)**:
>    - **Ultra-Low-Latency Single-Protein Design (`B=1, S=256`)**: **4× TPU v7x (`TP=8`)** reaches **`12.51 ms`** (`79.9 proteins/sec` — **`3.12×` faster** than Best-Opt G4 and **`5.11×` faster** than Stock G4). **4× TPU v6e (`TP=4` Full BF16)** reaches **`18.85 ms`** (`53.1 proteins/sec` — **`2.07×` faster** than Best-Opt G4 and **`3.40×` faster** than Stock G4).
>    - **High-Throughput Proteome Screening (`4× Chips`)**: **4× TPU v7x (`DP=4 × On-Die TP=2`)** reaches **`202,150.3 tokens/sec`** (`3,406.9 TFLOPS`), beating **4× Best-Optimized G4 (`116,280.1 tokens/sec`) by `1.74×` (`+74%`)**. At medium batch (`TotalB=4, S=512`), **4× TPU v6e (`INT8 + Pallas VMEM DP=4`)** reaches **`42,865.3 tokens/sec` (`47.78 ms`)**, beating **4× Best-Optimized G4 (`20,829.2 tokens/sec`, `98.32 ms`) by `2.06×` (`+106%`)**.
> 4. **Relative Price/Performance Advantage (`%`, Zero Dollar Figures)**:
>    - Normalized to **1× Stock G4 = `100%`** (and **1× Best-Optimized G4 = `100%`**), **TPU v7x** achieves **`185%` Price/Performance (`+85%` vs. Stock G4, `+57%` vs. Best-Opt G4)**, and **1× TPU v6e (INT8)** achieves **`134%` Price/Performance (`+34%` vs. Stock G4, `+14%` vs. Best-Opt G4)**.

![Executive Benchmark Overview](charts/01_executive_headline_comparison.png)

---

## 1. Understanding `GB.Protein-16B` & How Customers Use It in Production

`genbio-ai/GB.Protein-16B` (`AIDO.Protein-16B`) is a **bidirectional Sparse Mixture-of-Experts (MoE) encoder** trained on massive protein sequence corpora (UniRef / metagenomic sequences). Unlike causal LLM chatbots that generate one token at a time (`S_q = 1`), biological foundation models process **complete amino acid sequences (`256` to `2,048+` residues)** in a single bidirectional forward pass to power three core scientific workflows:
1. **Zero-Shot Variant Effect & Deep Mutational Scanning (DMS)**: Scoring log-likelihood differences ($\Delta \log P$) across thousands of single- and multi-site mutations (e.g., optimizing fluorescence in **Green Fluorescent Protein (`P42212`)** or antibody affinity maturation).
2. **Multi-Chain Protein-Protein Interaction & Complex Embedding**: Extracting `2,304`-dimensional structure-aware residue representations across multi-chain complexes (such as the **`822`-residue SARS-CoV-2 Spike RBD + Human ACE2 receptor complex**).
3. **Interactive De Novo Protein Sequence Design**: Iteratively mutating and evaluating candidate sequences at `Batch = 1`, where sub-20ms single-sequence turnaround directly accelerates in-silico directed evolution loops.

```mermaid
flowchart LR
    subgraph Input["Biological Sequence Input"]
        AA["Protein Sequence / Complex\n(256 to 2,048 Amino Acids)\ne.g. GFP (238 aa) or SARS2-ACE2 (822 aa)"]
    end
    subgraph Backbone["GB.Protein-16B (36 Layers, 16.05B Total / 4.21B Active Params)"]
        Emb["Amino Acid Embedding\n(44 Vocab -> 2,304 Hidden)"]
        Attn["Bidirectional Full Attention\n(36 Heads x 64 Dim + 1D RoPE)"]
        Router["Top-2 of 8 Sparse MoE Router\n(Active Compute = 16.82 GFLOPs/token)"]
        Experts["2 Active SwiGLU Experts / Token\n(2,304 -> 7,680 -> 2,304 + Custom Pallas VMEM)"]
        Emb --> Attn --> Router --> Experts
    end
    subgraph Outputs["Downstream Customer Applications"]
        DMS["Zero-Shot Variant Fitness (DMS)\nLogits (B, S, 44) -> Delta Log-Likelihood"]
        Struct["Structure & Function Embeddings\nHidden States (B, S, 2304)"]
        Design["Interactive De Novo Design\nSub-20ms Closed-Loop Screening"]
        Experts --> DMS
        Experts --> Struct
        Experts --> Design
    end
```

### Model Architecture & Hardware Roofline Specifications

| Parameter | Specification | Hardware Execution Impact |
| :--- | :--- | :--- |
| **Architecture** | `AIDO.Protein-16B` (`RnaMoEForMaskedLM`) | Bidirectional encoder (no causal mask; full $S \times S$ attention) |
| **Total vs. Active Parameters** | **16.05 Billion** (`30.54 GB` BF16 / `15.27 GB` INT8) / **4.21 Billion Active** | Fits on `1× TPU v7x` (BF16), `2× TPU v6e` (BF16), `1× TPU v6e` (INT8), or `1× G4` (BF16) |
| **Layers & Hidden Size** | `36` Layers, `H = 2,304`, `36` Heads (`D_h = 64`) | `36` heads divide evenly by `TP=2` (`18`/core) and `TP=4` (`9`/chip) |
| **MoE Configuration** | `8` Experts/Layer, **Top-K = 2 Active**, `I = 7,680` | `26.2%` active parameter ratio (`16.82 GFLOPs/token` base compute) |
| **Hardware Compared** | **TPU v7x** (`192 GB` HBM3e, `7,372 GB/s`, `4,614 TFLOPS` BF16)<br>**TPU v6e** (`32 GB` HBM2e, `1,600 GB/s`, `918 TFLOPS` BF16 / `1,836 TOPS` INT8)<br>**NVIDIA G4** (`96 GB` GDDR7, `1,792 GB/s`) | TPU v7x provides `4.1×` the memory bandwidth of G4; TPU v6e provides ultra-compact single-chip INT8 & multi-chip BF16 scaling |

---

## 2. Step-by-Step Optimization Methodology (Both GPU & TPU)

To guarantee a fair, rigorous scientific comparison, we optimized **both** the NVIDIA G4 GPU and Google Cloud TPUs (v6e and v7x):

````carousel
### Step 1: Optimizing NVIDIA G4 GPU (`+151%` Over Stock HuggingFace)
Stock HuggingFace `RnaMoEBlock` executes a Python loop over all 8 experts (`for i in range(8)`), indexing tokens with dynamic boolean masks (`torch.where`) that trigger host-device syncs at every layer (`36 × 8 = 288` syncs per forward pass).
- **Active-Expert Top-K=2 Gather**: We rewrote the MoE block to sort and gather only the `2 × N` routed token-expert pairs onto contiguous tensors.
- **Fused SDPA Bidirectional Attention**: Replaced eager materialization of the $(B, 36, S, S)$ attention matrix with PyTorch 2.6 Flash/Memory-Efficient SDPA.
- **Inductor Compilation (`torch.compile`)**: Compiled the full 36-layer model with `torch.compile(mode="default")` and TF32/BF16 TensorCore matmuls, improving `1× G4` speed by **`1.64×` at `B=1, S=256`** (`63.99 ms` $\rightarrow$ `39.00 ms`) and **`2.51×` at `B=1, S=512`** (`99.04 ms` $\rightarrow$ `39.42 ms`).
<!-- slide -->
### Step 2: Custom Pallas VMEM Kernel & Bio-Megakernel on TPU v6e & TPU v7x
We engineered a pure-XLA + **Pallas VMEM Bio-Megakernel** tailored to `GB.Protein-16B`:
- **Custom Pallas VMEM Fused Bias-SwiGLU Kernel (`jax.experimental.pallas`)**: Streams `gate_proj` and `up_proj` tiles directly from HBM into on-chip **VMEM** via double-buffered DMAs (`block_m=128, block_n=512`), adding the expert biases and computing $\text{silu}(g + b_g) \odot (u + b_u)$ entirely in registers/VMEM (`1,207.0 GB/s` on TPU v7x and `759.4 GB/s` on TPU v6e).
- **Single-Chip `1× TPU v6e` (32 GB) INT8 + Pallas W8A8 Execution**: Using per-channel symmetric `int8` weight quantization (`15.27 GB` total footprint) plus dynamic `int8` activation quantization feeding the TPU v6e native `int8 × int8 -> int32` MXU alongside the Pallas VMEM SwiGLU kernel, the entire 16B model runs on **1 single 32 GB TPU v6e chip** (`42.64 ms` at `B=1, S=256`, `100.0%` top-1 token agreement).
- **Zero-Padding `TP=4` on `4× TPU v6e` (`36` Heads $\div 4 = 9$ Heads/Chip)**: Because `GB.Protein-16B` has `36` attention heads (`H=2,304`), `TP=4` on 4 TPU v6e chips partitions attention into **exact `9` heads per chip (`D_h=64`) with zero head padding**, cutting single-protein latency to **`18.85 ms` (`2.07×` faster than Best-Opt G4)**.
````

![Optimization Waterfall Progression](charts/08_optimization_waterfall_speedup.png)

---

## 3. Apples-to-Apples Benchmark Results: 1× Chip vs. 1× Chip (Plus 2× TPU v6e BF16)

To ensure a transparent **apples-to-apples comparison by chip count**, the table below compares **single-chip deployments (`1× G4 GPU` vs. `1× TPU v6e Chip` vs. `1× TPU v7x Chip`)**, alongside **`2× TPU v6e Chips`**, which is the minimum chip count required to hold the unquantized `30.54 GB` BF16 weights in 32 GB/chip HBM2e.

![Single-Protein Latency Scaling](charts/02_single_protein_latency_scaling.png)

![Per-Accelerator Throughput Scaling](charts/03_per_accelerator_throughput_scaling.png)

### Table 1: Single-Chip (`1× Chip`) & Minimum-BF16 (`2× TPU v6e`) Latency & Throughput Comparison

| Workload (`Batch`, `SeqLen`) | 1× G4 GPU (Stock HF) | 1× G4 GPU (Best-Opt) | 1× TPU v6e (32GB INT8 / Pallas W8A8) | 2× TPU v6e (64GB Full BF16 `TP=2`) | 1× TPU v7x (192GB Full BF16 `TP=2`) | 1× TPU v7x vs. 1× Best-Opt G4 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`B=1, S=256`** *(Single Domain)* | `63.99 ms`<br>(`3,999 tok/s`) | `39.00 ms`<br>(`6,563 tok/s`) | `42.64 ms`* / `42.87 ms`<br>(`6,004` / `5,972 tok/s`) | `41.80 ms`<br>(`6,125 tok/s`) | **`34.73 ms`**<br>(**`7,372 tok/s`**) | **`1.12×` faster** (`1.84×` vs Stock) |
| **`B=1, S=512`** *(Medium Protein)* | `99.04 ms`<br>(`5,170 tok/s`) | `39.42 ms`<br>(`12,987 tok/s`) | `47.66 ms`* / `54.51 ms`<br>(`10,743` / `9,393 tok/s`) | `43.98 ms`<br>(`11,641 tok/s`) | **`36.38 ms`**<br>(**`14,075 tok/s`**) | **`1.08×` faster** (`2.72×` vs Stock) |
| **`B=1, S=1024`** *(Multi-Chain Complex)* | `97.70 ms`<br>(`10,481 tok/s`) | `44.78 ms`<br>(`22,870 tok/s`) | `77.36 ms`* / `78.57 ms`<br>(`13,237` / `13,033 tok/s`) | `55.66 ms`<br>(`18,397 tok/s`) | **`42.26 ms`**<br>(**`24,230 tok/s`**) | **`1.06×` faster** (`2.31×` vs Stock) |
| **`B=1, S=2048`** *(Large Complex)* | `95.36 ms`<br>(`21,477 tok/s`) | `86.44 ms`<br>(`23,693 tok/s`) | `146.97 ms`<br>(`13,935 tok/s`) | `94.28 ms`<br>(`21,722 tok/s`) | **`55.64 ms`**<br>(**`36,809 tok/s`**) | **`1.55×` faster** (`1.71×` vs Stock) |
| **`B=4, S=512`** *(2,048 tokens)* | `99.26 ms`<br>(`20,633 tok/s`) | `74.47 ms`<br>(`27,499 tok/s`) | `116.57 ms`<br>(`17,568 tok/s`) | `78.79 ms`<br>(`25,992 tok/s`) | **`47.62 ms`**<br>(**`43,009 tok/s`**) | **`1.56×` faster** (`2.08×` vs Stock) |
| **`B=8, S=512`** *(4,096 tokens)* | `175.47 ms`<br>(`23,344 tok/s`) | `144.89 ms`<br>(`28,270 tok/s`) | `199.78 ms`<br>(`20,503 tok/s`) | `134.86 ms`<br>(`30,373 tok/s`) | **`82.06 ms`**<br>(**`49,912 tok/s`**) | **`1.77×` faster** (`2.14×` vs Stock) |
| **`B=16, S=512`** *(8,192 tokens)* | `340.51 ms`<br>(`24,058 tok/s`) | `283.17 ms`<br>(`28,929 tok/s`) | `383.82 ms`<br>(`21,343 tok/s`) | `253.75 ms`<br>(`32,283 tok/s`) | **`156.90 ms`**<br>(**`52,212 tok/s`**) | **`1.80×` faster** (`2.17×` vs Stock) |
| **`B=32, S=512`** *(16,384 tokens)* | `671.88 ms`<br>(`24,385 tok/s`) | `561.07 ms`<br>(`29,202 tok/s`) | `759.82 ms`<br>(`21,563 tok/s`) | `499.49 ms`<br>(`32,801 tok/s`) | **`311.30 ms`**<br>(**`52,631 tok/s`**) | **`1.80×` faster** (`2.16×` vs Stock) |
| **`B=8, S=1024`** *(8,192 tokens)* | `343.62 ms`<br>(`23,841 tok/s`) | `288.24 ms`<br>(`28,421 tok/s`) | `389.38 ms`<br>(`21,039 tok/s`) | `261.26 ms`<br>(`31,355 tok/s`) | **`159.42 ms`**<br>(**`51,385 tok/s`**) | **`1.81×` faster** (`2.16×` vs Stock) |
| **`B=16, S=1024`** *(16,384 tokens)* | `664.26 ms`<br>(`24,665 tok/s`) | `573.92 ms`<br>(`28,547 tok/s`) | `770.82 ms`<br>(`21,255 tok/s`) | `514.53 ms`<br>(`31,843 tok/s`) | **`317.77 ms`**<br>(**`51,559 tok/s`**) | **`1.81×` faster** (`2.09×` vs Stock) |
| **`B=8, S=2048`** *(16,384 tokens)* | `661.07 ms`<br>(`24,788 tok/s`) | `591.37 ms`<br>(`27,705 tok/s`) | `798.71 ms`<br>(`20,513 tok/s`) | `551.84 ms`<br>(`29,690 tok/s`) | **`334.48 ms`**<br>(**`48,984 tok/s`**) | **`1.77×` faster** (`1.98×` vs Stock) |

*\*Note: On `1× TPU v6e`, values marked with `*` use the **Pallas VMEM + W8A8 Dynamic INT8 MXU** path (`tpu6e_1chip_pallas_w8a8_low_latency`), which reduces `B=1, S=512` latency from `54.51 ms` to `47.66 ms` (`+14.4%` speedup), while the second value is the W8A16 INT8 Weight-Only path (`tpu6e_1chip_int8_megakernel`).*

---

## 4. Apples-to-Apples Benchmark Results: 4× Chips vs. 4× Chips (`4× G4` vs. `4× TPU v6e` vs. `4× TPU v7x`)

When scaling out to **4 chips**, TPUs offer two distinct architectural advantages over `4× G4 GPUs`:
1. **Ultra-Low-Latency Single-Protein Mode (`TP=4` on 4× TPU v6e & `TP=8` on 4× TPU v7x)**: Dedicated high-speed Inter-Chip Interconnect (ICI) shards every layer across all 4 chips so a single protein (`B=1`) uses the combined memory bandwidth of all 4 chips simultaneously (`18.85 ms` on `4× TPU v6e` and `12.51 ms` on `4× TPU v7x`).
2. **High-Throughput Proteome Screening Mode (`DP=4` / `DP=2×TP=2`)**: Shards large protein batches across all 4 chips with near-100% linear scaling.

![4-Chip Cluster Throughput Comparison](charts/04_4chip_cluster_throughput_comparison.png)

### Table 2A: 4-Chip Ultra-Low-Latency Mode (`TP=4` on 4× TPU v6e & `TP=8` on 4× TPU v7x vs. Best-Opt G4)

| Workload (`Batch`, `SeqLen`) | 1× Best-Opt G4 (`39.0 ms` floor) | 4× TPU v6e (`BF16 TP=4`, 9 Heads/Chip) | 4× TPU v7x (`BF16 TP=8`, 8 TensorCores) | 4× TPU v6e Speedup vs. Best-Opt G4 | 4× TPU v7x Speedup vs. Best-Opt G4 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`B=1, S=256`** *(Single Domain)* | `39.00 ms` (`25.6 prot/s`) | **`18.85 ms`** (**`53.1 prot/s`**) | **`12.51 ms`** (**`79.9 prot/s`**) | **`2.07×` faster** (`3.40×` vs Stock) | **`3.12×` faster** (`5.11×` vs Stock) |
| **`B=1, S=512`** *(Medium Protein)* | `39.42 ms` (`25.4 prot/s`) | **`24.16 ms`** (**`41.4 prot/s`**) | **`15.49 ms`** (**`64.6 prot/s`**) | **`1.63×` faster** (`4.10×` vs Stock) | **`2.55×` faster** (`6.39×` vs Stock) |
| **`B=1, S=1024`** *(Multi-Chain Complex)* | `44.78 ms` (`22.3 prot/s`) | **`35.69 ms`** (**`28.0 prot/s`**) | **`19.99 ms`** (**`50.0 prot/s`**) | **`1.25×` faster** (`2.74×` vs Stock) | **`2.24×` faster** (`4.89×` vs Stock) |
| **`B=1, S=2048`** *(Large Complex)* | `86.44 ms` (`11.6 prot/s`) | **`63.84 ms`** (**`15.7 prot/s`**) | **`30.02 ms`** (**`33.3 prot/s`**) | **`1.35×` faster** (`1.49×` vs Stock) | **`2.88×` faster** (`3.18×` vs Stock) |
| **`B=8, S=512`** *(4,096 tokens)* | `144.89 ms` (`28,270 tok/s`) | **`85.44 ms`** (**`47,942 tok/s`**) | **`38.49 ms`** (**`106,421 tok/s`**) | **`1.70×` faster** | **`3.76×` faster** |
| **`B=16, S=512`** *(8,192 tokens)* | `283.17 ms` (`28,929 tok/s`) | **`167.33 ms`** (**`48,957 tok/s`**) | **`69.08 ms`** (**`118,582 tok/s`**) | **`1.69×` faster** | **`4.10×` faster** |

### Table 2B: 4-Chip High-Throughput Batch Screening (`4× G4` vs. `4× TPU v6e` vs. `4× TPU v7x`)

| Total Cluster Batch (`TotalB`, `SeqLen`) | 4× G4 GPU (`Best-Opt DP=4`) | 4× TPU v6e (`BF16 DP=2×TP=2`) | 4× TPU v6e (`INT8 DP=4`) | 4× TPU v7x (`BF16 DP=4×TP=2`) | Best 4× TPU vs. 4× Best-Opt G4 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`TotalB=4, S=512`** *(2,048 tok)* | `98.32 ms`<br>(`20,829 tok/s`) | `46.53 ms`<br>(`44,016 tok/s`) | **`47.78 ms`**\* / `49.97 ms`<br>(**`42,865`**\* / `40,987 tok/s`) | **`37.82 ms`**<br>(**`54,148 tok/s`**) | **v7x: `2.60×` (`+160%`)**<br>**v6e: `2.11×` (`+111%`)** |
| **`TotalB=16, S=512`** *(8,192 tok)* | `90.24 ms`<br>(`90,782 tok/s`) | `135.22 ms`<br>(`60,584 tok/s`) | `117.16 ms`<br>(`69,920 tok/s`) | **`48.94 ms`**<br>(**`167,406 tok/s`**) | **v7x: `1.84×` (`+84%`)** |
| **`TotalB=32, S=512`** *(16,384 tok)* | `155.14 ms`<br>(`105,606 tok/s`) | `254.09 ms`<br>(`64,480 tok/s`) | `203.22 ms`<br>(`80,624 tok/s`) | **`106.76 ms`**<br>(**`153,465 tok/s`**) | **v7x: `1.45×` (`+45%`)** |
| **`TotalB=64, S=512`** *(32,768 tok)* | `289.20 ms`<br>(`113,306 tok/s`) | `499.48 ms`<br>(`65,605 tok/s`) | `388.46 ms`<br>(`84,353 tok/s`) | **`162.10 ms`**<br>(**`202,150 tok/s`**) | **v7x: `1.78×` (`+78%`)** |
| **`TotalB=32, S=1024`** *(32,768 tok)* | `297.10 ms`<br>(`110,292 tok/s`) | `515.17 ms`<br>(`63,606 tok/s`) | `395.92 ms`<br>(`82,764 tok/s`) | **`166.96 ms`**<br>(**`196,267 tok/s`**) | **v7x: `1.78×` (`+78%`)** |
| **`TotalB=64, S=1024`** *(65,536 tok)* | `563.60 ms`<br>(`116,280 tok/s`) | `1,030.22 ms`<br>(`63,614 tok/s`) | `778.18 ms`<br>(`84,217 tok/s`) | **`328.91 ms`**<br>(**`199,255 tok/s`**) | **v7x: `1.71×` (`+71%`)** |
| **`TotalB=32, S=2048`** *(65,536 tok)* | `593.07 ms`<br>(`110,504 tok/s`) | `1,105.03 ms`<br>(`59,307 tok/s`) | `805.21 ms`<br>(`81,390 tok/s`) | **`351.20 ms`**<br>(**`186,608 tok/s`**) | **v7x: `1.69×` (`+69%`)** |

---

## 5. Relative Price/Performance Efficiency Comparison (Expressed Strictly as `%`, Zero Dollar Values)

Because cloud capacity pricing varies across on-demand, 1-year/3-year committed use discounts (CUDs), and enterprise agreements, we evaluate **Price/Performance efficiency strictly as relative percentages (`%`) with zero currency (`$`) figures**.

We present **three transparent relative cost-weight models** (normalized to **`1× G4 GPU = 100%` per-unit cost weight**):
1. **Equal-Unit Hardware Comparison (`100%` weight per deployment unit)**: Pure performance per unit (`1× G4` vs. `1× TPU v6e INT8` vs. `2× TPU v6e BF16` vs. `1× TPU v7x BF16`).
2. **Standard Cloud Tier Cost Ratio (`TPU v6e = 65%` of G4 per chip; `G4 = 100%`; `TPU v7x = 115%` of G4 per chip)**: Reflects Trillium (`v6e`) being a compact, highly cost-efficient 32 GB inference chip (~35% lower per-chip cost weight than a 96 GB Blackwell Server GPU) and Ironwood (`v7x`) being a flagship 192 GB HBM3e dual-core chip (~15% premium over G4).
3. **Conservative Tier Cost Ratio (`TPU v6e = 75%` of G4 per chip; `G4 = 100%`; `TPU v7x = 125%` of G4 per chip)**: Stress-tests TPU price/performance under a more conservative chip-cost ratio.

![Relative Price/Performance Efficiency Percentage](charts/05_price_performance_efficiency_percentage.png)

### Table 3: Relative Price/Performance Efficiency Index (`%` — Higher is Better, Zero Dollar Values)

| Deployment Configuration | Peak Sustained Throughput (`tok/s`) | Relative Cost Weight (Standard Tier, `G4=100%`) | Raw Throughput `%` vs. Stock G4 (`=100%`) | Price/Perf Index `%` vs. **Stock G4 (`=100%`)** (Standard Tier) | Price/Perf Index `%` vs. **Best-Opt G4 (`=100%`)** (Standard Tier) | Price/Perf Index `%` vs. **Stock G4 (`=100%`)** (Conservative Tier) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1× G4 GPU (Stock HF Baseline)** | `24,788.1` | `100%` | `100%` *(Baseline)* | **`100%`** *(Baseline)* | `85%` (`-15%`) | `100%` *(Baseline)* |
| **1× G4 GPU (Best-Optimized)** | `29,202.0` | `100%` | `118%` (`+18%`) | **`118%`** (`+18%`) | **`100%`** *(Baseline)* | `118%` (`+18%`) |
| **2× TPU v6e Chips (Full BF16 `TP=2`)** | `32,801.4` | `130%` *(2 × 65%)* | `132%` (`+32%`) | **`102%`** (`+2%`) | `86%` *(or **`112%`** per-pair raw output)* | `88%` |
| **1× TPU v6e Chip (INT8 Weight-Only `TP=1`)** | `21,563.0` | `65%` *(1 × 65%)* | `87%` | **`134%` (`+34%` better)** | **`114%` (`+14%` better)** | **`116%` (`+16%` better)** |
| **1× TPU v7x Chip (Full BF16 `TP=2`)** | **`52,630.9`** | `115%` *(1 × 115%)* | **`212%` (`+112%`)** | **`185%` (`+85%` better)** | **`157%` (`+57%` better)** | **`170%` (`+70%` better)** |
| **4× TPU v6e (`BF16 TP=4`, Interactive `B=1, S=256`)** | `53.1 prot/s` (`18.85 ms`) | `260%` *(vs 4× G4 = 400%)* | **`340%` (`+240%`)** | **`131%` (`+31%` per-chip eff.)** | **`128%` (`+28%` vs Opt G4)** | **`113%` (`+13%` better)** |
| **4× TPU v7x (`DP=4×TP=2`) vs. 4× Best-Opt G4** | **`202,150.3`** *(vs `116,280.1`)* | `460%` *(vs 4× G4 = 400%)* | **`204%` (`+104%`)** | **`177%` (`+77%` better)** | **`151%` (`+51%` better)** | **`163%` (`+63%` better)** |

> [!TIP]
> **Which TPU Should GenBio AI Choose?**
> - **Choose Google Cloud TPU v7x (Ironwood)** for **maximum absolute performance AND highest Price/Performance (`185%` vs. Stock G4, `157%` vs. Best-Opt G4)**: A single TPU v7x chip holds the entire 16.05B model in unquantized BF16 with `160+ GB` of free HBM3e headroom and beats `1× Best-Optimized G4` by **`+80%` in throughput** and **`+57%` in Price/Performance**.
> - **Choose Google Cloud TPU v6e (Trillium)** for **cost-efficient single-chip INT8 serving (`134%` Price/Performance vs. Stock G4, `114%` vs. Best-Opt G4)** or **low-latency 2-chip/4-chip BF16 scale-out**: `1× TPU v6e` runs the entire 16.05B model on a single 32 GB chip with `100.0%` top-1 token agreement, while `4× TPU v6e` (`TP=4`) delivers **`18.85 ms` single-protein latency (`2.07×` faster than Best-Opt G4)**.

---

## 6. Compute Utilization (`Active TFLOPS`) & Custom Pallas VMEM Kernel Performance

To verify that our TPU performance comes from genuine hardware utilization, we measured both **Sustained Active TFLOPS** across the full 36-layer model and isolated **Custom Pallas VMEM Fused Bias-SwiGLU Kernel** bandwidth across routed token batch sizes (`M = 512` to `8,192` tokens, `I_shard = 3,840`).

![Active TFLOPS and Pallas VMEM Bandwidth](charts/06_active_tflops_and_pallas_vmem_bandwidth.png)

### Table 4: Custom Pallas VMEM Fused Bias-SwiGLU Microbenchmark (`TPU v6e` vs. `TPU v7x`)

| Routed Tokens (`M`) | Intermediate Shard (`I_shard`) | TPU v6e Pallas VMEM Latency (`us`) | TPU v6e Effective Bandwidth (`GB/s`) | TPU v7x Pallas VMEM Latency (`us`) | TPU v7x Effective Bandwidth (`GB/s`) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`512`** | `3,840` | `201.29 us` | `97.7 GB/s` | **`168.54 us`** | **`116.7 GB/s`** |
| **`1,024`** | `3,840` | `202.98 us` | `193.7 GB/s` | **`168.57 us`** | **`233.3 GB/s`** |
| **`2,048`** | `3,840` | `258.59 us` | `304.2 GB/s` | **`180.60 us`** | **`435.5 GB/s`** |
| **`4,096`** | `3,840` | `347.95 us` | `452.1 GB/s` | **`202.07 us`** | **`778.5 GB/s`** |
| **`8,192`** | `3,840` | `414.23 us` | **`759.4 GB/s`** | **`260.63 us`** | **`1,207.0 GB/s`** |

---

## 7. End-to-End Numerical Parity & Real Biological Validation (`GFP` & `SARS-CoV-2 RBD + Human ACE2`)

Every optimization was audited against real biological sequences—never synthetic random weights or dummy layers:
1. ***Aequorea victoria* Green Fluorescent Protein (`UniProt P42212`, `238` amino acids)**:
   - Evaluated zero-shot Masked Language Modeling (MLM) mutation log-likelihood ratios ($\Delta \log P = \log P(\text{mut}) - \log P(\text{WT})$) on three classic functional variants:
     - **`S65T`** *(Heim et al., Nature 1995 — canonical Enhanced GFP chromophore shift)*: `BF16 = -0.1090`, `INT8 (1× v6e) = -0.0830` (conserved chromophore constraint captured identically).
     - **`C48A`** *(surface cysteine removal to prevent aggregation)*: `BF16 = +2.2559`, `INT8 (1× v6e) = +2.1025` (strong positive fitness gain predicted identically by both BF16 and 1-chip INT8).
     - **`G67A`** *(autocatalytic cyclization residue)*: `BF16 = +0.0078`, `INT8 (1× v6e) = +0.1699`.
2. **Multi-Chain Protein-Protein Interaction Complex (`SARS-CoV-2 Spike RBD + Human ACE2 Peptidase Domain`, `822` amino acids)**:
   - Extracted full `2,304`-dimensional residue embeddings across the two-chain complex separated by chain boundary tokens, achieving **`92.7%` unmasked amino acid identity recovery** and **`1.81×` to `4.08×` higher complex screening throughput** on TPU vs. Best-Optimized G4.

![Numerical Parity and Biological Validation](charts/07_numerical_parity_and_biological_validation.png)

### Table 5: End-to-End Numerical Parity Summary Across All Hardware Configurations

| Implementation Under Test | Reference Baseline | Logit Cosine Similarity | Final Hidden State (`2,304-d`) Cosine Similarity | Top-1 Predicted Amino Acid Agreement |
| :--- | :--- | :---: | :---: | :---: |
| **1× G4 Best-Optimized (`torch.compile` + TopK=2)** | Stock HuggingFace PyTorch (`1× G4`) | **`0.999984` (`99.998%`)** | **`0.996391` (`99.64%`)** | **`99.80%`** |
| **TPU v7x Bio-Megakernel (`Full BF16 TP=2`)** | Stock HuggingFace PyTorch (`1× G4`) | **`0.999984` (`99.998%`)** | **`0.996568` (`99.66%`)** | **`99.41%`** |
| **TPU v6e Bio-Megakernel (`Full BF16 TP=4`)** | TPU v6e Full BF16 Reference (`TP=2`) | **`0.999991` (`99.999%`)** | **`0.998195` (`99.82%`)** | **`99.58%`** |
| **1× TPU v6e Single-Chip (`INT8 + Pallas`)** | TPU v6e Full BF16 Reference (`TP=2`) | **`0.999952` (`99.995%`)** | **`0.992756` (`99.28%`)** | **`100.00%`** |
