# Part B — Capacity & Throughput Reconciliation Report (Audited)

**Scripts & Data Files:** `your-submission/partB/{capacity_reconciliation.py, partB_calculation_output.txt, partB_reconciliation.json}`  
**Source Inputs Audited (Unmodified):** `bench/model_spec.md`, `bench/bench_log.csv`, `REPORT_v0.md` (Section 2)

---

## 0. Executive Summary Mapped Directly to Rubric Questions B1–B4

- **B1 (7 pts) — Exact KV-Cache Bytes/Token & Maximum Concurrent `4096`-Token Sequences:**
  - **(a) KV-Cache Bytes per Token (GQA):** Using $L=28$ layers, $n_{kv}=8$ GQA KV heads (not $n_q=24$), $d_h=128$, and `fp16` (`2` B/element):
    $$\text{KV Bytes/Token} = 2_{\text{(K,V)}} \times 28 \times 8 \times 128 \times 2 = \mathbf{114,688 \text{ bytes/token}} \ (\mathbf{112.0 \text{ KiB/tok}} = \mathbf{0.114688 \text{ MB/tok}})$$
  - **(b) Approximate Max Concurrent `4096`-Token Sequences:** Usable GPU memory is $24.0\text{ GB} \times 0.92 = 22.08\text{ GB}$; subtracting `fp16` model weights ($4.2\text{ B} \times 2\text{ B} = 8.40\text{ GB}$) and non-KV runtime overhead ($1.60\text{ GB}$) leaves an idealized KV-cache pool of **$12.08\text{ GB}$** ($12,080,000,000\text{ B} \approx 11.2504\text{ GiB}$), or $\lfloor 12,080,000,000 / 114,688 \rfloor = \mathbf{105,329\text{ tokens}}$. Each `4096`-token sequence requires $4096 \times 114,688\text{ B} = \mathbf{469,762,048\text{ B}}$ (`469.76 MB` decimal / `448.00 MiB` = `0.4375 GiB` binary), yielding an **idealized capacity estimate of**:
    $$B_{\max,\text{ideal}}(4096) = \left\lfloor \frac{12,080,000,000}{469,762,048} \right\rfloor = \lfloor 25.715 \rfloor = \mathbf{25 \text{ sequences}}$$
  - **Check Against `bench_log.csv`:** At $B=24$ (`Row 11`), $24 / 25.715 = \mathbf{0.9333}$ (matches logged `kv_cache_util = 0.93`, `preempted_seqs = 0`). At $B=25$, utilization reaches $25 / 25.715 = \mathbf{0.9722}$ (rounds to **`0.97`**, the exact `kv_cache_util` ceiling logged at $B=32$ and $B=48$). At $B=32$ (`Row 12`) and $B=48$ (`Row 13`), the excess sequences over the 25-sequence cap are $32 - 25 = \mathbf{7}$ and $48 - 25 = \mathbf{23}$, matching logged `preempted_seqs = 7` and `preempted_seqs = 23`.

- **B2 (6 pts) — Long-Context (`prompt_len = 3584`) Throughput Anomaly, Mechanism, & Proposed Fix:**
  - **Anomaly:** In the long-context sweep (`Rows 8–13`, $p=3584, g=512$), `reported_tok_s` rises from `565.4` ($B=4$) to a peak of **`1607.4 tok/s` at $B=24$ (`Row 11`)**, but then **drops by `-13.9%` to `1384.0 tok/s` at $B=32$ (`Row 12`)** and **by `-19.2%` to `1298.5 tok/s` at $B=48$ (`Row 13`)**.
  - **Mechanism:** At $B=32$ and $B=48$, unthrottled KV-cache demand (`15.032 GB` = `124.44%` and `22.549 GB` = `186.66%` of the `12.08 GB` pool) exceeds physical KV capacity ($B_{\max,\text{ideal}} = 25$, `kv_cache_util = 0.97`), forcing the scheduler to preempt `7` ($B=32$) and `23` ($B=48$) sequences (`preempted_seqs`). Evicted/queued sequences must execute in a second wave (with re-prefill/recomputation overhead), stretching `wall_clock_s` from `61.16 s` ($B=24$) to `94.71 s` ($+54.9\%$) and `151.41 s` ($+147.6\%$), elevating `ttft_ms_p50` (`500.5 ms` $\rightarrow$ `636.9 ms` / `955.4 ms`), and inflating `e2e_ms_p95` (`69.22 s` $\rightarrow$ `97.47 s` / `105.43 s`).
  - **Proposed Config Change & Predicted Quantitative Effect:** Enable **`fp8` (`1` byte/element) KV-cache quantization** on the L4 GPU (or set scheduler `max_num_seqs = 24` to prevent over-admission). Quantizing KV cache from `fp16` to `fp8` halves per-token KV memory (`114,688 B` $\rightarrow$ `57,344 B = 56.0 KiB/tok`) and per-sequence memory (`469.76 MB` $\rightarrow$ `234.88 MB`), **doubling idealized `4096`-token capacity from `25` to `51` concurrent sequences**, reducing peak `kv_cache_util` to **`0.62` at $B=32$** (`7.516 GB / 12.08 GB`) and **`0.93` at $B=48$** (`11.274 GB / 12.08 GB`), and reducing `preempted_seqs` from `7` and `23` to **`0`**.

- **B3 (4 pts) — The Misread Column Behind `REPORT_v0.md` Section 2 & Two Independent Derivations of Batch-24 Goodput:**
  - **The Misread Column:** `reported_tok_s` is **Total-Token Throughput**:
    $$\texttt{reported\_tok\_s} = \frac{\texttt{num\_requests} \times (\texttt{prompt\_len} + \texttt{gen\_len})}{\texttt{wall\_clock\_s}}$$
    Because it includes prompt prefill tokens in the numerator, it multiplies true generated-output throughput by $\frac{p+g}{g} = \mathbf{3.0\times}$ on short prompts (`512/256`) vs. $\mathbf{8.0\times}$ on long prompts (`3584/512`). At $B=16$, long prompts appear higher on `reported_tok_s` (`1311.4` vs. `883.2 total tok/s`) even though **generated-output-token goodput is `44.33%` lower (`163.94` vs. `294.46 gen tok/s`)** and `itl_ms_p50` is `+59.74%` slower (`77.20 ms` vs. `48.33 ms`).*(Note: Rows 5 and 10 vary both `prompt_len` and `gen_len`, so they do not isolate `prompt_len` alone.)* Furthermore, `REPORT_v0.md` projected $B=48$ by linear extrapolation (`~3200 tok/s`) while ignoring `Row 13` ($B=48$), where `reported_tok_s` actually dropped to `1298.5 tok/s` (`162.31 gen tok/s`).
  - **Two Independent Ways to Derive Honest Generation "Goodput" for `Row 11` (`B=24, p=3584, g=512`), Plus Percentile-Based Decode Estimates:**
    1. **Way 1 — Direct End-to-End Generated-Output-Token Throughput (from `num_requests`, `gen_len`, and `wall_clock_s`):**
       $$\text{Goodput}_{\text{e2e}} = \frac{\texttt{num\_requests} \times \texttt{gen\_len}}{\texttt{wall\_clock\_s}} = \frac{24 \times 512}{61.16\text{ s}} = \frac{12,288}{61.16\text{ s}} = \mathbf{200.92 \text{ gen tok/s}} \quad \left(\text{Per-User Throughput} = \frac{512}{61.16\text{ s}} = \mathbf{8.37 \text{ gen tok/s/request}}\right)$$
    2. **Way 2 — Algebraic Decomposition of `reported_tok_s` Using Token-Length Proportions ($\frac{g}{p+g}$):**
       Because `model_spec.md:L30-L32` specifies that all requests use identical `(prompt_len, gen_len)` with no early stopping, generated output tokens are exactly $\frac{g}{p+g} = \frac{512}{3584 + 512} = \frac{1}{8}$ (`12.5%`) of all processed tokens:
       $$\text{Goodput}_{\text{from\_reported}} = \texttt{reported\_tok\_s} \times \frac{\texttt{gen\_len}}{\texttt{prompt\_len} + \texttt{gen\_len}} = 1607.4 \times \frac{512}{4096} = \frac{1607.4}{8} = \mathbf{200.925 \text{ gen tok/s}} \ (\mathbf{200.92 \text{ gen tok/s}})$$
       *(The `0.009 tok/s` difference between `200.9156` and `200.925` is strictly from 1-decimal rounding of `reported_tok_s = 1607.4` vs. unrounded $98,304 / 61.16 = 1607.325$.)*
    3. **Distinct Latency-Percentile Cross-Checks (From `ttft_ms_p50 = 500.5 ms` and `itl_ms_p50 = 96.07 ms` — *Not* Exact End-to-End Throughput):**
       - *Decode-Only Throughput Estimate (`202.57 gen tok/s`):* Subtracting median time-to-first-token (`0.5005 s`) from `wall_clock_s` (`61.16 s`) yields an estimated decode-only window of `60.6595 s` $\implies \frac{12,288}{60.6595\text{ s}} = \mathbf{202.57 \text{ gen tok/s}}$ (`8.44 gen tok/s/request`). This excludes prefill time and assumes all 24 simultaneous requests finish prefill near `ttft_ms_p50`.
       - *Instantaneous Step-Rate Estimate at Median ITL (`249.82 gen tok/s`):* Dividing active batch size by median inter-token latency gives $\frac{B}{\texttt{itl\_ms\_p50}/1000} = \frac{24}{0.09607\text{ s}} = \mathbf{249.82 \text{ gen tok/s}}$ (`10.41 gen tok/s/request`), and combining `ttft_ms_p50 + 512 * itl_ms_p50 = 49.688 s` gives a synthetic p50 request-level rate of $\frac{12,288}{49.688\text{ s}} = \mathbf{247.30 \text{ gen tok/s}}$. **Why this differs from observed end-to-end throughput (`200.92 gen tok/s`):** Per `model_spec.md:L37-L39`, `ttft_ms_p50` and `itl_ms_p50` are **medians (`p50`)**, whereas batch `wall_clock_s` (`61.16 s`) is determined by the **mean decode step duration plus tail completion across all 24 requests** (note that logged `e2e_ms_p95` is `69,221.3 ms`). Percentile-based step rates therefore overestimate sustained end-to-end batch goodput by `+23.1%` to `+24.3%` and must not be reported as observed end-to-end throughput.
  - **What `REPORT_v0.md` Should Have Said:** Packing `3584`-token prompts instead of `512`-token prompts (with `512` vs. `256` output tokens) reduces generated-output goodput at $B=16$ by **`44.3%`** (`294.5` $\rightarrow$ `163.9 gen tok/s`); long-prompt generation peaks at **$B=24$ (`200.9 gen tok/s`, `kv_cache_util = 0.93`)** and **degrades at $B=32$ (`173.0 gen tok/s`) and $B=48$ (`162.3 gen tok/s`)** because the `12.08 GB` KV cache saturates at ~25 sequences (`0.97` utilization), triggering sequence preemptions.

- **B4 (3 pts) — One Paragraph: The Single Serving-Stack Counter to Confirm the B2 Mechanism & Expected Values:**
  To confirm our B2 KV-cache exhaustion and sequence-preemption/recomputation diagnosis directly from the serving engine (e.g., vLLM), we would pull the engine's **cumulative recomputed/prefill token counter (`vllm:num_preemptions_total` paired with total executed prefill tokens `prefill_tokens_total` / `num_total_gpu_blocks`)**. Under `recompute` preemption, an unpreempted run at `batch_size = 24` (`Row 11`, `kv_cache_util = 0.93`) executes exactly $24 \times 3,584 = \mathbf{86,016\text{ prefill tokens}}$ with **`0` preemption events** (`num_total_gpu_blocks * block_size` $\approx \mathbf{102,400\text{--}105,329\text{ tokens}}$). By contrast, when KV demand exceeds the 25-sequence pool at `batch_size = 32` (`Row 12`) and `batch_size = 48` (`Row 13`), we expect `vllm:num_preemptions_total` to report **$\ge 7$** and **$\ge 23$** preemption events, respectively, and `prefill_tokens_total` to exceed the nominal input token count ($32 \times 3,584 = 114,688$ and $48 \times 3,584 = 172,032$) by at least the re-prefilled tokens of the `7` and `23` evicted sequences—showing **$\ge 114,688 + (7 \times 3,584) = \mathbf{139,776+\text{ prefill tokens}}$ at $B=32$** and **$\ge 172,032 + (23 \times 3,584) = \mathbf{254,464+\text{ prefill tokens}}$ at $B=48$** (or if `swap` mode is configured instead of `recompute`, we expect the GPU$\leftrightarrow$CPU swap traffic counter `swap_out_blocks_total` to jump from **`0` blocks at $B=24$** to **$\approx 7 \times 4,096\text{ tok} \approx 3.29\text{ GB}$ at $B=32$** and **$\approx 23 \times 4,096\text{ tok} \approx 10.80\text{ GB}$ at $B=48$**).

---

## 1. Extracted Specifications, Workload Assumptions, and Missing Parameters

### 1.1 Explicitly Supplied Parameters (`bench/model_spec.md` & `bench/bench_log.csv`)

| Category | Parameter / Field | Supplied Value | Source Location |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | Model Name & Type | `FLM-4B-Instruct (dense)` | `bench/model_spec.md:L3` |
| **Model Architecture** | Total Parameters | `4.2 B` | `bench/model_spec.md:L7` |
| **Model Architecture** | Transformer Layers ($L$) | `28` | `bench/model_spec.md:L8` |
| **Model Architecture** | Hidden Dimension ($d_{\text{model}}$) | `3072` | `bench/model_spec.md:L9` |
| **Model Architecture** | Query Attention Heads ($n_q$) | `24` | `bench/model_spec.md:L10` |
| **Model Architecture** | KV Attention Heads ($n_{kv}$, GQA) | `8` ($3:1$ Q-to-KV GQA ratio) | `bench/model_spec.md:L11` |
| **Model Architecture** | Head Dimension ($d_h$) | `128` ($24 \times 128 = 3072$) | `bench/model_spec.md:L12` |
| **Model Architecture** | Vocabulary Size ($V$) | `128k` (`128,000`) | `bench/model_spec.md:L13` |
| **Precision** | Model Weights Precision | `fp16` (`2` bytes/parameter) | `bench/model_spec.md:L14` |
| **Precision** | KV-Cache Precision | `fp16` (`2` bytes/element) | `bench/model_spec.md:L15` |
| **Hardware** | GPU Device & Total Memory | `1× NVIDIA L4 (24 GB)` | `bench/model_spec.md:L21` |
| **Hardware** | Peak Memory Bandwidth | `300 GB/s` | `bench/model_spec.md:L22` |
| **Hardware** | Peak FP16 Dense Compute | `~121 TFLOPS` | `bench/model_spec.md:L23` |
| **Serving Config** | Maximum Context Length (`max_model_len`) | `4096` tokens | `bench/model_spec.md:L24` |
| **Serving Config** | GPU Memory Utilization Cap (`gpu_memory_utilization`) | `0.92` | `bench/model_spec.md:L25` |
| **Serving Config** | Non-KV Runtime Overhead (activations, CUDA graphs, etc.) | `assume ~1.6 GB` | `bench/model_spec.md:L26` |
| **Load-Test Harness** | Arrival & Stopping Rule | `num_requests` identical requests submitted simultaneously; all generate exactly `gen_len` tokens (`no early stopping`), with `batch_size == num_requests` on all 13 rows | `bench/model_spec.md:L30-L32`, `bench/bench_log.csv:L2-L14` |
| **Load-Test Harness** | Short-Prompt Sweep (`Rows 1–7`) | `prompt_len = 512`, `gen_len = 256` ($S = 768$ tokens/req), `batch_size` $\in \{1, 2, 4, 8, 16, 32, 64\}$ | `bench/bench_log.csv:L2-L8` |
| **Load-Test Harness** | Long-Prompt Sweep (`Rows 8–13`) | `prompt_len = 3584`, `gen_len = 512` ($S = 4096 = \texttt{max\_model\_len}$ tokens/req), `batch_size` $\in \{4, 8, 16, 24, 32, 48\}$ | `bench/bench_log.csv:L9-L14` |

### 1.2 Missing / Undocumented Parameters in the Supplied Files

1. **Serving Engine, Version, and PagedAttention Block Size:** The serving framework (e.g., vLLM, TensorRT-LLM, SGLang), its KV-cache block size (e.g., `16` or `32` tokens/block), and block-table allocator metadata overhead are not specified in `model_spec.md`.
2. **Preemption Policy (`recompute` vs. `swap`) and Scheduler Admission Policy:** `model_spec.md` defines `preempted_seqs` as *"sequences the scheduler preempted at least once"*, without specifying whether preempted KV blocks are evicted and re-prefilled (`recompute`) or copied over PCIe to host RAM (`swap`), or how admission queueing interacts with `ttft_ms_p50`.
3. **Prefix Caching (`enable_prefix_caching`) and Chunked Prefill Settings:** Not specified in `model_spec.md`. Although `model_spec.md:L30` notes that `num_requests` *identical* requests are submitted simultaneously, the observed `kv_cache_util` values in `bench_log.csv` are consistent with the simple linear per-request allocation model ($B \times S$ where each request allocates independent KV blocks).
4. **Exact Feed-Forward Dimension (`d_ff`) and Embedding Tying:** Only the rounded parameter count (`4.2 B`) is given; exact MLP intermediate dimensions are omitted.
5. **Unit Convention (`Decimal GB` [$10^9$ B] vs. `Binary GiB` [$2^{30}$ B]):** `model_spec.md` writes `24 GB` and `~1.6 GB` without specifying decimal vs. binary units. In Section 2 below, we track both conventions explicitly and compare both against `bench_log.csv`.
6. **Hardware Profiler Telemetry & Full Latency Distributions:** No hardware performance counters (e.g., Nsight Compute / DCGM HBM bytes read/written per second) are provided, and `bench_log.csv` logs only median `ttft_ms_p50`, median `itl_ms_p50`, `e2e_ms_p95`, and unique `preempted_seqs` (rather than total preemption events).

---

## 2. KV-Cache Memory Derivation & Idealized Capacity Model

### 2.1 Per-Token KV-Cache Formula and Intermediate Arithmetic

For a Transformer using **Grouped-Query Attention (GQA)** with $L = 28$ layers, $n_{kv} = 8$ KV heads, head dimension $d_h = 128$, and `fp16` (`2` bytes/element) KV precision, each token stores one Key vector and one Value vector per layer and KV head:

$$\text{KV Bytes per Token (GQA)} = 2_{\text{(K,V)}} \times L \times n_{kv} \times d_h \times b_{\text{elem}}$$

$$\text{KV Bytes per Token (GQA)} = 2 \times 28 \times 8 \times 128 \times 2 = \mathbf{114,688 \text{ bytes/token}}$$

- **In Binary Units (`1 KiB = 1,024 B`, `1 MiB = 1,048,576 B`):**
  $$\frac{114,688}{1024} = \mathbf{112.0 \text{ KiB/token}} = \mathbf{0.109375 \text{ MiB/token}}$$
- **In Decimal Units (`1 KB = 1,000 B`, `1 MB = 1,000,000 B`):**
  $$\frac{114,688}{10^3} = \mathbf{114.688 \text{ KB/token}} = \mathbf{0.114688 \text{ MB/token}}$$
- **MHA Calculation Pitfall (if $n_q = 24$ were mistakenly used instead of $n_{kv} = 8$):**
  $$2 \times 28 \times 24 \times 128 \times 2 = 344,064 \text{ bytes/token} \ (336.0 \text{ KiB/token}) \quad \longrightarrow \quad \mathbf{3\times \text{ overestimate}}$$

### 2.2 Available KV-Cache Pool Budget (Decimal `GB` vs. Binary `GiB` Kept Distinct)

Applying the supplied memory parameters from `bench/model_spec.md` (`24 GB` GPU, `0.92` utilization, `4.2 B` `fp16` weights, `~1.6 GB` non-KV overhead):
1. **Total Usable GPU Memory (`gpu_memory_utilization = 0.92`):**
   $$M_{\text{usable}} = 24.0 \times 0.92 = \mathbf{22.08 \text{ GB}}$$
2. **Model Weights (`4.2 B` parameters in `fp16` = `2` bytes/parameter):**
   $$M_{\text{weights}} = 4.2 \times 10^9 \times 2 \text{ bytes} = \mathbf{8.40 \text{ GB}} \quad \left(\text{or } 7.8231 \text{ GiB}\right)$$
3. **Non-KV Runtime Overhead (activations, CUDA graphs, scratch):**
   $$M_{\text{overhead}} = \mathbf{1.60 \text{ GB}}$$
4. **Idealized Available KV-Cache Memory Pool ($M_{\text{KV}}$):**
   $$M_{\text{KV}} = 22.08 - 8.40 - 1.60 = \mathbf{12.08 \text{ GB}}$$
   - **Decimal `GB` Convention ($1 \text{ GB} = 10^9 \text{ B}$):**
     $$M_{\text{KV, dec}} = 12,080,000,000 \text{ bytes} = \mathbf{12.08 \text{ GB}} \approx \mathbf{11.2504 \text{ GiB}}$$
     $$\text{Idealized Max Tokens} = \left\lfloor \frac{12,080,000,000}{114,688} \right\rfloor = \mathbf{105,329 \text{ tokens}}$$
   - **Binary `GiB` Convention (if `12.08` denoted $12.08 \text{ GiB} = 12.08 \times 2^{30} \text{ B}$):**
     $$M_{\text{KV, GiB}} = 12,970,801,234 \text{ bytes} = \mathbf{12.08 \text{ GiB}} \approx \mathbf{12.9708 \text{ GB}}$$
     $$\text{Idealized Max Tokens (GiB)} = \left\lfloor \frac{12,970,801,234}{114,688} \right\rfloor = \mathbf{113,096 \text{ tokens}}$$

### 2.3 Per-Sequence KV Footprint & Idealized Sequence Capacity Estimate ($B_{\max}$)

Under the load-test rules (`model_spec.md:L30-L32`), every request reaches a completed sequence length of $S = \texttt{prompt\_len} + \texttt{gen\_len}$:

1. **Short-Prompt Sequence ($S = 512 + 256 = 768$ tokens):**
   $$\text{KV}_{\text{seq}}(768) = 768 \times 114,688 \text{ B} = \mathbf{88,080,384 \text{ B}} = \mathbf{88.08 \text{ MB (decimal)}} = \mathbf{84.00 \text{ MiB (binary)}}$$
   - **Idealized Capacity Estimate ($12.08 \text{ GB}$ decimal pool):**
     $$B_{\max,\text{ideal}}(768) = \left\lfloor \frac{12,080,000,000}{88,080,384} \right\rfloor = \lfloor 137.147 \rfloor = \mathbf{137 \text{ sequences}}$$
   - At the largest tested short-prompt batch (`batch_size = 64`, `Row 7`), linear KV demand is $64 \times 88.08 \text{ MB} = \mathbf{5.637 \text{ GB}}$ (`5.250 GiB`), which is **`0.4667` (`46.67%`)** of the `12.08 GB` decimal pool (matching logged **`kv_cache_util = 0.47`** and **`preempted_seqs = 0`**).

2. **Long-Prompt Sequence ($S = 3584 + 512 = 4096 = \texttt{max\_model\_len}$ tokens):**
   $$\text{KV}_{\text{seq}}(4096) = 4096 \times 114,688 \text{ B} = \mathbf{469,762,048 \text{ B}} = \mathbf{469.76 \text{ MB (decimal)}} = \mathbf{448.00 \text{ MiB}} \ (\mathbf{0.4375 \text{ GiB}})$$
   - **Idealized Capacity Estimate ($12.08 \text{ GB}$ decimal pool):**
     $$B_{\max,\text{ideal}}(4096) = \left\lfloor \frac{12,080,000,000}{469,762,048} \right\rfloor = \lfloor 25.715 \rfloor = \mathbf{25 \text{ sequences}}$$
     *(Note: Under the stated memory assumptions, `25` long-context sequences is the **idealized capacity estimate** assuming zero allocator/block fragmentation and exact `1.60 GB` non-KV overhead. In practice, PagedAttention block-size rounding, block-table metadata, reserved scratch blocks, or runtime overhead variations above `1.60 GB` can reduce actual usable capacity.)*
   - Comparing the idealized decimal-pool utilization ratios against the logged `kv_cache_util` and `preempted_seqs` columns in `bench_log.csv`:
     - **$B = 24$ sequences (`98,304` tokens, `Row 11`):** $24 \times 469,762,048 = 11,274,289,152 \text{ B} = \mathbf{11.274 \text{ GB}}$ (`10.500 GiB`) $\implies \frac{11.2743}{12.08} = \mathbf{0.9333}$ (logged `kv_cache_util = 0.93`, `preempted_seqs = 0`).
     - **$B = 25$ sequences (`102,400` tokens, idealized capacity limit):** $25 \times 469,762,048 = 11,744,051,200 \text{ B} = \mathbf{11.744 \text{ GB}}$ (`10.9375 GiB`) $\implies \frac{11.7441}{12.08} = \mathbf{0.9722}$ (rounds to **`0.97`**, matching the peak `kv_cache_util` logged at $B=32$ and $B=48$).
     - **$B = 26$ sequences (`106,496` tokens):** $26 \times 469,762,048 = 12,213,813,248 \text{ B} = \mathbf{12.214 \text{ GB}}$ (`11.375 GiB`) $\implies \frac{12.2138}{12.08} = \mathbf{1.0111 > 1.00}$ (exceeds the idealized `12.08 GB` pool).
     - **$B = 32$ (`Row 12`) & $B = 48$ (`Row 13`):** Unthrottled linear KV demand reaches **`15.032 GB` (`14.000 GiB`, `124.44%` of pool)** at $B=32$ and **`22.549 GB` (`21.000 GiB`, `186.66%` of pool)** at $B=48$. Under the idealized 25-sequence capacity estimate, the excess sequences ($32 - 25 = \mathbf{7}$ and $48 - 25 = \mathbf{23}$) match the logged **`preempted_seqs = 7`** and **`preempted_seqs = 23`**.

---

## 3. Complete Row-by-Row Reconciliation Table (`bench/bench_log.csv`)

Formulas applied across all 13 rows (`capacity_reconciliation.py`):
- **Total-Token Throughput (`Calc Total Tok/s`):** $\frac{\texttt{num\_requests} \times (\texttt{prompt\_len} + \texttt{gen\_len})}{\texttt{wall\_clock\_s}}$
- **Generated-Output-Token Throughput (`Calc Gen Output Tok/s`):** $\frac{\texttt{num\_requests} \times \texttt{gen\_len}}{\texttt{wall\_clock\_s}} = \texttt{reported\_tok\_s} \times \frac{\texttt{gen\_len}}{\texttt{prompt\_len} + \texttt{gen\_len}}$
- **Unthrottled KV Demand Ratio (`12.08 GB` dec / `12.08 GiB` bin):** $\frac{B \times (\texttt{prompt\_len} + \texttt{gen\_len}) \times 114,688}{M_{\text{KV}}}$
- **Idealized Capped KV Ratio (`12.08 GB` dec):** $\frac{\min(B, B_{\max,\text{ideal}}) \times (\texttt{prompt\_len} + \texttt{gen\_len}) \times 114,688}{12.08 \times 10^9}$
- **Analytical Traffic-Model HBM Estimate (`Traffic-Model Est. GB/s`):** $\frac{8.4 \text{ GB} + \min(B, B_{\max,\text{ideal}}) \times (\texttt{prompt\_len} + \texttt{gen\_len}/2) \times 114,688 \text{ B}}{\texttt{itl\_ms\_p50} / 1000}$ *(model-based estimate under assumed traffic formula, **not** hardware-measured bandwidth; see Section 5.3)*.

| Row | $B$ | $p$ | $g$ | `wall_clock_s` | `reported_tok_s` (Logged) | `Calc Total Tok/s` | **`Calc Gen Output Tok/s`** | $\frac{p+g}{g}$ | KV Demand (`GB` dec / `GiB` bin) | Unthrottled Ratio (`Dec` / `GiB`) | Idealized Capped Ratio (`Logged kvu`) | `preempted_seqs` (`Ideal` / `Logged`) | `ttft_ms_p50` | `itl_ms_p50` | `e2e_ms_p95` | Traffic-Model Est. (`GB/s`)\* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `1` | `512` | `256` | `10.94` | `70.2` | `70.20` | **`23.40`** | `3.0×` | `0.088` / `0.082` | `0.0073` / `0.0068` | `0.0073` (**`0.01`**) | `0` / `0` | `71.1` | `43.48` | `12863.7` | `194.9` |
| **2** | `2` | `512` | `256` | `11.61` | `132.3` | `132.30` | **`44.10`** | `3.0×` | `0.176` / `0.164` | `0.0146` / `0.0136` | `0.0146` (**`0.01`**) | `0` / `0` | `68.3` | `43.10` | `13045.7` | `198.3` |
| **3** | `4` | `512` | `256` | `11.77` | `261.0` | `261.00` | **`87.00`** | `3.0×` | `0.352` / `0.328` | `0.0292` / `0.0272` | `0.0292` (**`0.03`**) | `0` / `0` | `71.5` | `45.34` | `13409.5` | `191.7` |
| **4** | `8` | `512` | `256` | `12.40` | `495.4` | `495.48` | **`165.16`** | `3.0×` | `0.705` / `0.656` | `0.0583` / `0.0543` | `0.0583` (**`0.06`**) | `0` / `0` | `72.7` | `46.83` | `14137.1` | `191.9` |
| **5** | `16` | `512` | `256` | `13.91` | `883.2` | `883.39` | **`294.46`** | `3.0×` | `1.409` / `1.312` | `0.1167` / `0.1087` | `0.1167` (**`0.12`**) | `0` / `0` | `71.7` | `48.33` | `15592.3` | `198.1` |
| **6** | `32` | `512` | `256` | `16.50` | `1489.6` | `1489.45` | **`496.48`** | `3.0×` | `2.819` / `2.625` | `0.2333` / `0.2173` | `0.2333` (**`0.23`**) | `0` / `0` | `68.3` | `56.17` | `18502.8` | `191.4` |
| **7** | `64` | `512` | `256` | `21.68` | `2267.3` | `2267.16` | **`755.72`** | `3.0×` | `5.637` / `5.250` | `0.4667` / `0.4346` | `0.4667` (**`0.47`**) | `0` / `0` | `73.3` | `67.91` | `24323.7` | `192.9` |
| **8** | `4` | `3584` | `512` | `28.98` | `565.4` | `565.36` | **`70.67`** | `8.0×` | `1.879` / `1.750` | `0.1556` / `0.1449` | `0.1556` (**`0.16`**) | `0` / `0` | `483.2` | `51.33` | `32673.3` | `198.0` |
| **9** | `8` | `3584` | `512` | `36.30` | `902.6` | `902.70` | **`112.84`** | `8.0×` | `3.758` / `3.500` | `0.3111` / `0.2897` | `0.3111` (**`0.31`**) | `0` / `0` | `519.0` | `62.26` | `39982.9` | `191.5` |
| **10** | `16` | `3584` | `512` | `49.97` | `1311.4` | `1311.51` | **`163.94`** | `8.0×` | `7.516` / `7.000` | `0.6222` / `0.5795` | `0.6222` (**`0.62`**) | `0` / `0` | `498.3` | `77.20` | `54602.1` | `200.1` |
| **11** | `24` | `3584` | `512` | `61.16` | `1607.4` | `1607.33` | **`200.92`** | `8.0×` | `11.274` / `10.500` | `0.9333` / `0.8692` | `0.9333` (**`0.93`**) | `0` / `0` | `500.5` | `96.07` | `69221.3` | `197.5` |
| **12** | `32` | `3584` | `512` | `94.71` | `1384.0` | `1383.93` | **`172.99`** | `8.0×` | `15.032` / `14.000` | **`1.2444`** / `1.1589` | `0.9722` (**`0.97`**) | **`7` / `7`** | `636.9` | `101.79` | `97465.7` | `190.7` |
| **13** | `48` | `3584` | `512` | `151.41` | `1298.5` | `1298.51` | **`162.31`** | `8.0×` | `22.549` / `21.000` | **`1.8666`** / `1.7384` | `0.9722` (**`0.97`**) | **`23` / `23`** | `955.4` | `100.00` | `105427.5` | `194.1` |

*\*Note: `Traffic-Model Est. (GB/s)` is an analytical estimate derived from an assumed memory-traffic model, not a hardware counter measurement; see Section 5.3 for model assumptions and required telemetry.*

---

## 4. Investigation of the Throughput Anomaly in `REPORT_v0.md` Section 2

### 4.1 Total-Token Throughput vs. Generated-Output-Token Throughput at `batch_size = 16`

`REPORT_v0.md` Section 2 claims:
> *"From `bench_log.csv`: at batch 16, long prompts hit **1311 tok/s** vs only **883 tok/s** for short prompts. Longer prompts clearly give better GPU utilization. Recommendation: encourage clients to pack more context per request; throughput improves with prompt length."*

Reconciling `reported_tok_s` against `num_requests`, `prompt_len`, `gen_len`, and `wall_clock_s` establishes the exact arithmetic identity:

$$\texttt{reported\_tok\_s} = \frac{\texttt{num\_requests} \times (\texttt{prompt\_len} + \texttt{gen\_len})}{\texttt{wall\_clock\_s}} = \underbrace{\left(\frac{\texttt{num\_requests} \times \texttt{gen\_len}}{\texttt{wall\_clock\_s}}\right)}_{\text{Generated-Output-Token Throughput}} \times \underbrace{\left(1 + \frac{\texttt{prompt\_len}}{\texttt{gen\_len}}\right)}_{\text{Total-to-Output Token Multiplier}}$$

- **Why `1311.4` vs. `883.2 tok/s` is a numerator-mixing artifact:**
  1. In **Row 5 (`B=16, p=512, g=256`)**, total tokens (`12,288`) equal **$3.0\times$** generated output tokens (`4,096`). Over `wall_clock_s = 13.91 s`, `reported_tok_s = 883.2 tok/s` (`883.39` unrounded), consisting of `588.93 prompt tok/s` + **`294.46 generated output tok/s`**.
  2. In **Row 10 (`B=16, p=3584, g=512`)**, total tokens (`65,536`) equal **$8.0\times$** generated output tokens (`8,192`). Over `wall_clock_s = 49.97 s`, `reported_tok_s = 1311.4 tok/s` (`1311.51` unrounded), consisting of `1147.57 prompt tok/s` + **only `163.94 generated output tok/s`**.
  3. Because `reported_tok_s` adds parallel prefill tokens (`prompt_len`) to sequential autoregressive decode tokens (`gen_len`) in the numerator while dividing by `wall_clock_s` (which is `~96%–99%` decode time, since `ttft_ms_p50` is only `71.7 ms` out of `13.91 s` in Row 5 and `498.3 ms` out of `49.97 s` in Row 10), increasing the prompt-to-output ratio from `2:1` ($3.0\times$ total-to-output) to `7:1` ($8.0\times$ total-to-output) artificially inflates `reported_tok_s`.
  4. When measured by **Generated-Output-Token Throughput** ($\frac{\texttt{num\_requests} \times \texttt{gen\_len}}{\texttt{wall\_clock\_s}}$), long prompts at `batch_size = 16` generate **`163.94 gen tok/s` vs. `294.46 gen tok/s` for short prompts (`44.3%` lower output token rate, or `1.80×` slower)**, while median inter-token decode latency (`itl_ms_p50`) increases by **`+59.7%`** (`48.33 ms` $\rightarrow$ `77.20 ms`).

### 4.2 Which Numerator and Denominator Are Justified?

- **Numerator (`num_requests * gen_len` vs. `num_requests * (prompt_len + gen_len)`):**
  - **`num_requests * gen_len` (Generated Output Tokens)** is the justified numerator when comparing workloads with different prompt-to-output ratios, because user-facing generation rate and decode capacity depend on tokens generated.
  - **`num_requests * (prompt_len + gen_len)` (`reported_tok_s`)** is only comparable across runs that share the exact same `prompt_len` and `gen_len` (e.g., comparing `batch_size = 24, 32, 48` within `Rows 8–13`, where $\frac{p+g}{g} = 8.0\times$ is constant).
- **Denominator (`wall_clock_s` vs. Decode-Step `itl_ms_p50`):**
  - **`wall_clock_s` (End-to-End Batch Completion Time)** is the justified denominator for system capacity planning because the GPU is occupied for the entire `wall_clock_s` interval (capturing initial prefill, all decode steps, and any preemption/re-execution overhead).
  - Dividing by `itl_ms_p50` alone (e.g., nominal $\frac{B}{\texttt{itl\_ms\_p50}/1000}$) ignores prefill overhead and fails when `preempted_seqs > 0` (`Rows 12–13`), where `itl_ms_p50` stays near `~100–101.8 ms` for the active subset while `wall_clock_s` stretches from `61.16 s` to `94.71 s` and `151.41 s`.

---

## 5. Strict Separation of Observed Log Facts, Derived Calculations, and Hypotheses Requiring Telemetry

### 5.1 Tier 1 — Directly Observed Log Facts (`bench/bench_log.csv` & `bench/model_spec.md`)

1. **Short-Prompt Sweep (`Rows 1–7`, `p=512, g=256`):** Across `batch_size` $1 \rightarrow 64$, `preempted_seqs = 0` on all 7 rows, `kv_cache_util` rises from `0.01` to `0.47`, `reported_tok_s` rises monotonically from `70.2` to `2267.3`, `ttft_ms_p50` remains between `68.3 ms` and `73.3 ms`, and `itl_ms_p50` increases from `43.48 ms` ($B=1$) to `67.91 ms` ($B=64$).
2. **Long-Prompt Sweep Up to `B = 24` (`Rows 8–11`, `p=3584, g=512`):** From `batch_size` $4 \rightarrow 24$, `preempted_seqs = 0`, `kv_cache_util` rises from `0.16` ($B=4$) to `0.31` ($B=8$), `0.62` ($B=16$), and `0.93` ($B=24$); `reported_tok_s` peaks at **`1607.4`** at $B=24$; `ttft_ms_p50` is `483.2–519.0 ms`; and `itl_ms_p50` rises from `51.33 ms` ($B=4$) to `96.07 ms` ($B=24$).
3. **Long-Prompt Throughput Reversal & Preemption at `B = 32` and `B = 48` (`Rows 12–13`):**
   - Contrary to `REPORT_v0.md`'s projection that `batch_size = 48` should scale linearly to `~3200 tok/s`, `batch_size = 32` and `batch_size = 48` were already logged in `Rows 12–13`:
   - At **`batch_size = 32` (`Row 12`)**: `kv_cache_util` caps at **`0.97`**, **`preempted_seqs` jumps from `0` to `7`**, `wall_clock_s` increases from `61.16 s` to **`94.71 s`**, `ttft_ms_p50` increases to **`636.9 ms`**, `itl_ms_p50` is **`101.79 ms`**, `e2e_ms_p95` jumps from `69,221.3 ms` to **`97,465.7 ms`**, and `reported_tok_s` **drops to `1384.0`**.
   - At **`batch_size = 48` (`Row 13`)**: `kv_cache_util` remains at **`0.97`**, **`preempted_seqs` jumps to `23`**, `wall_clock_s` reaches **`151.41 s`**, `ttft_ms_p50` reaches **`955.4 ms`**, `itl_ms_p50` is **`100.00 ms`**, `e2e_ms_p95` reaches **`105,427.5 ms`**, and `reported_tok_s` **drops further to `1298.5`**.

### 5.2 Tier 2 — Calculations Derived from Supplied Data

1. **Exact Throughput Decomposition:**
   - Generated-output-token throughput (`num_requests * gen_len / wall_clock_s`) peaks at **`755.72 gen tok/s`** for short prompts ($B=64$) and **`200.92 gen tok/s`** for long prompts ($B=24$), then declines by **`-13.9%` to `172.99 gen tok/s` at $B=32$** and by **`-19.2%` to `162.31 gen tok/s` at $B=48$**.
2. **Idealized KV-Cache Demand & Capacity Over-Subscription:**
   - Using the supplied GQA parameters (`114,688 B/tok = 112.0 KiB/tok`) and memory budget (`24 GB * 0.92 - 8.40 GB - 1.60 GB = 12.08 GB`), unthrottled KV demand for $B$ full-length (`4096`-token) sequences is **`11.274 GB` (`93.33%` of `12.08 GB`) at $B=24$**, **`15.032 GB` (`124.44%` of `12.08 GB`) at $B=32$**, and **`22.549 GB` (`186.66%` of `12.08 GB`) at $B=48$**.
   - Under this idealized linear memory model, at most $B_{\max,\text{ideal}}(4096) = \lfloor 12,080,000,000 / 469,762,048 \rfloor = \mathbf{25 \text{ sequences}}$ (`11.744 GB` = `97.22%` $\rightarrow$ `0.97`) fit concurrently without exceeding `12.08 GB`, and the idealized excess sequence counts ($32 - 25 = 7$ and $48 - 25 = 23$) match the logged `preempted_seqs` (`7` and `23`).

### 5.3 Tier 3 — Hypotheses & Model-Based Estimates Requiring Additional Telemetry

1. **Analytical Traffic-Model HBM Bandwidth Estimate (`190.7–200.1 GB/s`) — Hypothesis of Decode Memory-Bandwidth Pressure:**
   - *Model-Based Calculation:* If we assume an idealized decode-step traffic model where every step reads (i) the full `8.40 GB` model weights once from HBM and (ii) the active KV cache at midpoint length $\min(B, 25) \times (p + g/2) \times 114,688\text{ B}$, dividing that byte volume by `itl_ms_p50 / 1000` yields **`190.7–200.1 GB/s`** across all 13 rows (`63.6%–66.7%` of the L4's `300 GB/s` peak specification).
   - *Why This Is a Hypothesis, Not Hardware-Measured Proof:* `bench_log.csv` does not include hardware performance counters. This simplified traffic model omits L2 cache hits, activation/workspace reads and writes, attention/MLP kernel launch overheads, tensor-core compute time, and exact dynamic token counts per step during preemption waves.
   - *Telemetry Required to Confirm:* NVIDIA Nsight Compute (`ncu`) or DCGM/NVML hardware profiling counters (`dram__bytes_read.sum.per_second`, `dram__bytes_write.sum.per_second`, `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) recorded during live decode steps.
2. **Preemption Policy (`Recompute` vs. `Swap`) and Queueing vs. Mid-Decode Eviction at $B=32, 48$:**
   - *Hypothesis:* When $B=32$ or $B=48$ requests are submitted simultaneously, KV-cache exhaustion triggers sequence preemption (`7` and `23` sequences), forcing evicted sequences to be re-prefilled (`recompute`) or swapped (`swap`) and executed in a subsequent wave, which explains both the jump in `wall_clock_s` (`61.16 s` $\rightarrow$ `94.71 s` / `151.41 s`) and the elevation in median `ttft_ms_p50` (`500.5 ms` $\rightarrow$ `636.9 ms` / `955.4 ms`).
   - *Telemetry Required to Confirm:* Engine scheduler logs reporting preemption mode (`recompute` vs. `swap`), per-request queue wait time vs. prefill execution time, and total preemption event counters.
3. **Prefix Caching, Chunked Prefill, and Block-Allocator Fragmentation:**
   - *Hypothesis:* Although all `num_requests` in a run are identical (`model_spec.md:L30`), `kv_cache_util` scales linearly with $B \times S$, which is consistent with `enable_prefix_caching` being disabled (or bypassed by the benchmark harness) and with negligible PagedAttention block fragmentation (`block_size` dividing `512`, `256`, `3584`, `4096`).
   - *Telemetry Required to Confirm:* Server launch flags (`--enable-prefix-caching`, `--block-size`, `--enable-chunked-prefill`) and block-allocator metrics (`num_total_gpu_blocks`, `num_free_gpu_blocks`).

---

## 6. Sensitivity Analyses Supported by the Supplied Data

### 6.1 Sensitivity to Sequence Length ($S$) and KV-Cache Precision (`fp16` vs. `fp8`/`int8`)

Using the idealized `12.08 GB` (`12,080,000,000` B) KV pool with $\text{KV}_{\text{tok, fp16}} = 114,688 \text{ B/tok}$ vs. quantized $\text{KV}_{\text{tok, fp8/int8}} = 57,344 \text{ B/tok}$:

| Sequence Length $S = p + g$ | `fp16` KV per Seq (`MB` dec / `MiB` bin) | Idealized $B_{\max}(S)$ (`fp16`) | `fp8` / `int8` KV per Seq (`MB` dec / `MiB` bin) | Idealized $B_{\max}(S)$ (`fp8` / `int8`) |
| :---: | :---: | :---: | :---: | :---: |
| **`512`** | `58.72 MB` / `56.00 MiB` | **`205`** | `29.36 MB` / `28.00 MiB` | **`411`** |
| **`768`** *(Short Bench)* | `88.08 MB` / `84.00 MiB` | **`137`** | `44.04 MB` / `42.00 MiB` | **`274`** |
| **`1024`** | `117.44 MB` / `112.00 MiB` | **`102`** | `58.72 MB` / `56.00 MiB` | **`205`** |
| **`1536`** | `176.16 MB` / `168.00 MiB` | **`68`** | `88.08 MB` / `84.00 MiB` | **`137`** |
| **`2048`** | `234.88 MB` / `224.00 MiB` | **`51`** | `117.44 MB` / `112.00 MiB` | **`102`** |
| **`3072`** | `352.32 MB` / `336.00 MiB` | **`34`** | `176.16 MB` / `168.00 MiB` | **`68`** |
| **`4096`** *(Long Bench)* | `469.76 MB` / `448.00 MiB` | **`25`** | `234.88 MB` / `224.00 MiB` | **`51`** |

### 6.2 Sensitivity to Non-KV Runtime Overhead (`model_spec.md` Assumes `~1.6 GB`)

Varying $M_{\text{overhead}} \in [1.0, 3.0]\text{ GB}$ in $M_{\text{KV}} = 22.08\text{ GB} - 8.40\text{ GB} - M_{\text{overhead}}$ shows how allocator/runtime overhead shifts idealized sequence capacity:

| Assumed Non-KV Overhead (`GB`) | Idealized KV Pool $M_{\text{KV}}$ (`GB` dec / `GiB` bin) | Idealized Max Tokens (`fp16`) | Idealized $B_{\max}(4096)$ (`fp16`) | Idealized $B_{\max}(768)$ (`fp16`) |
| :---: | :---: | :---: | :---: | :---: |
| `1.0 GB` | `12.68 GB` / `11.81 GiB` | `110,560` | `26` seqs | `143` seqs |
| `1.2 GB` | `12.48 GB` / `11.62 GiB` | `108,816` | `26` seqs | `141` seqs |
| **`1.6 GB` (Supplied Baseline)** | **`12.08 GB` / `11.25 GiB`** | **`105,329`** | **`25` seqs** | **`137` seqs** |
| `2.0 GB` | `11.68 GB` / `10.88 GiB` | `101,841` | `24` seqs | `132` seqs |
| `2.4 GB` | `11.28 GB` / `10.51 GiB` | `98,353` | `24` seqs | `128` seqs |
| `3.0 GB` | `10.68 GB` / `9.95 GiB` | `93,122` | `22` seqs | `121` seqs |

### 6.3 Cross-Part Sensitivity: Effect of Part A3 Multilingual Token Ratios on Idealized KV Concurrency

Applying the verified Part A3 **token-count-per-aligned-sentence ratios** (`a3_tokenizer_comparison.json`) to a base `768`-token English sequence (`512` prompt + `256` output) illustrates how tokenizer compression interacts with the idealized `12.08 GB` `fp16` KV pool (note: `FLM-4B-Instruct`'s actual `128k` tokenizer remains unmeasured; `o200k_base` and `cl100k_base` are shown as reference benchmarks):

| Language | `o200k_base` Token Ratio | Scaled $S$ (`o200k_base`) | Fits `4096` Limit? | Idealized $B_{\max}$ (`o200k_base`) | `cl100k_base` Token Ratio | Scaled $S$ (`cl100k_base`) | Fits `4096` Limit? | Idealized $B_{\max}$ (`cl100k_base`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **English (`eng`)** | `1.00×` | `768` tok | Yes | **`137` seqs** | `1.00×` | `768` tok | Yes | **`137` seqs** |
| **Hindi (`hin`)** | `1.57×` (`1.5719×`) | `1,207` tok | Yes | **`87` seqs (`-36.5%`)** | `4.77×` (`4.7666×`) | `3,661` tok | Yes | **`28` seqs (`-79.6%`)** |
| **Telugu (`tel`)** | `1.92×` (`1.9205×`) | `1,475` tok | Yes | **`71` seqs (`-48.2%`)** | `8.28×` (`8.2783×`) | `6,358` tok | **No (`>4096`)** | **`0` (Context Overflow)** |
| **Malayalam (`mal`)** | `1.95×` (`1.9542×`) | `1,501` tok | Yes | **`70` seqs (`-48.9%`)** | `8.93×` (`8.9346×`) | `6,862` tok | **No (`>4096`)** | **`0` (Context Overflow)** |
| **Kannada (`kan`)** | `1.95×` (`1.9549×`) | `1,501` tok | Yes | **`70` seqs (`-48.9%`)** | `8.84×` (`8.8428×`) | `6,791` tok | **No (`>4096`)** | **`0` (Context Overflow)** |
| **Tamil (`tam`)** | `1.98×` (`1.9765×`) | `1,518` tok | Yes | **`69` seqs (`-49.6%`)** | `7.64×` (`7.6406×`) | `5,868` tok | **No (`>4096`)** | **`0` (Context Overflow)** |
