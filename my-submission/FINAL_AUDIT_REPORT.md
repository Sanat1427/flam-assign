# Final Comprehensive Audit & Submission Report — AI Team Intern Assignment ("The Audit")

- **Canonical Submission Directory:** `my-submission/` (`37` verified files)
- **Source of Truth Audited Against:** `AI ASSIGNMENT 2026.pdf` (6 pages, 100 points total), `REPORT_v0.md`, `fertility.py`, `corpus_sample/{eng,hin}_sample.txt`, `bench/model_spec.md`, and `bench/bench_log.csv`
- **Audit Execution Date:** `2026-10-24` (after live re-execution of `audit_experiments.py`, `compare_tokenizers.py`, and `capacity_reconciliation.py`)
- **Starter-Kit Preservation Check:** **PASS** — All 6 original starter-kit files (`fertility.py` [3,063 B], `REPORT_v0.md` [1,443 B], `corpus_sample/eng_sample.txt` [458 B], `corpus_sample/hin_sample.txt` [774 B], `bench/bench_log.csv` [828 B], `bench/model_spec.md` [1,230 B]) retain their exact original byte lengths and timestamps (`2026-10-22 23:34:43`).

---

## A. Executive Verdict

### **Overall Readiness: `READY TO SUBMIT`**

1. **Resolution of All Source-of-Truth Conflicts (Parts A, B, and C):**
   - **Part B (`bench/model_spec.md` & `bench/bench_log.csv`):** Direct inspection of `bench/model_spec.md` confirms the exact supplied parameters are **`layers = 28`**, **`attention heads (Q) = 24`**, **`KV heads (GQA) = 8`** ($3:1$ GQA ratio), **`head_dim = 128`**, **`parameters = 4.2 B`** in `fp16` (**`8.40 GB` weights**), **`gpu_memory_utilization = 0.92`** on `1× NVIDIA L4 (24 GB)` (**`22.08 GB` usable**), and **`non-KV runtime overhead = assume ~1.6 GB`**. These yield **`114,688 B/token`** (`112.0 KiB/token`), **`469,762,048 B/seq`** (`469.76 MB` decimal / `448.00 MiB` binary) at $S=4096$, an idealized KV pool of **`12.08 GB`** (`12,080,000,000 B` $\approx 11.2504\text{ GiB}$, or `105,329` tokens), and an idealized maximum concurrency of **$B_{\max,\text{ideal}}(4096) = 25$ sequences** (`137` sequences at $S=768$). Every calculation in `capacity_reconciliation.py`, `partB_reconciliation.json`, `partB_calculation_output.txt`, `partB_report.md`, `partC/memo.md`, and this report now matches `bench/model_spec.md` and `bench/bench_log.csv` with zero discrepancies.
   - **Part A (`fertility.py`, `corpus_sample/`, and `FLORES-200` `N = 1,012`):** Live re-execution of `audit_experiments.py` and `compare_tokenizers.py` confirms that the canonical `a3_tokenizer_comparison.json`, `a3_analysis.md`, and `partA/memo.md` were already 100% consistent with the processed corpus (`my-submission/partA/corpus/processed/`). The earlier discrepancy in the draft audit summary (`1.928` vs. **`1.6473`** `o200k_base` Hindi micro tokens/word) was caused solely by a corrupted context-truncation summary that had been copied into the first draft of `FINAL_AUDIT_REPORT.md`—**not** by any change to the corpus, scripts, or `a3_tokenizer_comparison.json`.
   - **Part B3 Throughput Derivations:** `partB_report.md` (§0.B3 & §4) and Section C.2 below explicitly distinguish (i) **observed end-to-end generated-output-token throughput** (`200.92 gen tok/s` = `8.37 gen tok/s/request` at `Row 11`, derived both from $\frac{\texttt{num\_requests} \times \texttt{gen\_len}}{\texttt{wall\_clock\_s}} = \frac{24 \times 512}{61.16\text{ s}}$ and from $\texttt{reported\_tok\_s} \times \frac{g}{p+g} = \frac{1607.4}{8}$) from (ii) **percentile-based decode/step-rate estimates** (`202.57 gen tok/s` net decode-only estimate excluding `ttft_ms_p50 = 500.5 ms`, `249.82 gen tok/s` instantaneous step rate at `itl_ms_p50 = 96.07 ms`, and `247.30 gen tok/s` synthetic p50 request rate).
   - **Part C Reviewer Budget & Epistemic Labels:** `partC/memo.md` (§1.2 & §5.2) explicitly states that the `10` overlap re-ratings are **additional** to the `80` unique golden prompts (`80 unique + 10 additional overlap = 90 pairs/language`), totaling $90 \times 3.0\text{ min} = 4.5\text{ h}$ active rating $+ 0.5\text{ h}$ calibration = **`5.0 hours/week` for the Hindi reviewer** and **`5.0 hours/week` for the Kannada reviewer** (`10.0 hours/week` combined in Weeks 1 & 3; `2.0 hours/week` per language = `4.0 hours/week` combined in Week 2). All A100 VRAM (`~18–26 GB`) and training-time (`~3–6 GPU-hours/run`) figures are explicitly labeled as **unexecuted analytical estimates**.

---

## B. Requirement-by-Requirement Rubric Matrix

| Rubric Component | Weight | Assignment PDF Requirement (`AI ASSIGNMENT 2026.pdf`) | Deliverable File(s) | Status | Verification Evidence |
| :--- | :---: | :--- | :--- | :---: | :--- |
| **Gate: Integrity & Attribution** | Gate | Preserve starter kit; no fabricated numbers; honest `AI_USAGE.md`; reproducible scripts | `my-submission/AI_USAGE.md`, all `.py` scripts | **PASS** | All 6 starter-kit files untouched; all 3 Python scripts re-executed (`0` exit code); `AI_USAGE.md` documents 5 specific AI failure modes caught and corrected. |
| **Part A1: Corpus Assembly** | 10 pts | $\ge 200$ parallel sentences across `eng, hin, tam, tel, kan, mal` ($\ge 1,200$ total); provenance, license, preprocessing, limitations | `partA/build_corpus.py`, `partA/corpus_documentation.md`, `partA/corpus/raw/`, `partA/corpus/processed/` | **PASS** | `1,012` aligned `FLORES-200` `devtest` sentences per language (`6,072` total — $5.06\times$ minimum); CC-BY-SA-4.0 provenance, UTF-8, `NFC` normalized (`93` `hin` + `14` Dravidian raw lines fixed), whitespace-collapsed, cased. |
| **Part A2: Audit of `fertility.py`** | 15 pts | Identify all flaws in `fertility.py` & `corpus_sample/`; quantify each flaw individually (`E1–E6`) | `partA/audit_experiments.py`, `partA/e1_e6_raw_output.txt`, `partA/audit_results.md` | **PASS** | Reproduces `REPORT_v0.md` baseline (`eng = 1.27` [`1.265206`], `hin = 7.45` [`7.448452`], ratio `5.89x` [`5.887148`], `tok/char` `0.226` vs. `1.579`, ratio `7.00x`). Quantifies `E1` (`line.split(" ")` vs. `line.split()`), `E2` (`line.lower()` vs. cased), `E3` (macro vs. micro), `E4a/b` (codepoints vs. UTF-8 bytes vs. `\X` graphemes; refutes Finding #3 since `hin` has `4.68` vs. `eng` `5.67` codepoints/word), `E5` (`0/10` aligned lines in `corpus_sample/`), and `E6a/b/c` (harmless/correct choices). |
| **Part A3: Multi-Tokenizer & Multi-Metric Comparison** | 15 pts | Compare $\ge 2$ tokenizers across all 6 languages using $\ge 2$ denominators; macro vs. micro averages; statistical spread | `partA/compare_tokenizers.py`, `partA/a3_tokenizer_comparison.json`, `partA/a3_analysis.md` | **PASS** | Compares 3 `tiktoken` encodings (`gpt2` [`50,257`], `cl100k_base` [`100,277`], `o200k_base` [`200,019`]) across 5 denominators (whitespace words, `NFC` codepoints, UTF-8 bytes, `\X` graphemes, aligned sentences) + macro/micro means. Explicitly discloses `tiktoken`-only scope (`transformers` not installed). |
| **Part A4: Executive Memo** | 10 pts | $\le 1$-page memo answering whether to split Indic into a separate serving tier | `partA/memo.md`, `partA/a4_recommendation_memo.md` | **PASS** | `41` lines (`6,573` bytes); separates measured findings from recommendations; labels $1.57\times\text{–}1.98\times$ as token-count-per-aligned-sentence ratios (not cost multipliers); recommends benchmarking `FLM-4B-Instruct`'s actual `128k` tokenizer before routing changes. |
| **Part B1: KV-Cache Footprint & Capacity** | 7 pts | From `model_spec.md` alone, compute bytes/token and max concurrent `4096`-token sequences; check against `bench_log.csv` | `partB/partB_report.md` (§0.B1, §2), `partB/capacity_reconciliation.py`, `partB/partB_reconciliation.json` | **PASS** | $2 \times 28 \times 8 \times 128 \times 2 = \mathbf{114,688\text{ B/tok}}$ (`112.0 KiB/tok`); `469,762,048 B/seq` (`469.76 MB` dec / `448.00 MiB` bin); pool $= 24.0 \times 0.92 - 8.40 - 1.60 = \mathbf{12.08\text{ GB}}$ (`105,329` max tokens); $\lfloor 12.08\times 10^9 / 469,762,048 \rfloor = \mathbf{25\text{ sequences}}$. Matches `Row 11` (`B=24, util=0.93, preempt=0`) vs. `Row 12` (`B=32, util=0.97, preempt=7`) & `Row 13` (`B=48, util=0.97, preempt=23`). |
| **Part B2: Throughput Anomaly & Fix** | 6 pts | Identify `prompt_len=3584` throughput cliff, explain mechanism using rows/columns, propose 1 fix with quantitative effect | `partB/partB_report.md` (§0.B2, §3, §5, §6) | **PASS** | Peak at `Row 11` (`1607.4 tok/s`, `200.92 gen tok/s`), cliff at `Row 12` (`1384.0 tok/s`, `-13.9%`, `7` preemptions) and `Row 13` (`1298.5 tok/s`, `-19.2%`, `23` preemptions). Proposes `fp8` KV cache ($57,344\text{ B/tok} \rightarrow B_{\max}(4096)=51$ seqs, `0` preemptions, `kv_cache_util = 0.62` at $B=32$ and `0.93` at $B=48$) or capping `max_num_seqs=24`. |
| **Part B3: Misread Column & Two-Way Goodput** | 4 pts | Explain misread column in `REPORT_v0.md`; derive `Row 11` (`B=24`) goodput in **two independent ways**; state honest correction | `partB/partB_report.md` (§0.B3, §4) | **PASS** | `reported_tok_s` includes prompt tokens ($3.0\times$ output rate on `512/256` vs. $8.0\times$ on `3584/512`). **Way 1:** $\frac{24 \times 512}{61.16\text{ s}} = \mathbf{200.92\text{ gen tok/s}}$ (`8.37 gen tok/s/req`). **Way 2:** $1607.4 \times \frac{512}{4096} = \frac{1607.4}{8} = \mathbf{200.925\text{ gen tok/s}}$. Separately explains percentile-based decode estimates (`202.57 gen tok/s` net decode-only; `249.82 gen tok/s` p50 step rate). |
| **Part B4: Single Serving Counter** | 3 pts | One paragraph naming the single serving metric/counter to confirm B2 and expected values | `partB/partB_report.md` (§0.B4, §5.3) | **PASS** | Names `vllm:num_preemptions_total` (paired with `prefill_tokens_total` & `num_total_gpu_blocks`), predicting `0` preemptions / `86,016` prefill tokens at $B=24$, $\ge 7$ preemptions / $\ge 139,776$ prefill tokens at $B=32$, and $\ge 23$ preemptions / $\ge 254,464$ prefill tokens at $B=48$. |
| **Part C: Engineering Recommendation Memo** | 15 pts | Compare SFT, $\le 1\text{B}$ rewriter, and prompt engineering under constraints; label assumptions, arithmetic, threshold, kill criterion, Day-1 experiment | `partC/memo.md` (§1–§6) | **PASS** | Recommends staged path (`Option 3` Day-1 primary baseline + gated Week-2 `Option 1` LoRA SFT; rejects `Option 2` rewriter). Explicitly labels all 5 PDF elements in §1.2, scopes Week-3 launch to evaluated `hin` + `kan`, and budgets `5.0 h/wk/lang` (`80 unique + 10 additional overlap = 90 pairs/lang`). |
| **Lab Notebook & Process** | 15 pts | Chronological `NOTEBOOK.md` (including dead ends) + live defense readiness | `my-submission/NOTEBOOK.md` | **PASS** | 14 chronological entries (`Entries 1–14`) recording actual commands, outputs, dead ends, and audit corrections. |

---

## C. Part-by-Part Findings & Verified Recalculation Tables

### C.1 Part A — Tokenizer Audit (`fertility.py`, `E1–E6`, and `FLORES-200` `N = 1,012`)

#### 1. Root Cause of Earlier Summary Discrepancy (Requirement #2)
- **What Disagreed:** The previous draft of `FINAL_AUDIT_REPORT.md` listed Hindi `o200k_base` micro fertility as `1.9283` (`45,876 / 23,791`) instead of **`1.6473` (`42,241 / 25,643`)**, and described `fertility.py` as using `re.findall(r"\w+", ...)` on a 5-line sample instead of `line.split(" ")` on the 10-line `corpus_sample/`.
- **Exact Cause Identified via File & Code Inspection:** Those numbers came from an automated `<CONTEXT_SUMMARY>` block generated when the conversation context was compacted, **not** from the workspace files. Re-executing `my-submission/partA/audit_experiments.py` and `my-submission/partA/compare_tokenizers.py` against `fertility.py`, `corpus_sample/{eng,hin}_sample.txt`, and `my-submission/partA/corpus/processed/{eng,hin,tam,tel,kan,mal}.txt` confirmed that `my-submission/partA/e1_e6_raw_output.txt`, `my-submission/partA/audit_results.md`, `my-submission/partA/a3_tokenizer_comparison.json`, `my-submission/partA/a3_analysis.md`, and `my-submission/partA/memo.md` were **already 100% accurate and mutually consistent**.

#### 2. Controlled Single-Variable Ablation of `fertility.py` on `corpus_sample/` (`audit_experiments.py` / `audit_results.md`)

| Experiment ID | Single Variable Changed vs. `fertility.py` (`gpt2`, `10` lines/file) | `eng_sample.txt` Measured Result | `hin_sample.txt` Measured Result | Verified Impact on `REPORT_v0.md` Headline (`5.89x` fert, `7.00x` `tok/char`) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (`Base`)** | Unmodified `fertility.py` (`line.lower()`, `line.split(" ")`, `len(line)`, macro-average) | `99` tok, `79` words, `448` chars<br>Fert: **`1.265206` (`1.27`)**<br>`tok/char`: **`0.225636` (`0.226`)** | `459` tok, `62` words, `290` chars<br>Fert: **`7.448452` (`7.45`)**<br>`tok/char`: **`1.579108` (`1.579`)** | Reproduces `REPORT_v0.md` exactly: fertility ratio $= 7.448452 / 1.265206 = \mathbf{5.887148\times}$ (`5.89x`); `tok/char` ratio $= \mathbf{6.998478\times}$ (`7.00x`). |
| **`E1` (Word Split)** | `line.split(" ")` $\rightarrow$ `line.split()` | `78` words (`L07` `"books  in"` had `""`)<br>Fert: **`1.283063` (`1.28`, `+1.41%`)** | `61` words (`L10` `"किताबें  अलमारी"` had `""`)<br>Fert: **`7.598452` (`7.60`, `+2.01%`)** | `split(" ")` counts empty strings on double spaces, understating fertility; fixing to `split()` shifts `hin/eng` ratio to **`5.922121×` (`5.92x`)**. (On raw `FLORES-200`, `split(" ")` injects `+477` phantom words in `kan` and `+274` in `tel`.) |
| **`E2` (Casing)** | Remove `line = line.lower()` (preserve case) | `96` tokens (`-3` on `Thursday`, `NASA`, `GPU`)<br>Fert: **`1.229329` (`1.23`, `-2.84%`)** | `459` tokens (`0` change; Devanagari is unicameral)<br>Fert: **`7.448452` (`7.45`, `0.00%`)** | `.lower()` breaks cased `gpt2` merges in English (`99` $\rightarrow$ `96` tok), artificially inflating `eng` fertility; preserving case raises `hin/eng` ratio to **`6.058958×` (`6.06x`, `+2.92%`)**. |
| **`E3` (Aggregation)** | Macro-average (`mean(T_i/W_i)`) $\rightarrow$ Micro-average (`sum(T_i)/sum(W_i)`) | Micro Fert: `99 / 79` = **`1.253165` (`1.25`)**<br>Micro `tok/char`: **`0.220982` (`0.221`)** | Micro Fert: `459 / 62` = **`7.403226` (`7.40`)**<br>Micro `tok/char`: **`1.582759` (`1.583`)** | Macro-averaging weights a 4-word line (`hin` L08) equally with a 12-word line (`eng` L03); micro-averaging shifts fertility ratio to **`5.907625×` (`5.91x`)** and `tok/char` ratio to **`7.162382×` (`7.16x`)**. |
| **`E4a` (UTF-8 Bytes)** | `len(line)` (codepoints) $\rightarrow$ `len(line.encode("utf-8"))` | `448` UTF-8 bytes (`1.00` B/cp)<br>Macro `tok/B`: **`0.225636` (`0.226`)** | `764` UTF-8 bytes (`2.63` B/cp)<br>Macro `tok/B`: **`0.598992` (`0.599`)** | `hin/eng` ratio drops from **`7.00x` (`tok/cp`) to `2.65x` (`2.654683x` `tok/byte`)**. Also **refutes `REPORT_v0.md` Finding #3**: Hindi has **fewer** Unicode codepoints/word (`290/62 = 4.68`) than English (`448/79 = 5.67`, `-17.5%`)! |
| **`E4b` (Graphemes)** | `len(line)` $\rightarrow$ Extended Grapheme Clusters `len(regex.findall(r"\X", line))` | `448` graphemes<br>Macro `tok/G`: **`0.225636` (`0.226`)** | `188` graphemes (*aksharas*)<br>Macro `tok/G`: **`2.449732` (`2.450`)** | `hin/eng` ratio jumps to **`10.86x` (`10.857013x` `tok/grapheme`)** because `290` Hindi codepoints form `188` orthographic *aksharas*. |
| **`E5` (Alignment)** | All 10 non-parallel lines $\rightarrow$ 5 loosely matched topical pairs (`eng` L3,4,5,7,8 $\leftrightarrow$ `hin` L3,7,6,10,4) | `47` tokens across 5 lines<br>Macro Fert: **`1.228571` (`1.23`)** | `242` tokens across 5 lines<br>Macro Fert: **`7.585714` (`7.59`)** | `0/10` line indices align in `corpus_sample/`. Restricting to the 5 loosely matched pairs shifts macro fertility ratio to **`6.174419×` (`6.17x`)** and total token ratio from `459/99 = 4.64x` to `242/47 = 5.15x`. |
| **`E6a/b/c` (Controls)** | `random.seed(1337)`, `NFC` on sample, `add_special_tokens=False` | $\Delta = 0.000000$ (`0.00%`) | $\Delta = 0.000000$ (`0.00%`) | Confirmed non-bugs on `corpus_sample/` (`NFC` is good practice and fixes `107` non-NFC lines on raw `FLORES-200`; `add_special_tokens=False` prevents per-line BOS/EOS framing distortion). |

#### 3. Verified `FLORES-200` (`N = 1,012` Aligned Sentences/Language) Multi-Tokenizer Table (`a3_tokenizer_comparison.json` & `a3_analysis.md`)

| Language (`ISO 639-3`) | `.split()` Words | `NFC` Codepoints | UTF-8 Bytes | `\X` Graphemes | `gpt2` (`50,257` vocab)<br>`Total Tok` \| `Micro Tok/Wd` (`vs eng`) \| `Tok/Sent` (`vs eng`) | `cl100k_base` (`100,277` vocab)<br>`Total Tok` \| `Micro Tok/Wd` (`vs eng`) \| `Tok/Sent` (`vs eng`) | `o200k_base` (`200,019` vocab)<br>`Total Tok` \| `Micro Tok/Wd` (`vs eng`) \| `Tok/Sent` (`vs eng`) | `%` Token Reduction (`gpt2` $\rightarrow$ `o200k_base`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **English (`eng`)** | `21,901` | `131,966` | `132,096` | `131,966` | `27,044` \| `1.2348` (`1.00×`) \| `26.72` (`1.00×`) | `27,182` \| `1.2411` (`1.00×`) \| `26.86` (`1.00×`) | `26,873` \| `1.2270` (`1.00×`) \| `26.55` (**`1.00×`**) | `-0.63%` (`1.01×` less) |
| **Hindi (`hin`)** | `25,643` | `131,174` | `337,433` | `85,951` | `200,682` \| `7.8260` (`6.34×`) \| `198.30` (`7.42×`) | `129,567` \| `5.0527` (`4.07×`) \| `128.03` (`4.77×`) | `42,241` \| **`1.6473` (`1.34×`)** \| `41.74` (**`1.57×`**) | **`-78.95%` (`4.75×` less)** |
| **Tamil (`tam`)** | `16,775` | `154,031` | `421,535` | `99,624` | `420,071` \| `25.0415` (`20.28×`) \| `415.09` (`15.53×`) | `207,687` \| `12.3807` (`9.98×`) \| `205.22` (`7.64×`) | `53,114` \| **`3.1663` (`2.58×`)** \| `52.48` (**`1.98×`**) | **`-87.36%` (`7.91×` less)** |
| **Telugu (`tel`)** | `16,938` | `132,226` | `353,422` | `76,328` | `350,509` \| `20.6936` (`16.76×`) \| `346.35` (`12.96×`) | `225,022` \| `13.2850` (`10.70×`) \| `222.35` (`8.28×`) | `51,610` \| **`3.0470` (`2.48×`)** \| `51.00` (**`1.92×`**) | **`-85.28%` (`6.79×` less)** |
| **Kannada (`kan`)** | `16,100` | `137,650` | `374,964` | `89,994` | `366,989` \| `22.7943` (`18.46×`) \| `362.64` (`13.57×`) | `240,364` \| `14.9294` (`12.03×`) \| `237.51` (`8.84×`) | `52,534` \| **`3.2630` (`2.66×`)** \| `51.91` (**`1.95×`**) | **`-85.69%` (`6.99×` less)** |
| **Malayalam (`mal`)** | `14,930` | `149,261` | `411,670` | `79,357` | `409,912` \| `27.4556` (`22.23×`) \| `405.05` (`15.16×`) | `242,859` \| `16.2665` (`13.11×`) \| `239.98` (`8.93×`) | `52,515` \| **`3.5174` (`2.87×`)** \| `51.89` (**`1.95×`**) | **`-87.19%` (`7.81×` less)** |

---

### C.2 Part B — Capacity Reconciliation (`bench/model_spec.md` & `bench/bench_log.csv`)

#### 1. Verification of Source Parameters from `bench/model_spec.md` (Requirement #1)

| Parameter | Exact Value in `bench/model_spec.md` | Formula / Intermediate Arithmetic | Verified Result in `partB_reconciliation.json` & `partB_report.md` |
| :--- | :--- | :--- | :--- |
| **Model & Layers ($L$)** | `FLM-4B-Instruct (dense)`, `parameters = 4.2 B` (`L7`), **`layers = 28` (`L8`)** | $L = 28$ transformer layers | **`28` layers** |
| **Attention Heads ($n_q, n_{kv}, d_h$)** | **`attention heads (Q) = 24` (`L10`)**, **`KV heads (GQA) = 8` (`L11`)**, **`head_dim = 128` (`L12`)** | GQA ratio $= 24 / 8 = 3:1$ ($24 \times 128 = 3072 = d_{\text{model}}$) | **$n_q = 24$, $n_{kv} = 8$, $d_h = 128$** |
| **KV Bytes per Token (`B1a`)** | `KV cache precision = fp16` (`2` B/element, `L15`) | $2_{\text{(K,V)}} \times 28 \times 8 \times 128 \times 2\text{ B}$ (MHA pitfall with $n_q=24$ gives $344,064\text{ B} = 336\text{ KiB}$, a $3\times$ overestimate) | **`114,688 B/token`** (`112.0 KiB/tok` = `0.114688 MB/tok` dec = `0.109375 MiB/tok` bin) |
| **Weight Memory & Non-KV Overhead** | `4.2 B` params in `fp16` (`L7, L14`); **`non-KV runtime overhead = assume ~1.6 GB` (`L26`)** | $M_{\text{weights}} = 4.2 \times 10^9 \times 2 = \mathbf{8.40\text{ GB}}$; $M_{\text{overhead}} = \mathbf{1.60\text{ GB}}$ | **`8.40 GB` weights**, **`1.60 GB` non-KV overhead** |
| **GPU Budget & KV Pool Size** | `1× NVIDIA L4 (24 GB)` (`L21`), `gpu_memory_utilization = 0.92` (`L25`) | $M_{\text{usable}} = 24.0 \times 0.92 = 22.08\text{ GB}$; $M_{\text{KV}} = 22.08 - 8.40 - 1.60 = \mathbf{12.08\text{ GB}}$ | **`12.08 GB`** (`12,080,000,000 B` decimal $\approx \mathbf{11.2504\text{ GiB}}$ binary; **`105,329` max tokens**) |
| **Per-Sequence KV & Max Concurrency (`B1b`)** | `max_model_len = 4096` (`L24`); short $S=768$ (`512+256`), long $S=4096$ (`3584+512`) | $\text{KV}(4096) = 4096 \times 114,688 = \mathbf{469,762,048\text{ B}}$ (`469.76 MB` dec / `448.00 MiB` bin); $B_{\max,\text{ideal}}(4096) = \lfloor 12,080,000,000 / 469,762,048 \rfloor = \lfloor 25.715 \rfloor$ | **`25` sequences at $S=4096$** (`137` sequences at $S=768$, `88,080,384 B/seq`) |
| **Reconciliation with `bench_log.csv` (`B1b, B2`)** | `Row 11` ($B=24$): `kv_cache_util = 0.93`, `preempted_seqs = 0`<br>`Row 12` ($B=32$): `kv_cache_util = 0.97`, `preempted_seqs = 7`<br>`Row 13` ($B=48$): `kv_cache_util = 0.97`, `preempted_seqs = 23` | At $B=24$: $24 / 25.715 = \mathbf{0.9333} \rightarrow 0.93$, `0` preemptions.<br>At $B=25$ (pool cap): $25 / 25.715 = \mathbf{0.9722} \rightarrow \mathbf{0.97}$ (matches logged `0.97` ceiling at $B=32, 48$!).<br>Excess sequences: $32 - 25 = \mathbf{7}$ (`Row 12`) and $48 - 25 = \mathbf{23}$ (`Row 13`). | **Exact match across all 13 rows** (`kv_util_matches_rounded_decimal == true` for all 13 rows). |

#### 2. Audit of the Part B3 Throughput Derivations for `Row 11` (`batch_size = 24, prompt_len = 3584, gen_len = 512`) (Requirement #3)

Directly from `bench/bench_log.csv` Line 12 (`Row 11`):
- `batch_size = 24`, `prompt_len = 3584`, `gen_len = 512`, `num_requests = 24`, `wall_clock_s = 61.16`, `reported_tok_s = 1607.4`, `ttft_ms_p50 = 500.5`, `itl_ms_p50 = 96.07`, `e2e_ms_p95 = 69221.3`, `preempted_seqs = 0`, `kv_cache_util = 0.93`.

| Metric Name | Exact Formula & Inputs Used | Calculated Value | What It Physically Measures & Underlying Assumptions |
| :--- | :--- | :---: | :--- |
| **1. Observed End-to-End Generated-Token Goodput (Way 1: Direct from Request/Token Counts & Wall Clock)** | $$\frac{\texttt{num\_requests} \times \texttt{gen\_len}}{\texttt{wall\_clock\_s}} = \frac{24 \times 512}{61.16\text{ s}} = \frac{12,288}{61.16\text{ s}}$$ | **`200.92 gen tok/s`** (`200.9156` unrounded) | **Exact observed end-to-end system output goodput** over the full batch execution window (`61.16 s`), including initial prompt prefill and all 512 decode steps across all 24 completed requests. |
| **2. Observed End-to-End Generated-Token Goodput (Way 2: Algebraic Decomposition of `reported_tok_s`)** | $$\texttt{reported\_tok\_s} \times \frac{\texttt{gen\_len}}{\texttt{prompt\_len} + \texttt{gen\_len}} = 1607.4 \times \frac{512}{3584 + 512} = \frac{1607.4}{8}$$ | **`200.925 gen tok/s`** (`200.92` rounded) | **Exact observed end-to-end system output goodput** recovered from the harness's total-token counter (`reported_tok_s`) by removing the $8.0\times$ total-to-output token multiplier ($\frac{3584+512}{512} = 8.0$). Differs by `0.009 tok/s` from Way 1 solely because `1607.4` is rounded to 1 decimal place from $98,304 / 61.16 = 1607.325$. |
| **3. Observed Per-User (Per-Request) End-to-End Output Throughput** | $$\frac{\text{Goodput}_{\text{e2e}}}{\texttt{num\_requests}} = \frac{512\text{ gen tok}}{61.16\text{ s}}$$ | **`8.37 gen tok/s/req`** (`8.3715` unrounded) | Average end-to-end output token rate experienced per concurrent request over the `61.16 s` run. |
| **4. Derived Decode-Only Throughput Estimate (Excluding Median TTFT)** | $$\frac{\texttt{num\_requests} \times \texttt{gen\_len}}{\texttt{wall\_clock\_s} - (\texttt{ttft\_ms\_p50}/1000)} = \frac{12,288}{61.16 - 0.5005\text{ s}}$$ | **`202.57 gen tok/s`** (`8.44 tok/s/req`) | **Decode-phase-only estimate** that subtracts median time-to-first-token (`0.5005 s`) from `wall_clock_s` (`61.16 s`). *Assumption:* Assumes all 24 simultaneous requests complete prefill together in `~0.5005 s`. Because `ttft_ms_p50` is only `0.82%` of `wall_clock_s`, decode-only rate is within `+0.82%` of end-to-end goodput. |
| **5. Instantaneous Decode Step-Rate Estimate from Median ITL (`itl_ms_p50`)** | $$\frac{\texttt{batch\_size}}{\texttt{itl\_ms\_p50}/1000} = \frac{24}{0.09607\text{ s}}$$ *(or synthetic p50 rate $\frac{12,288}{0.5005 + 512 \times 0.09607} = \mathbf{247.30\text{ gen tok/s}}$)* | **`249.82 gen tok/s`** (`10.41 tok/s/req`) | **Instantaneous step rate at the median (`p50`) decode step**, **NOT** observed end-to-end throughput. Per `model_spec.md:L38`, `itl_ms_p50` (`96.07 ms`) is the **median** inter-token latency across decode steps, whereas `wall_clock_s` (`61.16 s`) reflects the **mean** step duration (`~118.7 ms/step`) and tail completion across all 24 requests (logged `e2e_ms_p95 = 69,221.3 ms`). |

---

### C.3 Part C — Verification of Reviewer Budget & Training Estimates (`partC/memo.md`, Requirement #4)

1. **Explicit Overlap & Per-Reviewer Weekly Hour Calculation (`partC/memo.md` §1.2 & §5.2):**
   - `partC/memo.md` explicitly states that in Week 1 (Round 1) and Week 3 (Round 3), each language's evaluation set consists of **`80` unique golden test prompts plus `10` *additional* blind duplicate overlap prompts = `90` total pairwise comparisons per language**.
   - At **`3.0 minutes per pairwise comparison`** (`20` comparisons/hour):
     - **Hindi Reviewer Weekly Hours:**
       - **Week 1:** $90\text{ pairs} \times 3.0\text{ min} = 270\text{ min} = \mathbf{4.5\text{ hours}}$ active rating (`4.0 h` for `80` unique prompts + `0.5 h` for `10` additional overlap prompts) $+ \mathbf{0.5\text{ hours}}$ rubric calibration = **`5.0 hours`** ($\le 10.0\text{ h/week}$).
       - **Week 2:** $40\text{ unique dev pairs} \times 3.0\text{ min} = 120\text{ min} = \mathbf{2.0\text{ hours}}$ ($\le 10.0\text{ h/week}$).
       - **Week 3:** $90\text{ pairs} \times 3.0\text{ min} = \mathbf{4.5\text{ hours}}$ active rating $+ \mathbf{0.5\text{ hours}}$ critical-defect sign-off = **`5.0 hours`** ($\le 10.0\text{ h/week}$).
     - **Kannada Reviewer Weekly Hours:**
       - **Week 1:** `90` pairs $\times 3.0\text{ min} = \mathbf{4.5\text{ h}} + \mathbf{0.5\text{ h}}$ calibration = **`5.0 hours`** ($\le 10.0\text{ h/week}$).
       - **Week 2:** `40` dev pairs $\times 3.0\text{ min} = \mathbf{2.0\text{ hours}}$ ($\le 10.0\text{ h/week}$).
       - **Week 3:** `90` pairs $\times 3.0\text{ min} = \mathbf{4.5\text{ h}} + \mathbf{0.5\text{ h}}$ sign-off = **`5.0 hours`** ($\le 10.0\text{ h/week}$).
     - **Combined Weekly Totals (Hindi + Kannada):** **`10.0 hours` (Week 1)**, **`4.0 hours` (Week 2)**, **`10.0 hours` (Week 3)** = **`24.0 hours` total**. Thus, neither individual reviewer nor the combined two-language total ever exceeds `10.0 hours/week`!
2. **Epistemic Labeling of Unmeasured A100 Training Figures & Proposed Experiments:**
   - All A100 training figures (`~18–26 GB` VRAM, `~3–6 GPU-hours/run`, `15,000–30,000` target examples) are explicitly labeled in `partC/memo.md` (§1.2, §2, §4.2) as **Unexecuted Analytical Estimates / Proposed Targets**, and both Option 3 (Concise Prompting) and Option 1 (LoRA SFT) are presented strictly as **proposed engineering plans with pre-registered acceptance gates**, never as completed experiments.

---

## D. Summary of What Was Verified, What Changed, and What Remains Unresolved

1. **What Was Verified from Original Sources & Live Code Execution:**
   - Inspected `bench/model_spec.md`, `bench/bench_log.csv`, `fertility.py`, `REPORT_v0.md`, and `corpus_sample/{eng,hin}_sample.txt` directly.
   - Re-executed `my-submission/partA/audit_experiments.py`, `my-submission/partA/compare_tokenizers.py`, and `my-submission/partB/capacity_reconciliation.py` in `.venv-x64` (`Python 3.12.10`, `tiktoken 0.14.0`, `regex 2026.9.29`).
   - Verified that `my-submission/partA/{a3_tokenizer_comparison.json, a3_analysis.md, audit_results.md, e1_e6_raw_output.txt, memo.md}`, `my-submission/partB/{capacity_reconciliation.py, partB_reconciliation.json, partB_calculation_output.txt}`, `my-submission/partC/memo.md`, and `my-submission/AI_USAGE.md` all match the original source files.
2. **What Changed During This Targeted Verification Pass:**
   - **`my-submission/partB/partB_report.md` (§0.B3):** Refined the B3 executive summary to explicitly separate the two exact observed end-to-end goodput derivations (`200.92 gen tok/s` from `num_requests * gen_len / wall_clock_s` and `200.925 gen tok/s` from `reported_tok_s / 8`) and per-user goodput (`8.37 gen tok/s/req`) from the percentile-based decode estimates (`202.57 gen tok/s` net decode-only estimate and `249.82 gen tok/s` instantaneous p50 step rate).
   - **`my-submission/FINAL_AUDIT_REPORT.md`:** Replaced all stale `<CONTEXT_SUMMARY>` numbers (`32` layers, `8.20 GB` weights, `1.80 GB` overhead, `1.928` Hindi `o200k` tok/word, `\w+` regex) with the exact verified values from `bench/model_spec.md` (`28` layers, `n_q=24`, `n_kv=8`, `8.40 GB` weights, `1.60 GB` overhead), `bench/bench_log.csv` (`kv_cache_util = 0.97` at $B=32, 48$; `e2e_ms_p95 = 69,221.3 ms` at $B=24$), `fertility.py` (`line.split(" ")`, `line.lower()`, `len(line)`), and `a3_tokenizer_comparison.json` (`1.6473` Hindi `o200k_base` micro tok/word).
   - **`my-submission/AI_USAGE.md` & `my-submission/NOTEBOOK.md`:** Added Item 5 to `AI_USAGE.md` and `Entry 14` to `NOTEBOOK.md` documenting how the `<CONTEXT_SUMMARY>` discrepancy was caught and resolved by re-verifying against the raw source files.
   - **Submission Directory Verified (`my-submission/`):** Verified all 37 files in `my-submission/`.
3. **Unresolved Issues:** **None (`0` unresolved discrepancies remain).**

---

## E. Canonical File Inventory (`my-submission/`)

| File Path (Relative to `my-submission/`) | Role in Submission | Verification Status |
| :--- | :--- | :---: |
| `NOTEBOOK.md` | Graded chronological lab notebook (`Entries 1–14`) | **VERIFIED** |
| `AI_USAGE.md` | Honest disclosure of where AI helped and 5 places where AI required correction | **VERIFIED** |
| `FINAL_AUDIT_REPORT.md` | This comprehensive rubric & source-of-truth verification report | **VERIFIED** |
| `MENTOR_REPORT.md` | Concise mentor-ready technical walkthrough of Parts A, B, and C | **VERIFIED** |
| `partA/memo.md` | Canonical Part A4 $\le 1$-page executive memo (`41` lines; mirrored at `partA/a4_recommendation_memo.md`) | **VERIFIED** |
| `partA/build_corpus.py`, `partA/corpus_documentation.md`, `partA/corpus/` | `6,072` aligned `FLORES-200` `devtest` sentences (`1,012`/lang) + validation JSON | **VERIFIED** |
| `partA/audit_experiments.py`, `partA/e1_e6_raw_output.txt`, `partA/audit_results.md` | Controlled `E1–E6` ablation of `fertility.py` on `corpus_sample/` | **VERIFIED** |
| `partA/compare_tokenizers.py`, `partA/a3_tokenizer_comparison.json`, `partA/a3_analysis.md` | 3 `tiktoken` encodings $\times$ 5 denominators across 6 languages | **VERIFIED** |
| `partB/partB_report.md`, `partB/capacity_reconciliation.py`, `partB/partB_reconciliation.json`, `partB/partB_calculation_output.txt` | Direct `B1–B4` rubric answers + 13-row reconciliation of `bench_log.csv` against `model_spec.md` | **VERIFIED** |
| `partC/memo.md` | Engineering recommendation memo with 5 explicit PDF labels & verified reviewer/capacity math | **VERIFIED** |

---

## F. Live-Defense Preparation: 10 Oral Exam Questions & Model Answers (Verified Against Source Files)

### Q1. *"What are the exact flaws in `fertility.py` and `corpus_sample/`, and how much does each one change the reported `5.89×` Hindi/English fertility ratio?"*
- **Evidence Status:** **Measured Fact (`partA/audit_results.md`, `partA/e1_e6_raw_output.txt`).**
- **Model Answer:**
  1. **`E1` — `line.split(" ")` vs. `line.split()` (`fertility.py:L62`):** `split(" ")` counts empty strings `""` as words whenever consecutive spaces appear (`eng_sample.txt` L07 `"books  in"` has `8` elements instead of `7` words; `hin_sample.txt` L10 `"किताबें  अलमारी"` has `6` elements instead of `5` words). Fixing to `line.split()` changes macro fertility from `1.2652` (`1.27`) to `1.2831` (`1.28`) on `eng` and from `7.4485` (`7.45`) to `7.5985` (`7.60`) on `hin`, shifting the ratio from **`5.89×` (`5.8871×`) to `5.92×` (`5.9221×`, `+0.59%`)**. On raw `FLORES-200`, `split(" ")` injects `+477` phantom empty words in Kannada and `+274` in Telugu.
  2. **`E2` — Unconditional Lowercasing `line = line.lower()` (`fertility.py:L60`):** `gpt2` is case-sensitive. Lowercasing splits capitalized English tokens (`" Thursday"`, `"NASA"`, `" GPU"`) from 1 token into 2 tokens each (`99` total tokens vs. `96` cased), while having **zero effect** on unicameral Devanagari (`459` tokens). Preserving case lowers `eng` fertility from `1.27` to `1.23` (`1.2293`) and raises the `hin/eng` ratio from **`5.89×` to `6.06×` (`6.0590×`, `+2.92%`)**.
  3. **`E3` — Macro-Averaging vs. Micro-Averaging (`fertility.py:L67`):** `sum(per_line_fertility) / n` weights a 4-word line (`hin` L08) equally with a 12-word line (`eng` L03). Corpus micro-averaging (`sum(T_i)/sum(W_i)`) gives `99/79 = 1.2532` (`eng`) and `459/62 = 7.4032` (`hin`), ratio **`5.91×` (`5.9076×`)**.
  4. **`E4` — `len(line)` Codepoint Denominator & Refutation of `REPORT_v0.md` Finding #3:** `REPORT_v0.md` claimed *"Hindi simply has more Unicode characters per word."* In fact, `hin_sample.txt` has **`4.68` codepoints/word (`290/62`) vs. `5.67` in `eng_sample.txt` (`448/79`)—17.5% fewer codepoints/word**! Changing the denominator from codepoints (`7.00×` ratio) to UTF-8 bytes (`764` vs. `448`) drops the ratio to **`2.65×`**, while extended grapheme clusters (`\X`, `188` vs. `448`) raises it to **`10.86×`**.
  5. **`E5` — Non-Parallel Sample Corpus:** `0/10` line indices in `eng_sample.txt` and `hin_sample.txt` align, and `5/10` sentences have no counterpart in the other file. Restricting to the 5 loosely matched pairs shifts the macro ratio to **`6.17×`** (`total token ratio = 242/47 = 5.15×`).

### Q2. *"Why do `Tokens / Word` and `Tokens / Aligned Sentence` rank Hindi and Dravidian languages differently under `o200k_base`?"*
- **Evidence Status:** **Measured Fact (`partA/a3_tokenizer_comparison.json`, `partA/a3_analysis.md` §3.3 & §5.2).**
- **Model Answer:**
  - Across `1,012` aligned `FLORES-200` `devtest` sentences, analytic/postpositional Hindi uses **`25,643` whitespace words** (because postpositions like `में`, `को`, `से` are written as separate words), whereas agglutinative Dravidian languages fuse suffixes onto stems and use only **`14,930` (`mal`) to `16,938` (`tel`) whitespace words**.
  - Dividing by whitespace words (`Micro Tok/Word`) therefore makes Malayalam (`3.5174` tok/word, **`2.87×`** `eng` `1.2270`) look **`2.14×` more expensive per word** than Hindi (`1.6473` tok/word, **`1.34×`** `eng`).
  - However, across the **exact same `1,012` translated sentences**, Malayalam uses `52,515` total tokens vs. `42,241` in Hindi—only **`1.24×`** Hindi's token count! Measured by **Token-Count-per-Aligned-Sentence Ratio vs. English** ($\sum T_{\text{lang}} / \sum T_{\text{eng}}$, where `eng = 26,873` tokens = `26.55` tok/sent), Hindi is **`1.57×`** (`41.74` tok/sent) and all four Dravidian languages cluster tightly between **`1.92×` and `1.98×`** (`tel` `1.92×`, `mal` `1.95×`, `kan` `1.95×`, `tam` `1.98×`).

### Q3. *"Walk through the exact Part B1 KV-cache calculation from `bench/model_spec.md`. What if you mistakenly used `attention heads (Q) = 24` instead of `KV heads (GQA) = 8`?"*
- **Evidence Status:** **Measured Specification + Exact Arithmetic (`bench/model_spec.md`, `partB/partB_report.md` §0.B1 & §2).**
- **Model Answer:**
  - From `bench/model_spec.md`: `layers = 28`, `KV heads (GQA) = 8`, `head_dim = 128`, `KV cache precision = fp16` (`2` bytes/element):
    $$\text{KV Bytes/Token} = 2_{\text{(K,V)}} \times 28 \times 8 \times 128 \times 2 = \mathbf{114,688\text{ B/token}} \ (\mathbf{112.0\text{ KiB/tok}} = \mathbf{0.114688\text{ MB/tok}})$$
  - At $S = 4096$ (`3584` prompt + `512` gen), one sequence requires $4096 \times 114,688 = \mathbf{469,762,048\text{ B}}$ (`469.76 MB` decimal / `448.00 MiB` = `0.4375 GiB` binary).
  - Available KV pool on `1× NVIDIA L4 (24 GB)`:
    $$M_{\text{KV}} = (24.0\text{ GB} \times 0.92) - (4.2\text{ B} \times 2\text{ B}) - 1.60\text{ GB} = 22.08 - 8.40 - 1.60 = \mathbf{12.08\text{ GB}} \ (12,080,000,000\text{ B} \approx 11.2504\text{ GiB})$$
  - Idealized maximum concurrent `4096`-token sequences:
    $$B_{\max,\text{ideal}}(4096) = \left\lfloor \frac{12,080,000,000}{469,762,048} \right\rfloor = \lfloor 25.7151 \rfloor = \mathbf{25\text{ sequences}} \ (\text{`105,329` max tokens})$$
  - **MHA Pitfall ($n_q = 24$):** Using $n_q = 24$ instead of $n_{kv} = 8$ overestimates KV memory by **$3\times$** (`344,064 B/tok = 336.0 KiB/tok`, `1.4093 GB/seq`), which would falsely predict $B_{\max}(4096) = \lfloor 12.08 / 1.4093 \rfloor = \mathbf{8\text{ sequences}}$—contradicting `bench_log.csv` where $B=16$ (`Row 10`, `kv_cache_util = 0.62`) and $B=24$ (`Row 11`, `kv_cache_util = 0.93`) run with **`0` preemptions**.

### Q4. *"How do `kv_cache_util = 0.97` and `preempted_seqs = 7` and `23` in Rows 12–13 of `bench_log.csv` confirm your `12.08 GB` decimal pool calculation?"*
- **Evidence Status:** **Measured Benchmark Data + Exact Arithmetic (`bench/bench_log.csv`, `partB/partB_report.md` §2.3 & §3).**
- **Model Answer:**
  - At $B = 24$ (`Row 11`), $24 \times 0.469762\text{ GB} = 11.2743\text{ GB}$, which is $11.2743 / 12.08 = \mathbf{0.9333}$ (logged `kv_cache_util = 0.93`, `preempted_seqs = 0`).
  - At the idealized ceiling of **$B = 25$ concurrent sequences**, active KV memory is $25 \times 0.469762\text{ GB} = 11.7441\text{ GB}$, which is $11.7441 / 12.08 = \mathbf{0.9722}$—rounding to **`0.97`**, the exact `kv_cache_util` value logged in both `Row 12` ($B=32$) and `Row 13` ($B=48$) because a 26th sequence (`12.2138 GB` = `101.11%`) cannot fit!
  - Furthermore, the number of sequences exceeding the 25-sequence pool capacity is $32 - 25 = \mathbf{7}$ at $B=32$ and $48 - 25 = \mathbf{23}$ at $B=48$, matching logged `preempted_seqs = 7` and `preempted_seqs = 23` to the exact integer.

### Q5. *"In Part B3, explain the two ways to compute Row 11 (`B=24`) goodput and why the `itl_ms_p50` step-rate calculation gives `249.82 gen tok/s` instead of `200.92 gen tok/s`."*
- **Evidence Status:** **Measured Benchmark Data + Exact Arithmetic (`partB/partB_report.md` §0.B3 & §4).**
- **Model Answer:**
  - **Observed End-to-End Goodput (`200.92 gen tok/s` = `8.37 gen tok/s/req`):**
    1. *From request/token counts & wall clock:* $\frac{\texttt{num\_requests} \times \texttt{gen\_len}}{\texttt{wall\_clock\_s}} = \frac{24 \times 512}{61.16\text{ s}} = \frac{12,288}{61.16\text{ s}} = \mathbf{200.92\text{ gen tok/s}}$.
    2. *From `reported_tok_s` and token proportions:* Because `reported_tok_s = 1607.4` includes `3584` prompt tokens + `512` generated tokens per request ($\frac{p+g}{g} = 8.0\times$), multiplying by the output token fraction $\frac{512}{4096} = \frac{1}{8}$ gives $\frac{1607.4}{8} = \mathbf{200.925\text{ gen tok/s}}$.
  - **Why Instantaneous Median Step Rate (`249.82 gen tok/s`) Differs:**
    - Dividing $B=24$ by `itl_ms_p50 = 96.07 ms` gives $\frac{24}{0.09607\text{ s}} = \mathbf{249.82\text{ gen tok/s}}$ (and using `ttft_ms_p50 + 512 * itl_ms_p50 = 49.688 s` gives `247.30 gen tok/s`).
    - However, per `model_spec.md:L38`, `itl_ms_p50` is a **50th-percentile (median)** step latency, not the **mean** step latency, and it ignores tail decode steps across the batch (whereas `wall_clock_s = 61.16 s` waits for all 24 requests to finish, and `e2e_ms_p95 = 69,221.3 ms`). Thus, `249.82 gen tok/s` is an instantaneous median decode step rate, whereas **`200.92 gen tok/s`** is the true observed end-to-end goodput.

### Q6. *"When comparing Row 5 (`B=16, p=512, g=256`) and Row 10 (`B=16, p=3584, g=512`), can we attribute the `-44.33%` drop in output throughput (`294.46` $\rightarrow$ `163.94 gen tok/s`) solely to longer prompts?"*
- **Evidence Status:** **Methodological Qualification (`partB/partB_report.md` §0.B3, `partC/memo.md` §2).**
- **Model Answer:**
  - No. While `Row 5` (`294.46 gen tok/s`) and `Row 10` (`163.94 gen tok/s`) both run at `batch_size = 16` with `0` preemptions, they change **both** `prompt_len` (`512` $\rightarrow$ `3584`) **and** `gen_len` (`256` $\rightarrow$ `512`). Because KV-cache size and attention cost grow across the decode phase from $p$ to $p+g$, this comparison demonstrates that the long-context workload (`3584/512`) has `44.33%` lower output goodput and `+59.74%` higher `itl_ms_p50` (`48.33 ms` $\rightarrow$ `77.20 ms`) than the short-context workload (`512/256`), but isolating the pure effect of `prompt_len` at fixed `gen_len` requires holding `gen_len = 256` constant (as in our Part C L4 load-test protocol).

### Q7. *"Why did you reclassify the `190.7–200.1 GB/s` HBM bandwidth numbers in Part B as analytical traffic-model estimates?"*
- **Evidence Status:** **Methodological Qualification (`partB/partB_report.md` §3 & §5.3).**
- **Model Answer:**
  - Dividing `8.40 GB` weights plus midpoint active KV cache ($\min(B, 25) \times (p + g/2) \times 114,688\text{ B}$) by `itl_ms_p50` gives `190.7–200.1 GB/s` (`63.6%–66.7%` of L4's `300 GB/s` peak).
  - Because `bench_log.csv` has no hardware performance counters (`dram__bytes_read.sum.per_second`), this formula is an **analytical traffic-model estimate**, not measured hardware bandwidth. What `bench_log.csv` directly proves is that the throughput cliff at $B=32, 48$ is caused by **KV-cache saturation (`kv_cache_util = 0.97`) and sequence preemption (`preempted_seqs = 7` and `23`)**.

### Q8. *"In Part C, why is Option 3 (Concise Prompt Engineering, $\Delta p \le 128$ tokens) your Primary Committed Launch Baseline while Option 1 (Merged `fp16` LoRA SFT) is a Gated Week-2 Experiment?"*
- **Evidence Status:** **Engineering Decision Under Constraints (`partC/memo.md` §1–§3).**
- **Model Answer:**
  - Three prerequisites for SFT are **unestablished** in the starter kit: (1) no Hindi/Kannada SFT dataset is supplied locally and our external API budget is **`$0`**; (2) `FLM-4B-Instruct`'s actual `128k` tokenizer and baseline failure modes are unmeasured; and (3) we have only **2 weeks** and **10 reviewer hours/week**.
  - **Option 3 ($\Delta p \le 128$ tokens)** requires zero training data, has `0%` weight-regression risk on English, bounds L4 KV impact ($S = 768 \rightarrow 896$ shifts $B_{\max,\text{ideal}}$ from `137` to `117` sequences), and is human-evaluated in Week 1.
  - **Option 1 (LoRA SFT merged into `8.40 GB` `fp16` weights)** runs in Week 2 *only if* Day 3 Gate #1 verifies $\ge 10,000$ clean decontaminated permissive/internal pairs, and replaces Option 3 at Week 3 *only if* it beats Option 3 by $\ge +15.0\text{ pp}$ net win margin with zero L4 preemptions at $B=24$.

### Q9. *"Why did you reject Option 2 ($\le 1\text{B}$ inference-time rewriter) in Part C?"*
- **Evidence Status:** **Derived Capacity & Latency Analysis (`partC/memo.md` §1.2 & §3).**
- **Model Answer:**
  - Co-locating a $\le 1\text{B}$ `fp16` rewriter (`~2.0 GB` weights + `~0.5–1.5 GB` KV/activations = `~2.5–3.5 GB`) on the `24 GB` L4 reduces the `12.08 GB` KV pool to `~8.6–9.6 GB`, dropping idealized `4096`-token capacity from **`25` sequences to `18–20` sequences** (causing immediate preemptions at `Row 11` $B=24$), adds a second sequential decode pass, and risks corrupting entities, numbers, and JSON formatting.

### Q10. *"How does your Part C reviewer schedule stay within the `10 hours/week` limit while still measuring inter-annotator reliability?"*
- **Evidence Status:** **Verified Arithmetic (`partC/memo.md` §5.2).**
- **Model Answer:**
  - Each main evaluation round (Week 1 Round 1 and Week 3 Round 3) rates **`80` unique golden test prompts plus `10` *additional* blind duplicate overlap prompts = `90` total pairwise comparisons per language**.
  - At `3.0 minutes/pair` (`20` pairs/hour), `90` pairs take **`4.5 hours`** (`4.0 h` unique + `0.5 h` overlap) plus **`0.5 hours`** of calibration/sign-off = **`5.0 hours/week` for Hindi and `5.0 hours/week` for Kannada (`10.0 hours/week` combined)**. Week 2 rates `40` dev-split pairs per language = **`2.0 hours/week` per language (`4.0 hours/week` combined)**. Neither individual reviewer nor the combined total ever exceeds `10.0 hours/week`.

---

## G. Final Submission Checklist

- [x] **Part B Source-of-Truth Verified:** `28` layers, `n_q=24`, `n_kv=8`, `head_dim=128`, `8.40 GB` weights, `1.60 GB` overhead, `114,688 B/tok`, `12.08 GB` pool, $B_{\max,\text{ideal}}(4096) = 25$ sequences, `kv_cache_util = 0.97` at $B=32, 48$, `preempted_seqs = 7` and `23`.
- [x] **Part A Numbers Reconciled:** `a3_tokenizer_comparison.json`, `a3_analysis.md`, `partA/memo.md`, and `FINAL_AUDIT_REPORT.md` all report identical numbers (`o200k_base` Hindi micro `tok/word = 1.6473`, sentence ratio `1.57×`; Dravidian sentence ratios `1.92×–1.98×`).
- [x] **Part B3 Throughput Derivations Audited:** Observed end-to-end goodput (`200.92 gen tok/s` = `8.37 gen tok/s/req`) is clearly separated from percentile-based decode estimates (`202.57` and `249.82 gen tok/s`).
- [x] **Part C Reviewer Budget & Epistemic Labels Verified:** `80 unique + 10 additional overlap = 90 pairs/lang` (`4.5 h + 0.5 h = 5.0 h/wk/lang`, `10.0 h/wk` combined); all A100 training figures labeled as unexecuted estimates.
- [x] **Starter-Kit Originals Untouched & Submission Directory Verified:** All 6 starter-kit files untouched; all 37 files in `my-submission/` verified and reproducible.
