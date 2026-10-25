# Mentor Technical Report — AI Team Intern Assignment (“The Audit”)

**Prepared For:** Engineering Mentor / Technical Reviewer  
**Prepared By:** AI Team Intern  
**Date:** 2026-10-24  
**Canonical Submission Directory:** `your-submission/` *(mirrored byte-for-byte in `my-submission/`)*  
**Status:** Complete, Reconciled Against Original Source Files, and Ready for Review

---

## A. Executive Summary

### 1. Overall Objective of the Assignment
The assignment (**“The Audit”**) places us in the role of an AI engineering intern asked to audit an outgoing intern's preliminary benchmark deck (`REPORT_v0.md`), measurement script (`fertility.py`), sample data (`corpus_sample/`), and serving load-test logs (`bench/model_spec.md` and `bench/bench_log.csv`) before leadership commits to three high-cost infrastructure and product decisions:
1. **Part A (Tokenizer Audit):** `REPORT_v0.md` Section 1 claimed that Hindi requires **`5.89×` more tokens per word** (`7.45` vs. `1.27`) and **`7.00×` more tokens per character** (`1.579` vs. `0.226`) than English because *"Hindi simply has more Unicode characters per word, so any tokenizer will struggle,"* and recommended splitting Indian-language traffic into a separate serving tier with a 6× token budget.
2. **Part B (Capacity Reconciliation):** `REPORT_v0.md` Section 2 claimed from `bench_log.csv` that at `batch_size = 16`, `3584`-token prompts achieve higher throughput (`1311 tok/s`) than `512`-token prompts (`883 tok/s`), recommended that clients pack longer prompts per request, and projected linear scaling to `batch_size = 48` (`~3200 tok/s`).
3. **Part C (Engineering Recommendation):** The product team wants assistant replies in Indian languages (`Hindi, Kannada, Tamil, Telugu, Bengali, Marathi`) to sound natural and conversational rather than formal/textbook, under strict constraints: **1× NVIDIA A100 80 GB GPU**, **2 weeks** for implementation and evaluation (**launch review in 3 weeks**), **native-speaker reviewer coverage for Hindi + Kannada only (`10 hours/week`)**, and a **`$0` external API budget**.

### 2. Summary of Work Investigated and Delivered
Across `your-submission/`, we delivered reproducible scripts, machine-readable JSON outputs, raw execution logs, and concise engineering memos:
- **Part A (`your-submission/partA/`):**
  - Assembled a **6-language, 1,012-sentence aligned parallel corpus** (`6,072` total sentences across English `eng`, Hindi `hin`, Tamil `tam`, Telugu `tel`, Kannada `kan`, and Malayalam `mal`) from official **FLORES-200 (`devtest`)** (`build_corpus.py`, `corpus_documentation.md`).
  - Conducted controlled single-variable experiments (`E1–E6`) isolating every implementation and methodological flaw in `fertility.py` and `corpus_sample/` (`audit_experiments.py`, `audit_results.md`).
  - Benchmarked three `tiktoken` byte-level BPE encodings (`gpt2` [`50,257` vocab], `cl100k_base` [`100,277` vocab], and `o200k_base` [`200,019` vocab]) across five denominators (`compare_tokenizers.py`, `a3_tokenizer_comparison.json`, `a3_analysis.md`) and synthesized a 1-page routing memo (`partA/memo.md`).
- **Part B (`your-submission/partB/`):**
  - Built an automated capacity and throughput reconciliation pipeline (`capacity_reconciliation.py`, `partB_reconciliation.json`, `partB_report.md`) auditing all 13 rows of `bench/bench_log.csv` against `bench/model_spec.md`.
- **Part C & Supporting Documentation (`your-submission/partC/memo.md`, `NOTEBOOK.md`, `AI_USAGE.md`, `FINAL_AUDIT_REPORT.md`):**
  - Designed an evidence-gated 3-week engineering and evaluation plan (`partC/memo.md`), maintained a 14-entry chronological lab notebook (`NOTEBOOK.md`), and documented where AI tooling helped and where initial AI assumptions required human correction (`AI_USAGE.md`).

### 3. Headline Findings & Recommended Engineering Direction
1. **`REPORT_v0.md` Section 1 Is Factually and Methodologically Flawed (Part A):**
   - **Hindi has *fewer* Unicode codepoints per word than English** (`4.68` vs. `5.67` on `corpus_sample/`, **`-17.5%`**; `5.12` vs. `6.03` on `FLORES-200`).
   - `gpt2`'s extreme Indic token inflation is caused by its **50k English-centric vocabulary falling back to raw UTF-8 byte tokens**, not an inescapable property of Indic scripts. Switching within `tiktoken` from `gpt2` to `o200k_base` reduces total tokens on `1,012` aligned `FLORES-200` sentences by **`-78.95%` (`4.75×` fewer tokens) in Hindi** and **`-85.28%` to `-87.36%` (`6.79×–7.91×` fewer tokens) across Tamil, Telugu, Kannada, and Malayalam**, bringing **token-count-per-aligned-sentence ratios vs. English** down from `7.42×` (`hin`) and `12.96×–15.53×` (Dravidian) under `gpt2` to **`1.57×` (`hin`) and `1.92×–1.98×` (Dravidian) under `o200k_base`**.
   - **Recommendation:** `gpt2` fertility alone cannot justify separate Indic routing, nor do static `tiktoken` compression ratios alone prove that a unified deployment is sufficient. Before changing routing architecture, benchmark **`FLM-4B-Instruct`'s actual `128k` production tokenizer** on our aligned `FLORES-200` corpus alongside Indic model quality and latency tests.
2. **`REPORT_v0.md` Section 2 Misread Total-Token Throughput and Missed KV-Cache Preemption Already in the Log (Part B):**
   - From `bench/model_spec.md` (`L = 28`, $n_{kv} = 8$ GQA heads, $d_h = 128$, `fp16`), `FLM-4B-Instruct`'s KV cache requires **`114,688 bytes/token` (`112.0 KiB/token`)** or **`469.76 MB` (`448.00 MiB`) per `4096`-token sequence**. Inside the L4's **`12.08 GB` (`11.2504 GiB`) idealized KV pool** (`24.0 GB × 0.92 − 8.40 GB weights − 1.60 GB overhead`), the **idealized maximum concurrency at `4096` tokens is `25` sequences**.
   - `reported_tok_s` is **total-token throughput** ($\frac{N(p+g)}{\text{wall\_clock\_s}}$), which inflates output generation rate by **`3.0×`** on short prompts (`512/256`) vs. **`8.0×`** on long prompts (`3584/512`). Measured by **generated-output-token goodput** ($\frac{N \cdot g}{\text{wall\_clock\_s}}$), `Row 10` (`B=16, 3584/512`) is actually **`44.33%` slower** than `Row 5` (`B=16, 512/256`): **`163.94` vs. `294.46 gen tok/s`** (`itl_ms_p50` `+59.74%` slower: `77.20 ms` vs. `48.33 ms`).
   - Long-context throughput peaks at **`batch_size = 24` (`Row 11`: `1607.4 total tok/s`, `200.92 gen tok/s`, `kv_cache_util = 0.93`, `preempted_seqs = 0`)** and **degrades at `batch_size = 32` (`Row 12`: `1384.0 tok/s`, `172.99 gen tok/s`, `preempted_seqs = 7`) and `batch_size = 48` (`Row 13`: `1298.5 tok/s`, `162.31 gen tok/s`, `preempted_seqs = 23`)** as `kv_cache_util` hits its **`0.97`** ceiling ($25 / 25.715 = 0.9722$) and the scheduler preempts every sequence above 25 ($32 - 25 = 7$; $48 - 25 = 23$).
3. **Staged, Evidence-Gated Engineering Recommendation for Conversational Hindi & Kannada (Part C):**
   - **Primary Committed Launch Baseline (Stage 1 — Week 1):** Implement and human-evaluate **Option 3 (Concise System-Prompt Engineering, $\Delta p \le 128$ tokens, no weight changes)** for **Hindi and Kannada**.
   - **Gated Experimental Upgrade (Stage 2 — Week 2):** Execute **Option 1 (LoRA SFT of `FLM-4B-Instruct` on `1× A100 80 GB`, merged into `8.40 GB` `fp16` weights)** *only if* a Day 3 Data & Tokenizer Gate confirms $\ge 10,000$ clean, decontaminated, permissively licensed conversational/instruction pairs (`$0` API cost), and promote it over Option 3 at Week 3 *only if* it beats Option 3 by $\ge +15.0\text{ percentage points}$ net human win margin without English or L4 serving regression.
   - **Rejected Option:** Reject **Option 2 ($\le 1\text{B}$ Inference-Time Rewriter)** because co-locating it on the `24 GB` L4 shrinks the KV pool from `12.08 GB` to `~8.6–9.6 GB` (dropping $B_{\max,\text{ideal}}(4096)$ from `25` to `18–20` sequences), adds a second sequential decode pass, and risks entity/number corruption.
   - **Language Scope Rule:** Scope Week 3 launch strictly to **Hindi and Kannada** (where human reviewer coverage exists); do not fine-tune or launch unreviewed **Tamil, Telugu, Bengali, or Marathi** changes blindly.

### 4. Epistemic Status Classification (Measured vs. Analytical vs. Proposed vs. Unvalidated)

| Category | Specific Items in This Report | Epistemic Status |
| :--- | :--- | :--- |
| **Directly Measured / Observed** | • `fertility.py` ablations (`E1–E6`) on `corpus_sample/` (`audit_experiments.py`)<br>• `FLORES-200` `devtest` (`N = 1,012` aligned sentences/language) counts & `tiktoken` (`gpt2`, `cl100k_base`, `o200k_base`) results across 5 denominators (`compare_tokenizers.py`)<br>• All 13 benchmark rows of `bench/bench_log.csv` (`wall_clock_s`, `reported_tok_s`, `ttft_ms_p50`, `itl_ms_p50`, `e2e_ms_p95`, `preempted_seqs`, `kv_cache_util`) | **Measured Fact** (Verified from executed code and starter-kit logs) |
| **Analytically Calculated from Supplied Specs/Logs** | • Per-token (`114,688 B`) & per-sequence (`469.76 MB` at $S=4096$) GQA KV footprint (`model_spec.md`)<br>• Idealized KV pool (`12.08 GB` dec = `11.2504 GiB` bin) & idealized capacity ($B_{\max,\text{ideal}}(4096) = 25$ seqs, $B_{\max,\text{ideal}}(768) = 137$ seqs)<br>• Observed end-to-end output goodput ($\frac{N \cdot g}{\text{wall\_clock\_s}} = \mathbf{200.92\text{ gen tok/s}}$ at `Row 11`) vs. percentile-based decode estimates (`202.57` & `249.82 gen tok/s`)<br>• Traffic-model decode HBM estimate (`190.7–200.1 GB/s`, labeled as a model estimate, not hardware telemetry)<br>• Part C reviewer-hour budget (`90` pairs/lang $\times 3.0\text{ min} + 0.5\text{ h} = \mathbf{5.0\text{ h/wk/lang}} = \mathbf{10.0\text{ h/wk}}$ combined) | **Analytical Calculation** (Exact arithmetic from supplied parameters; assumptions explicitly stated) |
| **Proposed & Conditional (Not Yet Tested)** | • Part B `fp8` KV-cache quantization (`57,344 B/tok`, predicted $B_{\max,\text{ideal}}(4096) = 51$ seqs and `0` preemptions at $B \le 48$) — **conditional on engine support and quality/latency validation**<br>• PartB/C scheduler concurrency cap (`max_num_seqs = 24` for long contexts)<br>• Part C staged plan (`Option 3` concise system prompt + gated `Option 1` LoRA SFT), `15k–30k` target SFT mixture, A100 training VRAM (`~18–26 GB`) and time (`~3–6 GPU-hrs/run`) estimates, and pre-registered acceptance thresholds | **Proposed / Analytical Estimate** (Not yet executed or empirically validated) |
| **Not Yet Validated / Unmeasured** | • `FLM-4B-Instruct`'s actual `128k` production tokenizer behavior on Indic text (model weights/tokenizer files not included in starter kit)<br>• Hugging Face `hf:xlm-roberta-base` fertility (`transformers` not installed in `.venv-x64`)<br>• Hardware profiler counters (Nsight Compute / DCGM HBM bytes/s) and serving engine preemption counters (`vllm:num_preemptions_total`) | **Unmeasured** (Explicitly disclosed as unexecuted) |

---

## B. Part A — Tokenizer Audit (`your-submission/partA/`)

### 1. Flaws Identified in `fertility.py` and `corpus_sample/` (`audit_experiments.py` & `audit_results.md`)
We ran controlled single-variable experiments (`E1–E6`) against the unmodified `fertility.py` baseline on the 10-line `corpus_sample/eng_sample.txt` and `corpus_sample/hin_sample.txt` files:

| Exp ID | Variable Audited in `fertility.py` / `corpus_sample/` | Baseline Value (`gpt2`, Unrounded / Rounded) | Single-Variable Experiment Value | Measured Delta ($\Delta$ / `%`) | Technical Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base** | Unmodified `fertility.py` (`line.lower()`, `line.split(" ")`, `len(line)`, macro-average) | `eng` fert: `1.265206` (`1.27`)<br>`hin` fert: `7.448452` (`7.45`)<br>ratio: `5.887148` (`5.89x`)<br>`eng` `tok/char`: `0.225636` (`0.226`)<br>`hin` `tok/char`: `1.579108` (`1.579`)<br>`tok/char` ratio: `6.998478` (`7.00x`) | Identical | `0.00%` | Reproduces all 5 numbers in `REPORT_v0.md`. Note that `fertility.py` computes `7.448452 / 1.265206 = 5.887148` (`5.89x`) before rounding (`7.45 / 1.27 = 5.866`). |
| **`E1`** | **Word splitting:** `line.split(" ")` (`L62`) $\rightarrow$ `line.split()` | `eng` words: `79` (`1.265206`)<br>`hin` words: `62` (`7.448452`)<br>ratio: `5.887148` (`5.89x`) | `eng` words: `78` (`1.283063` / `1.28`)<br>`hin` words: `61` (`7.598452` / `7.60`)<br>ratio: `5.922121` (`5.92x`) | `eng` fert: `+1.41%`<br>`hin` fert: `+2.01%`<br>ratio: `+0.59%` | **Code Bug:** `split(" ")` counts empty strings `""` as words on consecutive spaces (`eng` L07 `"books  in"`, `hin` L10 `"किताबें  अलमारी"`). On raw `FLORES-200`, `split(" ")` injects **`+477` phantom words in Kannada** (`213` lines) and **`+274` in Telugu** (`136` lines) vs. `0` in English. |
| **`E2`** | **Lowercasing:** `line = line.lower()` (`L60`) $\rightarrow$ preserve original casing | `eng` tokens: `99` (`1.265206`)<br>`hin` tokens: `459` (`7.448452`)<br>ratio: `5.887148` (`5.89x`) | `eng` tokens: `96` (`1.229329` / `1.23`)<br>`hin` tokens: `459` (`7.448452` / `7.45`)<br>ratio: `6.058958` (`6.06x`) | `eng` tok: `-3.03%`<br>`eng` fert: `-2.84%`<br>`hin`: `0.00%`<br>ratio: `+2.92%` | **Methodological Bug:** `gpt2` is case-sensitive. Lowercasing breaks single-token English merges (`" Thursday"` $\rightarrow$ `2` tok on L02, `"NASA"` $\rightarrow$ `2` tok on L06, `" GPU"` $\rightarrow$ `2` tok on L10) while having **zero effect** on unicameral Devanagari (`459` tokens). |
| **`E3`** | **Aggregation:** Macro-average across lines (`L67`) $\rightarrow$ Micro-average (`sum(T_i)/sum(W_i)`) | `eng` fert: `1.265206` (`1.27`)<br>`hin` fert: `7.448452` (`7.45`)<br>ratio: `5.887148` (`5.89x`) | `eng` micro: `99/79 = 1.253165` (`1.25`)<br>`hin` micro: `459/62 = 7.403226` (`7.40`)<br>ratio: `5.907625` (`5.91x`) | `eng` fert: `-0.95%`<br>`hin` fert: `-0.61%`<br>ratio: `+0.35%` | **Metric-Design Flaw:** Macro-averaging weights a 4-word line (`hin` L08) equally with a 12-word line (`eng` L03). Total serving/KV cost over a workload scales with the sum of tokens $\sum T_i$, requiring **micro-averages**. |
| **`E4a/b`** | **Character denominator:** `len(line)` codepoints (`L63`) $\rightarrow$ UTF-8 bytes (`E4a`) or `\X` extended grapheme clusters (`E4b`) | `eng` denom: `448` cp (`0.226` tok/cp)<br>`hin` denom: `290` cp (`1.579` tok/cp)<br>ratio: `6.998478` (`7.00x`) | **UTF-8 bytes (`E4a`):** `hin = 764` B (`0.599` tok/B), ratio **`2.65x`** (`2.654683x`)<br>**Graphemes `\X` (`E4b`):** `hin = 188` G (`2.450` tok/G), ratio **`10.86x`** (`10.857013x`) | Byte ratio: `-62.07%`<br>Grapheme ratio: `+55.13%` | **Refutes `REPORT_v0.md` Finding #3:** Hindi has **`4.68` codepoints/word (`290/62`) vs. `5.67` in English (`448/79`, `-17.5%`)**! In English, `448` bytes = `448` codepoints = `448` graphemes; in Hindi, `764` UTF-8 bytes = `290` codepoints = `188` orthographic *aksharas* (`\X`). |
| **`E5`** | **Corpus alignment:** All 10 lines $\rightarrow$ 5 loosely matched pairs (`eng` L3,4,5,7,8 $\leftrightarrow$ `hin` L3,7,6,10,4) | Macro ratio: `5.887148` (`5.89x`)<br>Total token ratio: `459/99 = 4.64x` | Macro ratio: `6.174419` (`6.17x`)<br>Total token ratio: `242/47 = 5.15x` | Macro ratio: `+4.88%`<br>Token ratio: `+11.06%` | **Fatal Sample Flaw:** `0/10` line indices align between `eng_sample.txt` and `hin_sample.txt`, and `5/10` sentences have no counterpart in the other file at all. |
| **`E6a/b/c`** | **Controls:** `random.seed(1337)` (`L25`), `NFC` normalization (`L49`), `add_special_tokens=False` (`L33`) | `1.265206` / `7.448452` | Identical (`\Delta = 0.000000`) | `0.00%` | **Verified Non-Bugs:** `random` is unused (BPE is deterministic); `NFC` is a no-op on `corpus_sample/` (`0/10` lines altered) but **essential on raw `FLORES-200`** (where `93` `hin` lines and `14` Dravidian lines are non-NFC); disabling framing tokens prevents per-line BOS/EOS distortion. |

### 2. Multilingual Evaluation Corpus (`build_corpus.py` & `corpus_documentation.md`)
- **Source & Provenance:** Official **FLORES-200 `devtest` split** (`flores200_dataset.tar.gz`, `25,585,843` bytes, SHA-256 `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6`), licensed under **CC-BY-SA 4.0**.
- **Scale & Alignment:** **1,012 strictly aligned parallel sentences per language** (`6,072` sentences total across `281` Wikimedia source articles: `34.68%` WikiBooks, `33.70%` WikiNews, `31.62%` WikiVoyage) covering **English (`eng_Latn`)**, **Hindi (`hin_Deva`)**, **Tamil (`tam_Taml`)**, **Telugu (`tel_Telu`)**, **Kannada (`kan_Knda`)**, and **Malayalam (`mal_Mlym`)**.
- **Preprocessing (`your-submission/partA/corpus/processed/`):** Strict UTF-8 validation (`0` replacement/control chars), line stripping, **Unicode `NFC` normalization** (which canonically decomposes legacy Devanagari nukta singletons `U+0958–U+095F` on `93` Hindi lines [`+122` codepoints] and composes split vowel signs on `14` Dravidian lines), **internal whitespace collapsing** (`" ".join(line.split())`, fixing `6` `hin`, `75` `tam`, `136` `tel`, `213` `kan`, and `54` `mal` multi-space lines), and **original casing preserved**.
- **Corpus Limitations:** (1) Formal translated Wikimedia prose (a proxy for comparable message content, not mathematically identical propositions due to translationese and minor translator edits); (2) excludes **Romanized/code-switched chat (Hinglish/Tanglish/Kanglish)**, conversational imperatives, and JSON/code; (3) retains minor upstream FLORES-200 CSV-escaped quote artifacts (`""` on `58–68` `tam`/`tel`/`mal` lines) so extraction remains 100% deterministic.

### 3. Multi-Tokenizer & Multi-Denominator Comparison (`compare_tokenizers.py` & `a3_analysis.md`)
- **Explicit Library Scope Disclosure:** In `.venv-x64`, `tiktoken 0.14.0` (and `regex 2026.9.29`) is installed, whereas Hugging Face `transformers`, `tokenizers`, and `sentencepiece` are **not installed** (`ModuleNotFoundError: No module named 'transformers'`). Rather than altering the locked environment, we evaluated three `tiktoken` encodings—**`gpt2`** (`50,257` vocab), **`cl100k_base`** (`100,277` vocab), and **`o200k_base`** (`200,019` vocab)—and explicitly disclosed that this compares **three BPE vocabularies within `tiktoken`**, not independent tokenizer libraries or algorithms. We also documented how `tiktoken`'s `enc.encode(line)` (`allowed_special=set()`, no post-processor framing template) differs from Hugging Face's `add_special_tokens=False`.

#### Summary of Verified Corpus-Level (Micro) Results Across `1,012` Aligned Sentences (`a3_tokenizer_comparison.json`)

| Language (`ISO 639-3`) | `.split()` Words | `NFC` Codepoints (`CP/Word`) | UTF-8 Bytes | `\X` Graphemes | **`gpt2` (`50k` Vocab)**<br>`Total Tok` \| `Tok/Word` (`vs eng`) \| `Tok/Sent` (`vs eng`) | **`cl100k_base` (`100k` Vocab)**<br>`Total Tok` \| `Tok/Word` (`vs eng`) \| `Tok/Sent` (`vs eng`) | **`o200k_base` (`200k` Vocab)**<br>`Total Tok` \| `Tok/Word` (`vs eng`) \| **`Tok/Sent` (`vs eng`)** | `o200k` `Tok/Byte` (`B/Tok`) | Token Reduction (`gpt2` $\rightarrow$ `o200k`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`eng` (English)** | `21,901` | `131,966` (`6.03`) | `132,096` | `131,966` | `27,044` \| `1.2348` (`1.00×`) \| `26.72` (`1.00×`) | `27,182` \| `1.2411` (`1.00×`) \| `26.86` (`1.00×`) | `26,873` \| `1.2270` (`1.00×`) \| **`26.55` (`1.00×`)** | `0.2034` (`4.92`) | `-0.63%` (`1.01×` less) |
| **`hin` (Hindi)** | `25,643` | `131,174` (`5.12`) | `337,433` | `85,951` | `200,682` \| `7.8260` (`6.34×`) \| `198.30` (`7.42×`) | `129,567` \| `5.0527` (`4.07×`) \| `128.03` (`4.77×`) | `42,241` \| `1.6473` (`1.34×`) \| **`41.74` (`1.57×`)** | `0.1252` (`7.99`) | **`-78.95%` (`4.75×` less)** |
| **`tam` (Tamil)** | `16,775` | `154,031` (`9.18`) | `421,535` | `99,624` | `420,071` \| `25.0415` (`20.28×`) \| `415.09` (`15.53×`) | `207,687` \| `12.3807` (`9.98×`) \| `205.22` (`7.64×`) | `53,114` \| `3.1663` (`2.58×`) \| **`52.48` (`1.98×`)** | `0.1260` (`7.94`) | **`-87.36%` (`7.91×` less)** |
| **`tel` (Telugu)** | `16,938` | `132,226` (`7.81`) | `353,422` | `76,328` | `350,509` \| `20.6936` (`16.76×`) \| `346.35` (`12.96×`) | `225,022` \| `13.2850` (`10.70×`) \| `222.35` (`8.28×`) | `51,610` \| `3.0470` (`2.48×`) \| **`51.00` (`1.92×`)** | `0.1460` (`6.85`) | **`-85.28%` (`6.79×` less)** |
| **`kan` (Kannada)** | `16,100` | `137,650` (`8.55`) | `374,964` | `89,994` | `366,989` \| `22.7943` (`18.46×`) \| `362.64` (`13.57×`) | `240,364` \| `14.9294` (`12.03×`) \| `237.51` (`8.84×`) | `52,534` \| `3.2630` (`2.66×`) \| **`51.91` (`1.95×`)** | `0.1401` (`7.14`) | **`-85.69%` (`6.99×` less)** |
| **`mal` (Malayalam)** | `14,930` | `149,261` (`10.00`) | `411,670` | `79,357` | `409,912` \| `27.4556` (`22.23×`) \| `405.05` (`15.16×`) | `242,859` \| `16.2665` (`13.11×`) \| `239.98` (`8.93×`) | `52,515` \| `3.5174` (`2.87×`) \| **`51.89` (`1.95×`)** | `0.1276` (`7.84`) | **`-87.19%` (`7.81×` less)** |

*(Macro-averaged `Tokens / Word` is slightly higher than micro-averaged `Tokens / Word` across all languages—e.g., under `o200k_base`: `eng` `1.2360` vs. `1.2270`, `hin` `1.6695` vs. `1.6473`, `tam` `3.1946` vs. `3.1663`, `tel` `3.0797` vs. `3.0470`, `kan` `3.3010` vs. `3.2630`, `mal` `3.5539` vs. `3.5174`.)*

### 4. Key Insights & Methodological Caveats (`partA/memo.md`)
1. **Why `Tokens / Whitespace Word` Distorts Agglutinative Dravidian Languages:**
   - Under `o200k_base`, Malayalam has a micro word fertility of **`3.5174` tok/word (`2.87×` `eng`)** vs. Hindi's **`1.6473` tok/word (`1.34×` `eng`)**, making Malayalam appear **`2.14×` more expensive per word** than Hindi.
   - Yet across the **same `1,012` aligned sentences**, Malayalam uses **`52,515` tokens vs. `42,241` in Hindi (`1.24×`)**! Because Dravidian languages are **agglutinative** (`14,930–16,938` words across `1,012` sentences vs. `25,643` words in analytic/postpositional Hindi), dividing by whitespace words penalizes agglutinative morphology. **Token-Count-per-Aligned-Sentence Ratio vs. English** ($\sum T_{\text{lang}} / \sum T_{\text{eng}}$: **`1.57×` for Hindi** and **`1.92×–1.98×` for Dravidian** under `o200k_base`) is the appropriate cross-language compression metric because it holds the aligned sentence count constant.
2. **Why Tokenizer Compression Alone Does Not Establish Quality, Latency, or Unified-Serving Sufficiency:**
   - The `1.57×–1.98×` `o200k_base` values are **static token-count-per-aligned-sentence ratios**, **not** serving-cost multipliers. Lower token counts reduce static sequence length and per-sentence KV footprint for a fixed text, but they do **not** measure downstream language modeling quality, factual accuracy, generated-output verbosity/repetition loops, or end-to-end latency.
   - Therefore, while `REPORT_v0.md`'s `gpt2` numbers are insufficient to justify a separate Indic serving pool, **static `tiktoken` compression alone does not prove that a unified production deployment is sufficient**.
3. **Why the Production Tokenizer Must Be Benchmarked First:**
   - We did **not** test `FLM-4B-Instruct`'s `128k` tokenizer (`bench/model_spec.md:L13`) and must not assume it behaves like `o200k_base` (`200k`). Benchmarking `FLM-4B-Instruct`'s exact production tokenizer on `your-submission/partA/corpus/processed/` is mandatory before making routing or capacity commitments.

---

## C. Part B — Capacity Reconciliation (`your-submission/partB/`)

### 1. Serving Model and Hardware Assumptions (`bench/model_spec.md`)
From `bench/model_spec.md` (with undocumented parameters such as the serving engine, PagedAttention block size, preemption mode `recompute` vs. `swap`, and prefix-caching flags explicitly listed as unspecified):
- **Model (`FLM-4B-Instruct`, dense):** `4.2 B` parameters, **`L = 28` layers**, `d_model = 3072`, **`n_q = 24` query heads**, **`n_kv = 8` KV heads (GQA, 3:1 ratio)**, **`head_dim = 128`**, `vocab = 128k`, `fp16` weights (`2` B/param), `fp16` KV cache (`2` B/elem).
- **Hardware & Memory Budget:** **`1× NVIDIA L4 (24 GB)`**, `300 GB/s` peak memory bandwidth, `~121 TFLOPS` peak `fp16` compute, `max_model_len = 4096`, **`gpu_memory_utilization = 0.92`**, and **`non-KV runtime overhead = assume ~1.6 GB`**.
- **Load-Test Harness (`bench/bench_log.csv`):** `num_requests` identical requests (`batch_size == num_requests` on all 13 rows) submitted simultaneously with fixed `(prompt_len, gen_len)` and no early stopping:
  - **Short-context sweep (`Rows 1–7`):** `prompt_len = 512`, `gen_len = 256` ($S = 768$), $B \in \{1, 2, 4, 8, 16, 32, 64\}$.
  - **Long-context sweep (`Rows 8–13`):** `prompt_len = 3584`, `gen_len = 512` ($S = 4096 = \texttt{max\_model\_len}$), $B \in \{4, 8, 16, 24, 32, 48\}$.

### 2. KV-Cache Memory Formula, Role of GQA, and Idealized Capacity Derivation (`B1`)
1. **Per-Token KV-Cache Footprint (Grouped-Query Attention):**
   $$\text{KV Bytes/Token (GQA)} = 2_{\text{(K,V)}} \times L \times n_{kv} \times d_h \times b_{\text{elem}} = 2 \times 28 \times 8 \times 128 \times 2 = \mathbf{114,688\text{ bytes/token}} \ (\mathbf{112.0\text{ KiB/tok}} = \mathbf{0.114688\text{ MB/tok}})$$
   - *Role of GQA ($n_{kv}=8$ vs. $n_q=24$):* GQA shares each KV head across $24 / 8 = 3$ query heads, reducing KV memory by **$3\times$** compared to standard Multi-Head Attention ($2 \times 28 \times 24 \times 128 \times 2 = 344,064\text{ B/tok} = 336.0\text{ KiB/tok}$).
2. **Idealized Available KV-Cache Pool ($M_{\text{KV}}$):**
   - Usable GPU memory: $24.0\text{ GB} \times 0.92 = \mathbf{22.08\text{ GB}}$
   - Less `fp16` model weights: $4.2 \times 10^9 \times 2\text{ B} = \mathbf{8.40\text{ GB}}$ (`7.8231 GiB`)
   - Less non-KV runtime overhead: $\mathbf{1.60\text{ GB}}$
   - **Idealized KV Pool:** $M_{\text{KV}} = 22.08 - 8.40 - 1.60 = \mathbf{12.08\text{ GB}}$ (`12,080,000,000 B` decimal $\approx \mathbf{11.2504\text{ GiB}}$ binary), holding at most $\lfloor 12,080,000,000 / 114,688 \rfloor = \mathbf{105,329\text{ tokens}}$ (or `113,096` tokens if `12.08` were binary `GiB`).
3. **Per-Sequence KV Footprint & Idealized Maximum Concurrency ($B_{\max,\text{ideal}}$):**
   - **Short sequence ($S = 768$ tokens):** $768 \times 114,688\text{ B} = \mathbf{88,080,384\text{ B}}$ (`88.08 MB` dec / `84.00 MiB` bin) $\implies B_{\max,\text{ideal}}(768) = \lfloor 12,080,000,000 / 88,080,384 \rfloor = \mathbf{137\text{ sequences}}$.
   - **Long sequence ($S = 4096$ tokens):** $4096 \times 114,688\text{ B} = \mathbf{469,762,048\text{ B}}$ (`469.76 MB` dec / `448.00 MiB` = `0.4375 GiB` bin) $\implies$
     $$B_{\max,\text{ideal}}(4096) = \left\lfloor \frac{12,080,000,000}{469,762,048} \right\rfloor = \lfloor 25.7151 \rfloor = \mathbf{25\text{ sequences}}$$
   - *Qualification:* `25` sequences is the **idealized capacity estimate** assuming zero PagedAttention block-allocator fragmentation and exact `1.60 GB` non-KV overhead; allocator metadata or runtime overhead $> 1.60\text{ GB}$ can reduce actual capacity (e.g., to `24` sequences if overhead is `2.0–2.4 GB`).
   - *Empirical Check Against `bench_log.csv`:*
     - At **$B = 24$ (`Row 11`)**: $24 \times 0.469762\text{ GB} = 11.2743\text{ GB} \implies 11.2743 / 12.08 = \mathbf{0.9333}$ (matches logged **`kv_cache_util = 0.93`**, **`preempted_seqs = 0`**).
     - At **$B = 25$ (idealized cap)**: $25 \times 0.469762\text{ GB} = 11.7441\text{ GB} \implies 11.7441 / 12.08 = \mathbf{0.9722}$ (rounds to **`0.97`**, the exact `kv_cache_util` ceiling logged in `Row 12` [$B=32$] and `Row 13` [$B=48$]!).
     - At **$B = 32$ (`Row 12`) & $B = 48$ (`Row 13`)**: Excess sequences over the 25-sequence cap are $32 - 25 = \mathbf{7}$ and $48 - 25 = \mathbf{23}$, matching logged **`preempted_seqs = 7`** and **`preempted_seqs = 23`**.

### 3. Total-Token Throughput vs. Generated-Token Throughput & Benchmark Findings (`B2`, `B3`)
1. **Why `REPORT_v0.md` Section 2 Was Misled by `reported_tok_s` (`B3`):**
   - Across all 13 rows, `reported_tok_s` equals **Total-Token Throughput**:
     $$\texttt{reported\_tok\_s} = \frac{\texttt{num\_requests} \times (\texttt{prompt\_len} + \texttt{gen\_len})}{\texttt{wall\_clock\_s}} = \text{Generated-Output Goodput} \times \left(1 + \frac{\texttt{prompt\_len}}{\texttt{gen\_len}}\right)$$
   - Because `reported_tok_s` includes prompt prefill tokens in the numerator, it multiplies true output generation rate by **`3.0×`** on `512/256` (`Rows 1–7`) vs. **`8.0×`** on `3584/512` (`Rows 8–13`).
   - Comparing `Row 5` (`B=16, p=512, g=256`) vs. `Row 10` (`B=16, p=3584, g=512`): although `reported_tok_s` is higher on `Row 10` (`1311.4` vs. `883.2 total tok/s`), **observed generated-output-token goodput is `44.33%` lower (`163.94` vs. `294.46 gen tok/s`)** and median decode step latency `itl_ms_p50` is **`+59.74%` slower (`77.20 ms` vs. `48.33 ms`)**. *(Note: Rows 5 and 10 change both `prompt_len` and `gen_len`, so this comparison does not isolate `prompt_len` alone.)*
2. **Two Independent Derivations of Observed Goodput vs. Percentile Decode Estimates for `Row 11` (`B=24, p=3584, g=512`):**
   - **Observed End-to-End Goodput (Way 1 — Direct from request/token counts & wall clock):**
     $$\frac{\texttt{num\_requests} \times \texttt{gen\_len}}{\texttt{wall\_clock\_s}} = \frac{24 \times 512}{61.16\text{ s}} = \frac{12,288}{61.16\text{ s}} = \mathbf{200.92\text{ gen tok/s}} \quad \left(\text{Per-User Goodput} = \frac{512}{61.16\text{ s}} = \mathbf{8.37\text{ gen tok/s/req}}\right)$$
   - **Observed End-to-End Goodput (Way 2 — Algebraic decomposition of `reported_tok_s`):**
     $$\texttt{reported\_tok\_s} \times \frac{\texttt{gen\_len}}{\texttt{prompt\_len} + \texttt{gen\_len}} = 1607.4 \times \frac{512}{4096} = \frac{1607.4}{8} = \mathbf{200.925\text{ gen tok/s}} \ (\mathbf{200.92\text{ gen tok/s}})$$
   - **Distinct Percentile-Based Decode Estimates (Not Observed End-to-End Throughput):**
     - *Decode-Only Rate (excluding `ttft_ms_p50 = 500.5 ms`):* $\frac{12,288}{61.16 - 0.5005\text{ s}} = \mathbf{202.57\text{ gen tok/s}}$ (`8.44 tok/s/req`, assuming all 24 simultaneous requests complete prefill together in `~0.5005 s`).
     - *Instantaneous Median Step Rate (from `itl_ms_p50 = 96.07 ms`):* $\frac{24}{0.09607\text{ s}} = \mathbf{249.82\text{ gen tok/s}}$ (`10.41 tok/s/req`; synthetic p50 request rate $\frac{12,288}{0.5005 + 512 \times 0.09607} = \mathbf{247.30\text{ gen tok/s}}$). Because `itl_ms_p50` is the **median** step latency whereas `wall_clock_s` (`61.16 s`) is governed by the **mean** step duration (`~118.7 ms/step`) and tail completion across all 24 requests (`e2e_ms_p95 = 69,221.3 ms`), percentile-based step rates must not be presented as observed end-to-end throughput.
3. **Long-Context Throughput Cliff at $B = 32$ and $B = 48$ (`B2`):**

| Row | `batch_size` ($B$) | `prompt_len` / `gen_len` | `wall_clock_s` | `reported_tok_s` (Total) | **Observed Output Goodput (`gen tok/s`)** | Unthrottled KV Demand (`GB` / `% Pool`) | Logged `kv_cache_util` (`Calc Cap`) | Logged `preempted_seqs` (`Ideal` $B-25$) | `ttft_ms_p50` / `itl_ms_p50` | `e2e_ms_p95` (`ms`) | Traffic-Model Est. (`GB/s`)\* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5** | `16` | `512 / 256` | `13.91 s` | `883.2` | **`294.46`** | `1.409 GB` (`11.7%`) | `0.12` (`0.1167`) | `0` (`0`) | `71.7` / `48.33` | `15,592.3` | `198.1` |
| **7** | `64` | `512 / 256` | `21.68 s` | `2267.3` | **`755.72`** | `5.637 GB` (`46.7%`) | `0.47` (`0.4667`) | `0` (`0`) | `73.3` / `67.91` | `24,323.7` | `192.9` |
| **8** | `4` | `3584 / 512` | `28.98 s` | `565.4` | **`70.67`** | `1.879 GB` (`15.6%`) | `0.16` (`0.1556`) | `0` (`0`) | `483.2` / `51.33` | `32,673.3` | `198.0` |
| **9** | `8` | `3584 / 512` | `36.30 s` | `902.6` | **`112.84`** | `3.758 GB` (`31.1%`) | `0.31` (`0.3111`) | `0` (`0`) | `519.0` / `62.26` | `39,982.9` | `191.5` |
| **10** | `16` | `3584 / 512` | `49.97 s` | `1311.4` | **`163.94`** | `7.516 GB` (`62.2%`) | `0.62` (`0.6222`) | `0` (`0`) | `498.3` / `77.20` | `54,602.1` | `200.1` |
| **11** | **`24` (Peak)** | `3584 / 512` | `61.16 s` | **`1607.4`** | **`200.92`** | `11.274 GB` (`93.3%`) | **`0.93`** (`0.9333`) | **`0`** (`0`) | `500.5` / `96.07` | `69,221.3` | `197.5` |
| **12** | **`32` (Cliff)** | `3584 / 512` | `94.71 s` | `1384.0` (`-13.9%`) | **`172.99` (`-13.9%`)** | `15.032 GB` (**`124.4%`**) | **`0.97`** (`0.9722`) | **`7`** (`7`) | `636.9` / `101.79` | `97,465.7` | `190.7` |
| **13** | **`48` (Cliff)** | `3584 / 512` | `151.41 s` | `1298.5` (`-19.2%`) | **`162.31` (`-19.2%`)** | `22.549 GB` (**`186.7%`**) | **`0.97`** (`0.9722`) | **`23`** (`23`) | `955.4` / `100.00` | `105,427.5` | `194.1` |

*\*Note: `Traffic-Model Est. (GB/s)` (`190.7–200.1 GB/s`) is an **analytical estimate** derived from $\frac{8.40\text{ GB} + \min(B, 25)\times(p+g/2)\times 114,688\text{ B}}{\texttt{itl\_ms\_p50}/1000}$, **not** hardware-measured bandwidth; confirming actual HBM saturation requires Nsight Compute / DCGM DRAM counters. Similarly, whether `enable_prefix_caching` is disabled is not specified in `model_spec.md`; observed `kv_cache_util` is consistent with the simple linear model ($B \times S$).*

### 4. Practical Capacity Recommendations & Serving Telemetry (`B2`, `B4`)
1. **Immediate Scheduler Guardrail (Zero Hardware/Precision Change):** Cap concurrent long-context ($S=4096$) admission at **`max_num_seqs = 24`** on the `24 GB` L4 so the server operates at its observed peak (`Row 11`: `200.92 gen tok/s`, `kv_cache_util = 0.93`, `preempted_seqs = 0`) instead of thrashing at $B \ge 32$.
2. **Proposed KV-Cache Quantization (`fp8` / `int8`) — *Proposed & Conditional on Engine Support and Validation (Not Yet Tested)*:**
   - **Analytical Prediction:** If the serving engine and L4 Ada Lovelace hardware support **`fp8` (`1` byte/element) KV-cache quantization** without unacceptable quality or latency degradation, per-token KV memory halves from `114,688 B` (`112.0 KiB`) to **`57,344 B` (`56.0 KiB`)** and per-sequence memory at $S=4096$ halves from `469.76 MB` to **`234.88 MB`**. Under the `12.08 GB` idealized pool, this **predicted (untested)** change would double idealized `4096`-token concurrency from **`25` to `51` sequences**, lowering predicted `kv_cache_util` to **`0.62` at $B=32$** (`7.516 GB / 12.08 GB`) and **`0.93` at $B=48$** (`11.274 GB / 12.08 GB`) and reducing predicted `preempted_seqs` from `7` and `23` to **`0`**. Because this has **not been empirically tested**, it must be validated for both Hindi/Kannada generation quality and kernel dequantization latency before production rollout.
3. **Single Serving-Stack Counter to Confirm the B2 Preemption Mechanism (`B4`):**
   - Pull the engine's cumulative preemption and prefill token counters (**`vllm:num_preemptions_total`** paired with **`prefill_tokens_total`** and **`num_total_gpu_blocks`**). Under `recompute` preemption, we expect `0` preemptions and exactly $24 \times 3,584 = \mathbf{86,016\text{ prefill tokens}}$ at $B=24$, jumping to **$\ge 7$ preemptions ($\ge 139,776\text{ prefill tokens}$)** at $B=32$ and **$\ge 23$ preemptions ($\ge 254,464\text{ prefill tokens}$)** at $B=48$ (or if `swap` mode is used, `swap_out_blocks_total` jumping from `0` at $B=24$ to $\approx 3.29\text{ GB}$ at $B=32$ and $\approx 10.80\text{ GB}$ at $B=48$).

---

## D. Part C — Engineering Recommendation (`your-submission/partC/memo.md`)

### 1. Comparison of the Three Candidate Approaches Under Assignment Constraints
Constraints: **`1× NVIDIA A100 80 GB` (R&D)**, **`1× NVIDIA L4 24 GB` (Production Serving)**, **2 weeks implementation + evaluation (launch review in 3 weeks)**, **native-speaker reviewer (`Hindi + Kannada only`, `10 hours/week`)**, and **`$0` external API budget**.

| Evaluation Dimension | **Option 3 (Path c): Concise System-Prompt Engineering ($\Delta p \le 128$ tokens, No Training)**<br>***PRIMARY COMMITTED LAUNCH BASELINE (Stage 1)*** | **Option 1 (Path a): LoRA SFT of `FLM-4B-Instruct` (`4.2B`, Merged into `fp16` Weights)**<br>***GATED WEEK-2 EXPERIMENT (Stage 2)*** | **Option 2 (Path b): Lightweight Inference-Time Rewriter ($\le 1\text{B}$ Parameters)**<br>***REJECTED*** |
| :--- | :--- | :--- | :--- |
| **Why Selected / Rejected** | **Selected as Primary Committed Baseline:** Zero training-data dependency (`$0` API risk), zero weight-regression risk on English (`0%` catastrophic forgetting), executable in Days 2–3, and human-evaluated in Week 1. | **Selected as Gated Week-2 Experiment:** Can adapt weights directly to Kannada agglutinative morphology and conversational register with **zero extra prompt tokens ($\Delta p = 0$)** at serving time once merged into `fp16` weights—*if and only if* clean permissive/internal SFT data is verified by Day 3. | **Rejected:** Violates L4 memory headroom, adds sequential decode latency, and compounds factual/formatting errors. |
| **L4 (`24 GB`) Serving Capacity & Latency Impact (Derived from Part B)** | Adding $\Delta p \le 128$ prompt tokens to short requests ($S = 768 \rightarrow 896$) uses `102.76 MB/seq`, shifting idealized concurrency $B_{\max,\text{ideal}}$ from **`137` to `117` sequences** (safe for $B \le 64$). *(Verbose $\ge 1,024$-token few-shot prompts are prohibited because they cut $B_{\max}$ to $\le 58$.)* | Merging LoRA ($W' = W_0 + \frac{\alpha}{r}BA$ in `fp16`) preserves `8.40 GB` weights, `114,688 B/tok` KV footprint, and **$B_{\max,\text{ideal}}(4096) = 25$ sequences** (`137` at $S=768$), assuming output length (`gen_len`) does not inflate. | Co-locating a `1B` `fp16` model (`~2.0 GB` weights + `~0.5–1.5 GB` KV/activations) shrinks the L4 KV pool from `12.08 GB` to **`~8.6–9.6 GB`**, dropping $B_{\max,\text{ideal}}(4096)$ from **`25` to `18–20` sequences** (causing immediate preemptions at `Row 11` $B=24$) and adding a **second sequential decode pass**. |
| **Data & Schedule Risk (`2 Weeks`, `$0` API)** | **Low:** Requires no SFT corpus; leaves full schedule for evaluation and prompt iteration. | **Moderate-to-High (Mitigated by Day 3 Kill Gate):** Starter kit contains **no SFT dataset** and `$0` API budget prohibits commercial synthetic data generation; relies on auditing internal logs and permissive open datasets (`Apache-2.0` / `CC-BY-4.0`). | **High:** Requires training/validating a rewriter *plus* building dual-model orchestration in 2 weeks. |

### 2. Why Prompt Engineering Is the Primary Committed Baseline and LoRA SFT Is Gated
1. **No Pre-Existing SFT Corpus in the Starter Kit (`$0` API Budget):** Only the 10-line `corpus_sample/` and the single-sentence `FLORES-200` Wikipedia translation corpus exist in the workspace. Committing exclusively to SFT without first verifying $\ge 10,000$ clean, decontaminated, permissively licensed conversational pairs would risk arriving at Week 3 with an overfitted or translationese-degraded checkpoint.
2. **Unmeasured Baseline Failure Modes:** Because `FLM-4B-Instruct` weights are not in the starter kit, we have not yet observed whether its "too formal/textbook" Hindi and Kannada replies are a surface register choice (solvable via a $\le 128$-token system prompt) or a deeper morphological/weight limitation.
3. **Strict Six-Language Scope Rule (`hin, kan` vs. `tam, tel, ben, mar`):** Although the product team requested casualization across 6 languages (`Hindi, Kannada, Tamil, Telugu, Bengali, Marathi`), human reviewer coverage exists **only for Hindi and Kannada**. Because conversational register ("casual vs. textbook") and agglutinative grammar cannot be validated by regexes alone, **Week 3 launch and any SFT promotion are scoped strictly to Hindi and Kannada**; `tam, tel, ben, mar` remain on the base configuration behind a feature flag until native reviewer coverage is allocated.

### 3. Back-of-Envelope Arithmetic, Reviewer Schedule, Kill Gates, and Day-1 Experiment
- **Day-1 First Experiment:** Run `compare_tokenizers.py` with `FLM-4B-Instruct`'s actual `128k` production tokenizer on the `1,012` aligned `FLORES-200` `devtest` sentences (`your-submission/partA/corpus/processed/{eng,hin,kan,tam,tel,mal}.txt`) to measure its true token-count-per-aligned-sentence ratio, and generate a `20`-prompt baseline sample (`Base` vs. draft `Option 3` system prompt) in Hindi and Kannada.
- **Analytical Training Estimate on `1× A100 80 GB` (Explicitly Labeled as Unexecuted Estimate):** For a proposed target mixture of `15,000–30,000` decontaminated examples (`40%` Kannada, `35%` Hindi, `25%` English replay buffer; `2` epochs $\approx 30\text{M}–60\text{M}$ tokens at $S \le 2048$), `bf16`/`fp16` LoRA ($r=16, \alpha=32$) with gradient checkpointing is estimated to use **`~18–26 GB` VRAM** (`8.4 GB` base weights + `~1–2 GB` LoRA AdamW/grad states + `~8–16 GB` activations) and **`~3–6 GPU-hours/run`** at `~3,000–5,000 tok/s`.
- **Golden Evaluation Set & 3-Layer Decontamination (Frozen Day 2):** `240` golden test prompts (`80` Hindi + `80` Kannada + `80` English regression, stratified into 4 buckets $\times$ `20` prompts: native-script QA, multi-step instructions, Romanized Hinglish/Kanglish, and JSON/table/code formatting) plus a disjoint `40` Hindi + `40` Kannada dev split. All candidate SFT data is decontaminated via SHA-256 hash freeze, `NFC` substring match, 13-gram overlap, and 5-gram MinHash Jaccard $\ge 0.70$.
- **Exact Reviewer-Hour Arithmetic ($\le 10.0$ Hours/Week Cap — Verified Both Per-Reviewer and Combined):**
  - Each main round (Week 1 and Week 3) evaluates **`80` unique golden test prompts plus `10` *additional* blind duplicate overlap prompts = `90` total pairwise comparisons per language** at **`3.0 minutes/pair`** (`20` pairs/hour):
    - **Week 1 (Round 1 — `Base` vs. `Option 3`):** `90` pairs $\times 3.0\text{ min} = \mathbf{4.5\text{ h}}$ active rating (`4.0 h` unique + `0.5 h` additional overlap) $+ \mathbf{0.5\text{ h}}$ rubric calibration = **`5.0 hours` Hindi + `5.0 hours` Kannada = `10.0 hours` combined**.
    - **Week 2 (Round 2 — Dev-Split Diagnostic on `40` Dev Prompts/Lang):** `40` pairs $\times 3.0\text{ min} = \mathbf{2.0\text{ hours}}$ **Hindi + `2.0 hours` Kannada = `4.0 hours` combined**.
    - **Week 3 (Round 3 — `Option 3 Baseline` vs. `Locked LoRA SFT`):** `90` pairs $\times 3.0\text{ min} = \mathbf{4.5\text{ h}}$ active rating $+ \mathbf{0.5\text{ h}}$ critical-defect sign-off = **`5.0 hours` Hindi + `5.0 hours` Kannada = `10.0 hours` combined**.
  - Neither individual reviewer (`5.0 h`, `2.0 h`, `5.0 h`) nor the combined total (`10.0 h`, `4.0 h`, `10.0 h`) ever exceeds `10.0 hours/week`.
- **Pre-Registered Proposed Acceptance Gates & Kill Criteria:**
  - **Kill Gate #1 (Day 3 EOD):** Abort Week-2 LoRA SFT if $< 10,000$ clean decontaminated permissive/internal pairs exist locally or if the `128k` tokenizer exhibits $> 5\times$ token inflation vs. English.
  - **Kill Gate #2 (Day 8 EOD / Week 3 Round 3):** Discard SFT and ship **Option 3** if SFT fails to achieve **$\text{Win} + \text{Tie} \ge 65.0\%$ ($\ge 52/80$)** AND **$\text{Win Rate} - \text{Loss Rate} \ge +15.0\text{ pp}$ ($\ge +12/80$ net wins)** over Option 3 in both Hindi and Kannada, or if critical defects exceed `2.5%` (`2/80`), native script purity drops below `98.0%`, `rep-4` loops exceed `1.0%`, English/format compliance drops by `> 1.0%`, P95 `gen_len` inflates by `> +15%`, or L4 `preempted_seqs > 0` at $B=24$ (`5` repeated runs within `±3.0%` of Base `calc_gen_tok_s` and `itl_ms_p50`).

---

## E. Complete Submission Map, Reproducibility Commands, and Status

### 1. Canonical File Inventory (`your-submission/`)

| Path (Relative to `your-submission/`) | Description | Status |
| :--- | :--- | :---: |
| `MENTOR_REPORT.md` | *This mentor-ready technical summary report* | **Complete** |
| `FINAL_AUDIT_REPORT.md` | Comprehensive rubric reconciliation, verification tables, and 10 oral-defense Q&As | **Complete** |
| `NOTEBOOK.md` | 14-entry chronological lab notebook documenting commands, dead ends, and audit fixes | **Complete** |
| `AI_USAGE.md` | Honest disclosure of where AI assisted and 5 specific places where AI output was corrected | **Complete** |
| `partA/memo.md` | Canonical $\le 1$-page Part A4 executive memo (`41` lines; mirrored at `partA/a4_recommendation_memo.md`) | **Complete** |
| `partA/build_corpus.py`, `partA/corpus_documentation.md`, `partA/corpus/` | Reproducible `FLORES-200` `devtest` pipeline (`1,012` aligned sentences $\times 6$ languages = `6,072` lines) | **Complete** |
| `partA/audit_experiments.py`, `partA/e1_e6_raw_output.txt`, `partA/audit_results.md` | Controlled `E1–E6` single-variable ablation of `fertility.py` and `corpus_sample/` | **Complete** |
| `partA/compare_tokenizers.py`, `partA/a3_tokenizer_comparison.json`, `partA/a3_analysis.md` | Multi-tokenizer (`gpt2`, `cl100k_base`, `o200k_base`) & 5-denominator analysis | **Complete** |
| `partB/partB_report.md`, `partB/capacity_reconciliation.py`, `partB/partB_reconciliation.json`, `partB/partB_calculation_output.txt` | Direct `B1–B4` answers, KV-cache & goodput derivations, and 13-row reconciliation of `bench_log.csv` | **Complete** |
| `partC/memo.md` | Engineering recommendation memo with 5 explicit rubric labels, option trade-offs, and 3-week plan | **Complete** |

### 2. Reproducible Verification Commands (From Workspace Root `c:\Users\TANNU\OneDrive\Desktop\starter_kit`)

```powershell
# 1. Part A1: Verify FLORES-200 corpus integrity and NFC/whitespace normalization summary
.\.venv-x64\Scripts\python.exe your-submission/partA/build_corpus.py

# 2. Part A2: Run controlled E1-E6 single-variable ablations on fertility.py & corpus_sample/
.\.venv-x64\Scripts\python.exe your-submission/partA/audit_experiments.py

# 3. Part A3: Run 3-tokenizer (gpt2, cl100k_base, o200k_base) x 5-denominator comparison on 6,072 sentences
.\.venv-x64\Scripts\python.exe your-submission/partA/compare_tokenizers.py

# 4. Part B: Run KV-cache capacity and 13-row throughput reconciliation on bench/model_spec.md & bench/bench_log.csv
.\.venv-x64\Scripts\python.exe your-submission/partB/capacity_reconciliation.py
```

### 3. Current Readiness Status
- **Starter-Kit Preservation:** All 6 original starter-kit files (`fertility.py`, `REPORT_v0.md`, `corpus_sample/eng_sample.txt`, `corpus_sample/hin_sample.txt`, `bench/bench_log.csv`, `bench/model_spec.md`) are verified unmodified (`2026-10-22 23:34:43`).
- **Cross-Document Numerical Consistency:** Every number in `MENTOR_REPORT.md`, `FINAL_AUDIT_REPORT.md`, `NOTEBOOK.md`, `partA/memo.md`, `partA/a3_analysis.md`, `partB/partB_report.md`, and `partC/memo.md` is reconciled directly against `a3_tokenizer_comparison.json`, `e1_e6_raw_output.txt`, and `partB_reconciliation.json`.
- **Submission Status:** **`READY TO SUBMIT`**.
