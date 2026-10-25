# AI Usage Disclosure (`AI_USAGE.md`)

**Assignment:** AI Team Intern Assignment — *The Audit*  
**Canonical Submission Path:** `my-submission/AI_USAGE.md`

---

## 1. Where AI Tools Helped

We used an AI coding assistant (pair-programming agent) across Parts A, B, and C for the following tasks:

1. **Experiment Scripting & Data Pipeline Automation (Part A):**
   - Generated the boilerplate CLI and tabular reporting structure for `my-submission/partA/audit_experiments.py` (testing single-variable perturbations `E1–E6` on `fertility.py`), `my-submission/partA/build_corpus.py` (downloading and extracting `flores200_dataset.tar.gz`, verifying SHA-256 checksums, applying `NFC` and whitespace normalization, and computing corpus statistics), and `my-submission/partA/compare_tokenizers.py` (running `tiktoken` `gpt2`, `cl100k_base`, and `o200k_base` across five denominators: whitespace words, Unicode codepoints, extended grapheme clusters `\X`, UTF-8 bytes, and aligned sentences).
2. **Capacity & Throughput Reconciliation Scripting (Part B):**
   - Wrote `my-submission/partB/capacity_reconciliation.py` to parse `bench/model_spec.md` and `bench/bench_log.csv`, compute per-token and per-sequence GQA KV-cache memory in both decimal (`GB`/`MB`) and binary (`GiB`/`MiB`/`KiB`) units, reconcile `reported_tok_s` against total-token throughput vs. generated-output-token throughput, and generate the sensitivity tables in `my-submission/partB/partB_reconciliation.json`.
3. **Document Structuring & Arithmetic Cross-Checking (Parts A, B, C):**
   - Assisted in formatting Markdown comparison tables, checking floating-point vs. rounded consistency across JSON outputs and prose reports (`a3_analysis.md`, `partA/memo.md`, `partB_report.md`, `partC/memo.md`), and maintaining chronological entries in `my-submission/NOTEBOOK.md`.

---

## 2. Where AI Initially Misled Us or Required Human Audit & Correction

In alignment with the assignment's ground rules (*"What we're checking is whether you know which parts of your submission you understand"*), several initial AI-generated assumptions and framings were caught and corrected during our validation passes:

1. **Part A1/A2 — Assuming `unicodedata.normalize("NFC")` Was a No-Op Everywhere Because It Had Zero Effect on `corpus_sample/`:**
   - *Initial AI Misstep:* Because Experiment `E6b` showed `0/10` lines changed under `NFC` on `corpus_sample/hin_sample.txt`, an early draft treated `NFC` normalization as practically inert for Devanagari corpora.
   - *Correction (`NOTEBOOK.md` Entry 4):* When we inspected the raw FLORES-200 `devtest` files in `build_corpus.py`, we discovered that **`93` of `1,012` lines in `hin_Deva.devtest`** (`+122` codepoints) and **`14` lines across the four Dravidian languages** were **not** in `NFC`! Specifically, raw `hin_Deva.devtest` contained precomposed Devanagari nukta singletons (`U+095B` `ज़`, `U+095C` `ड़`), which are Unicode *Canonical Composition Exclusions* that decompose into two codepoints (`base + U+093C`) under `NFC`.
2. **Part A3/A4 — Over-Claiming Tokenizer Library Independence and Serving-Cost Multipliers:**
   - *Initial AI Misstep:* Early prose drafts referred to the `1.57×` (Hindi) and `1.92×–1.98×` (Dravidian) `o200k_base` figures as "serving-cost multipliers", implied `FLM-4B-Instruct`'s unmeasured `128k` tokenizer would resemble `o200k_base`, and suggested unified deployment was proven sufficient.
   - *Correction (`NOTEBOOK.md` Entries 6–8):* We explicitly disclosed that all three tested encodings (`gpt2`, `cl100k_base`, `o200k_base`) come from a single library (`tiktoken 0.14.0`) because `transformers` (`hf:xlm-roberta-base`) is not installed in `.venv-x64`; relabeled `1.57×–1.98×` strictly as **static token-count-per-aligned-sentence ratios**; removed any assumption about `FLM-4B-Instruct`'s `128k` tokenizer; and clarified that while `gpt2` fertility alone cannot justify separate Indic routing, static subword compression alone does **not** prove model quality, output conciseness, or end-to-end latency.
3. **Part B — Treating Analytical HBM Traffic Estimates (`190.7–200.1 GB/s`) and Prefix Caching as Proven Hardware/Runtime Facts:**
   - *Initial AI Misstep:* The first draft of `partB_report.md` presented the `190.7–200.1 GB/s` decode bandwidth calculation as direct proof of HBM saturation, claimed `enable_prefix_caching` was "proven disabled", and did not note that comparing `Row 5` (`p=512, g=256`) with `Row 10` (`p=3584, g=512`) changes *both* `prompt_len` and `gen_len`.
   - *Correction (`NOTEBOOK.md` Entry 10):* We reclassified `190.7–200.1 GB/s` as an **analytical traffic-model estimate** (`(weights + midpoint_active_KV) / itl_ms_p50`) requiring Nsight Compute / DCGM DRAM counters to verify; changed the prefix-caching conclusion to *"not specified in `model_spec.md`; observed `kv_cache_util` is consistent with the simple linear model ($B \times S$)"*; qualified `25` sequences as an **idealized capacity estimate** assuming zero block-allocator fragmentation; and noted the `(prompt_len, gen_len)` confounder between Rows 5 and 10.
4. **Part C — Defaulting to Unconditioned SFT Without Establishing Training-Data Availability:**
   - *Initial AI Misstep:* The first draft of `partC/memo.md` recommended LoRA SFT as the unconditioned primary path and cited `30,000` training examples (`40% kan / 35% hin / 25% eng`), `18–26 GB` VRAM, and `4–6 GPU-hours` without highlighting that **no SFT corpus exists in the starter kit** and external APIs are prohibited (`$0` budget).
   - *Correction (`NOTEBOOK.md` Entry 12):* We restructured the recommendation so **Option 3 (Concise Prompt Engineering, $\Delta p \le 128$ tokens)** is the **Primary Committed Launch Baseline (Week 1)** and **Option 1 (Merged `fp16` LoRA SFT)** is a **Gated Week-2 Experimental Upgrade** contingent on passing a Day 3 Data & Tokenizer Kill Gate ($\ge 10,000$ clean decontaminated permissive/internal pairs), while explicitly labeling all training-VRAM, GPU-hour, and data-volume figures as back-of-envelope estimates.
5. **Final Audit — Relying on Automated Context-Truncation Summary Instead of Re-Reading Source Files:**
   - *Initial AI Misstep:* After a long-conversation context compaction, the first draft of `FINAL_AUDIT_REPORT.md` copied corrupted values from the auto-generated context summary (`32` layers, `8.20 GB` weights, `1.80 GB` overhead, `1.928` Hindi `o200k_base` tok/word, and a `\w+` regex description of `fertility.py`) even though the actual scripts and reports on disk (`capacity_reconciliation.py`, `partB_reconciliation.json`, `partB_report.md`, `a3_tokenizer_comparison.json`, `a3_analysis.md`, `partA/memo.md`) contained the exact source values.
   - *Correction (`NOTEBOOK.md` Entry 14):* We re-inspected `bench/model_spec.md` (`28` layers, `n_q=24`, `n_kv=8`, `8.40 GB` weights, `1.60 GB` overhead), `bench/bench_log.csv` (`kv_cache_util = 0.97` at $B=32, 48$), and `fertility.py` (`line.split(" ")`, `line.lower()`, `len(line)`), re-executed `audit_experiments.py`, `compare_tokenizers.py`, and `capacity_reconciliation.py`, and updated `FINAL_AUDIT_REPORT.md` and `partB_report.md` (§0.B3) so every number matches the canonical source files and outputs.
