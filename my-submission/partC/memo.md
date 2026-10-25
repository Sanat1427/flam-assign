# Part C — Engineering Recommendation Memo: Improving Hindi & Kannada Performance (Audited)

**To:** Engineering Leadership & Launch Review Committee  
**From:** AI Team Intern (The Audit)  
**Canonical Path:** `your-submission/partC/memo.md` *(mirrored identically at `my-submission/partC/memo.md`)*  
**Supporting Audit Evidence:** `REPORT_v0.md`, `bench/model_spec.md`, `bench/bench_log.csv`, `your-submission/partA/{memo.md, a3_analysis.md}`, `your-submission/partB/{partB_report.md, partB_reconciliation.json}`

---

## 1. Executive Decision & Risk-Adjusted Recommendation

**Primary Recommendation (Staged, Evidence-Gated Decision):**
1. **Primary Committed Launch Approach (Stage 1 — Week 1 Deliverable):** Implement and validate **Option 3 (Path c) — Concise System-Prompt Engineering ($\Delta p \le 128$ tokens, no model weight changes)** on `FLM-4B-Instruct` (`4.2B`) as our **committed primary launch baseline** for shifting assistant replies from formal/textbook prose to a natural, conversational register.
2. **Gated Experimental Upgrade (Stage 2 — Week 2 Experiment):** Evaluate **Option 1 (Path a) — Parameter-Efficient Supervised Fine-Tuning (LoRA SFT, merged into `fp16` weights) of `FLM-4B-Instruct`** on our single **NVIDIA A100 80 GB GPU** during Week 2 **only if** the Week 1 Day 3 **Data & Tokenizer Feasibility Gate** passes, and promote the merged SFT checkpoint over Option 3 at the Week 3 Launch Review **only if** it clears all pre-registered human-preference, English non-regression, and L4 serving-capacity gates. *(Scope Rule: Because human reviewer coverage exists **only for Hindi and Kannada**, any launch or SFT promotion at Week 3 must be scoped strictly to evaluated languages [Hindi & Kannada]; unreviewed languages [Tamil, Telugu, Bengali, Marathi] must not be fine-tuned or launched blindly without native reviewer coverage.)*
3. **Rejected Approach:** **Reject Option 2 (Path b — $\le 1\text{B}$ Lightweight Inference-Time Rewriter Model).**

**Quick-Reference Rubric Summary (Explicitly Labelled Elements):**
- **Explicit Assumptions:** See Section 2 & Section 6 (no SFT dataset or `FLM-4B-Instruct` tokenizer/weights exist in the starter kit; training VRAM/time and data volume are unexecuted estimates; L4 serving parameters come from `bench/model_spec.md`).
- **Back-of-Envelope Arithmetic (Data Volume, Training/Serving Cost, Reviewer Throughput):**
  - *Data Volume (Proposed Target for Gate #1):* Minimum `10,000` (target `15,000–30,000`) decontaminated instruction/conversational pairs (`40%` Kannada, `35%` Hindi, `25%` English replay) from permissive open datasets/internal logs (`$0` API cost).
  - *Training Cost (Analytical Estimate on `1× A100 80 GB`):* `~18–26 GB` VRAM (`8.4 GB` `bf16`/`fp16` base weights + `~1–2 GB` LoRA $r=16$ states + `~8–16 GB` checkpointed activations at $S \le 2048$) and **`~3–6 GPU-hours/run`** (`30M–60M` tokens at `~3k–5k tok/s`).
  - *Serving Cost on `1× L4 24 GB` (`bench/model_spec.md`):* Base/merged SFT uses `8.40 GB` weights, leaving a `12.08 GB` (`11.2504 GiB`) idealized KV pool ($B_{\max,\text{ideal}}(4096) = 25$ seqs, $B_{\max,\text{ideal}}(768) = 137$ seqs). Option 3 ($\Delta p \le 128$ prompt tokens) shifts short-request length $S = 768 \rightarrow 896$ ($B_{\max,\text{ideal}} = 117$ seqs, `+16.7%` KV/seq). Option 2 ($\le 1\text{B}$ rewriter, `~2.5–3.5 GB` VRAM) shrinks the L4 KV pool to `~8.6–9.6 GB` ($B_{\max,\text{ideal}}(4096)$ drops to `18–20` seqs) and adds a second sequential decode pass.
  - *Reviewer Throughput (`<= 10.0 h/week` cap):* At **`3.0 min/pair`** (`20` pairs/hour), `80` unique golden prompts + `10` additional overlap re-ratings = **`90` total ratings/language** (`4.5 h` active rating + `0.5 h` calibration = **`5.0 h/week/language`**; see Section 5.2).
- **Success Metric with Numeric Threshold (Pre-Registered Proposed Gate):** On the `80` Hindi and `80` Kannada golden test prompts, blinded human pairwise evaluation must achieve **$\text{Win} + \text{Tie} \ge 65.0\%$ ($\ge 52/80$)** AND **$\text{Win Rate} - \text{Loss Rate} \ge +15.0\text{ percentage points}$ ($\ge +12$ net wins/80)** with critical defect rate **$\le 2.5\%$ ($\le 2/80$)**, native script purity **$\ge 98.0\%$**, `rep-4` loop rate **$\le 1.0\%$**, English/format regression **$\le 1.0\%$**, and L4 `preempted_seqs == 0` at $B=24$ (`calc_gen_tok_s` and `itl_ms_p50` within `±3.0%`).
- **Kill Criterion (Observation & Deadline to Abandon Path):**
  - *Kill Gate #1 (Day 3 EOD):* Abandon the Week-2 LoRA SFT experiment immediately if $< 10,000$ clean, decontaminated, permissively licensed conversational/instruction pairs exist locally (`$0` API budget) or if `FLM-4B-Instruct`'s `128k` tokenizer exhibits severe UTF-8 byte fallback (`> 5×` token-count-per-aligned-sentence ratio vs. English).
  - *Kill Gate #2 (Day 8 EOD / Week 3 Round 3):* Abandon the SFT checkpoint and ship **Option 3 (Concise System Prompt)** if SFT degrades English accuracy/formatting by `> 1.0%`, inflates P95 output token length by `> +15%`, or fails to beat Option 3 by `>= +15.0 pp` net win margin in Week 3 blinded human review.
- **First Experiment to Run on Day 1:** Run `your-submission/partA/compare_tokenizers.py` with `FLM-4B-Instruct`'s actual `128k` production tokenizer on the `1,012` aligned FLORES-200 `devtest` sentences (`your-submission/partA/corpus/processed/{eng,hin,kan,tam,tel,mal}.txt`) to measure its exact token-count-per-aligned-sentence ratio and generate a `20`-prompt baseline sample (`Base` vs. `Option 3` draft prompt) in Hindi and Kannada to diagnose whether "too formal/textbook" outputs are solvable via system-prompt steering or stem from base weight limitations.

**Why This Decision Is Justified by the Actual Evidence and Constraints (Rather Than Defaulting Blindly to SFT):**
- **Distinguishing Recommendations from Validated Outcomes:** No prompt engineering, SFT run, or Hindi/Kannada generation quality benchmark has been executed on `FLM-4B-Instruct` in this workspace.
- **Why Unconditioned SFT Cannot Be Assumed as the Sole Primary Path:** While LoRA SFT theoretically offers a higher quality ceiling for agglutinative Kannada morphology and Hindi instruction following without adding prompt tokens at inference time, **three prerequisites for SFT are currently unestablished in the supplied files**: (a) **no Hindi/Kannada instruction-tuning dataset exists in the starter kit** (only the 10-line `corpus_sample/` and the `FLORES-200` Wikipedia single-sentence translation corpus built in Part A1 exist locally), and we have a **`$0` external API budget** (ruling out synthetic data generation via commercial LLMs); (b) `FLM-4B-Instruct`'s actual `128k` tokenizer files and base Hindi/Kannada failure modes are unmeasured; and (c) we have only **two weeks** for implementation and evaluation with **10 reviewer hours/week**.
- **Why Concise Prompt Engineering (Option 3) as Primary Launch Path + Gated LoRA SFT (Option 1) as Week-2 Experiment Minimizes Launch Risk:**
  - **Option 3 ($\Delta p \le 128$ tokens)** requires zero unverified training corpora, preserves `FLM-4B-Instruct`'s exact English weights (`0%` catastrophic forgetting risk), can be implemented and human-evaluated in Week 1, and bounds prompt-token overhead on our production `1× NVIDIA L4 (24 GB)` serving GPU.
  - **Option 1 (LoRA SFT merged to `fp16`)** is the right **Week-2 experimental upgrade** because—if clean permissive/internal SFT data is verified in Week 1 and if base weights struggle with Kannada morphology or output verbosity—merging a LoRA adapter into `FLM-4B-Instruct`'s `8.40 GB` `fp16` weights improves generation without adding permanent system-prompt tokens ($\Delta p = 0$) or a second model hop.

---

## 2. Epistemic Audit: Verified Facts vs. Reclassified Estimates & Assumptions

| Claim / Parameter | Audit Status | Verified Evidence or Explicit Assumption / Qualification |
| :--- | :--- | :--- |
| **Part A3 Tokenizer Compression (`tiktoken`)** | **Verified Calculation (`a3_analysis.md`)** | On `1,012` aligned FLORES-200 `devtest` sentences, `o200k_base` reduces total tokens vs. `gpt2` by **`-78.95%` (`4.75×` less) in Hindi** and **`-85.69%` (`6.99×` less) in Kannada**, yielding **token-count-per-aligned-sentence ratios vs. English** of **`1.57×` (`hin`) and `1.95×` (`kan`)** (vs. `4.77×` and `8.84×` on `cl100k_base`). **Caveat:** Static `tiktoken` compression does *not* prove generation quality or lower latency, and `FLM-4B-Instruct`'s actual `128k` tokenizer was not tested. |
| **Part B Row 5 vs. Row 10 Output Throughput (`-44.33%`) & ITL (`+59.74%`)** | **Verified Calculation with Confounder Qualification (`bench_log.csv`)** | At `batch_size = 16`, generated-output-token throughput ($\frac{\texttt{num\_requests} \times \texttt{gen\_len}}{\texttt{wall\_clock\_s}}$) is **`294.46 gen tok/s`** (`4,096 / 13.91 s`, `itl_ms_p50 = 48.33 ms`) in Row 5 (`p=512, g=256`) vs. **`163.94 gen tok/s`** (`8,192 / 49.97 s`, `itl_ms_p50 = 77.20 ms`) in Row 10 (`p=3584, g=512`)—a **`-44.33%`** difference. **Critical Qualification:** Rows 5 and 10 change *both* `prompt_len` (`512` $\rightarrow$ `3584`) and `gen_len` (`256` $\rightarrow$ `512`), so this comparison **does not prove that longer prompts alone caused the `-44.33%` drop**. |
| **Production Serving VRAM (`1× L4 24 GB`) vs. Training VRAM (`1× A100 80 GB`)** | **Verified Spec (`model_spec.md`) vs. Analytical Estimate** | **Production L4 (`24 GB`):** `24.0 * 0.92 = 22.08 GB` usable, `8.40 GB` `fp16` weights (`4.2B * 2 B`), `~1.60 GB` assumed non-KV overhead $\implies$ **`12.08 GB` (`11.2504 GiB`) idealized KV pool**, supporting an **idealized capacity estimate of $B_{\max,\text{ideal}}(4096) = 25$ sequences** (`137` at $S=768$; allocator/block overhead may reduce actual capacity). **Training A100 (`80 GB`):** Distinct R&D hardware specified by Part C constraints. |
| **Merged LoRA SFT Inference Memory Impact** | **Qualified Engineering Property** | Merging a LoRA adapter ($W' = W_0 + \frac{\alpha}{r}BA$ in `fp16`) preserves the base model's `4.2B` parameter count (`8.40 GB` weights) and `114,688 B/tok` (`112.0 KiB/tok`) KV-cache formula—adding no *extra* weight memory relative to Base—**provided** the base model fits, vocabulary size (`128k`) is unchanged, and the merged checkpoint is empirically tested on L4 to confirm output length (`gen_len`) does not inflate KV utilization. |
| **LoRA SFT Training VRAM (`~18–26 GB`) & Run Time (`~3–6 GPU-hrs`)** | **Reclassified as Unexecuted Analytical Estimate** | Not measured by a training run. Estimated for `1× A100 80 GB` assuming: `4.2B` `bf16`/`fp16` base weights (`8.4 GB`), LoRA rank $r=16$ (`~40–80M` trainable params, `~1–2 GB` FP32 master weights + AdamW states + gradients), **gradient checkpointing enabled**, sequence length $S \le 2048$, micro-batch size `2–4`, gradient accumulation `8–16` (`~8–16 GB` activations/workspace), and `~3,000–5,000 tok/s` effective training throughput over `30M–60M` total training tokens (`15k–30k` examples $\times$ `2` epochs). |
| **SFT Training Corpus (`15,000–30,000` Examples, `40% kan / 35% hin / 25% eng`)** | **Reclassified as Proposed Target (Not Confirmed Existing Data)** | **No SFT dataset exists in the starter kit.** This mixture is a **proposed target** contingent on auditing internal logs and open permissively licensed Indic datasets (`Apache-2.0` / `CC-BY-4.0`) on Days 1–3 without paid APIs. |

---

## 3. Structured Comparison of the Three Engineering Options

| Dimension | **Option 3: Concise Prompt Engineering ($\Delta p \le 128$ tok, No Training) *(PRIMARY COMMITTED LAUNCH PATH)*** | **Option 1: LoRA SFT of `FLM-4B-Instruct` (`4.2B`, Merged `fp16`) *(GATED WEEK-2 EXPERIMENT)*** | **Option 2: Lightweight Rewriter Model ($\le 1\text{B}$ Parameters) *(REJECTED)*** |
| :--- | :--- | :--- | :--- |
| **Expected Hindi & Kannada Quality Impact** | **Low to Moderate *(Hypothesis)*:** Steers language/script selection, tone, and conciseness if `FLM-4B-Instruct` base weights already possess Hindi/Kannada competence; cannot fix deep morphological gaps in weights. | **Moderate to High *(Hypothesis)*:** Can adapt weights directly to Kannada agglutinative suffixes/sandhi, Hindi postpositions, and concise output—*if* high-quality decontaminated SFT data is available. | **Low / Negative (High Error-Compounding Risk):** A $\le 1\text{B}$ model has weaker reasoning, entity fidelity, and Kannada grammar than the `4.2B` base model; rewriting risks corrupting numbers, entities, code, and JSON. |
| **Data & Schedule Risk (2-Week Window, `$0` API)** | **Lowest Risk:** Zero training data required; implemented in Days 2–3; leaves full time for offline and human evaluation. | **Moderate-to-High Data Risk:** Requires curating and decontaminating `10k–30k` permissive/internal examples with `$0` API budget; mitigated by making SFT a gated Week-2 experiment. | **Highest Schedule & System Risk:** Requires training/validating a $\le 1\text{B}$ rewriter *plus* building and load-testing a dual-model serving pipeline in 2 weeks. |
| **Fit with `1× A100 80 GB` (R&D) & `1× L4 24 GB` (Production Serving)** | **High (If Concise, $\Delta p \le 128$ tok):** Uses A100 for batch evaluation. On L4 (`12.08 GB` idealized KV pool), adding $\Delta p = 128$ tok to $S=768$ ($S'=896$) shifts $B_{\max,\text{ideal}}$ from `137` to `117` seqs (safe for $B \le 64$). *(Verbose $\ge 1,024$-tok few-shot prompts are avoided as they cut $B_{\max}$ to $\le 58$ and eat context headroom.)* | **High (If Merged to `fp16`):** Training fits on `1× A100 80 GB` (`~18–26 GB` estimated VRAM with grad checkpointing). Merged `fp16` checkpoint keeps `8.40 GB` weights and $\Delta p = 0$ extra prompt tokens on the L4. | **Poor on L4 (`24 GB`):** Co-locating a `1B` `fp16` model (`~2.0 GB` weights + `~0.5–1.5 GB` KV/activations) cuts the L4 KV pool from `12.08 GB` to `~8.6–9.6 GB` (dropping $B_{\max,\text{ideal}}(4096)$ from `25` to `18–20` seqs) and adds a **second sequential decode pass**. |
| **Regression & Rollback Profile** | **Zero Weight Regression / Instant Rollback:** English requests bypass the Indic system prompt entirely; rollback is a config change (`< 1 min`). | **Manageable Weight Regression Risk:** SFT can degrade English or formatting if unbalanced; requires a 25% English replay buffer, hard non-regression gates, and weight-pointer rollback (`< 5 min`) to Option 3. | **High Operational Risk:** Dual-model orchestration timeouts and factual drift between `FLM-4B` draft and $\le 1\text{B}$ rewrite. |
| **Evidence Required Before Launch** | Blinded human evaluation (`Base` vs. `Option 3`), offline script/format/repetition checks, and L4 load test with $\Delta p$ tokens included. | All Option 3 checks *plus* Week 1 data audit pass, blinded human superiority over Option 3 (`Win - Loss >= +15 pp`), English non-regression, and L4 load test on merged weights. | Dual-model L4 latency/preemption test, factual fidelity audit (`FLM-4B` vs. rewritten output), and blinded human evaluation. |

---

## 4. Executable Three-Week Plan & Stop Conditions

### 4.1 Week 1 — Baseline Audit, Golden Eval Freeze, Option 3 Primary Baseline, and SFT Data Gate
- **Days 1–2 (Tokenizer Audit & Golden Evaluation Set Freeze):**
  1. **Audit `FLM-4B-Instruct`'s Actual `128k` Tokenizer:** Run `your-submission/partA/compare_tokenizers.py` on `your-submission/partA/corpus/processed/{eng,hin,kan}.txt` (`1,012` aligned FLORES-200 sentences) to measure `FLM-4B-Instruct`'s exact token-count-per-aligned-sentence ratio for Hindi and Kannada.
  2. **Freeze the Representative Evaluation Set (`240` Total Prompts: `80` Hindi + `80` Kannada + `80` English Regression):** Commit SHA-256 hashes of all `240` golden test prompts (plus a separate `40` Hindi + `40` Kannada dev split for Week 2 tuning) to Git *before* inspecting any training data (Section 5.1).
- **Day 3 (Option 3 Implementation & SFT Feasibility Gate):**
  1. **Implement Option 3 (Primary Launch Baseline):** Create language-routed concise system prompts ($\Delta p \le 128$ tokens) for Hindi and Kannada enforcing target script fidelity (native Devanagari/Kannada vs. Romanized Hinglish/Kanglish matching the user's query), concise non-repetitive generation, and strict preservation of code/JSON/numbers.
  2. **Execute Early Stop Gate #1 (SFT Data & Compute Feasibility Gate):** Audit local/internal logs and permissively licensed open Indic instruction datasets (`Apache-2.0` / `CC-BY-4.0`) and run a 100-step smoke test of the `bf16` LoRA training script on `1× A100 80 GB`.
     - **Early Stop Condition #1:** If we cannot verify **$\ge 10,000$ clean, decontaminated, permissively licensed instruction pairs** (`>= 3,500` Kannada, `>= 3,500` Hindi, `>= 2,500` English) by Day 3 EOD, or if the training stack fails on the A100 environment, **cancel the Week 2 LoRA SFT experiment immediately** and dedicate Week 2 to iterative refinement and validation of **Option 3 (Prompt Engineering)**.
- **Days 4–5 (Week 1 Offline & Human Evaluation of Option 3 — `10.0` Combined Reviewer Hours):**
  1. Run offline metrics (Section 5.3) on `Base FLM-4B` and `Option 3` across the `240` golden prompts.
  2. Conduct **Round 1 Blinded Human Evaluation (`Base FLM-4B` vs. `Option 3`)** on all `80` Hindi and `80` Kannada golden test prompts (**`5.0 hours` Hindi + `5.0 hours` Kannada = `10.0 hours` combined**; Section 5.2).

### 4.2 Week 2 — Gated LoRA SFT Experiment (Or Option 3 Refinement if Stopped Early)
- **Days 6–8 (Decontaminated LoRA SFT Training on `1× A100 80 GB`):**
  1. *Target Mixture (Proposed Target, Contingent on Gate #1):* Up to `15,000–30,000` decontaminated examples (`40%` Kannada, `35%` Hindi, `25%` English replay buffer).
  2. *Training Config & Estimated Budget:* Train LoRA (`r=16, alpha=32`, linear layers, `bf16`/`fp16` with gradient checkpointing, $S \le 2048$; estimated `~18–26 GB` VRAM and `~3–6 GPU-hours/run`), merge adapter into `fp16` weights (`8.40 GB`), and evaluate on the `40` Hindi + `40` Kannada dev split and `80` English regression prompts.
  3. **Early Stop Condition #2 (Day 8 Offline Gate):** If the merged LoRA SFT checkpoint fails the offline English non-regression gate (`> 1.0%` drop on the `80` English prompts) or exhibits `rep-4` repetition loops `> 1.0%` after two hyperparameter runs, **halt SFT** and lock **Option 3** as the Week 3 launch candidate.
- **Days 9–10 (Week 2 Dev-Split Human Error Analysis — `4.0` Combined Reviewer Hours):**
  - Reviewers score `40` Hindi + `40` Kannada dev-split pairs (`Option 3` vs. `LoRA SFT`, **`2.0 hours` Hindi + `2.0 hours` Kannada = `4.0 hours` combined**, well within the `10.0 hr/wk` cap) to verify that SFT resolves Week 1 failure modes without introducing hallucinations. Lock the final candidate for Week 3 testing by Day 10 EOD.

### 4.3 Week 3 — Final Blinded Evaluation, L4 Serving Verification, and Launch Decision
- **Days 11–12 (Final Blinded Human Review — `10.0` Combined Reviewer Hours):**
  - Reviewers perform the **definitive double-blind pairwise evaluation (`Option 3 Primary Baseline` vs. `Locked LoRA SFT Candidate`)** on the held-out **`80` Hindi and `80` Kannada golden test prompts** (**`5.0 hours` Hindi + `5.0 hours` Kannada = `10.0 hours` combined**; Section 5.2).
- **Days 13–14 (Repeated-Run L4 Capacity Verification & Rollback Drill):**
  - Execute the **5-run repeated L4 load-test verification (`batch_size = 16` at `512/256` and `batch_size = 24` at `3584/512`)** (Section 6.1) and verify `< 5 minute` config/weight rollback.
- **Day 15 (Launch Review Decision):**
  - **Promote Merged LoRA SFT** if and only if it passes all pre-registered gates against Option 3.
  - **Otherwise Launch Option 3 (Concise System-Prompt Baseline)** if Option 3 passes its Week 1 gates over Base `FLM-4B`.

---

## 5. Evaluation Protocol, Sampling, Reviewer Arithmetic, and Leakage Prevention

### 5.1 Sampling the Representative Golden Evaluation Set & Preventing Data Leakage
- **Fixed Golden Test Set (`240` Prompts Total = `80` Hindi + `80` Kannada + `80` English Regression, Frozen Day 2):**
  - Each Indic language (`80` prompts) is stratified into **4 buckets $\times$ `20` prompts** (plus a disjoint **Dev Split of `40` prompts/language = `10` prompts/bucket** for Week 2 iteration):
    1. *Native-Script Informational QA & Summarization* (`20` test, `10` dev — Devanagari `hin` / Kannada `kan`).
    2. *Multi-Step Instruction Following & Constraints* (`20` test, `10` dev).
    3. *Conversational & Romanized / Code-Switched Queries* (`20` test, `10` dev — Hinglish / Kanglish in Latin script).
    4. *Structured Output & Entity/Format Preservation* (`20` test, `10` dev — JSON, Markdown tables, code blocks, dates/numbers).
  - **How Prompts Are Sampled Without Leakage:** Sampled on Days 1–2 from held-out production query logs (anonymized, PII-stripped) or newly authored/curated test prompts by the engineering/reviewer team, **never** drawn from any candidate SFT training dataset.
- **3-Layer Decontamination Protocol (Applied Before Any SFT Run):**
  1. Freeze and record SHA-256 hashes of all `240` golden test prompts, `80` dev prompts, and `1,012` `FLORES-200` `devtest` sentences (`your-submission/partA/corpus/processed/`) on Day 2.
  2. Remove any SFT candidate that matches an eval prompt/reference under: (a) Unicode `NFC` + whitespace-normalized substring match, (b) **13-gram character/word overlap**, or (c) **5-gram MinHash Jaccard similarity $\ge 0.70$** (in both native script and Latin transliteration).

### 5.2 Exact Reviewer-Hour Arithmetic (Strictly $\le 10.0$ Combined Hours/Week Across Both Languages)
To avoid any ambiguity in the prompt's phrase *"a native-speaker reviewer (Hindi + Kannada only) for 10 hours/week"*, we budget **combined reviewer time across Hindi + Kannada at $\le 10.0$ hours/week** (`5.0 hours/week` Hindi + `5.0 hours/week` Kannada) using **3.0 minutes per blinded pairwise comparison** (`20` comparisons/hour). Each main evaluation round rates **`80` unique golden prompts plus `10` *additional* blind duplicate re-ratings (`90` total ratings per language)** to measure intra-annotator reliability (Cohen's $\kappa$):

| Week | Evaluation Round & Purpose | Total Ratings / Language (`Unique + Additional Overlap`) | Active Rating Time (`3.0 min/pair`) | Calibration & Sign-off Time | Hours / Language | **Combined Weekly Total (`hin` + `kan`)** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Week 1** | **Round 1 (Golden Test Set):** `Base FLM-4B` vs. `Option 3 (Concise Prompt)` | `80` unique + `10` additional overlap = **`90` pairs/lang** | $90 \times 3.0\text{ min} = \mathbf{4.5\text{ h}}$ (`4.0 h` unique + `0.5 h` overlap) | $0.5\text{ h}$ rubric calibration | `5.0 h` `hin` + `5.0 h` `kan` | **`10.0 hours`** ($\le 10\text{ h/wk}$) |
| **Week 2** | **Round 2 (Dev Split Diagnostic):** `Option 3` vs. `LoRA SFT v1` error taxonomy | `40` unique dev-split prompts = **`40` pairs/lang** | $40 \times 3.0\text{ min} = \mathbf{2.0\text{ h}}$ | Included in $2.0\text{ h}$ dev review | `2.0 h` `hin` + `2.0 h` `kan` | **`4.0 hours`** ($\le 10\text{ h/wk}$) |
| **Week 3** | **Round 3 (Final Golden Test Set):** `Option 3 Baseline` vs. `Locked LoRA SFT Candidate` | `80` unique + `10` additional overlap = **`90` pairs/lang** | $90 \times 3.0\text{ min} = \mathbf{4.5\text{ h}}$ (`4.0 h` unique + `0.5 h` overlap) | $0.5\text{ h}$ critical-defect sign-off | `5.0 h` `hin` + `5.0 h` `kan` | **`10.0 hours`** ($\le 10\text{ h/wk}$) |
| **Total** | **3-Week Cumulative Totals** | **`200` unique + `20` overlap = `220` pairs/lang** | **`11.0 h` / lang** | **`1.0 h` / lang** | **`12.0 h` / lang** | **`24.0 hours` total** |

*(Note: If two separate language reviewers are available at `10.0 h/week` each [`20.0 h/week` combined], every per-language count above doubles directly to `160` unique + `20` additional overlap = `180` pairs/language [`9.0 h` active rating + `1.0 h` calibration = `10.0 h/week/language`].)*

- **Blinded Human Scoring Criteria (Recorded per Pair in Randomized A/B Order):**
  1. **Overall Preference:** `Model A Wins` / `Tie` / `Model B Wins`.
  2. **1–3 Rubric Scores:** (i) *Instruction & Factual Accuracy* (`1=incorrect/hallucinated, 2=minor omission, 3=fully accurate`), (ii) *Morphological & Grammatical Fluency* (`1=broken syntax/wrong Kannada case suffix or Hindi postposition, 2=understandable translationese, 3=natural native phrasing`), and (iii) *Script & Register Match* (`1=wrong script/language, 2=inconsistent code-switching, 3=matches user script/register`).
  3. **Binary Critical Error Flag (`0/1`):** Factual/numeric fabrication, wrong output language/script, unrecovered repetition loop, or unsafe output.

### 5.3 Offline Automated Metrics (`$0` API Cost, Local Python Execution)
1. **Primary Instruction-Set Automated Metrics (on the `240` Golden Prompts):**
   - **Target Script Purity Rate:** Fraction of native-script prompts (`Buckets 1, 2, 4`) where $\ge 95\%$ of alphabetic characters belong to the target Unicode block (Devanagari `U+0900–U+097F` or Kannada `U+0C80–U+0CBF`) via Python `unicodedata` / `regex`.
   - **4-Gram Repetition Loop Rate (`rep-4`):** Fraction of outputs where unique 4-grams / total 4-grams $< 0.70$ or any 4-gram repeats $\ge 5$ times consecutively.
   - **Deterministic Format Compliance Rate:** Pass rate of `json.loads()`, Markdown table column count check, and code-block delimiter check on Bucket 4 (`20` prompts/language) and the `80` English regression prompts.
   - **Sequence Length & Verbosity Profile:** P50/P95 generated token count (`gen_len`) and output-to-prompt token ratio using `FLM-4B-Instruct`'s actual `128k` tokenizer.
2. **Auxiliary Translation Non-Regression Check (`FLORES-200` `devtest`):**
   - **Task & Subset:** Zero-shot English $\rightarrow$ Hindi and English $\rightarrow$ Kannada sentence translation on the held-out `1,012` aligned sentences in `your-submission/partA/corpus/processed/{eng,hin,kan}.txt`.
   - **Reference Format & Metric:** Single-reference, UTF-8 `NFC`-normalized target lines (`hin.txt`, `kan.txt`), scored via character 6-gram + word 2-gram F-score (`chrF++`, $\beta=2$) computed against the **Day-1 measured `Base FLM-4B-Instruct` baseline score** (used strictly to verify no cross-lingual regression; no baseline score is assumed in advance).

---

## 6. Pre-Registered Acceptance Gates, Serving Verification Procedure, and Rollout Prerequisites

### 6.1 Pre-Registered Proposed Acceptance Gates (Not Measured Outcomes)

| Gate | Evaluated Comparison | Proposed Pass Threshold (Both Hindi & Kannada Required) | Action if Threshold Is Not Met |
| :--- | :--- | :--- | :--- |
| **Gate 1: Human Preference (`80` Golden Prompts/Lang)** | Round 1: `Option 3` vs. `Base` <br> Round 3: `LoRA SFT` vs. `Option 3` | **$\text{Win} + \text{Tie} \ge 65.0\%$** ($\ge 52/80$ prompts) AND **$\text{Win Rate} - \text{Loss Rate} \ge +15.0\text{ pp}$** ($\ge +12$ net wins out of $80$), with mean rubric $\ge 2.4/3.0$ | If `LoRA SFT` fails Round 3 against `Option 3`, **discard SFT and ship `Option 3`**. If `Option 3` fails Round 1 against `Base`, hold launch and escalate. |
| **Gate 2: Critical Error Ceiling** | Human binary critical-defect flag on `80` prompts/language | **$\le 2.5\%$** ($\le 2 / 80$ prompts per language) and **$\le$ Base `FLM-4B` count** | Hard launch blocker; any candidate exceeding `2 / 80` critical errors is rejected. |
| **Gate 3: Offline Script, Repetition & Format Gates** | `240` golden prompts (`80 hin, 80 kan, 80 eng`) + `FLORES-200` `devtest` | Native script purity **$\ge 98.0\%$**; `rep-4` loop rate **$\le 1.0\%$**; English & format parser pass-rate drop **$\le 1.0\%$** ($\le 1/80$ prompts); `FLORES-200` `chrF++` **$\ge \text{Base} - 0.5$** | Reject candidate; if SFT degrades English or format compliance by $> 1.0\%$, fall back to `Option 3` (which leaves English prompts untouched). |
| **Gate 4: L4 Serving Capacity & Latency Gate** | $K=5$ repeated runs on `1× L4 24 GB` (`bench_log.csv` harness) | **`preempted_seqs == 0`** across all 5 runs at $B=24$ (`S=4096`, `kv_cache_util <= 0.95`);Indic P95 `gen_len` inflation **$\le +15\%$**; 5-run mean `calc_gen_tok_s` and `itl_ms_p50` within **$\pm 3.0\%$ (or $\pm 2\text{ SE}$)** of Base | Enforce scheduler `max_concurrency = 24` for long contexts (`S=4096`); if Option 3's $\Delta p \le 128$ tokens pushes long prompts over `4096`, cap user prompt input at $4096 - 512 - \Delta p = 3456$ tokens. |

### 6.2 Repeated-Run L4 Load-Test Procedure for the `±3.0%` Serving Threshold
Because `bench_log.csv` provides single-run point measurements (`wall_clock_s` rounded to `0.01 s`), we define the following procedure on `1× NVIDIA L4 (24 GB)`:
1. Run `1` warm-up pass followed by **$K = 5$ timed repetitions** at (a) `batch_size = 16, prompt_len = 512, gen_len = 256` (`Row 5` baseline: `294.46 gen tok/s`, `itl_ms_p50 = 48.33 ms`, `kv_cache_util = 0.12`) and (b) `batch_size = 24, prompt_len = 3584, gen_len = 512` (`Row 11` baseline: `200.92 gen tok/s`, `itl_ms_p50 = 96.07 ms`, `kv_cache_util = 0.93`, `preempted_seqs = 0`).
2. Compute the sample mean $\bar{x}$ and standard error $\text{SE} = s / \sqrt{5}$ of **generated-output-token throughput** ($\frac{\texttt{num\_requests} \times \texttt{gen\_len}}{\texttt{wall\_clock\_s}}$) and `itl_ms_p50`. A candidate passes if `preempted_seqs == 0` on all 5 runs and $|\bar{x}_{\text{cand}} - \bar{x}_{\text{base}}| / \bar{x}_{\text{base}} \le 3.0\%$ (or the 95% confidence interval $\bar{x}_{\text{cand}} - \bar{x}_{\text{base}} \pm 2.776 \cdot \text{SE}_{\text{diff}}$ includes $0$).
3. For **Option 3**, also run the $B=16$ (`p = 512 + \Delta p, g = 256`) test with the actual $\Delta p \le 128$ system prompt prepended to measure the exact prefill/decode cost of the system prompt on L4.

### 6.3 Rollout Prerequisites, Offline Minimum Requirement, and Production Monitoring
- **Offline Evaluation + L4 Harness Is the Mandatory Minimum:** Because `bench/model_spec.md` does not establish that live production traffic, shadow-replay pipelines, or A/B routing infrastructure currently exist, **passing the offline golden evaluation set (`240` prompts), the `10.0 hr/wk` blinded human evaluation, and the L4 load-test harness is the required and sufficient baseline for Week 3 launch review**.
- **Conditional Shadow / Canary Rollout (If Infrastructure Exists):** *If* production request-routing and telemetry infrastructure is available at launch, deploy behind a language-routed feature flag (`5% -> 25% -> 100%`) while monitoring:
  1. **P50/P95 Generated Output Tokens per Turn (`gen_len`) and Output-to-Prompt Token Ratio** segmented by language (`eng`, `hin`, `kan`) and script (`native` vs. `Romanized`).
  2. **Generated-Output-Token Throughput** ($\frac{\sum \text{gen\_tokens}}{\text{wall\_clock\_s}}$), **`kv_cache_util`**, **`preempted_seqs`**, **`ttft_ms_p50/p95`**, **`itl_ms_p50/p95`**, and **`e2e_ms_p95`**.
  3. **Automatic Rollback Condition:** Revert immediately (`< 5 min`) to the previous state (`Option 3` or `Base`) if `preempted_seqs > 0` at $B \le 24$, P95 output token length or `e2e_ms_p95` increases by `> 20%`, or repetition/script-mismatch rates exceed `1.5%`.
