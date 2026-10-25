# Part A4 — Tokenizer Audit & Routing Memo (≤ 1 Page)

**Canonical Memo:** `my-submission/partA/memo.md` *(identical copy at `my-submission/partA/a4_recommendation_memo.md`)*  
**Full Evidence & Per-Language Tables:** `my-submission/partA/{a3_analysis.md, a3_tokenizer_comparison.json, audit_results.md, e1_e6_raw_output.txt, corpus_documentation.md}`

---

## 1. Central Measured Findings (Parts A1–A3)

`REPORT_v0.md` Section 1 recommended separate Indic routing and a 6× Hindi budget after running `fertility.py` (`gpt2`) on a 10-line sample (`hin/eng` word fertility `7.45 / 1.27 = 5.89×`), claiming *"Hindi simply has more Unicode characters per word, so any tokenizer will struggle. This is a property of the script, not the tokenizer."* Our measurements refute that diagnosis:

1. **Hindi has *fewer* Unicode codepoints per word than English:** On the starter-kit sample (`audit_results.md`), Hindi averages **`4.68` codepoints/word** (`290/62`) vs. **`5.67` in English** (`448/79`, **17.5% fewer**). On `1,012` aligned FLORES-200 `devtest` sentences (`corpus_validation_summary.json`), Hindi also has fewer codepoints/word (`5.12` vs. `6.03`) and fewer total codepoints (`131,174` vs. `131,966`).
2. **High `gpt2` fertility is a property of its 50k English-centric vocabulary, not Indic scripts:** Across `1,012` aligned FLORES-200 sentences in 6 languages (`eng`, `hin`, `tam`, `tel`, `kan`, `mal`; cased, `line.split()`, no framing tokens; full tables in `a3_analysis.md`), replacing `tiktoken`'s `gpt2` (`50,257` vocab) with `cl100k_base` (`100,277` vocab) and `o200k_base` (`200,019` vocab) reduces total tokens by **`-78.95%` (`4.75×` fewer tokens) in Hindi** (`200,682` $\rightarrow$ `42,241`) and **`-85.28%` to `-87.36%` (`6.79×–7.91×` fewer tokens) across Tamil, Telugu, Kannada, and Malayalam**, while English changes by **`-0.63%`** (`27,044` $\rightarrow$ `26,873`).
3. **Token-Count-per-Aligned-Sentence Ratios vs. English (`1.00×`):**
   - **`gpt2` (`50k`):** Hindi **`7.42×`** (`198.30` vs. `26.72` tok/sent); Dravidian **`12.96×–15.53×`** (`tel` `12.96×`, `kan` `13.57×`, `mal` `15.16×`, `tam` `15.53×`).
   - **`cl100k_base` (`100k`):** Hindi **`4.77×`** (`128.03` vs. `26.86` tok/sent); Dravidian **`7.64×–8.93×`** (`tam` `7.64×`, `tel` `8.28×`, `kan` `8.84×`, `mal` `8.93×`).
   - **`o200k_base` (`200k`):** Hindi **`1.57×`** (`41.74` vs. `26.55` tok/sent); Dravidian **`1.92×–1.98×`** (`tel` `1.92×`, `mal` `1.95×`, `kan` `1.95×`, `tam` `1.98×`).

---

## 2. Key Measurement Pitfalls (`audit_results.md` & `a3_analysis.md`)

- **Script Bugs (`E1`, `E2`, `E5`):** (1) `line.split(" ")` counts empty strings on double spaces (`7.45` $\rightarrow$ `7.60` on `hin_sample`; injects `+477` phantom words in raw FLORES-200 Kannada vs. `0` in English); (2) `line.lower()` breaks cased `gpt2` merges in English (`99` $\rightarrow$ `96` tokens, inflating `hin/eng` ratio from `5.89×` to `6.06×`) without affecting unicameral Devanagari; (3) `eng_sample.txt` and `hin_sample.txt` are not parallel (`0/10` indices align; only `5/10` loosely match). *(Note: `random.seed(1337)`, `NFC`, and `add_special_tokens=False` caused zero sample distortion, though `NFC` fixes `107` raw FLORES-200 lines.)*
- **Denominator & Aggregation Pitfalls (`E3`, `E4`, A3):**
  - **Macro vs. Micro (`E3`):** `fertility.py` macro-averages per-line ratios (`mean(T_i/W_i)`); corpus token volume requires **micro-averages** ($\sum T_i / \sum D_i$).
  - **Character vs. Byte vs. Grapheme (`E4`):** On `hin_sample`, changing the denominator shifts `hin/eng` `gpt2` ratio from **`7.00×` (codepoints)** to **`2.65×` (UTF-8 bytes)** or **`10.86×` (graphemes `\X`)**. Under `o200k_base`, Indic `Tokens / UTF-8 Byte` (`0.1252–0.1460`) is lower than English (`0.2034`) only because Indic codepoints use 3 UTF-8 bytes vs. 1 ASCII byte.
  - **Morphology Confounds `Tokens / Word`:** Agglutinative Dravidian languages use **`14,930–16,938` words** across `1,012` sentences vs. **`25,643` words** in analytic/postpositional Hindi. Under `o200k_base`, `Tokens / Word` makes Malayalam (`3.5174`, `2.87×` `eng`) look **2.14× worse** than Hindi (`1.6473`, `1.34×` `eng`), yet Malayalam uses only **`1.24×`** as many total tokens (`52,515` vs. `42,241`). **Token-Count-per-Aligned-Sentence Ratio** ($\sum T_{\text{lang}} / \sum T_{\text{eng}}$) is the appropriate cross-language compression metric because it holds the aligned sentence count constant.

---

## 3. Practical Recommendations (Distinct from Measured Findings)

1. **`gpt2` Fertility Alone Cannot Justify Separate Indic Routing—Nor Does Compression Alone Prove Unified Serving Is Sufficient:** `REPORT_v0.md`'s `gpt2` numbers reflect an obsolete 50k English-centric vocabulary and flawed script logic, and are **insufficient to justify a separate Indic routing pool**. However, **these static `tiktoken` compression results do not establish that a unified production deployment is sufficient**, as routing also depends on the production tokenizer, Indic model quality, output verbosity, and latency.
2. **Benchmark `FLM-4B-Instruct`'s Exact Production Tokenizer on the Aligned Corpus:** We did not evaluate `FLM-4B-Instruct`'s `128k` tokenizer and do not assume it resembles `o200k_base`. Run `compare_tokenizers.py` with `FLM-4B-Instruct`'s exact production tokenizer on `my-submission/partA/corpus/processed/`, alongside Indic quality and latency benchmarks, before setting routing or capacity policy.

---

## 4. Methodological Caveats & Production Metric to Monitor

- **Caveats:** (a) **Library Scope:** All three tested encodings (`gpt2`, `cl100k_base`, `o200k_base`) use `tiktoken 0.14.0`; `hf:xlm-roberta-base` was not run because `transformers` is unavailable in `.venv-x64`. (b) **Domain & Script:** FLORES-200 is formal translated Wikimedia prose (a proxy for comparable content, not identical propositions) and excludes **Romanized/code-switched chat (Hinglish/Tanglish)**. (c) **Compression $\neq$ Cost, Quality, or Latency:** The `1.57×–1.98×` `o200k_base` values are **token-count-per-aligned-sentence ratios**, not serving-cost multipliers; they do not measure **generated-output length**, **model quality**, or **end-to-end latency**.
- **Production Metric to Monitor:** Track **P50/P95 Generated Output Tokens per Completed Turn (and Output-to-Prompt Token Ratio) segmented by language and script (native vs. Romanized)** to detect verbosity, repetition loops, or code-switched fragmentation that directly drive decode FLOPs and KV-cache occupancy.
