# Part A3 — Multilingual Tokenizer and Measurement Comparison (Validated)

## 1. Environment, Tokenizer Selection, and Settings Verification

### 1.1 Environment & Package Check (Verified)
- **Interpreter:** `Python 3.12.10 (AMD64)` in `.venv-x64`
- **Installed Tokenizer Library:** `tiktoken 0.14.0` (with `regex 2026.9.29`)
- **HuggingFace `transformers` Check:** Probing `fertility.py --tokenizer hf:xlm-roberta-base` (or `import transformers`) in `.venv-x64` fails with `ModuleNotFoundError: No module named 'transformers'` (`tokenizers` and `sentencepiece` are also not installed in `.venv-x64`). Consequently, the Hugging Face `xlm-roberta-base` comparison was not completed.
- **Explicit Scope Disclosure (Same Library Implementation):** All three tested encodings (`gpt2`, `cl100k_base`, `o200k_base`) use the **`tiktoken` byte-level BPE library** (`tiktoken 0.14.0`). This experiment compares **three BPE vocabulary/merge-table encodings within `tiktoken`** (varying from 50k English-centric BPE to 200k multilingual BPE), **not** independent tokenizer library implementations or alternative algorithms (such as SentencePiece Unigram or WordPiece):

| Tokenizer ID | Model Family | Vocab Size (`n_vocab`) | Provenance / Cache Status (`data-gym-cache`) | Special-Token Behavior (`enc.encode`) | Case Handling |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **`gpt2`** | GPT-2 / `r50k_base` (2019) | `50,257` | Cached locally (`6d1cbeee...`, `456,318` B + `6c7ea1a7...`, `1,042,301` B) | No BOS/EOS framing (`allowed_special=set()`) | Case-sensitive (`lowercase=False`) |
| **`cl100k_base`** | GPT-3.5 / GPT-4 (2022–2023) | `100,277` | Downloaded legitimately via `tiktoken` (`9b5ad71b...`, `1,681,126` B) | No BOS/EOS framing (`allowed_special=set()`) | Case-sensitive (`lowercase=False`) |
| **`o200k_base`** | GPT-4o Multilingual (2024) | `200,019` | Downloaded legitimately via `tiktoken` (`fb374d41...`, `3,613,922` B) | No BOS/EOS framing (`allowed_special=set()`) | Case-sensitive (`lowercase=False`) |

### 1.2 Precise Meaning of "No Special Tokens Added" in `tiktoken` vs. Hugging Face
- **Hugging Face `transformers` (`AutoTokenizer`):** `tok.encode(text, add_special_tokens=True)` (the default) applies a post-processor template that automatically prepends/appends sequence framing tokens (such as `[CLS]`/`[SEP]` or `<bos>`/`<eos>`) to every encoded string. That is why `fertility.py:L33` explicitly passes `add_special_tokens=False` in its `hf:` branch.
- **`tiktoken` (`Encoding.encode`):** `tiktoken` has a **different API and behavior** (`enc.encode(text, *, allowed_special=set(), disallowed_special='all')`). It has **no post-processor template** and **never** automatically prepends or appends BOS/EOS framing tokens to an input string. Instead, each `tiktoken` encoding defines a set of special-token literals (`{'<|endoftext|>'}` in `gpt2`; `{'<|endoftext|>', '<|fim_prefix|>', '<|fim_middle|>', '<|fim_suffix|>', '<|endofprompt|>'}` in `cl100k_base`; `{'<|endoftext|>', '<|endofprompt|>'}` in `o200k_base`), and `allowed_special=set(), disallowed_special='all'` only controls whether literal occurrences of those strings inside `text` are encoded as special tokens or raise a `ValueError`. Thus, calling `enc.encode(line)` encodes strictly the literal characters of `line` into BPE subword/byte token IDs with zero framing overhead.

---

## 2. Implementation-Level Pitfalls Audited in `fertility.py` & Invariant Corpus Denominators

### 2.0 Implementation-Level Evidence Moved from Executive Memo (Full Details in `audit_results.md`)
Before running our 6-language comparison, Part A2 (`audit_experiments.py`, raw output in `e1_e6_raw_output.txt`) audited `fertility.py` on the 10-line starter-kit samples (`eng_sample.txt`, `hin_sample.txt`, baseline `gpt2` macro fertility `1.27` [`1.2652`] vs. `7.45` [`7.4485`] tok/word, ratio `5.89×` [`5.8871×`]):
- **E1 — Word Counting (`line.split(" ")` vs. `line.split()`):** `fertility.py:L62` uses `line.split(" ")`, which counts empty strings `""` as words whenever consecutive spaces occur (`eng_sample` L07, `hin_sample` L10). Switching to `line.split()` lowers word counts (`eng` `79` $\rightarrow$ `78`, `hin` `62` $\rightarrow$ `61`) and changes macro fertility to `1.28` (`1.2831`) and `7.60` (`7.5985`), ratio `5.92×` (`5.9221×`). On raw FLORES-200 `devtest`, `split(" ")` injects `+477` phantom empty words across `213` Kannada lines and `+274` across `136` Telugu lines vs. `0` in English.
- **E2 — Lowercasing (`line.lower()` vs. Cased):** `fertility.py:L60` lowercases every line (`if True:`), breaking cased `gpt2` BPE merges in bicameral English (`Thursday` 1 $\rightarrow$ 2 tok on L02, `NASA` 1 $\rightarrow$ 2 tok on L06, `GPU` 1 $\rightarrow$ 2 tok on L10; total `99` $\rightarrow$ `96` tokens, fertility `1.27` $\rightarrow$ `1.23` [`1.2293`], inflating `hin/eng` ratio from `5.89×` to `6.06×` [`6.0590×`]), while having zero effect on unicameral Devanagari (`459` tokens).
- **E3 — Macro vs. Micro Averaging:** `fertility.py:L69` macro-averages per-line ratios (`mean(T_i/W_i)`), weighting a 4-word line (`hin` L02) equally with a 15-word line (`eng` L09). Corpus-level micro-averaging (`sum(T_i)/sum(W_i)`) yields `1.25` (`99/79 = 1.2532`) for `eng` and `7.40` (`459/62 = 7.4032`) for `hin` (ratio `5.91×` [`5.9076×`]).
- **E4 — Character Denominators (`Codepoints` vs. `UTF-8 Bytes` vs. `Graphemes \X`):** In `eng_sample`, `448` codepoints = `448` UTF-8 bytes = `448` graphemes (`0.226` tok/unit). In `hin_sample`, `290` codepoints (`1.579` tok/cp, `7.00×` `eng`) correspond to `764` UTF-8 bytes (`0.599` tok/B, `2.65×` `eng`) and `188` extended grapheme clusters (`2.450` tok/grapheme, `10.86×` `eng`). Furthermore, `hin_sample` has **fewer codepoints per word (`290/62 = 4.68`) than `eng_sample` (`448/79 = 5.67`, `-17.5%`)**, directly refuting `REPORT_v0.md` Finding #3.
- **E5 — Non-Parallel Sample Corpus:** `0/10` line indices in `eng_sample.txt` and `hin_sample.txt` align; only `5/10` lines have loose topical counterparts (`eng` L3,4,5,7,8 $\leftrightarrow$ `hin` L3,7,6,10,4; matched macro ratio `6.17×`, matched total token ratio `242/47 = 5.15×` vs. `459/99 = 4.64×`).
- **E6 — Harmless/Correct Choices:** `random.seed(1337)` (`\Delta = 0`, deterministic BPE), `NFC` normalization (`\Delta = 0` on sample, though essential on raw FLORES-200 where `107` lines across `hin`/`tam`/`tel`/`kan`/`mal` are non-NFC), and `add_special_tokens=False` (`\Delta = 0` for `tiktoken`).

All three tokenizers in Part A3 processed the **exact same 1,012 aligned sentences** (`IDs 1..1012`) per language from `your-submission/partA/corpus/processed/<lang>.txt` (UTF-8, NFC-normalized, consecutive internal whitespace collapsed to a single ASCII space `' '`, original casing preserved).

### 2.1 Explicit Metric Definitions
For a corpus of $N = 1,012$ aligned sentences where sentence $i$ produces $T_i$ tokens and has $W_i$ whitespace-delimited words (`line.split()`), $C_i$ Unicode codepoints (`len(line)`), $G_i$ extended grapheme clusters (`len(regex.findall(r"\X", line))`), and $B_i$ UTF-8 bytes (`len(line.encode("utf-8"))`):

1. **Corpus-Level Micro-Averages (Ratios of Corpus Totals):**
   - **Tokens per Aligned Sentence:** $\frac{\sum_{i=1}^N T_i}{N}$
   - **Sentence-Parity Ratio vs. English:** $\frac{\sum_{i=1}^N T_{i,\text{lang}}}{\sum_{i=1}^N T_{i,\text{eng}}} = \frac{\text{Tokens/Sentence}(\text{lang})}{\text{Tokens/Sentence}(\text{eng})}$ (holds the translated sentence count $N=1,012$ constant across languages).
   - **Word-Based Fertility (`Tokens / Word`):** $\frac{\sum_{i=1}^N T_i}{\sum_{i=1}^N W_i}$, and **Word-Fertility Ratio vs. English:** $\frac{\text{Tokens/Word}(\text{lang})}{\text{Tokens/Word}(\text{eng})}$.
   - **Tokens per Extended Grapheme Cluster (`Tokens / Grapheme`):** $\frac{\sum_{i=1}^N T_i}{\sum_{i=1}^N G_i}$.
   - **Tokens per Unicode Codepoint (`Tokens / Codepoint`):** $\frac{\sum_{i=1}^N T_i}{\sum_{i=1}^N C_i}$.
   - **Tokens per UTF-8 Byte (`Tokens / UTF-8 Byte`):** $\frac{\sum_{i=1}^N T_i}{\sum_{i=1}^N B_i}$ — measures token emission rate per raw UTF-8 byte (higher values indicate greater fragmentation / less compression).
   - **UTF-8 Bytes per Token (`UTF-8 Bytes / Token`):** $\frac{\sum_{i=1}^N B_i}{\sum_{i=1}^N T_i} = \left(\text{Tokens / UTF-8 Byte}\right)^{-1}$ — the exact reciprocal of `Tokens / UTF-8 Byte`, measuring the average number of UTF-8 bytes spanned by each token (higher values indicate greater byte compression).
2. **Per-Line Macro-Averages (Unweighted Means of Per-Sentence Ratios):**
   - $\text{Macro}(T/D) = \frac{1}{N} \sum_{i=1}^N \frac{T_i}{D_i}$ for denominator $D \in \{W, C, G, B\}$.
3. **Rounding Convention:** All cross-language ratios and differences in `a3_tokenizer_comparison.json` and the tables below are computed from **exact unrounded IEEE-754 floats** before rounding for display.

### 2.2 Verified Invariant Denominator Counts (`your-submission/partA/corpus/processed/`)
| Lang | Language | Script | Aligned Sentences ($N$) | Whitespace Words (`line.split()`) | Unicode Codepoints (`len(line)`) | Non-Space Codepoints | Extended Grapheme Clusters (`\X`) | UTF-8 Bytes (`len(line.encode('utf-8'))`) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`eng`** | English | Latin | `1,012` | `21,901` | `131,966` | `111,077` | `131,966` | `132,096` |
| **`hin`** | Hindi | Devanagari | `1,012` | `25,643` | `131,174` | `106,543` | `85,951` | `337,433` |
| **`tam`** | Tamil | Tamil | `1,012` | `16,775` | `154,031` | `138,268` | `99,624` | `421,535` |
| **`tel`** | Telugu | Telugu | `1,012` | `16,938` | `132,226` | `116,300` | `76,328` | `353,422` |
| **`kan`** | Kannada | Kannada | `1,012` | `16,100` | `137,650` | `122,562` | `89,994` | `374,964` |
| **`mal`** | Malayalam | Malayalam | `1,012` | `14,930` | `149,261` | `135,343` | `79,357` | `411,670` |

---

## 3. Per-Language Totals & Corpus-Level Micro-Averages (Reconciled with `a3_tokenizer_comparison.json`)

### 3.1 Tokenizer 1: `gpt2` (`50,257` vocab — English-Centric Byte-Level BPE)
| Lang | Total Tokens | Tokens / Sentence | **Sentence-Parity Ratio vs. `eng`** | `Tokens / Word` (Micro) | **Word-Fertility Ratio vs. `eng`** | `Tokens / Grapheme` (`vs. eng`) | `Tokens / Codepoint` (`vs. eng`) | `Tokens / UTF-8 Byte` (`vs. eng`) | `UTF-8 Bytes / Token` (Reciprocal) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`eng`** | `27,044` | `26.7233` | **`1.00x`** (`1.0000x`) | `1.2348` | **`1.00x`** (`1.0000x`) | `0.2049` (`1.00x`) | `0.2049` (`1.00x`) | `0.2047` (`1.00x`) | `4.8845` (`4.88`) |
| **`hin`** | `200,682` | `198.3024` | **`7.42x`** (`7.4206x`) | `7.8260` | **`6.34x`** (`6.3377x`) | `2.3348` (`11.39x`) | `1.5299` (`7.47x`) | `0.5947` (`2.90x`) | `1.6814` (`1.68`) |
| **`tam`** | `420,071` | `415.0899` | **`15.53x`** (`15.5329x`) | `25.0415` | **`20.28x`** (`20.2793x`) | `4.2166` (`20.58x`) | `2.7272` (`13.31x`) | `0.9965` (`4.87x`) | `1.0035` (`1.00`) |
| **`tel`** | `350,509` | `346.3528` | **`12.96x`** (`12.9607x`) | `20.6936` | **`16.76x`** (`16.7583x`) | `4.5921` (`22.41x`) | `2.6508` (`12.94x`) | `0.9918` (`4.84x`) | `1.0083` (`1.01`) |
| **`kan`** | `366,989` | `362.6374` | **`13.57x`** (`13.5701x`) | `22.7943` | **`18.46x`** (`18.4595x`) | `4.0779` (`19.90x`) | `2.6661` (`13.01x`) | `0.9787` (`4.78x`) | `1.0217` (`1.02`) |
| **`mal`** | `409,912` | `405.0514` | **`15.16x`** (`15.1572x`) | `27.4556` | **`22.23x`** (`22.2343x`) | `5.1654` (`25.21x`) | `2.7463` (`13.40x`) | `0.9957` (`4.86x`) | `1.0043` (`1.00`) |

### 3.2 Tokenizer 2: `cl100k_base` (`100,277` vocab — GPT-3.5 / GPT-4 BPE)
| Lang | Total Tokens | Tokens / Sentence | **Sentence-Parity Ratio vs. `eng`** | `Tokens / Word` (Micro) | **Word-Fertility Ratio vs. `eng`** | `Tokens / Grapheme` (`vs. eng`) | `Tokens / Codepoint` (`vs. eng`) | `Tokens / UTF-8 Byte` (`vs. eng`) | `UTF-8 Bytes / Token` (Reciprocal) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`eng`** | `27,182` | `26.8597` | **`1.00x`** (`1.0000x`) | `1.2411` | **`1.00x`** (`1.0000x`) | `0.2060` (`1.00x`) | `0.2060` (`1.00x`) | `0.2058` (`1.00x`) | `4.8597` (`4.86`) |
| **`hin`** | `129,567` | `128.0306` | **`4.77x`** (`4.7666x`) | `5.0527` | **`4.07x`** (`4.0711x`) | `1.5075` (`7.32x`) | `0.9877` (`4.80x`) | `0.3840` (`1.87x`) | `2.6043` (`2.60`) |
| **`tam`** | `207,687` | `205.2243` | **`7.64x`** (`7.6406x`) | `12.3807` | **`9.98x`** (`9.9754x`) | `2.0847` (`10.12x`) | `1.3483` (`6.55x`) | `0.4927` (`2.39x`) | `2.0297` (`2.03`) |
| **`tel`** | `225,022` | `222.3538` | **`8.28x`** (`8.2783x`) | `13.2850` | **`10.70x`** (`10.7040x`) | `2.9481` (`14.31x`) | `1.7018` (`8.26x`) | `0.6367` (`3.09x`) | `1.5706` (`1.57`) |
| **`kan`** | `240,364` | `237.5138` | **`8.84x`** (`8.8428x`) | `14.9294` | **`12.03x`** (`12.0289x`) | `2.6709` (`12.97x`) | `1.7462` (`8.48x`) | `0.6410` (`3.12x`) | `1.5600` (`1.56`) |
| **`mal`** | `242,859` | `239.9792` | **`8.93x`** (`8.9346x`) | `16.2665` | **`13.11x`** (`13.1062x`) | `3.0603` (`14.86x`) | `1.6271` (`7.90x`) | `0.5899` (`2.87x`) | `1.6951` (`1.70`) |

### 3.3 Tokenizer 3: `o200k_base` (`200,019` vocab — GPT-4o Multilingual / Indic-Aware BPE)
| Lang | Total Tokens | Tokens / Sentence | **Sentence-Parity Ratio vs. `eng`** | `Tokens / Word` (Micro) | **Word-Fertility Ratio vs. `eng`** | `Tokens / Grapheme` (`vs. eng`) | `Tokens / Codepoint` (`vs. eng`) | `Tokens / UTF-8 Byte` (`vs. eng`) | `UTF-8 Bytes / Token` (Reciprocal) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`eng`** | `26,873` | `26.5543` | **`1.00x`** (`1.0000x`) | `1.2270` | **`1.00x`** (`1.0000x`) | `0.2036` (`1.00x`) | `0.2036` (`1.00x`) | `0.2034` (`1.00x`) | `4.9156` (`4.92`) |
| **`hin`** | `42,241` | `41.7401` | **`1.57x`** (`1.5719x`) | `1.6473` | **`1.34x`** (`1.3425x`) | `0.4915` (`2.41x`) | `0.3220` (`1.58x`) | `0.1252` (`0.62x`) | `7.9883` (`7.99`) |
| **`tam`** | `53,114` | `52.4842` | **`1.98x`** (`1.9765x`) | `3.1663` | **`2.58x`** (`2.5804x`) | `0.5331` (`2.62x`) | `0.3448` (`1.69x`) | `0.1260` (`0.62x`) | `7.9364` (`7.94`) |
| **`tel`** | `51,610` | `50.9980` | **`1.92x`** (`1.9205x`) | `3.0470` | **`2.48x`** (`2.4832x`) | `0.6762` (`3.32x`) | `0.3903` (`1.92x`) | `0.1460` (`0.72x`) | `6.8479` (`6.85`) |
| **`kan`** | `52,534` | `51.9111` | **`1.95x`** (`1.9549x`) | `3.2630` | **`2.66x`** (`2.6593x`) | `0.5838` (`2.87x`) | `0.3816` (`1.87x`) | `0.1401` (`0.69x`) | `7.1375` (`7.14`) |
| **`mal`** | `52,515` | `51.8923` | **`1.95x`** (`1.9542x`) | `3.5174` | **`2.87x`** (`2.8666x`) | `0.6618` (`3.25x`) | `0.3518` (`1.73x`) | `0.1276` (`0.63x`) | `7.8391` (`7.84`) |

---

## 4. Macro-Average vs. Micro-Average Comparison

- **Macro-Average** ($\frac{1}{N}\sum_{i=1}^N \frac{T_i}{D_i}$) computes the unweighted mean of sentence-level ratios, giving a short sentence the same weight as a long sentence.
- **Micro-Average** ($\frac{\sum_{i=1}^N T_i}{\sum_{i=1}^N D_i}$) weights each sentence by its denominator $D_i$ and equals the corpus-level token emission rate.

*(Note: Differences $\Delta = \text{Macro} - \text{Micro}$ are computed from unrounded floats before rounding to 4 decimal places; for example, on `o200k_base` `hin`, $1.669549 - 1.647272 = +0.022276 \rightarrow +0.0223$, and on `gpt2` `kan`, $22.995088 - 22.794348 = +0.200741 \rightarrow +0.2007$.)*

| Lang | `gpt2` Macro `Tok/Word` | `gpt2` Micro `Tok/Word` | `gpt2` $\Delta$ ($\text{Macro}-\text{Micro}$) | `cl100k` Macro `Tok/Word` | `cl100k` Micro `Tok/Word` | `cl100k` $\Delta$ ($\text{Macro}-\text{Micro}$) | `o200k` Macro `Tok/Word` | `o200k` Micro `Tok/Word` | `o200k` $\Delta$ ($\text{Macro}-\text{Micro}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`eng`** | `1.2444` | `1.2348` | `+0.0096` | `1.2504` | `1.2411` | `+0.0093` | `1.2360` | `1.2270` | `+0.0090` |
| **`hin`** | `7.8663` | `7.8260` | `+0.0403` | `5.0847` | `5.0527` | `+0.0320` | `1.6695` | `1.6473` | `+0.0223` |
| **`tam`** | `25.2448` | `25.0415` | `+0.2033` | `12.4772` | `12.3807` | `+0.0965` | `3.1946` | `3.1663` | `+0.0283` |
| **`tel`** | `20.8099` | `20.6936` | `+0.1163` | `13.3688` | `13.2850` | `+0.0837` | `3.0797` | `3.0470` | `+0.0327` |
| **`kan`** | `22.9951` | `22.7943` | `+0.2007` | `15.0702` | `14.9294` | `+0.1407` | `3.3010` | `3.2630` | `+0.0380` |
| **`mal`** | `27.6843` | `27.4556` | `+0.2287` | `16.3933` | `16.2665` | `+0.1268` | `3.5539` | `3.5174` | `+0.0365` |

| Lang | `o200k` Macro `Tok/CP` | `o200k` Micro `Tok/CP` | `o200k` Macro `Tok/Grapheme` | `o200k` Micro `Tok/Grapheme` | `o200k` Macro `Tok/UTF-8 Byte` | `o200k` Micro `Tok/UTF-8 Byte` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`eng`** | `0.2058` | `0.2036` | `0.2058` | `0.2036` | `0.2056` | `0.2034` |
| **`hin`** | `0.3250` | `0.3220` | `0.4962` | `0.4915` | `0.1266` | `0.1252` |
| **`tam`** | `0.3460` | `0.3448` | `0.5350` | `0.5331` | `0.1267` | `0.1260` |
| **`tel`** | `0.3926` | `0.3903` | `0.6816` | `0.6762` | `0.1473` | `0.1460` |
| **`kan`** | `0.3838` | `0.3816` | `0.5869` | `0.5838` | `0.1412` | `0.1401` |
| **`mal`** | `0.3540` | `0.3518` | `0.6657` | `0.6618` | `0.1287` | `0.1276` |

- **When to use each:**
  - **Corpus-Level Ratio (Micro-Average)** is the appropriate estimator for **total token volume, KV-cache footprint, and aggregate prefill/decode compute across a workload**, because those quantities depend on the sum of tokens $\sum T_i$ across requests.
  - **Macro-Average** is an unweighted sentence-level average; across all 6 languages and 3 tokenizers on this corpus, macro-averaged `tok/word` is slightly higher than micro-averaged `tok/word` by `+0.0090` to `+0.2287` tok/word, reflecting a slight negative covariance between sentence word count $W_i$ and per-sentence ratio $T_i / W_i$.

---

## 5. Interpretation: Tokenizer Compression & Denominator Selection

### 5.1 Interpreting `gpt2`'s Near-1.0 `Tokens / UTF-8 Byte` on Dravidian Languages
- On `gpt2`, the observed corpus-level `Tokens / UTF-8 Byte` ratios on Dravidian languages are **`0.9787` (`kan`, `1.0217` B/tok), `0.9918` (`tel`, `1.0083` B/tok), `0.9957` (`mal`, `1.0043` B/tok), and `0.9965` (`tam`, `1.0035` B/tok)**, compared to **`0.5947` (`hin`, `1.6814` B/tok)** and **`0.2047` (`eng`, `4.8845` B/tok)**.
- **What these ratios suggest (and what they do not imply):**
  - Because every non-ASCII Dravidian codepoint occupies 3 UTF-8 bytes while ASCII spaces and punctuation occupy 1 byte, an aggregate rate of `0.9787–0.9965` tokens per UTF-8 byte (`1.0035–1.0217` UTF-8 bytes per token) does **not** mean every character becomes a fixed number of tokens. Rather, it is an aggregate average across 1-byte ASCII spaces/punctuation/digits, unmerged single-byte UTF-8 fallback tokens, and occasional 2-byte BPE merges of recurring UTF-8 prefix bytes (or space + byte pairs).
  - Even with those occasional multi-byte merges, averaging `1.0035–1.0217` UTF-8 bytes per token indicates that `gpt2`'s English-centric vocabulary achieves almost no multi-byte subword compression on Dravidian text, whereas `o200k_base` spans **`6.8479` (`tel`) to `7.9364` (`tam`) UTF-8 bytes per token** (`0.1260–0.1460` tokens per UTF-8 byte).

### 5.2 Which Single Metric Should Drive Routing and Cost Decisions — And Why?
To select a metric for cross-language routing and serving-cost estimation, we must ask what each denominator holds constant across languages:

1. **Why `Tokens / Whitespace Word` (`Word Fertility`) is confounded by morphological typology:**
   - Under `o200k_base`, Malayalam has a micro-averaged word fertility of **`3.5174` tok/word (`2.87x` English)** while Hindi has **`1.6473` tok/word (`1.34x` English)**—making Malayalam appear **2.14× (`3.5174 / 1.6473`) more expensive per word** than Hindi.
   - However, across the **same 1,012 translated sentences**, Malayalam consumes **`52,515` total tokens** vs. **`42,241` in Hindi**—only **`1.24×` (`52,515 / 42,241`)** Hindi's token count!
   - The discrepancy arises because Dravidian languages (`tam`, `tel`, `kan`, `mal`) are **agglutinative** (`14,930` words in Malayalam vs. `25,643` words in analytic/postpositional Hindi). Dividing by whitespace words penalizes agglutinative languages for fusing suffixes and postpositions onto stems without spaces.
2. **Why `Tokens / Unicode Codepoint` and `Tokens / Grapheme Cluster` are script-dependent:**
   - Alphabetic Latin and abugida Indic scripts represent vowels and consonant clusters at different structural granularities, so neither codepoints nor grapheme clusters hold message content constant across scripts.
3. **Why `Tokens / UTF-8 Byte` (and its reciprocal `UTF-8 Bytes / Token`) cannot drive serving cost:**
   - Under `o200k_base`, Hindi (`0.1252` tok/byte = `7.9883` B/tok) and Tamil (`0.1260` tok/byte = `7.9364` B/tok) have **lower** `Tokens / UTF-8 Byte` (`0.62x` of English's `0.2034` tok/byte = `4.9156` B/tok) because UTF-8 uses 3 bytes per Indic codepoint vs. 1 byte for ASCII. Transformer attention and KV-cache capacity scale with **token count**, not UTF-8 byte count.
4. **The Appropriate Tokenizer Compression Metric — `Tokens per Aligned Sentence` (and `Token-Count-per-Aligned-Sentence Ratio vs. English` $\frac{\sum T_{\text{lang}}}{\sum T_{\text{eng}}}$):**
   - While human translations across `1,012` FLORES-200 sentences are not mathematically identical propositions (due to translationese and minor translator edits), **aligned translated sentences serve as the best available proxy for comparable translated message content** across analytic and agglutinative languages.
   - Measured by **Token-Count-per-Aligned-Sentence Ratio vs. English** (`Total Tokens(lang) / Total Tokens(eng)`, which measures static input/reference token volume per aligned sentence rather than end-to-end serving cost):
     - Under `gpt2`: Hindi requires **`7.42x`** (`200,682 / 27,044`) and Dravidian languages require **`12.96x–15.53x`** as many tokens per aligned sentence as English for the same 1,012 translated sentences.
     - Under `o200k_base`: Hindi's **token-count-per-aligned-sentence ratio** is **`1.57x`** (`42,241 / 26,873`) and all four Dravidian languages converge tightly to **`1.92x–1.98x`** (`tel` `1.92x`, `mal` `1.95x`, `kan` `1.95x`, `tam` `1.98x`)—a **`4.75×` (`-78.95%`) token reduction on Hindi** and a **`6.79×–7.91×` (`-85.28%` to `-87.36%`) token reduction on Dravidian languages** relative to `gpt2`.

---

## 6. Limitations & Scope Caveats

1. **Subword Compression $\neq$ Model Quality or Guaranteed End-to-End Latency:**
   - Lower token counts under `o200k_base` demonstrate **substantially higher subword compression under that encoding**, which directly reduces token-level KV-cache footprint and sequence length for a given text.
   - However, tokenizer compression alone does **not** prove superior downstream language modeling quality, factual accuracy, or conversational fluency, nor does it guarantee lower end-to-end latency in isolation (since end-to-end latency also depends on model parameter count, larger vocabulary embedding/LM-head projection overhead for `200,019` vs. `50,257` tokens, scheduler/batching behavior, and generation verbosity).
2. **Benchmark Domain Scope:**
   - FLORES-200 `devtest` (`1,012` sentences) is formal encyclopedic, news, and travel prose translated from English. It does not measure Romanized Indic (Hinglish/Tanglish), conversational chat, or source code.

---

## 7. Reproducible Commands (Verified from Starter-Kit Root)

All commands below were executed and verified from `c:\Users\TANNU\OneDrive\Desktop\starter_kit`:

```powershell
# 1. Run full Part A3 multi-tokenizer & multi-denominator comparison (writes JSON + stdout)
.\.venv-x64\Scripts\python.exe your-submission/partA/compare_tokenizers.py

# 2. Run unmodified starter-kit fertility.py on the 6 processed FLORES-200 languages (gpt2 vs o200k_base)
.\.venv-x64\Scripts\python.exe fertility.py --corpus eng=your-submission/partA/corpus/processed/eng.txt --corpus hin=your-submission/partA/corpus/processed/hin.txt --corpus tam=your-submission/partA/corpus/processed/tam.txt --corpus tel=your-submission/partA/corpus/processed/tel.txt --corpus kan=your-submission/partA/corpus/processed/kan.txt --corpus mal=your-submission/partA/corpus/processed/mal.txt --tokenizer gpt2

.\.venv-x64\Scripts\python.exe fertility.py --corpus eng=your-submission/partA/corpus/processed/eng.txt --corpus hin=your-submission/partA/corpus/processed/hin.txt --corpus tam=your-submission/partA/corpus/processed/tam.txt --corpus tel=your-submission/partA/corpus/processed/tel.txt --corpus kan=your-submission/partA/corpus/processed/kan.txt --corpus mal=your-submission/partA/corpus/processed/mal.txt --tokenizer o200k_base
```
*(Note: Running `fertility.py --tokenizer hf:xlm-roberta-base` was not executed to completion because `transformers` is not installed in `.venv-x64`, raising `ModuleNotFoundError: No module named 'transformers'`; therefore any Hugging Face tokenizer numbers remain unmeasured.)*
