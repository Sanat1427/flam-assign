# Chronological Lab Notebook — AI Team Intern Assignment (The Audit)

This notebook records every hypothesis, command, observed result, interpretation, dead end, and limitation in chronological order.

---

## Entry 1 — Initial Inspection & Environment Check
**Timestamp:** `2026-10-23 01:33 IST`

### Objective
Inspect the `starter_kit/` directory, understand `fertility.py`, `REPORT_v0.md`, `corpus_sample/`, `bench/model_spec.md`, and `bench/bench_log.csv`, and verify the local Python environment without changing any files.

### Commands Run
```powershell
Get-ChildItem -Recurse -Force c:\Users\TANNU\OneDrive\Desktop\starter_kit | Select-Object FullName, Length
python --version
pip list
```

### Observed Result
- Files present: `fertility.py` (3063 B), `REPORT_v0.md` (1443 B), `corpus_sample/eng_sample.txt` (458 B, 10 lines), `corpus_sample/hin_sample.txt` (774 B, 10 lines), `bench/model_spec.md` (1230 B), `bench/bench_log.csv` (828 B, 14 lines).
- Default Python is `Python 3.12.10` on Windows `ARM64` (Qualcomm Snapdragon X).
- Neither `tiktoken` nor `transformers` is installed in the base environment.
- Attempting to run `python fertility.py --corpus eng=corpus_sample/eng_sample.txt --corpus hin=corpus_sample/hin_sample.txt --tokenizer gpt2` failed immediately with `ModuleNotFoundError: No module named 'tiktoken'`.

---

## Entry 2 — Dead End: Installing `tiktoken` on Windows ARM64 Python & Resolving via x64 Python `.venv-x64`
**Timestamp:** `2026-10-23 02:01 IST`

### Hypothesis
Running `python -m pip install tiktoken` in the existing Python 3.12 environment will install a prebuilt wheel for `tiktoken`.

### Command & Observed Result (Dead End)
```powershell
python -m pip install tiktoken
```
- **Observed Result:** Failed (`exit code 1`). PyPI had prebuilt `win_arm64` wheels for `regex` (`regex-2026.9.29-cp312-cp312-win_arm64.whl`), but `tiktoken 0.14.0` had no `win_arm64` wheel on PyPI, so `pip` downloaded `tiktoken-0.14.0.tar.gz` and failed at `running build_rust: error: can't find Rust compiler`.
- Checked `py -0p` for an existing x64 interpreter; only `-V:3.12-arm64` was installed.

### Revision & Resolution (`2026-10-23 02:06 IST`)
With approval, installed official `Python 3.12.10 (AMD64)` into `C:\Users\TANNU\AppData\Local\Programs\Python\Python312-x64` (`PrependPath=0`, keeping system default ARM64 Python untouched), created `.venv-x64`, and installed `tiktoken 0.14.0` from the prebuilt `tiktoken-0.14.0-cp312-cp312-win_amd64.whl` wheel (which also installed `regex 2026.9.29`).

```powershell
C:\Users\TANNU\AppData\Local\Programs\Python\Python312-x64\python.exe -m venv .venv-x64
.\.venv-x64\Scripts\Activate.ps1
python -m pip install tiktoken
python fertility.py --corpus eng=corpus_sample/eng_sample.txt --corpus hin=corpus_sample/hin_sample.txt --tokenizer gpt2
```

### Observed Baseline Output
```text
tokenizer: gpt2
lang      fertility (tok/word)    tok/char
------------------------------------------
eng                       1.27       0.226
hin                       7.45       1.579

hin is 5.89x the fertility of eng (worse tokenization)
```
All 5 numbers match `REPORT_v0.md` Section 1 identically.

---

## Entry 3 — Part A2 Controlled Experiments (`E1`–`E6`) on `fertility.py`
**Timestamp:** `2026-10-23 14:09 IST`

### Command & Script Run
```powershell
.\.venv-x64\Scripts\python.exe my-submission/partA/audit_experiments.py
```
- **Script:** `my-submission/partA/audit_experiments.py`
- **Raw Output Saved:** `my-submission/partA/e1_e6_raw_output.txt`

---

### 0. Baseline Unrounded Reference Values
- **Observed Result:**
  - `eng`: `tot_tokens = 99`, `tot_words (split(" ")) = 79`, `tot_chars = 448`
    - Unrounded macro fertility = `1.265206` (formatted `.2f` $\rightarrow$ `1.27`)
    - Unrounded macro `tok/char` = `0.225636` (formatted `.3f` $\rightarrow$ `0.226`)
  - `hin`: `tot_tokens = 459`, `tot_words (split(" ")) = 62`, `tot_chars = 290`
    - Unrounded macro fertility = `7.448452` (formatted `.2f` $\rightarrow$ `7.45`)
    - Unrounded macro `tok/char` = `1.579108` (formatted `.3f` $\rightarrow$ `1.579`)
  - Unrounded fertility ratio (`hin / eng`) = `7.448452 / 1.265206 = 5.887148` (formatted `.2f` $\rightarrow$ `5.89x`).
  - Unrounded `tok/char` ratio (`hin / eng`) = `1.579108 / 0.225636 = 6.998478` (formatted `.2f` $\rightarrow$ `7.00x`).
- **Interpretation:** Dividing the rounded display numbers gives `7.45 / 1.27 = 5.8661` (`5.87x`), whereas `fertility.py` divides the unrounded IEEE-754 floats before formatting (`5.887148` $\rightarrow$ `5.89x`), which is mathematically sound practice.

---

### 1. Experiment E1 — Word Counting (`line.split(" ")` vs. `line.split()`)
- **Command / Function:** `compute_metrics(..., split_mode="whitespace")` in `my-submission/partA/audit_experiments.py` (changing only `line.split(" ")` at `fertility.py:L62` to `line.split()`).
- **Observed Result:**
  - **Total word counts:**
    - `eng`: `79` (`split(" ")`) $\rightarrow$ `78` (`split()`), $\Delta = -1$ word.
    - `hin`: `62` (`split(" ")`) $\rightarrow$ `61` (`split()`), $\Delta = -1$ word.
  - **Exact lines affected:**
    - `eng` L07 (`"Please keep the books  in the cupboard."`): contains a double space between `books` and `in`. `split(" ")` returns `8` elements `['please', 'keep', 'the', 'books', '', 'in', 'the', 'cupboard.']` (`fert = 10/8 = 1.250000`), whereas `split()` returns `7` words (`fert = 10/7 = 1.428571`).
    - `hin` L10 (`"किताबें  अलमारी में रखी हैं।"`): contains a double space between `किताबें` and `अलमारी`. `split(" ")` returns `6` elements `['किताबें', '', 'अलमारी', 'में', 'रखी', 'हैं।']` (`fert = 45/6 = 7.500000`), whereas `split()` returns `5` words (`fert = 45/5 = 9.000000`).
  - **Recomputed Fertility (Macro-Averaged):**
    - `eng` fertility: `1.265206` (`1.27`) $\rightarrow$ `1.283063` (`1.28`), $\Delta = +0.017857$ (`+1.41%`).
    - `hin` fertility: `7.448452` (`7.45`) $\rightarrow$ `7.598452` (`7.60`), $\Delta = +0.150000$ (`+2.01%`).
    - `hin/eng` fertility ratio: `5.887148` (`5.89x`) $\rightarrow$ `5.922121` (`5.92x`), $\Delta = +0.034973$ (`+0.59%`).
    - `tok/char` is unchanged (`0.226` and `1.579`).
- **Interpretation:** `line.split(" ")` is a genuine implementation bug. Consecutive spaces inject empty string tokens `""` into the word list (inflating the word-count denominator) while `gpt2` still tokenizes the extra space as a separate token (`' '`, ID `220`) in the numerator. Fixing this raises `hin` fertility from `7.45` to `7.60` and `eng` fertility from `1.27` to `1.28`, changing the reported numbers slightly (`5.89x` $\rightarrow$ `5.92x`), though the overall $\sim 5.9\times$ conclusion remains intact.
- **Limitations:** `line.split()` still treats attached punctuation (e.g., `"cupboard."`, `"हैं।"`) as part of the word rather than using Unicode word segmentation (`UAX #29`), and does not normalize away the extra whitespace token in `encode(line)`.

---

### 2. Experiment E2 — Lowercasing (`line.lower()` vs. Original Casing)
- **Command / Function:** `compute_metrics(..., lowercase=False)` in `my-submission/partA/audit_experiments.py` (removing `line = line.lower()` at `fertility.py:L60`).
- **Observed Result:**
  - **Total token counts:**
    - `eng`: `99` tokens (lowercased) $\rightarrow$ `96` tokens (original casing), $\Delta = -3$ tokens (`-3.03%`).
    - `hin`: `459` tokens (lowercased) $\rightarrow$ `459` tokens (original casing), $\Delta = 0$ tokens (`0.00%`).
  - **Per-line token count changes:**
    - `eng` L02 (`"The Quarterly Review meeting moved to Thursday."`): `9` tokens (lowercased) $\rightarrow$ `8` tokens (cased), $\Delta = -1$. Specifically, cased `' Thursday'` (1 token) splits into `[' th', 'ursday']` (2 tokens) when lowercased.
    - `eng` L06 (`"NASA and ISRO announced a joint mission update."`): `11` tokens (lowercased) $\rightarrow$ `10` tokens (cased), $\Delta = -1$. Specifically, cased `'NASA'` (1 token) splits into `['n', 'asa']` (2 tokens) when lowercased (whereas `' ISRO'` is 2 tokens in both: `[' IS', 'RO']` cased vs. `[' is', 'ro']` lowercased).
    - `eng` L10 (`"The GPU cluster ran out of memory during the night job."`): `13` tokens (lowercased) $\rightarrow$ `12` tokens (cased), $\Delta = -1$. Specifically, cased `' GPU'` (1 token) splits into `[' g', 'pu']` (2 tokens) when lowercased.
    - All other 7 English lines (`L01, L03, L04, L05, L07, L08, L09`) have identical token counts (`\Delta = 0`) between cased and lowercased (e.g., `'Bengaluru'` is 4 tokens in both `['B', 'eng', 'al', 'uru']` and `['b', 'eng', 'al', 'uru']`; `' Mysuru'` is 3 tokens in both `[' M', 'ys', 'uru']` and `[' my', 'sur', 'u']`), though their token IDs differ on capitalized words.
    - In `hin`, `0/10` lines change token counts or token IDs.
  - **Recomputed Fertility & `tok/char`:**
    - `eng` fertility: `1.265206` (`1.27`) $\rightarrow$ `1.229329` (`1.23`), $\Delta = -0.035877$ (`-2.84%`).
    - `hin` fertility: `7.448452` (`7.45`) $\rightarrow$ `7.448452` (`7.45`), $\Delta = 0.000000$ (`0.00%`).
    - `hin/eng` fertility ratio: `5.887148` (`5.89x`) $\rightarrow$ `6.058958` (`6.06x`), $\Delta = +0.171810$ (`+2.92%`).
    - `eng` `tok/char`: `0.225636` (`0.226`) $\rightarrow$ `0.219562` (`0.220`), $\Delta = -0.006074$ (`-2.69%`).
- **Interpretation:** Forcing `line.lower()` before a case-sensitive BPE tokenizer (`gpt2`) is a preprocessing bug. Because Latin is bicameral and Devanagari is unicameral, `.lower()` is a strict no-op on Hindi while breaking single-token BPE merges for capitalized English proper nouns and acronyms (`Thursday`, `NASA`, `GPU`), artificially inflating English fertility (`1.23` $\rightarrow$ `1.27`) and understating the `hin/eng` fertility ratio (`6.06x` $\rightarrow$ `5.89x`).
- **Limitations:** Measured on a 10-line sample; on corpora with higher acronym or proper-noun density, the distortion from lowercasing cased BPE tokenizers can be larger, whereas on uncased tokenizers (`bert-base-uncased`) it would have zero effect.

---

### 3. Experiment E3 — Averaging (`Macro-Average` vs. `Micro-Average`)
- **Command / Function:** `compute_metrics(..., agg_mode="micro")` in `my-submission/partA/audit_experiments.py` (replacing `sum(per_line_fertility)/n` at `fertility.py:L67` with `tot_tokens / tot_words` and `tot_tokens / tot_chars`).
- **Observed Result:**
  - `eng` fertility: `1.265206` (`1.27` macro) $\rightarrow$ `99 / 79 = 1.253165` (`1.25` micro), $\Delta = -0.012041$ (`-0.95%`).
  - `hin` fertility: `7.448452` (`7.45` macro) $\rightarrow$ `459 / 62 = 7.403226` (`7.40` micro), $\Delta = -0.045227$ (`-0.61%`).
  - `hin/eng` fertility ratio: `5.887148` (`5.89x` macro) $\rightarrow$ `5.907625` (`5.91x` micro), $\Delta = +0.020477$ (`+0.35%`).
  - `eng` `tok/char`: `0.225636` (`0.226` macro) $\rightarrow$ `99 / 448 = 0.220982` (`0.221` micro), $\Delta = -0.004654$ (`-2.06%`).
  - `hin` `tok/char`: `1.579108` (`1.579` macro) $\rightarrow$ `459 / 290 = 1.582759` (`1.583` micro), $\Delta = +0.003650$ (`+0.23%`).
  - `hin/eng` `tok/char` ratio: `6.998478` (`7.00x` macro) $\rightarrow$ `7.162382` (`7.16x` micro), $\Delta = +0.163904$ (`+2.34%`).
- **Interpretation:** Macro-averaging (`mean(T_i / W_i)`) weights every line equally regardless of length—so `hin` L08 (4 words, fertility `8.75`) has the same weight as `eng` L03 (12 words, fertility `1.083`). Micro-averaging (`sum(T_i) / sum(W_i)`) weights each line by its word count and directly measures total tokens generated per word across a workload, which is the relevant metric for serving capacity and cost. However, because line lengths in `corpus_sample/` do not vary by orders of magnitude (`4–12` words), changing macro to micro only shifts fertility by `<1%` (`1.27` $\rightarrow$ `1.25`, `7.45` $\rightarrow$ `7.40`, ratio `5.89x` $\rightarrow$ `5.91x`).
- **Limitations:** On a production dataset containing both 1-word queries and 500-word paragraphs, macro- vs. micro-averaging can diverge much more strongly than on these relatively uniform 4–12 word sentences.

---

### 4. Experiment E4 — Character Measurement (Unicode Codepoints vs. UTF-8 Bytes vs. Grapheme Clusters)
- **Command / Function:** `compute_metrics(..., char_mode="utf8_bytes")` and `compute_metrics(..., char_mode="graphemes")` using `regex.findall(r"\X", line)` (`regex 2026.9.29` is already installed in `.venv-x64` as a dependency of `tiktoken`).
- **Observed Result:**
  - **Corpus character/unit totals:**
    - `eng`: `448` Unicode codepoints (`len(line)`) = `448` UTF-8 bytes (`len(line.encode("utf-8"))`) = `448` extended grapheme clusters (`\X`). (Excluding spaces: `379` non-space codepoints.)
    - `hin`: `290` Unicode codepoints (`len(line)`) vs. `764` UTF-8 bytes (`len(line.encode("utf-8"))`) vs. `188` extended grapheme clusters (`\X`). (Excluding spaces: `238` non-space codepoints.)
  - **Tokens per character-unit (Macro-average / Micro-average):**
    - **Unicode codepoints (Baseline):** `eng = 0.225636` (`0.226` macro) / `0.220982` (`0.221` micro); `hin = 1.579108` (`1.579` macro) / `1.582759` (`1.583` micro); Ratio = **`7.00x`** macro (`7.16x` micro).
    - **UTF-8 bytes (`E4a`):** `eng = 0.225636` (`0.226` macro) / `0.220982` (`0.221` micro); `hin = 0.598992` (`0.599` macro) / `0.600785` (`0.601` micro); Ratio = **`2.65x`** macro (`2.72x` micro), $\Delta = -62.07\%$.
    - **Extended Grapheme Clusters `\X` (`E4b`):** `eng = 0.225636` (`0.226` macro) / `0.220982` (`0.221` micro); `hin = 2.449732` (`2.450` macro) / `2.441489` (`2.441` micro); Ratio = **`10.86x`** macro (`11.05x` micro), $\Delta = +55.13\%$.
  - **Observed Characters per Word (Testing `REPORT_v0.md` Finding #3):**
    - With `split(" ")`: `eng` = `448 / 79 = 5.671` codepoints/word; `hin` = `290 / 62 = 4.677` codepoints/word.
    - With `split()`: `eng` = `448 / 78 = 5.744` codepoints/word; `hin` = `290 / 61 = 4.754` codepoints/word.
- **Interpretation:**
  1. Changing the character definition does **not** change `fertility (tok/word)` (`1.27` vs `7.45`), which uses words in the denominator, but it **dramatically changes `tok/char`** for Hindi from `1.579` (`7.00x` English) to `0.599` tok/byte (`2.65x` English) or `2.450` tok/grapheme (`10.86x` English).
  2. In ASCII English, 1 grapheme = 1 codepoint = 1 UTF-8 byte (`448` across all three). In Devanagari Hindi, an orthographic syllable (*akshara*, e.g., `फ़्ते` in `हफ़्ते`) combines multiple Unicode codepoints (consonants, vowel *matras*, virama `\u094d`, nukta `\u093c`), and each Devanagari codepoint encodes to 3 UTF-8 bytes (`238 * 3 + 50 ASCII = 764` bytes).
  3. **Direct Refutation of `REPORT_v0.md` Finding #3:** The report claims *"Root cause: Hindi simply has more Unicode characters per word, so any tokenizer will struggle. This is a property of the script, not the tokenizer."* Empirically, Hindi has **17.5% fewer Unicode characters per word** than English (`4.68` vs `5.67`). `gpt2` produces `1.579` tokens per Unicode character (`>1` token/char!) only because `gpt2` is a byte-level BPE tokenizer trained almost exclusively on English that falls back to raw UTF-8 byte tokens (`0.599` tokens/byte $\approx 1.67$ bytes/token) on Devanagari.
- **Limitations:** Extended grapheme clusters (`\X`, UAX #29) approximate orthographic *aksharas* closely, though traditional Indic typographic syllable rules can occasionally differ on complex conjunct clusters.

---

### 5. Experiment E5 — Corpus Comparability (`eng_sample.txt` vs. `hin_sample.txt`)
- **Command / Function:** Line-by-line inspection and `compute_metrics` on the 5 semantically matched pairs in `my-submission/partA/audit_experiments.py`.
- **Observed Result:**
  - **0 out of 10 lines match at the same line index:**
    - `L01`: `eng` = *"Bengaluru International Airport handled record traffic in March."* vs. `hin` = *"मुझे सुबह की चाय बहुत पसंद है।"* (*"I like morning tea very much."*)
    - `L02`: `eng` = *"The Quarterly Review meeting moved to Thursday."* vs. `hin` = *"बेंगलुरु में आज हल्की बारिश हो रही है।"* (*"It is raining lightly in Bengaluru today."*)
    - `L09`: `eng` = *"Do you want tea or coffee?"* vs. `hin` = *"आज दफ़्तर में बहुत काम था।"* (*"There was a lot of work at the office today."*)
  - **Only 5 out of 10 lines have any semantic counterpart in the other file (and 4 of those 5 are at different line numbers):**
    1. `eng` L03 (*"I bought this book yesterday from a small shop near MG Road."*, 12 words, 13 tokens) $\leftrightarrow$ `hin` L03 (*"यह किताब मैंने कल ही खरीदी थी।"* = *"I bought this book just yesterday."*, 7 words, 47 tokens) — **partial match**: English has an extra 6-word prepositional phrase (*"from a small shop near MG Road"*) missing in Hindi.
    2. `eng` L04 (*"Children are playing cricket on the ground."*, 8 tokens) $\leftrightarrow$ `hin` L07 (*"बच्चे मैदान में क्रिकेट खेल रहे हैं।"*, 59 tokens).
    3. `eng` L05 (*"The train arrived exactly on time."*, 7 tokens) $\leftrightarrow$ `hin` L06 (*"ट्रेन ठीक समय पर पहुँची।"*, 40 tokens).
    4. `eng` L07 (*"Please keep the books  in the cupboard."*, 10 tokens) $\leftrightarrow$ `hin` L10 (*"किताबें  अलमारी में रखी हैं।"* = *"The books are kept in the cupboard."*, 45 tokens) — **modality mismatch** (imperative *"Please keep..."* vs. declarative *"are kept"*).
    5. `eng` L08 (*"We are visiting Mysuru next week."*, 9 tokens) $\leftrightarrow$ `hin` L04 (*"हम अगले हफ़्ते मैसूर जा रहे हैं।"*, 51 tokens).
  - **5 lines in each file have no counterpart at all:**
    - Unmatched `eng` lines: `L01` (Airport traffic), `L02` (Quarterly Review), `L06` (NASA and ISRO), `L09` (tea or coffee), `L10` (GPU cluster).
    - Unmatched `hin` lines: `L01` (morning tea), `L02` (rain in Bengaluru), `L05` (*"क्या तुमने खाना खा लिया?"* = *"Have you eaten food?"*), `L08` (*"मुझे थोड़ा पानी चाहिए।"* = *"I want some water."*), `L09` (work at the office).
  - **Quantitative impact of restricting to the 5 matched pairs:**
    - `eng` fertility: `1.265206` (`1.27`) $\rightarrow$ `1.228571` (`1.23`), $\Delta = -0.036634$ (`-2.90%`).
    - `hin` fertility: `7.448452` (`7.45`) $\rightarrow$ `7.585714` (`7.59`), $\Delta = +0.137262$ (`+1.84%`).
    - `hin/eng` macro fertility ratio: `5.887148` (`5.89x`) $\rightarrow$ `6.174419` (`6.17x`), $\Delta = +0.287271$ (`+4.88%`).
    - Corpus total token ratio (`hin_tokens / eng_tokens`): `459 / 99 = 4.636x` (all 10 lines) $\rightarrow$ `242 / 47 = 5.149x` (5 matched pairs), $\Delta = +11.06\%$.
- **Interpretation:** The corpora are **not semantically parallel**. The English sample has longer, technical/proper-noun sentences (`78` actual words vs. `61` in Hindi—28% more words in English), which confounds any direct comparison of token counts per request (`REPORT_v0.md` Finding #1 claims *"Serving Hindi will cost us roughly 6x more per request than English"*). Evaluating cross-language serving cost per request requires a line-by-line parallel corpus holding semantic content constant.
- **Limitations:** Even the 5 matched pairs are small (`n = 5`) and include minor translation divergences (`eng` L03's extra clause and `eng` L07's imperative phrasing).

---

### 6. Experiment E6 — Other Implementation Choices (`random.seed`, `NFC` Normalization, `add_special_tokens=False`)
- **Command / Function:** Section `E6` of `my-submission/partA/audit_experiments.py`.
- **Observed Result:**
  - **E6a (`random.seed(1337)` at `fertility.py:L25`):** Changing the seed (`random.seed(999999)`) or removing it yields `eng_fert_diff = 0.0000000000` and `hin_fert_diff = 0.0000000000`. The `random` module is imported at `L21` and seeded at `L25`, but never called anywhere in `fertility.py`.
  - **E6b (`unicodedata.normalize("NFC", line)` at `fertility.py:L49`):** Comparing raw lines (`read_lines_raw`) against NFC-normalized lines (`read_lines_nfc`) yields `0/10` lines altered in `eng_sample.txt` and `0/10` lines altered in `hin_sample.txt`, with `eng_fert_diff = 0.0000000000` and `hin_fert_diff = 0.0000000000`. Furthermore, a counterfactual `NFD` normalization on `hin_sample.txt` also produces `290` codepoints and `459` tokens (`0` difference), because Devanagari nukta characters (`\u093c` in `हफ़्ते`, `दफ़्तर`, `थोड़ा`) are excluded from canonical composition in Unicode and are therefore already decomposed in both `NFC` and `NFD`.
  - **E6c (`add_special_tokens=False` at `fertility.py:L33`):** When `--tokenizer gpt2` is passed, `spec.startswith("hf:")` is `False`, so `fertility.py` executes the `else` branch (`tiktoken.get_encoding("gpt2").encode`), which never executes line 33 and adds `0` special tokens by default. (Additionally, `transformers` is not installed in `.venv-x64`.)
- **Interpretation:**
  - **`random.seed(1337)`:** Completely irrelevant dead code; tokenization is deterministic.
  - **`unicodedata.normalize("NFC", line)`:** Has zero measurable effect on `corpus_sample/` (files are already NFC), and is actually good defensive engineering practice for general corpora so canonically equivalent strings are tokenized consistently.
  - **`add_special_tokens=False`:** Irrelevant to the `--tokenizer gpt2` run (unexecuted branch), and **methodologically correct** when `hf:` tokenizers are used—it matches `tiktoken` (which adds no BOS/EOS tokens) and prevents per-line `[CLS]`/`[SEP]` or `<bos>`/`<eos>` framing tokens from artificially inflating subword fertility on short lines.
  - None of these three implementation choices is a bug.
- **Limitations:** While NFC vs. NFD has zero effect on `eng_sample.txt` and `hin_sample.txt`, on external corpora containing precomposed Latin accented characters (e.g., `é` `U+00E9` vs. `e` + `\u0301`) or non-canonical Devanagari presentation forms (`U+0958`–`U+095F`), NFC normalization would standardize character and token counts.

---

## Summary Table of Experiments E1–E6

| Experiment | Command / Script | Before $\rightarrow$ After Result (Observed) | Conclusion (Observed vs. Interpretation) | Materially Changes Reported Fertility? |
| :--- | :--- | :--- | :--- | :--- |
| **E1 — Word counting** (`split(" ")` vs. `split()`) | `.\.venv-x64\Scripts\python.exe my-submission/partA/audit_experiments.py` (`E1`) | **Words:** `eng` `79` $\rightarrow$ `78`; `hin` `62` $\rightarrow$ `61`<br>**Fertility:** `eng` `1.27` (`1.2652`) $\rightarrow$ `1.28` (`1.2831`); `hin` `7.45` (`7.4485`) $\rightarrow$ `7.60` (`7.5985`)<br>**Ratio:** `5.89x` (`5.8871`) $\rightarrow$ `5.92x` (`5.9221`) | **Confirmed Code Bug:** Double spaces in `eng` L07 and `hin` L10 create phantom empty strings `""` in `split(" ")`, inflating word counts by $+1$ and depressing fertility by $1.4\text{--}2.0\%$. | **Minor / Moderate** ($\Delta\text{hin} = +0.15$ tok/word, ratio `5.89x` $\rightarrow$ `5.92x`; does not overturn the $\sim 6\times$ gap). |
| **E2 — Lowercasing** (`line.lower()` vs. cased) | `.\.venv-x64\Scripts\python.exe my-submission/partA/audit_experiments.py` (`E2`) | **Tokens:** `eng` `99` $\rightarrow$ `96` (`L02, L06, L10` $-1$ each); `hin` `459` $\rightarrow$ `459`<br>**Fertility:** `eng` `1.27` (`1.2652`) $\rightarrow$ `1.23` (`1.2293`); `hin` `7.45` $\rightarrow$ `7.45`<br>**Ratio:** `5.89x` (`5.8871`) $\rightarrow$ `6.06x` (`6.0590`) | **Confirmed Preprocessing Bug:** `.lower()` breaks cased `gpt2` merges (`Thursday`, `NASA`, `GPU`) in bicameral English ($+3$ tokens) while having zero effect on unicameral Hindi. | **Moderate** ($\Delta\text{eng} = -0.04$ tok/word (`-2.84%`), pushes `hin/eng` ratio from `5.89x` over `6.06x` (`+2.92%`)). |
| **E3 — Averaging** (Macro vs. Micro) | `.\.venv-x64\Scripts\python.exe my-submission/partA/audit_experiments.py` (`E3`) | **Fertility:** `eng` `1.27` (`1.2652`) $\rightarrow$ `1.25` (`99/79 = 1.2532`); `hin` `7.45` (`7.4485`) $\rightarrow$ `7.40` (`459/62 = 7.4032`)<br>**Ratio:** `5.89x` (`5.8871`) $\rightarrow$ `5.91x` (`5.9076`)<br>**`tok/char`:** `eng` `0.226` $\rightarrow$ `0.221`; `hin` `1.579` $\rightarrow$ `1.583` | **Metric-Design Mismatch:** Macro-average weights a 4-word line (`hin` L08) equally with a 12-word line (`eng` L03); micro-average (`tot_tokens / tot_words`) is required for serving cost. | **No (Small on this sample)** ($<1\%$ shift in fertility: `1.27` $\rightarrow$ `1.25`, `7.45` $\rightarrow$ `7.40`, ratio `5.89x` $\rightarrow$ `5.91x`). |
| **E4 — Character measurement** (Codepoints vs. UTF-8 bytes & Graphemes `\X`) | `.\.venv-x64\Scripts\python.exe my-submission/partA/audit_experiments.py` (`E4a`, `E4b`) | **Counts:** `eng` `448` codepoints = `448` bytes = `448` graphemes (`0.226` across all 3); `hin` `290` codepoints (`1.579` tok/cp, `7.00x`) $\rightarrow$ `764` UTF-8 bytes (`0.599` tok/B, **`2.65x`**) $\rightarrow$ `188` graphemes (`2.450` tok/G, **`10.86x`**)<br>**Chars/word:** `eng` `5.67` vs. `hin` `4.68` | **Major Conceptual Flaw in `tok/char` & Refutes Report Finding #3:** Hindi has **17.5% fewer** Unicode chars/word (`4.68`) than English (`5.67`). High `tok/char` is due to 3-byte UTF-8 fallback in `gpt2`, not longer Hindi words. | **No change to `tok/word` (`1.27`, `7.45`), but MASSIVELY changes `tok/char`** (`1.579` $\rightarrow$ `0.599` tok/byte (`-62%`) or `2.450` tok/grapheme (`+55%`)) and **invalidates Report Finding #2 & #3**. |
| **E5 — Corpus comparability** (10 non-parallel lines vs. 5 matched pairs) | `.\.venv-x64\Scripts\python.exe my-submission/partA/audit_experiments.py` (`E5`) | **Alignment:** `0/10` lines align by index; only `5/10` have loose counterparts (`eng` L3,4,5,7,8 $\leftrightarrow$ `hin` L3,7,6,10,4)<br>**Fertility (5-pair):** `eng` `1.27` $\rightarrow$ `1.23`; `hin` `7.45` $\rightarrow$ `7.59`; **Ratio:** `5.89x` $\rightarrow$ **`6.17x`** (`+4.88%`)<br>**Total tok ratio:** `459/99 = 4.64x` $\rightarrow$ `242/47 = 5.15x` (`+11.1%`) | **Confirmed Corpus Flaw:** Corpora are not parallel (`eng` has 28% more words and different topics like NASA/GPU/Airport vs. tea/rain/water), confounding per-request cost claims. | **Materially affects cross-lingual ratio & per-request cost** (shifts fertility ratio from `5.89x` to `6.17x` (`+4.9%`) and total token ratio from `4.64x` to `5.15x` (`+11.1%`)). |
| **E6 — Other choices** (`NFC`, `random.seed(1337)`, `add_special_tokens=False`) | `.\.venv-x64\Scripts\python.exe my-submission/partA/audit_experiments.py` (`E6a–c`) | **`random.seed`:** $\Delta = 0.000000$<br>**`NFC` vs Raw / `NFD`:** `0/10` lines changed, $\Delta = 0.000000$<br>**`add_special_tokens=False`:** Branch unexecuted for `gpt2`; adds `0` framing tokens | **Harmless / Correct (Not Bugs):** `random` is unused; sample files are already NFC (and NFC is good practice); `add_special_tokens=False` is unexecuted for `gpt2` and methodologically right for `hf:`. | **No** ($\Delta = 0.00\%$ across all metrics). |
| **Combined Code Fixes (`E1 + E2 + E3`)** | `.\.venv-x64\Scripts\python.exe my-submission/partA/audit_experiments.py` (`Combined`) | **Fertility:** `eng` `1.27` (`1.2652`) $\rightarrow$ `1.23` (`96/78 = 1.2308`); `hin` `7.45` (`7.4485`) $\rightarrow$ `7.52` (`459/61 = 7.5246`)<br>**Ratio:** `5.89x` (`5.8871`) $\rightarrow$ **`6.11x`** (`6.1137`, `+3.85%`) | Applying `split()` + cased text + micro-averaging simultaneously lowers `eng` fertility and raises `hin` fertility. | **Moderate on `tok/word` (`5.89x` $\rightarrow$ `6.11x`, `+3.85%`)**; `gpt2`'s severe Hindi penalty is real, but its root cause in `REPORT_v0.md` is misdiagnosed. |


---

## Entry 4 — Part A1: Multilingual Parallel Corpus Construction & Validation
**Timestamp:** `2026-10-23 15:40 IST`

### Objective
Construct a documented, reproducible, multi-way parallel evaluation corpus covering English (`eng`), Hindi (`hin`), and at least two Dravidian languages (`tam` Tamil, `tel` Telugu, `kan` Kannada, `mal` Malayalam) to replace the 10-line non-parallel `corpus_sample/`. Keep raw and processed text separate, validate encoding/empty-line/duplicate/whitespace/normalization properties, and document corpus limitations.

### Commands Run
```powershell
.\.venv-x64\Scripts\python.exe my-submission/partA/build_corpus.py
```
- **Script:** `my-submission/partA/build_corpus.py`
- **Raw Output Saved:** `my-submission/partA/corpus_validation_output.txt`
- **Structured JSON Summary:** `my-submission/partA/corpus/corpus_validation_summary.json`
- **Corpus Documentation:** `my-submission/partA/corpus_documentation.md`

### Observed Facts & Measurements

1. **Source Archive & Provenance (Verified):**
   - Downloaded official FLORES-200 benchmark archive from `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`.
   - **Size:** `25,585,843` bytes (`SHA-256: b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6`).
   - **License:** `CC-BY-SA-4.0`.
   - **Split extracted:** `devtest` (`1,012` multi-way parallel sentences per language across `281` unique English Wikimedia source articles: `351` `wikibooks`, `341` `wikinews`, `320` `wikivoyage`).
   - **6 Languages extracted:** `eng` (`eng_Latn`), `hin` (`hin_Deva`), `tam` (`tam_Taml`), `tel` (`tel_Telu`), `kan` (`kan_Knda`), `mal` (`mal_Mlym`).

2. **Unexpected Anomalies Discovered in Raw FLORES-200 (`my-submission/partA/corpus/raw/*.txt`):**
   - **Zero empty lines, zero duplicate lines, and 100% valid UTF-8** across all 6 languages (`1,012` lines each).
   - **Non-NFC Unicode characters in raw FLORES-200:**
     - `hin_Deva.devtest` has **93 lines** containing non-NFC characters (`+122` codepoints added under NFC). Specifically, raw `hin_Deva` contains precomposed Devanagari nukta singletons (`U+095B` `ज़`, `U+095C` `ड़`, etc.) which are Unicode canonical composition exclusions that decompose into base consonant + `\u093c` nukta under `NFC`!
     - The four Dravidian languages contain **14 non-NFC lines** (`kan`: 7 lines, $\Delta = -13$ codepoints; `mal`: 4 lines, $\Delta = -4$ codepoints; `tam`: 2 lines, $\Delta = -2$ codepoints; `tel`: 1 line, $\Delta = -5$ codepoints) where decomposed two-part vowel signs compose into single codepoints under `NFC`.
   - **Repeated internal spaces (`"  "`) in raw FLORES-200:**
     - `eng`: `0` lines (`+0` phantom empty words in `split(" ")`)
     - `hin`: `6` lines (`+27` phantom empty words in `split(" ")`)
     - `tam`: `75` lines (`+100` phantom empty words in `split(" ")`)
     - `tel`: `136` lines (`+274` phantom empty words in `split(" ")`)
     - `kan`: `213` lines (`+477` phantom empty words in `split(" ")`)
     - `mal`: `54` lines (`+71` phantom empty words in `split(" ")`)
   - **CSV-escaped double quotes (`""` / `"""`) in raw FLORES-200:**
     - Present in `60` `tam` lines, `68` `tel` lines, and `58` `mal` lines (e.g., Sentence #0001), versus `0` in `eng`, `hin`, and `kan`.

3. **Validated Counts on Processed Corpus (`my-submission/partA/corpus/processed/*.txt`):**
   - Preprocessing applied: `line.strip()` $\rightarrow$ `unicodedata.normalize("NFC", line)` $\rightarrow$ `" ".join(line.split())` (preserving original casing).
   - All 6 processed files have **`1,012` lines, `0` empty lines, `0` duplicate lines, `0` non-NFC lines, and `0` multi-space lines**:
     - **`eng` (English, Latin):** `1,012` lines | `21,901` words | `131,966` codepoints (`111,077` non-space) | `131,966` graphemes | `132,096` UTF-8 line bytes (`133,108` file bytes) | `6.03` CP/word | `6.03` graphemes/word.
     - **`hin` (Hindi, Devanagari):** `1,012` lines | `25,643` words | `131,174` codepoints (`106,543` non-space) | `85,951` graphemes | `337,433` UTF-8 line bytes (`338,445` file bytes) | `5.12` CP/word | `3.35` graphemes/word.
     - **`tam` (Tamil, Tamil):** `1,012` lines | `16,775` words | `154,031` codepoints (`138,268` non-space) | `99,624` graphemes | `421,535` UTF-8 line bytes (`422,547` file bytes) | `9.18` CP/word | `5.94` graphemes/word.
     - **`tel` (Telugu, Telugu):** `1,012` lines | `16,938` words | `132,226` codepoints (`116,300` non-space) | `76,328` graphemes | `353,422` UTF-8 line bytes (`354,434` file bytes) | `7.81` CP/word | `4.51` graphemes/word.
     - **`kan` (Kannada, Kannada):** `1,012` lines | `16,100` words | `137,650` codepoints (`122,562` non-space) | `89,994` graphemes | `374,964` UTF-8 line bytes (`375,976` file bytes) | `8.55` CP/word | `5.59` graphemes/word.
     - **`mal` (Malayalam, Malayalam):** `1,012` lines | `14,930` words | `149,261` codepoints (`135,343` non-space) | `79,357` graphemes | `411,670` UTF-8 line bytes (`412,682` file bytes) | `10.00` CP/word | `5.32` graphemes/word.

### Interpretation
1. **Why Raw FLORES-200 Validates Our Part A2 Findings (`E1` & `E6b`):**
   - While NFC normalization had zero effect on the 10-line `corpus_sample/`, on real FLORES-200 `devtest` it normalizes **107 lines** across Hindi and the four Dravidian languages.
   - Similarly, `fertility.py`'s `line.split(" ")` bug would inject **+477 phantom empty words in Kannada (21.0% of lines affected)** and **+274 phantom empty words in Telugu (13.4% of lines affected)** while injecting **0 in English**, severely distorting raw cross-lingual word fertility if whitespace is not collapsed or split via `line.split()`.
2. **Morphological Typology Confounds `tokens / word` Across Language Families:**
   - Across the exact same `1,012` parallel sentences, analytic/postpositional Hindi uses **`25,643` words** (`5.12` codepoints/word, fewer than English's `6.03`), whereas agglutinative Dravidian languages fuse suffixes onto stems and use only **`14,930` (`mal`) to `16,938` (`tel`) words** (`7.81` to `10.00` codepoints/word).

### Assumptions & Limitations
- **Domain & Register Caveat:** FLORES-200 consists of formal encyclopedic/news/travel prose (`wikibooks`, `wikinews`, `wikivoyage`) translated from English; it does not cover conversational chat, code-switching (Hinglish/Tanglish in Latin script), or source code/math.
- **Upstream Translation & Quote Artifacts:** Raw FLORES-200 contains CSV-escaped quotes (`""`) in ~6% of `tam`/`tel`/`mal` lines and occasional translator clause omissions (e.g., `tel` L1012). We preserved punctuation without manual editing to keep preprocessing strictly deterministic and documented these upstream artifacts.


---

## Entry 5 — Part A3: Multilingual Tokenizer & Measurement Comparison
**Timestamp:** `2026-10-23 15:58 IST`

### Objective
Compare multiple tokenizers across all 6 languages (`eng`, `hin`, `tam`, `tel`, `kan`, `mal`) on the `1,012` aligned sentences of our processed FLORES-200 `devtest` corpus (`my-submission/partA/corpus/processed/`). Report per-language totals, corpus-level micro-averages, and sentence-level macro-averages across five denominators (`whitespace words`, `Unicode codepoints`, `Extended Grapheme Clusters \X`, `UTF-8 bytes`, and `parallel sentence`), and determine which single metric should drive a routing-and-cost decision.

### Environment & Tokenizer Availability Check (Observed)
```powershell
.\.venv-x64\Scripts\python.exe -m pip list
.\.venv-x64\Scripts\python.exe -c "import tiktoken; print(tiktoken.__version__, tiktoken.list_encoding_names())"
```
- **Observed Result:**
  - `tiktoken 0.14.0` and `regex 2026.9.29` are installed in `.venv-x64`.
  - `transformers`, `tokenizers`, and `sentencepiece` are **not** installed (`ModuleNotFoundError: No module named 'transformers'` when probing `hf:xlm-roberta-base`).
  - `tiktoken` provides three BPE encodings spanning three generations of multilingual/Indic support:
    1. **`gpt2`** (`n_vocab = 50,257`): Already cached locally (`6d1cbeee...` + `6c7ea1a7...`).
    2. **`cl100k_base`** (`n_vocab = 100,277`, GPT-3.5 / GPT-4): Downloaded legitimately over HTTPS from OpenAI's public blob store (`9b5ad71b...`, `1,681,126` B).
    3. **`o200k_base`** (`n_vocab = 200,019`, GPT-4o Indic-aware BPE): Downloaded legitimately over HTTPS from OpenAI's public blob store (`fb374d41...`, `3,613,922` B).
  - All three encodings use `enc.encode(line)` with `0` special tokens added (`add_special_tokens=False` equivalent) and preserve original casing (`lowercase=False`).

### Commands & Scripts Run
```powershell
# 1. Full corrected multi-tokenizer, multi-denominator evaluation (cased, split(), micro & macro)
.\.venv-x64\Scripts\python.exe my-submission/partA/compare_tokenizers.py

# 2. Unmodified starter-kit fertility.py on the 6 processed FLORES-200 languages (gpt2 vs o200k_base)
.\.venv-x64\Scripts\python.exe fertility.py --corpus eng=my-submission/partA/corpus/processed/eng.txt --corpus hin=my-submission/partA/corpus/processed/hin.txt --corpus tam=my-submission/partA/corpus/processed/tam.txt --corpus tel=my-submission/partA/corpus/processed/tel.txt --corpus kan=my-submission/partA/corpus/processed/kan.txt --corpus mal=my-submission/partA/corpus/processed/mal.txt --tokenizer gpt2

.\.venv-x64\Scripts\python.exe fertility.py --corpus eng=my-submission/partA/corpus/processed/eng.txt --corpus hin=my-submission/partA/corpus/processed/hin.txt --corpus tam=my-submission/partA/corpus/processed/tam.txt --corpus tel=my-submission/partA/corpus/processed/tel.txt --corpus kan=my-submission/partA/corpus/processed/kan.txt --corpus mal=my-submission/partA/corpus/processed/mal.txt --tokenizer o200k_base
```
- **Script:** `my-submission/partA/compare_tokenizers.py`
- **Raw Output Saved:** `my-submission/partA/a3_tokenizer_comparison_output.txt`
- **JSON Data Saved:** `my-submission/partA/a3_tokenizer_comparison.json`
- **Detailed Analysis Writeup:** `my-submission/partA/a3_analysis.md`

### Observed Results (1,012 Aligned Sentences per Language, Original Casing Preserved)

#### 1. Corpus-Level Totals & Micro-Averaged Metrics (`sum(tokens) / sum(denom)`)
- **`gpt2` (`50,257` vocab):**
  - `eng`: `27,044` tokens | `26.72` tok/sent (`1.00x` parity) | `1.235` tok/word (`1.00x`) | `0.205` tok/grapheme | `0.205` tok/CP | `0.205` tok/B (`4.88` B/tok)
  - `hin`: `200,682` tokens | `198.30` tok/sent (**`7.42x` parity**) | `7.826` tok/word (`6.34x`) | `2.335` tok/grapheme (`11.39x`) | `1.530` tok/CP (`7.47x`) | `0.595` tok/B (`1.68` B/tok)
  - `tam`: `420,071` tokens | `415.09` tok/sent (**`15.53x` parity**) | `25.041` tok/word (`20.28x`) | `4.217` tok/grapheme (`20.58x`) | `2.727` tok/CP (`13.31x`) | `0.997` tok/B (`1.00` B/tok)
  - `tel`: `350,509` tokens | `346.35` tok/sent (**`12.96x` parity**) | `20.694` tok/word (`16.76x`) | `4.592` tok/grapheme (`22.41x`) | `2.651` tok/CP (`12.94x`) | `0.992` tok/B (`1.01` B/tok)
  - `kan`: `366,989` tokens | `362.64` tok/sent (**`13.57x` parity**) | `22.794` tok/word (`18.46x`) | `4.078` tok/grapheme (`19.90x`) | `2.666` tok/CP (`13.01x`) | `0.979` tok/B (`1.02` B/tok)
  - `mal`: `409,912` tokens | `405.05` tok/sent (**`15.16x` parity**) | `27.456` tok/word (`22.23x`) | `5.165` tok/grapheme (`25.21x`) | `2.746` tok/CP (`13.40x`) | `0.996` tok/B (`1.00` B/tok)
- **`cl100k_base` (`100,277` vocab):**
  - `eng`: `27,182` tokens | `26.86` tok/sent (`1.00x` parity) | `1.241` tok/word (`1.00x`) | `0.206` tok/grapheme | `0.206` tok/CP | `0.206` tok/B (`4.86` B/tok)
  - `hin`: `129,567` tokens | `128.03` tok/sent (**`4.77x` parity**) | `5.053` tok/word (`4.07x`) | `1.507` tok/grapheme (`7.32x`) | `0.988` tok/CP (`4.80x`) | `0.384` tok/B (`2.60` B/tok)
  - `tam`: `207,687` tokens | `205.22` tok/sent (**`7.64x` parity**) | `12.381` tok/word (`9.98x`) | `2.085` tok/grapheme (`10.12x`) | `1.348` tok/CP (`6.55x`) | `0.493` tok/B (`2.03` B/tok)
  - `tel`: `225,022` tokens | `222.35` tok/sent (**`8.28x` parity**) | `13.285` tok/word (`10.70x`) | `2.948` tok/grapheme (`14.31x`) | `1.702` tok/CP (`8.26x`) | `0.637` tok/B (`1.57` B/tok)
  - `kan`: `240,364` tokens | `237.51` tok/sent (**`8.84x` parity**) | `14.929` tok/word (`12.03x`) | `2.671` tok/grapheme (`12.97x`) | `1.746` tok/CP (`8.48x`) | `0.641` tok/B (`1.56` B/tok)
  - `mal`: `242,859` tokens | `239.98` tok/sent (**`8.93x` parity**) | `16.267` tok/word (`13.11x`) | `3.060` tok/grapheme (`14.86x`) | `1.627` tok/CP (`7.90x`) | `0.590` tok/B (`1.70` B/tok)
- **`o200k_base` (`200,019` vocab — Multilingual / Indic-aware):**
  - `eng`: `26,873` tokens | `26.55` tok/sent (`1.00x` parity) | `1.227` tok/word (`1.00x`) | `0.204` tok/grapheme | `0.204` tok/CP | `0.203` tok/B (`4.92` B/tok)
  - `hin`: `42,241` tokens | `41.74` tok/sent (**`1.57x` parity**) | `1.647` tok/word (`1.34x`) | `0.491` tok/grapheme (`2.41x`) | `0.322` tok/CP (`1.58x`) | `0.125` tok/B (`7.99` B/tok)
  - `tam`: `53,114` tokens | `52.48` tok/sent (**`1.98x` parity**) | `3.166` tok/word (`2.58x`) | `0.533` tok/grapheme (`2.62x`) | `0.345` tok/CP (`1.69x`) | `0.126` tok/B (`7.94` B/tok)
  - `tel`: `51,610` tokens | `51.00` tok/sent (**`1.92x` parity**) | `3.047` tok/word (`2.48x`) | `0.676` tok/grapheme (`3.32x`) | `0.390` tok/CP (`1.92x`) | `0.146` tok/B (`6.85` B/tok)
  - `kan`: `52,534` tokens | `51.91` tok/sent (**`1.95x` parity**) | `3.263` tok/word (`2.66x`) | `0.584` tok/grapheme (`2.87x`) | `0.382` tok/CP (`1.87x`) | `0.140` tok/B (`7.14` B/tok)
  - `mal`: `52,515` tokens | `51.89` tok/sent (**`1.95x` parity**) | `3.517` tok/word (`2.87x`) | `0.662` tok/grapheme (`3.25x`) | `0.352` tok/CP (`1.73x`) | `0.128` tok/B (`7.84` B/tok)

#### 2. Observed Token Reduction (`o200k_base` vs. `gpt2`)
- `eng`: `27,044` $\rightarrow$ `26,873` (`-0.63%`, `1.01x` reduction)
- `hin`: `200,682` $\rightarrow$ `42,241` (**`-78.95%`**, **`4.75x` fewer tokens**)
- `tam`: `420,071` $\rightarrow$ `53,114` (**`-87.36%`**, **`7.91x` fewer tokens**)
- `tel`: `350,509` $\rightarrow$ `51,610` (**`-85.28%`**, **`6.79x` fewer tokens**)
- `kan`: `366,989` $\rightarrow$ `52,534` (**`-85.69%`**, **`6.99x` fewer tokens**)
- `mal`: `409,912` $\rightarrow$ `52,515` (**`-87.19%`**, **`7.81x` fewer tokens**)

### Interpretation
1. **Why `REPORT_v0.md` Finding #3 Is Completely Refuted:**
   - `REPORT_v0.md` claimed that high Hindi fertility is *"a property of the script, not the tokenizer"* so *"any tokenizer will struggle"*.
   - On the exact same 1,012 sentences, switching from `gpt2` to `o200k_base` reduces Hindi token volume by **4.75× (`-78.95%`)** and Dravidian token volume by **6.79×–7.91× (`-85.3%` to `-87.4%`)**!
   - On `gpt2`, Dravidian scripts experience **98–99.7% raw UTF-8 byte fallback** (`0.979–0.997` tokens/byte, i.e., `1.00–1.02` bytes/token). On `o200k_base`, Indic tokens span `6.85–7.99` UTF-8 bytes (`0.125–0.146` tokens/byte).
2. **Why `Tokens / Parallel Sentence` (Cross-Lingual Token Parity $\frac{\sum T_{\text{lang}}}{\sum T_{\text{eng}}}$) Is the Right Metric for Routing & Cost:**
   - On `o200k_base`, `tok/word` makes agglutinative Malayalam (`3.517` tok/word = `2.87x` English) look **2.14× worse** than analytic/postpositional Hindi (`1.647` tok/word = `1.34x` English), only because Malayalam packs the same 1,012 sentences into `14,930` long agglutinated words while Hindi uses `25,643` words.
   - When we hold the **semantic message (parallel sentence)** constant, all four Dravidian languages cost almost identically: **`1.92x–1.98x` English (`51.0–52.5` tok/sent)** vs. **`1.57x` English (`41.7` tok/sent)** for Hindi!
   - Meanwhile, `tok/byte` (`0.125` on Hindi vs `0.203` on English) makes Hindi look *cheaper* than English because each Devanagari character is 3 UTF-8 bytes of Unicode block overhead.
   - Therefore, **Corpus-Level Micro-Averaged Token Parity per Parallel Sentence** ($\frac{\sum T_{\text{lang}}}{\sum T_{\text{eng}}}$) is the single number that directly predicts GPU prefill/decode FLOPs and KV-cache footprint per semantically equivalent user request.

### Limitations
- **Benchmark Domain vs. Live Production:** FLORES-200 `devtest` (`1,012` sentences) is formal encyclopedic/news/travel prose; live chat traffic includes code, JSON, and Romanized Indic (Hinglish/Tanglish).
- **Tokenization Efficiency $\neq$ Model Quality:** Lower fertility (`o200k_base`) proves vocabulary compression efficiency and lower serving cost per sentence, not downstream generation quality or accuracy in Indic languages.


---

## Entry 6 — Part A3 Focused Validation & Reconciliation Pass
**Timestamp:** `2026-10-23 16:23 IST`

### Objective
Perform a focused validation pass on the Part A3 results in `my-submission/partA/a3_tokenizer_comparison.json`, `my-submission/partA/a3_tokenizer_comparison_output.txt`, and `my-submission/partA/a3_analysis.md` without unnecessarily rerunning corpus generation or modifying original starter-kit files.

### Validation Checks Run
```powershell
.\.venv-x64\Scripts\python.exe -c "import json, inspect, tiktoken; ..."
```

### Verified Findings & Corrections Applied to `a3_analysis.md`

1. **Measurement Reconciliation:**
   - Programmatically verified all mathematical identities in `my-submission/partA/a3_tokenizer_comparison.json` across all 3 tokenizers (`gpt2`, `cl100k_base`, `o200k_base`) and all 6 languages (`eng`, `hin`, `tam`, `tel`, `kan`, `mal`):
     - `micro_tok_per_sent == tot_tokens / 1012`
     - `micro_tok_per_word == tot_tokens / tot_words`
     - `micro_tok_per_cp == tot_tokens / tot_codepoints`
     - `micro_tok_per_grapheme == tot_tokens / tot_graphemes`
     - `micro_tok_per_byte == tot_tokens / tot_utf8_bytes`
     - `micro_bytes_per_tok == tot_utf8_bytes / tot_tokens`
     - `parity_ratio_vs_eng == tot_tokens / eng['tot_tokens']`
   - **Display-Rounding Clarifications Added to `a3_analysis.md`:**
     - Expanded rate columns in `a3_analysis.md` Section 3 from 3 decimals to **4 decimals** alongside unrounded 4-decimal ratios (e.g., `1.0035` B/tok on `gpt2` `tam` and `1.0043` B/tok on `gpt2` `mal` rather than only 2-decimal `1.00` B/tok, and `3.12x` [`3.1152x`] for `cl100k_base` `kan` `tok/byte` ratio vs. `eng` where `a3_analysis.md` previously had a 1-ulp typo `3.11x`).
     - Added an explicit note in Section 4 explaining that `Diff (Macro - Micro)` is computed from unrounded floats before 4-decimal rounding (e.g., `o200k_base` `hin`: `1.669549 - 1.647272 = +0.022276` $\rightarrow$ `+0.0223`, whereas subtracting the 4-decimal rounded columns `1.6695 - 1.6473` yields `+0.0222`; `gpt2` `kan`: `22.995088 - 22.794348 = +0.200741` $\rightarrow$ `+0.2007`).
     - Standardized the exact token-reduction range in prose to `4.75x–7.91x` (`-78.95%` to `-87.36%`) rather than rounding to `4.8x–7.9x`.

2. **Precise `tiktoken` Special-Token Semantics Verified:**
   - Inspected `inspect.signature(enc.encode)` and `enc.special_tokens_set` for all three `tiktoken` encodings:
     - Signature: `(text: 'str', *, allowed_special: "Literal['all'] | AbstractSet[str]" = set(), disallowed_special: "Literal['all'] | Collection[str]" = 'all') -> 'list[int]'`
     - Special token sets: `gpt2` = `{'<|endoftext|>'}`; `cl100k_base` = `{'<|endofprompt|>', '<|fim_middle|>', '<|fim_prefix|>', '<|fim_suffix|>', '<|endoftext|>'}`; `o200k_base` = `{'<|endoftext|>', '<|endofprompt|>'}`.
   - Clarified in `a3_analysis.md` Section 1.2 that `tiktoken` does **not** share Hugging Face `AutoTokenizer`'s API or post-processor template: `tiktoken`'s `enc.encode(line)` never prepends/appends sequence framing tokens (`<bos>`/`<eos>` or `[CLS]`/`[SEP]`); its `allowed_special`/`disallowed_special` flags only govern whether literal special-token strings inside `text` are allowed or raise a `ValueError`.

3. **Tightened Interpretations & Metric Labels:**
   - **Separated `Tokens / UTF-8 Byte` and `UTF-8 Bytes / Token`:** Placed them in distinct table columns and defined them explicitly as exact reciprocals ($\frac{\sum T_i}{\sum B_i}$ vs. $\frac{\sum B_i}{\sum T_i}$). Kept `Sentence-Parity Ratio vs. eng` in a separate column from `Word-Fertility Ratio vs. eng`.
   - **Aligned Sentences as Proxy:** Qualified that the `1,012` aligned FLORES-200 sentences are a **proxy for comparable translated content**, not mathematically identical semantic propositions.
   - **Removed Unmeasured Punctuation/Boundary Claim:** Removed the unmeasured speculation that short sentences have higher punctuation/boundary density per word; replaced it with the exact empirical observation that macro-averaged `tok/word` exceeds micro-averaged `tok/word` by `+0.0090` to `+0.2287` across all 6 languages, reflecting a slight negative covariance between sentence word count $W_i$ and line-level $T_i / W_i$.
   - **Qualified `gpt2` Dravidian `Tokens / UTF-8 Byte` (`0.9787`–`0.9965`):** Clarified that an aggregate rate near `1.0` token per UTF-8 byte (`1.0035`–`1.0217` bytes/token) is a corpus-level average across 3-byte Dravidian codepoints, 1-byte ASCII spaces/punctuation/digits, unmerged byte fallback tokens, and occasional 2-byte BPE merges of recurring UTF-8 prefix bytes—not a fixed token count per character.
   - **Compression vs. Quality & Latency:** Explicitly distinguished subword compression under `o200k_base` from downstream model quality/fluency and end-to-end serving latency (which also depends on model architecture, 200k-vocab LM-head softmax cost, batching, and generation length).


---

## Entry 7 — Part A3 Final Disclosure Check & Part A4 Recommendation Memo
**Timestamp:** `2026-10-23 21:24 IST`

### Objective
1. Complete the final pre-A4 check on `my-submission/partA/a3_analysis.md`, confirming all referenced files exist and explicitly disclosing that all three tested encodings (`gpt2`, `cl100k_base`, `o200k_base`) use the `tiktoken` library (not independent tokenizer library implementations) because Hugging Face `transformers` (`hf:xlm-roberta-base`) is unavailable in `.venv-x64`.
2. Draft the ≤1-page **Part A4 Recommendation Memo** (`my-submission/partA/memo.md` and `my-submission/partA/a4_recommendation_memo.md`) synthesizing the verified Part A1, A2, and A3 evidence, clearly separating measured findings from practical recommendations, stating the biggest caveat, and specifying the single production metric to monitor.

### Files Updated / Created
- Updated `my-submission/partA/a3_analysis.md` (Section 1.1 explicit `tiktoken`-library disclosure).
- Created `my-submission/partA/memo.md` (and identical copy `my-submission/partA/a4_recommendation_memo.md`).

### Verification of Referenced Files & Starter-Kit Integrity
- Confirmed all referenced files exist under `my-submission/partA/` and all original starter-kit files (`fertility.py`, `REPORT_v0.md`, `corpus_sample/eng_sample.txt`, `corpus_sample/hin_sample.txt`, `bench/bench_log.csv`, `bench/model_spec.md`) remain unmodified.


---

## Entry 8 — Part A4 Memo Revision & Final Validation Pass
**Timestamp:** `2026-10-23 21:32 IST`

### Objective
Revise the Part A4 memo (`my-submission/partA/memo.md`, canonical; mirrored identically in `my-submission/partA/a4_recommendation_memo.md`) and `my-submission/partA/a3_analysis.md` before submission:
1. Keep all measured A2/A3 values unchanged and verified against `e1_e6_raw_output.txt` and `a3_tokenizer_comparison.json`.
2. Reframe the routing recommendation: our results show that `gpt2` fertility alone is insufficient to justify separate Indic routing, **but** static `tiktoken` compression measurements alone do **not** establish that a unified production deployment is sufficient.
3. Remove any unverified implication that `FLM-4B-Instruct`'s `128k` tokenizer will resemble `o200k_base`; recommend benchmarking `FLM-4B-Instruct`'s exact production tokenizer on the same aligned 6-language FLORES-200 corpus.
4. Label the `1.57×–1.98×` `o200k_base` figures strictly as **token-count-per-aligned-sentence ratios**, not serving-cost multipliers.
5. Keep measured findings strictly separate from recommendations and retain all caveats (`tiktoken`-only library scope, formal translated Wikimedia prose, Romanized/code-switched Indic text, generated-output length, downstream model quality, and end-to-end latency).
6. Reduce `my-submission/partA/memo.md` to a concise one-page memo (`41` lines, `6,573` B), moving the detailed per-language tables and implementation-level A2 summary into `my-submission/partA/a3_analysis.md` (Section 2.0 & Section 3).
7. Designate `my-submission/partA/memo.md` as canonical while keeping `my-submission/partA/a4_recommendation_memo.md` byte-for-byte identical.

### Validation Commands Executed
```powershell
# Verify byte-for-byte identity of memo.md and a4_recommendation_memo.md, check referenced files, sync my-submission/, and verify starter-kit timestamps/sizes
.\.venv-x64\Scripts\python.exe -c "import pathlib, hashlib; p1=pathlib.Path('my-submission/partA/memo.md'); p2=pathlib.Path('my-submission/partA/a4_recommendation_memo.md'); assert p1.read_bytes()==p2.read_bytes(); text=p1.read_text(encoding='utf-8'); print('SHA256 match:', hashlib.sha256(p1.read_bytes()).hexdigest(), '| Lines:', len(text.splitlines()), '| Words:', len(text.split()), '| Bytes:', len(p1.read_bytes()))"
Get-Item fertility.py, REPORT_v0.md, corpus_sample\eng_sample.txt, corpus_sample\hin_sample.txt, bench\bench_log.csv, bench\model_spec.md | Select-Object Name, Length, LastWriteTime
```

### Verification Results
- Confirmed `my-submission/partA/memo.md` and `my-submission/partA/a4_recommendation_memo.md` are byte-for-byte identical (`SHA-256: 56781af242a2b6829e7fd9c8900a67488a8623d2870ae48b2fa028b18f9c8513`, `41` lines, `6,573` B).
- Confirmed all referenced files (`audit_experiments.py`, `e1_e6_raw_output.txt`, `audit_results.md`, `build_corpus.py`, `corpus_validation_output.txt`, `corpus_documentation.md`, `corpus/corpus_validation_summary.json`, `compare_tokenizers.py`, `a3_tokenizer_comparison.json`, `a3_tokenizer_comparison_output.txt`, `a3_analysis.md`, `memo.md`, `a4_recommendation_memo.md`) exist in `my-submission/partA/`.
- Confirmed all original starter-kit files (`fertility.py`, `REPORT_v0.md`, `corpus_sample/eng_sample.txt`, `corpus_sample/hin_sample.txt`, `bench/bench_log.csv`, `bench/model_spec.md`) remain completely untouched (`LastWriteTime = 22-10-2026 23:34:43`).


---

## Entry 9 — Part B: Capacity & Serving Throughput Reconciliation
**Timestamp:** `2026-10-24 00:08 IST`

### Objective
Audit `REPORT_v0.md` Section 2 (*"Serving throughput"*) against the supplied specifications in `bench/model_spec.md` and the 13 benchmark runs in `bench/bench_log.csv`:
1. Extract all supplied model (`FLM-4B-Instruct`), hardware (`1× NVIDIA L4 24 GB`), precision (`fp16` weights and KV cache), memory budget, and load-test assumptions, and explicitly list undocumented parameters.
2. Derive KV-cache memory usage per token, per sequence, and per batch; reconcile calculated KV utilization and sequence capacity against `kv_cache_util` and `preempted_seqs` across all 13 rows of `bench_log.csv`.
3. Investigate the throughput anomaly at `batch_size = 16` (`1311.4` vs. `883.2 tok/s`) by distinguishing **Total-Token Throughput** (`reported_tok_s`) from **Generated-Output-Token Throughput** (`num_requests * gen_len / wall_clock_s`).
4. Assess preemption and memory-bandwidth saturation at `batch_size = 24, 32, 48`, separating conclusions proved by the logs from hypotheses requiring additional runtime telemetry.
5. Run sensitivity analyses supported by the supplied specifications (sequence length, KV precision `fp16` vs. `fp8`/`int8`, non-KV runtime overhead around `1.6 GB`, and Part A3 multilingual token-count-per-aligned-sentence ratios).

### Commands & Scripts Executed
```powershell
.\.venv-x64\Scripts\python.exe my-submission/partB/capacity_reconciliation.py
```
- **Script:** `my-submission/partB/capacity_reconciliation.py`
- **Raw/Derived Text Output:** `my-submission/partB/partB_calculation_output.txt`
- **Structured JSON Output:** `my-submission/partB/partB_reconciliation.json`
- **Detailed Findings Report:** `my-submission/partB/partB_report.md`

### Key Measured & Derived Results
1. **KV-Cache Memory Derivation (`bench/model_spec.md`):**
   - **Per-token KV cache (GQA, $L=28, n_{kv}=8, d_h=128$, `fp16` = 2 B):**
     $$2 \times 28 \times 8 \times 128 \times 2 = \mathbf{114,688 \text{ B/tok}} = \mathbf{112.0 \text{ KiB/tok}} = \mathbf{0.114688 \text{ MB/tok}}$$
     *(Using $n_q=24$ MHA by mistake would give `344,064 B/tok = 336.0 KiB/tok`, a $3\times$ overestimate.)*
   - **Available KV-Cache Pool ($M_{\text{KV}}$):**
     - Usable VRAM: $24.0 \text{ GB} \times 0.92 = 22.08 \text{ GB}$
     - Model weights (`4.2 B` params in `fp16`): $4.2 \times 10^9 \times 2 \text{ B} = 8.40 \text{ GB}$
     - Non-KV runtime overhead: $1.60 \text{ GB}$
     - Available KV pool: $22.08 - 8.40 - 1.60 = \mathbf{12.08 \text{ GB}}$ ($12,080,000,000\text{ B} \approx 11.2504\text{ GiB}$), holding at most $\lfloor 12,080,000,000 / 114,688 \rfloor = \mathbf{105,329 \text{ tokens}}$.
   - **Per-Sequence Footprint & Idealized Sequence Capacity Estimate ($B_{\max,\text{ideal}}$):**
     - Short workload ($S = 512 + 256 = 768$ tok): **`88,080,384 B` (`88.08 MB` / `84.00 MiB`)** per sequence $\implies B_{\max,\text{ideal}}(768) = \lfloor 12.08\text{ GB} / 88.08\text{ MB} \rfloor = \mathbf{137 \text{ sequences}}$ (`preempted_seqs = 0` for all tested $B \le 64$).
     - Long workload ($S = 3584 + 512 = 4096$ tok): **`469,762,048 B` (`469.76 MB` / `448.00 MiB` = `0.4375 GiB`)** per sequence $\implies B_{\max,\text{ideal}}(4096) = \lfloor 12,080,000,000 / 469,762,048 \rfloor = \lfloor 25.715 \rfloor = \mathbf{25 \text{ sequences}}$ *(idealized capacity estimate under the stated memory assumptions; allocator/block overhead may reduce actual capacity)*.
   - **Reconciliation with `kv_cache_util` and `preempted_seqs` (`13/13` rows):**
     - At $B=24$ (`S=4096`): $24 \times 469.76\text{ MB} = 11.274\text{ GB} = \mathbf{93.33\%}$ of `12.08 GB` $\implies$ logged `kv_cache_util = 0.93`, `preempted_seqs = 0`.
     - At $B=25$ (`S=4096` idealized capacity estimate): $25 \times 469.76\text{ MB} = 11.744\text{ GB} = \mathbf{97.22\%}$ of `12.08 GB` $\implies$ rounds to **`0.97`**.
     - At $B=32$ (`Row 12`): unthrottled demand is `15.032 GB` (`124.44%` of pool); idealized excess sequences ($32 - 25 = 7$) match `preempted_seqs = 7` (`kv_cache_util = 0.97`).
     - At $B=48$ (`Row 13`): unthrottled demand is `22.549 GB` (`186.66%` of pool); idealized excess sequences ($48 - 25 = 23$) match `preempted_seqs = 23` (`kv_cache_util = 0.97`).

2. **Resolution of the Throughput Anomaly (`REPORT_v0.md` Section 2):**
   - Verified across all 13 rows that `reported_tok_s` equals **Total-Token Throughput**:
     $$\texttt{reported\_tok\_s} = \frac{\texttt{num\_requests} \times (\texttt{prompt\_len} + \texttt{gen\_len})}{\texttt{wall\_clock\_s}}$$
   - For short prompts (`512/256`), $\frac{p+g}{g} = 3.0\times$; for long prompts (`3584/512`), $\frac{p+g}{g} = 8.0\times$.
   - At `batch_size = 16`, `reported_tok_s` appears higher on long prompts (`1311.4` vs. `883.2 tok/s`) solely because it includes `3,584` parallel prefill tokens per request ($8\times$ multiplier) vs. `512` ($3\times$ multiplier).
   - Measured by **Generated-Output-Token Throughput** ($\frac{\texttt{num\_requests} \times \texttt{gen\_len}}{\texttt{wall\_clock\_s}}$), long prompts at `batch_size = 16` generate **`163.94 gen tok/s`** vs. **`294.46 gen tok/s`** for short prompts—**`44.3%` slower (`1.80×` lower output rate)**, with median inter-token decode latency `itl_ms_p50` increasing **`+59.7%`** (`48.33 ms` $\rightarrow$ `77.20 ms`).
   - Furthermore, `REPORT_v0.md`'s projection that `batch_size = 48` will scale linearly to `~3200 tok/s` is directly contradicted by `Row 13` (`batch_size = 48`), which was already executed in `bench_log.csv` and achieved **`1298.5 total tok/s` (`162.31 gen tok/s`)**—a **`-19.2%` throughput drop** below `batch_size = 24` (`1607.4 total tok/s`, `200.92 gen tok/s`).

### Unresolved Uncertainties Explicitly Noted
- Exact serving framework/version, PagedAttention block size, prefix-caching/chunked-prefill settings, hardware HBM counters, and whether preemption uses `recompute` (eviction + re-prefill) or PCIe `swap` are not specified in `model_spec.md` and require runtime telemetry.


---

## Entry 10 — Part B Focused Technical Audit & Epistemic Separation
**Timestamp:** `2026-10-24 00:18 IST`

### Objective
Perform a focused technical audit of `my-submission/partB/` (`capacity_reconciliation.py`, `partB_calculation_output.txt`, `partB_reconciliation.json`, `partB_report.md`) to ensure strict epistemic separation between directly observed log facts, calculations derived from supplied specifications, and hypotheses requiring additional telemetry.

### Revisions & Verification Checks Executed
```powershell
.\.venv-x64\Scripts\python.exe my-submission/partB/capacity_reconciliation.py
.\.venv-x64\Scripts\python.exe -c "import csv, json, pathlib; ..."
```
1. **Reclassified `190.7–200.1 GB/s` HBM Values as Analytical Traffic-Model Estimates:**
   - Moved the `190.7–200.1 GB/s` values out of "proven conclusions" into **Section 5.3 (Tier 3 — Hypotheses & Model-Based Estimates Requiring Additional Telemetry)** in `partB_report.md` and relabeled the column in `capacity_reconciliation.py` / `partB_calculation_output.txt` as `Model_BW` (`Traffic-Model Est. GB/s`).
   - Explicitly documented the traffic-model assumptions (reading `8.40 GB` weights once per step + midpoint active KV cache $\min(B, 25) \times (p + g/2) \times 114,688\text{ B}$, divided by `itl_ms_p50 / 1000`, omitting L2 cache hits, activation traffic, kernel launch overhead, and compute overlap) and the hardware profiler counters required to verify an HBM bottleneck (`ncu` / DCGM `dram__bytes_read.sum.per_second`).
2. **Updated Prefix-Caching Statement:**
   - Changed the prefix-caching description in `capacity_reconciliation.py` and `partB_report.md` from *"proves prefix caching is disabled"* to *"not specified in `model_spec.md`; observed `kv_cache_util` values are consistent with the simple linear per-request allocation model ($B \times S$)."*
3. **Qualified the `25`-Sequence Long-Context Capacity as an Idealized Estimate:**
   - Explicitly described $B_{\max,\text{ideal}}(4096) = 25$ sequences (and $B_{\max,\text{ideal}}(768) = 137$ sequences) as the **idealized capacity estimate under the stated memory assumptions** (`24.0 GB` decimal, `0.92` utilization, `8.40 GB` FP16 weights, `1.60 GB` non-KV overhead, and zero allocator fragmentation), noting that PagedAttention block granularity, block-table metadata, or runtime overhead variations may reduce actual capacity.
4. **Verified Consistent Decimal `GB` vs. Binary `GiB` Distinction & All 13 Rows of `bench_log.csv`:**
   - Confirmed all 13 rows match `capacity_reconciliation.py` across `reported_tok_s`, `calc_total_tok_s`, `calc_gen_tok_s`, KV demand (`GB` decimal and `GiB` binary), `kv_cache_util`, and `preempted_seqs`, and confirmed all original starter-kit files remain unmodified (`LastWriteTime = 22-10-26 23:34:43`).


---

## Entry 11 — Part C: Engineering Recommendation Memo
**Timestamp:** `2026-10-24 00:28 IST`

### Objective
Draft a defensible, decision-oriented engineering recommendation memo (`my-submission/partC/memo.md`) for improving Hindi and Kannada performance under the assignment's constraints:
- **Compute:** `1× NVIDIA A100 80 GB` GPU (with production serving on `1× NVIDIA L4 24 GB` per `bench/model_spec.md`)
- **Timeline:** 2 weeks for implementation & evaluation (`Weeks 1–2`); launch review in 3 weeks (`Week 3`)
- **Reviewers:** Hindi and Kannada native reviewers, limited to `10 hours per week`
- **Budget:** `$0` budget for external APIs
- **Options Evaluated:** (1) Supervised fine-tuning (SFT) of `FLM-4B-Instruct` (`4.2B`), (2) Lightweight rewriter model ($\le 1\text{B}$ parameters), and (3) Prompt engineering without model training.

### Inputs Inspected & Synthesized
- `REPORT_v0.md`, `bench/model_spec.md`, `bench/bench_log.csv`
- `my-submission/partA/memo.md`, `my-submission/partA/a3_analysis.md`
- `my-submission/partB/partB_report.md`, `my-submission/partB/partB_reconciliation.json`
- `my-submission/NOTEBOOK.md`

### Rationale & Key Design Decisions
1. **Primary Recommendation — Option 1 (LoRA SFT of `FLM-4B-Instruct`, merged into `fp16` base weights), with Option 3 ($\le 128$-token Concise System Prompt) as the Day-2 Control Baseline & Launch Fallback; Option 2 ($\le 1\text{B}$ Rewriter) Rejected:**
   - **Serving & Capacity Grounding (from Part B):** Merging a LoRA SFT adapter into `FLM-4B-Instruct`'s `8.40 GB` `fp16` weights adds **zero VRAM overhead** (keeping the L4's `12.08 GB` [`11.2504 GiB`] idealized KV pool and $B_{\max,\text{ideal}}(4096) = 25$ sequence capacity intact), **zero extra prompt prefill tokens**, and **zero additional autoregressive decode steps**. By contrast, co-locating a $\le 1\text{B}$ `fp16` rewriter (`~2.5–3.5 GB` weights + KV/activations) on the `24 GB` L4 shrinks the KV pool to `~8.6–9.6 GB` ($B_{\max,\text{ideal}}(4096)$ drops to `18–20` sequences), adds a second sequential decode pass, and risks semantic/factual corruption by a smaller model. Meanwhile, verbose few-shot prompt engineering (Option 3 alone) inflates `prompt_len`, which Part B (`Row 5` vs. `Row 10`) proved degrades generated-output-token throughput by `44.3%` (`294.46` $\rightarrow$ `163.94 gen tok/s`) and increases median `itl_ms_p50` by `+59.7%` (`48.33` $\rightarrow$ `77.20 ms`).
   - **Compute & `$0` API Feasibility:** On `1× A100 80 GB`, `bf16`/`fp16` LoRA SFT of a `4.2B` model uses `~18–26 GB` VRAM and completes `30,000` examples in `~4–6 GPU-hours`, allowing 2–3 iterations across Weeks 1–2 using permissive open datasets (`Apache-2.0` / `CC-BY-4.0`) and internal logs with zero external API calls.
2. **Reviewer Budget Compliance ($\le 10$ Hours / Week):**
   - Budgeted the human evaluation protocol at **`5.0 hours/week` Hindi + `5.0 hours/week` Kannada = `10.0 hours/week` combined** (`108–120` blinded pairwise comparisons/language/week at `2.5 min/pair`, plus `30 min` calibration), ensuring strict compliance even under the stricter combined `10 hr/wk` cap (and noting that if `10 hr/wk per language` is available, double-annotation coverage expands from `17%` to `100%`).
3. **Strict Epistemic Separation & Data Leakage Prevention:**
   - Explicitly distinguished measured Part A/B facts from unmeasured Part C hypotheses (disclosing that Part A's `tiktoken` compression is not proof of generation quality or latency, `FLM-4B-Instruct`'s actual `128k` tokenizer is unmeasured until Day 1, and Part B's `190.7–200.1 GB/s` values are analytical traffic-model estimates rather than hardware telemetry).
   - Specified Day-2 chronological freeze and SHA-256 hashing of the `360`-prompt stratified golden eval set (`120` Hindi, `120` Kannada, `120` English regression, covering native script, multi-step instructions, Romanized Hinglish/Kanglish, and JSON/code formatting) plus 3-layer decontamination (exact NFC match, 13-gram overlap, and 5-gram MinHash Jaccard $\ge 0.70$) before training.
4. **AI Assistance Disclosure:**
   - Used AI pair-programming assistance to structure the option comparison matrix, verify arithmetic consistency across Parts A/B/C, and format the 3-week execution schedule and acceptance gates in `my-submission/partC/memo.md`.

### Validation Commands Executed
```powershell
New-Item -ItemType Directory -Force my-submission\partC | Out-Null
.\.venv-x64\Scripts\python.exe -c "import pathlib, hashlib; p1=pathlib.Path('my-submission/partC/memo.md'); print('Part C SHA256:', hashlib.sha256(p1.read_bytes()).hexdigest(), '| Lines:', len(p1.read_text(encoding='utf-8').splitlines()), '| Bytes:', len(p1.read_bytes()))"
Get-Item fertility.py, REPORT_v0.md, corpus_sample\eng_sample.txt, corpus_sample\hin_sample.txt, bench\bench_log.csv, bench\model_spec.md | Select-Object Name, Length, LastWriteTime
```


---

## Entry 12 — Part C Final Evidence & Feasibility Audit
**Timestamp:** `2026-10-24 00:43 IST`

### Objective
Perform a final evidence and feasibility audit of `my-submission/partC/memo.md` against `REPORT_v0.md`, `bench/model_spec.md`, `bench/bench_log.csv`, `my-submission/partA/memo.md`, `my-submission/partA/a3_analysis.md`, and `my-submission/partB/partB_report.md`.

### Key Audit Revisions & Corrections Applied to `my-submission/partC/memo.md`
1. **Reframed Primary Recommendation to Match Established Evidence:**
   - Because no Hindi/Kannada SFT dataset exists in the starter kit (`$0` API budget) and `FLM-4B-Instruct`'s actual `128k` tokenizer and base weights are unmeasured, we revised the primary recommendation so that **Option 3 (Concise System-Prompt Engineering, $\Delta p \le 128$ tokens, no model training)** is the **Primary Committed Launch Approach (Stage 1, built and human-evaluated in Week 1)**, while **Option 1 (Merged `fp16` LoRA SFT of `FLM-4B-Instruct` on `1× A100 80 GB`)** is a **Gated Week-2 Experimental Upgrade (Stage 2)**—executed *only if* the Day 3 Data & Tokenizer Feasibility Gate verifies $\ge 10,000$ clean, decontaminated, permissively licensed instruction pairs, and promoted over Option 3 in Week 3 *only if* it beats Option 3 on blinded human evaluation and passes all English non-regression and L4 serving gates. **Option 2 ($\le 1\text{B}$ Rewriter)** remains rejected.
2. **Verified & Qualified Quantitative Claims:**
   - **SFT Training Corpus (`15k–30k` examples, `40% kan / 35% hin / 25% eng`):** Explicitly labeled as a **proposed target mixture**, not confirmed existing data, with a hard **Early Stop Gate #1 on Day 3** if $< 10,000$ clean decontaminated pairs exist.
   - **A100 Training VRAM (`~18–26 GB`) and Run Time (`~3–6 GPU-hours`):** Reclassified as **unexecuted analytical engineering estimates** and documented the exact assumptions (`1× A100 80 GB`, `4.2B` `bf16`/`fp16` weights = `8.4 GB`, LoRA $r=16$ `~40–80M` trainable params = `~1–2 GB` optimizer/grad states, gradient checkpointing enabled, $S \le 2048$, micro-batch `2–4`, grad accum `8–16`, `~3,000–5,000 tok/s`).
   - **Production L4 (`24 GB`) vs. R&D A100 (`80 GB`) & Merged LoRA Memory:** Explicitly separated the L4 serving environment (`12.08 GB` = `11.2504 GiB` idealized KV pool, $B_{\max,\text{ideal}}(4096) = 25$ sequences) from the A100 training GPU, and clarified that merged `fp16` LoRA preserves the `8.40 GB` base weight footprint only assuming the base model fits, vocabulary size (`128k`) is unchanged, and output length (`gen_len`) is validated on L4. Also calculated the exact L4 KV impact of Option 3's $\Delta p \le 128$ prompt tokens ($S=768 \rightarrow 896$ shifts $B_{\max,\text{ideal}}$ from `137` to `117` sequences).
   - **Part B Row 5 vs. Row 10 Output Throughput Reduction (`-44.33%`):** Verified the exact arithmetic (`294.46 gen tok/s` [`4,096 / 13.91 s`] vs. `163.94 gen tok/s` [`8,192 / 49.97 s`], `-44.33%`; `itl_ms_p50` `48.33 ms` $\rightarrow$ `77.20 ms`, `+59.74%`) and added the explicit qualification that **Rows 5 and 10 vary both `prompt_len` (`512` $\rightarrow$ `3584`) and `gen_len` (`256` $\rightarrow$ `512`)**, so this comparison does not isolate prompt length alone.
   - **Human Reviewer Arithmetic (`<= 10.0` Combined Hours/Week):** Defined a fixed golden test set of **`80` Hindi + `80` Kannada prompts** (`20` prompts $\times 4$ buckets/language, plus `80` English regression prompts = `240` golden test prompts, and a separate `40` Hindi + `40` Kannada dev split for Week 2). At **`3.0 minutes/comparison`**, rating `80` pairs takes **`4.0 hours`** + `0.5h` calibration + `0.5h` (`10` overlap items) = **`5.0 hours/language/week` (`10.0 hours/week` combined)** in Week 1 (`Base` vs. `Option 3`) and Week 3 (`Option 3` vs. `LoRA SFT`), and `40` dev pairs $\times 3.0\text{ min} = \mathbf{2.0\text{ hours/language}}$ (**`4.0 hours` combined**) in Week 2 (`24.0` combined hours across 3 weeks).
   - **Replaced Arbitrary `+2.0 chrF++` Gate:** Made deterministic instruction-set checks (script purity $\ge 98.0\%$, `rep-4` $\le 1.0\%$, format parser compliance, length ratio) the primary offline gates, and specified `FLORES-200` `devtest` (`1,012` aligned sentences in `partA/corpus/processed/{eng,hin,kan}.txt`, single-reference `NFC`, character 6-gram / word 2-gram `chrF++`) strictly as an auxiliary translation non-regression check ($\ge \text{Day-1 Base} - 0.5$) without inventing baseline scores.
   - **Pre-Registered Thresholds, $K=5$ L4 Load-Test Protocol & Offline Minimum:** Labeled all thresholds as pre-registered proposals, defined a 5-repetition L4 load-test protocol (`±3.0%` or `±2 SE` on mean `calc_gen_tok_s` and `itl_ms_p50` at $B=16$ and $B=24$), and established offline evaluation + L4 harness verification as the mandatory minimum launch gate (treating shadow/canary rollout as conditional on live routing infrastructure).
3. **AI Assistance Disclosure:**
   - Used AI pair-programming assistance to audit quantitative consistency across Parts A, B, and C, verify reviewer-hour and KV-capacity arithmetic, and update `my-submission/partC/memo.md` and `my-submission/NOTEBOOK.md`.

### Validation Commands Executed
```powershell
.\.venv-x64\Scripts\python.exe -c "import pathlib, hashlib; p1=pathlib.Path('my-submission/partC/memo.md'); print('Part C Audited SHA256:', hashlib.sha256(p1.read_bytes()).hexdigest(), '| Lines:', len(p1.read_text(encoding='utf-8').splitlines()), '| Bytes:', len(p1.read_bytes()))"
Get-Item fertility.py, REPORT_v0.md, corpus_sample\eng_sample.txt, corpus_sample\hin_sample.txt, bench\bench_log.csv, bench\model_spec.md | Select-Object Name, Length, LastWriteTime
```

---

## Entry 13 — Final Assignment Audit Against Original PDF (`AI ASSIGNMENT 2026.pdf`) & Submission Package Finalization
**Timestamp:** `2026-10-24 00:58 IST`

### Objective
Conduct a comprehensive final audit of the entire submission against the original assignment PDF (`C:\Users\TANNU\Downloads\AI ASSIGNMENT 2026.pdf`, 6 pages) and evaluation rubric (`Part A: 50 pts`, `Part B: 20 pts`, `Part C: 15 pts`, `Live Defense + Lab Notebook: 15 pts`), close any remaining structural or rubric-mapping gaps, and produce `my-submission/FINAL_AUDIT_REPORT.md`.

### Key Findings & Final Fixes Applied
1. **Created `my-submission/AI_USAGE.md` (Required Gate Deliverable on Page 5 of PDF):**
   - Documented where AI pair-programming assistance accelerated execution (FLORES-200 ingestion/NFC normalization, `E1–E6` ablation scripting, `bench_log.csv` reconciliation verification) and four specific instances where initial AI assumptions had to be audited and corrected (Hugging Face `transformers` availability assumption, over-interpreting analytical HBM traffic estimates `190.7–200.1 GB/s` as hardware telemetry, over-claiming prefix caching as proven disabled rather than unspecified, and initially ranking unverified SFT ahead of Day-1 prompt engineering).
2. **Added Direct `B1–B4` Rubric Mapping (`Section 0`) to `my-submission/partB/partB_report.md`:**
   - Mapped every Part B calculation directly to the PDF's four graded sub-questions:
     - **`B1 (7 pts)`:** `114,688 B/tok` (`112 KiB/tok`), `469,762,048 B/seq` (`448 MiB/seq` at $S=4096$), `12.08 GB` KV pool, and $B_{\max}(4096) = 25$ sequences, reconciled against `Row 11` ($B=24$, `0` preemptions) and `Row 12` ($B=32$, `7` preemptions $= 32 - 25$).
     - **`B2 (6 pts)`:** Long-context throughput cliff at `Row 12` (`1384.0 tok/s`, `-13.90%`) and `Row 13` (`1298.5 tok/s`, `-19.22%`, `23` preemptions $= 48 - 25$), plus quantitative fix (`fp8` KV cache doubling $B_{\max}(4096)$ from `25` to `51` sequences and reducing preemptions at $B=48$ from `23` to `0`, or capping `max_num_seqs = 24`).
     - **`B3 (4 pts)`:** Explained the `reported_tok_s` total-token inflation ($3\times$ on `512/256` vs. $8\times$ on `3584/512`) and derived `Row 11` (`B=24`) goodput in **two independent ways**: (Way 1) from `wall_clock_s`: $12,288 / 61.16 = 200.92\text{ gen tok/s}$ ($1607.4 / 8$); (Way 2) from latency columns `ttft_ms_p50 = 500.5 ms` and `itl_ms_p50 = 96.07 ms`: $12,288 / (61.16 - 0.5005) = 202.57\text{ gen tok/s}$ net decode goodput and $24 / 0.09607 = 249.82\text{ gen tok/s}$ instantaneous median step rate ($247.30\text{ gen tok/s}$ p50 request-level rate).
     - **`B4 (3 pts)`:** Provided a concise one-paragraph serving-counter verification answer naming `vllm:num_preemptions_total` (and `prefill_tokens_total` alongside `num_total_gpu_blocks`) with exact expected values at $B=24, 32, 48$.
3. **Enhanced `my-submission/partC/memo.md` with Explicit PDF Labels, 6-Language Scope Rule, and Unambiguous Overlap Rating Arithmetic:**
   - Added the 5 explicit PDF-required labels in Section 1.2 (`Explicit Assumptions`, `Back-of-Envelope Arithmetic`, `Success Metric with Numeric Threshold`, `Kill Criterion`, `First Experiment on Day 1`), explicitly scoped Week-3 casualization launch to Hindi + Kannada while holding Tamil, Telugu, Bengali, and Marathi behind a feature flag until native reviewer coverage is allocated, and clarified in Section 5.2 that each main round rates `80` unique golden prompts + `10` additional blind overlap pairs = `90` total pairs/language ($90 \times 3.0\text{ min} = 4.5\text{ h} + 0.5\text{ h}$ calibration $= 5.0\text{ h/wk/lang} = 10.0\text{ h/wk}$ combined).
4. **Generated `my-submission/FINAL_AUDIT_REPORT.md`:**
   - Created `FINAL_AUDIT_REPORT.md` containing the executive readiness verdict, full rubric matrix, independent recalculation tables for Parts A/B/C, 10 live-defense oral exam questions & model answers, and final checklist in `my-submission/`.

---

## Entry 14 — Targeted Source-of-Truth Verification & Resolution of Summary Discrepancies
**Timestamp:** `2026-10-24 01:15 IST`

### Objective
Resolve discrepancies introduced into the first draft of `FINAL_AUDIT_REPORT.md` by re-inspecting `bench/model_spec.md`, `bench/bench_log.csv`, `fertility.py`, `corpus_sample/{eng,hin}_sample.txt`, `my-submission/partA/a3_tokenizer_comparison.json`, `my-submission/partB/partB_reconciliation.json`, and `my-submission/partC/memo.md`, and re-executing all three verification scripts (`audit_experiments.py`, `compare_tokenizers.py`, and `capacity_reconciliation.py`).

### Root Cause Diagnosis & Corrections Applied
1. **Part B Source-of-Truth Verification (`bench/model_spec.md` & `bench/bench_log.csv`):**
   - Direct inspection of `bench/model_spec.md` and `my-submission/partB/capacity_reconciliation.py` confirmed that our script, JSON (`partB_reconciliation.json`), and report (`partB_report.md`) already used the exact supplied source parameters: `layers = 28`, `attention heads (Q) = 24`, `KV heads (GQA) = 8` ($3:1$ GQA ratio), `head_dim = 128`, `parameters = 4.2 B` in `fp16` (`8.40 GB` weights), `gpu_memory_utilization = 0.92` (`22.08 GB` usable), and `non-KV runtime overhead = assume ~1.6 GB` $\implies$ `114,688 B/token` (`112.0 KiB/token`), `12.08 GB` KV pool (`105,329` tokens), and $B_{\max,\text{ideal}}(4096) = 25$ sequences (`0.97` capped `kv_cache_util` at $B=32$ and $B=48$, `7` and `23` preempted sequences).
   - The erroneous `32` layers / `8.20 GB` weights / `1.80 GB` overhead figures in the first draft of `FINAL_AUDIT_REPORT.md` had come from a corrupted context-truncation summary rather than from the workspace files. We corrected `FINAL_AUDIT_REPORT.md` to match `bench/model_spec.md` and `partB_reconciliation.json`.
2. **Part A Numerical Verification (`fertility.py` & `a3_tokenizer_comparison.json`):**
   - Live re-execution of `audit_experiments.py` and `compare_tokenizers.py` confirmed that `a3_tokenizer_comparison.json`, `a3_analysis.md`, `audit_results.md`, and `partA/memo.md` were already 100% consistent with the processed `FLORES-200` `devtest` corpus (`o200k_base` Hindi micro `tok/word = 1.6473` [`42,241 / 25,643`], sentence-parity ratio `1.57×` [`42,241 / 26,873`]; Dravidian sentence-parity ratios `1.92×–1.98×`). Corrected `FINAL_AUDIT_REPORT.md` to match `a3_tokenizer_comparison.json` and `audit_results.md`.
3. **Part B3 Throughput Derivations Refined (`my-submission/partB/partB_report.md` §0.B3):**
   - Updated Section 0 (`B3`) of `partB_report.md` so the two independent derivations of **observed end-to-end generated-token goodput** at `Row 11` (`B=24, p=3584, g=512`) are presented first: (Way 1) $\frac{24 \times 512}{61.16\text{ s}} = \mathbf{200.92\text{ gen tok/s}}$ (`8.37 gen tok/s/req`), and (Way 2) $\texttt{reported\_tok\_s} \times \frac{512}{3584+512} = \frac{1607.4}{8} = \mathbf{200.925\text{ gen tok/s}}$ (`200.92` rounded), clearly separated from the percentile-based decode estimates (`202.57 gen tok/s` net decode-only estimate excluding `ttft_ms_p50 = 500.5 ms`, and `249.82 gen tok/s` instantaneous step rate at `itl_ms_p50 = 96.07 ms`).
4. **Part C Reviewer Budget & Epistemic Labels Verified (`my-submission/partC/memo.md`):**
   - Confirmed `partC/memo.md` explicitly states that the `10` overlap comparisons are additional to the `80` unique golden prompts (`80 unique + 10 additional overlap = 90 pairs/language`), yielding $90 \times 3.0\text{ min} = 4.5\text{ h}$ active rating $+ 0.5\text{ h}$ calibration = **`5.0 h/week` for Hindi and `5.0 h/week` for Kannada (`10.0 h/week` combined)** in Weeks 1 and 3 (`2.0 h/week` per language = `4.0 h/week` combined in Week 2), and that all A100 training figures remain labeled as unexecuted analytical estimates.

### Commands Executed & Verified
```powershell
.\.venv-x64\Scripts\python.exe my-submission/partA/audit_experiments.py
.\.venv-x64\Scripts\python.exe my-submission/partA/compare_tokenizers.py
.\.venv-x64\Scripts\python.exe my-submission/partB/capacity_reconciliation.py
```


