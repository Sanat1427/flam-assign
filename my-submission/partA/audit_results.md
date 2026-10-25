# Part A2 — Controlled Tokenizer Audit (`fertility.py` & `REPORT_v0.md`)

**Environment:** `Python 3.12.10 (AMD64)` in `.venv-x64` (`tiktoken 0.14.0`, `regex 2026.9.29`)  
**Reproducible Command:**
```powershell
.\.venv-x64\Scripts\python.exe my-submission/partA/audit_experiments.py
```

---

## 1. Summary Table of Hypotheses, Controlled Experiments, Measured Effects, and Conclusions

Every experiment below modifies **strictly one variable** relative to the unmodified `fertility.py` baseline (`gpt2` encoding on `corpus_sample/eng_sample.txt` and `corpus_sample/hin_sample.txt`).

| ID | Hypothesis (Tested) | Single Variable Changed | Baseline Value (Unrounded / Rounded) | Experiment Value (Unrounded / Rounded) | Measured Delta ($\Delta$ / %) | Classification & Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Base** | `fertility.py` reproduces `REPORT_v0.md` exactly. | None (unmodified `fertility.py`) | `eng` fert: `1.265206` (`1.27`)<br>`hin` fert: `7.448452` (`7.45`)<br>fert ratio: `5.887148` (`5.89x`)<br>`eng` tpc: `0.225636` (`0.226`)<br>`hin` tpc: `1.579108` (`1.579`)<br>tpc ratio: `6.998478` (`7.00x`) | Same | `0.000000` (`0.00%`) | **Verified Baseline:** Matches all 5 numbers in `REPORT_v0.md`. Note that `7.45 / 1.27 = 5.866`, whereas `fertility.py` computes `7.448452 / 1.265206 = 5.887148` (`5.89x`) before rounding. |
| **E1** | `line.split(" ")` miscounts words when lines contain consecutive spaces. | `line.split(" ")` $\rightarrow$ `line.split()` | `eng` words: `79`<br>`hin` words: `62`<br>`eng` fert: `1.265206` (`1.27`)<br>`hin` fert: `7.448452` (`7.45`)<br>ratio: `5.887148` (`5.89x`) | `eng` words: `78`<br>`hin` words: `61`<br>`eng` fert: `1.283063` (`1.28`)<br>`hin` fert: `7.598452` (`7.60`)<br>ratio: `5.922121` (`5.92x`) | `eng` fert: `+0.017857` (`+1.41%`)<br>`hin` fert: `+0.150000` (`+2.01%`)<br>ratio: `+0.034973` (`+0.59%`) | **Confirmed Code Bug:** `eng` L07 (`"books  in"`) and `hin` L10 (`"किताबें  अलमारी"`) contain double spaces. `split(" ")` injects an empty string `""` into `words`, inflating word count by $+1$ on those lines and understating fertility (while `gpt2` still tokenizes the extra space). |
| **E2** | `line.lower()` alters BPE merges on cased scripts (English) while having zero effect on unicameral scripts (Devanagari). | `line = line.lower()` $\rightarrow$ preserve original casing | `eng` tokens: `99`<br>`hin` tokens: `459`<br>`eng` fert: `1.265206` (`1.27`)<br>`hin` fert: `7.448452` (`7.45`)<br>ratio: `5.887148` (`5.89x`)<br>`eng` tpc: `0.225636` (`0.226`) | `eng` tokens: `96`<br>`hin` tokens: `459`<br>`eng` fert: `1.229329` (`1.23`)<br>`hin` fert: `7.448452` (`7.45`)<br>ratio: `6.058958` (`6.06x`)<br>`eng` tpc: `0.219562` (`0.220`) | `eng` tok: `-3` (`-3.03%`)<br>`eng` fert: `-0.035877` (`-2.84%`)<br>`hin` fert: `0.000000` (`0.00%`)<br>ratio: `+0.171810` (`+2.92%`) | **Confirmed Methodological / Preprocessing Bug:** `gpt2` is a case-sensitive tokenizer with dedicated vocabulary entries for `" Thursday"`, `"NASA"`, and `" GPU"`. Lowercasing splits them into `[' th', 'ursday']`, `['n', 'asa']`, and `[' g', 'pu']` ($+3$ extra English tokens), artificially inflating English fertility while leaving Hindi untouched. |
| **E3** | Macro-averaging per-line ratios (`sum(r_i)/n`) differs from corpus micro-average (`sum(tok)/sum(words)`). | Macro-average across lines $\rightarrow$ Micro-average (`tot_tokens / tot_words`) | `eng` fert: `1.265206` (`1.27`)<br>`hin` fert: `7.448452` (`7.45`)<br>ratio: `5.887148` (`5.89x`)<br>`eng` tpc: `0.225636` (`0.226`)<br>`hin` tpc: `1.579108` (`1.579`) | `eng` fert: `1.253165` (`1.25`)<br>`hin` fert: `7.403226` (`7.40`)<br>ratio: `5.907625` (`5.91x`)<br>`eng` tpc: `0.220982` (`0.221`)<br>`hin` tpc: `1.582759` (`1.583`) | `eng` fert: `-0.012041` (`-0.95%`)<br>`hin` fert: `-0.045227` (`-0.61%`)<br>ratio: `+0.020477` (`+0.35%`)<br>tpc ratio: `+0.163904` (`+2.34%`) | **Metric-Design Choice (Misaligned with Cost Goal):** Macro-averaging weights a 4-word line (`hin` L08, fert `8.75`) equally with a 12-word line (`eng` L03, fert `1.08`). For total serving/token cost over a traffic stream, micro-averaging (`total_tokens / total_words`) is the mathematically appropriate estimator. |
| **E4a** | `len(line)` counts Unicode codepoints (`290` in `hin`) rather than UTF-8 bytes (`764` in `hin`), masking byte-level BPE behavior. | `chars = len(line)` $\rightarrow$ `len(line.encode("utf-8"))` | `eng` denom: `448` chars<br>`hin` denom: `290` chars<br>`eng` tpc: `0.225636` (`0.226`)<br>`hin` tpc: `1.579108` (`1.579`)<br>ratio: `6.998478` (`7.00x`) | `eng` denom: `448` bytes<br>`hin` denom: `764` bytes<br>`eng` tok/B: `0.225636` (`0.226`)<br>`hin` tok/B: `0.598992` (`0.599`)<br>ratio: `2.654683` (`2.65x`) | `hin` denom: `+474` (`+163.4%`)<br>`hin` rate: `-0.980116` (`-62.07%`)<br>ratio: `-4.343796` (`-62.07%`) | **Confirmed Conceptual Metric Flaw & Report Contradiction:** Each Devanagari character is 3 UTF-8 bytes (`1.0` byte for ASCII space). `gpt2` falls back to byte tokens on Devanagari (`0.599` tok/byte $\approx 1.67$ bytes/tok vs `4.43` bytes/tok in English). Also **refutes Report Finding #3**: Hindi has *fewer* Unicode codepoints/word (`4.68`) than English (`5.67`)! |
| **E4b** | `len(line)` splits Indic *aksharas* (orthographic syllables) into separate consonant + vowel-matra/virama/nukta codepoints. | `chars = len(line)` $\rightarrow$ Extended Grapheme Clusters `len(regex.findall(r"\X", line))` | `eng` denom: `448`<br>`hin` denom: `290`<br>`eng` tpc: `0.225636` (`0.226`)<br>`hin` tpc: `1.579108` (`1.579`)<br>ratio: `6.998478` (`7.00x`) | `eng` graphemes: `448`<br>`hin` graphemes: `188`<br>`eng` tok/G: `0.225636` (`0.226`)<br>`hin` tok/G: `2.449732` (`2.450`)<br>ratio: `10.857013` (`10.86x`) | `hin` denom: `-102` (`-35.17%`)<br>`hin` rate: `+0.870624` (`+55.13%`)<br>ratio: `+3.858534` (`+55.13%`) | **Confirmed Conceptual Distinction:** In English, `448` bytes = `448` codepoints = `448` graphemes. In Hindi, `764` UTF-8 bytes = `290` codepoints = `188` grapheme clusters. Treating "character" as interchangeable across scripts is conceptually invalid. |
| **E5** | `eng_sample.txt` and `hin_sample.txt` are not parallel line-by-line, confounding cross-language comparison. | All 10 lines (non-parallel) $\rightarrow$ 5 semantically matched pairs (`eng` L3,4,5,7,8 $\leftrightarrow$ `hin` L3,7,6,10,4) | `eng` fert: `1.265206` (`1.27`)<br>`hin` fert: `7.448452` (`7.45`)<br>fert ratio: `5.887148` (`5.89x`)<br>total tok ratio: `459/99 = 4.636x` | `eng` fert: `1.228571` (`1.23`)<br>`hin` fert: `7.585714` (`7.59`)<br>fert ratio: `6.174419` (`6.17x`)<br>total tok ratio: `242/47 = 5.149x` | `eng` fert: `-0.036634` (`-2.90%`)<br>`hin` fert: `+0.137262` (`+1.84%`)<br>fert ratio: `+0.287271` (`+4.88%`)<br>tok ratio: `+0.513x` (`+11.06%`) | **Confirmed Corpus Flaw:** 0 of 10 line indices align (`L01` is Bengaluru Airport in `eng` vs morning tea in `hin`), and 5 of 10 lines have no translation counterpart in the other file at all. |
| **E6a** | Unused `random.seed(1337)` (`fertility.py:L25`) looks suspicious. | Change/remove `random.seed` | `eng` fert: `1.265206`<br>`hin` fert: `7.448452` | `eng` fert: `1.265206`<br>`hin` fert: `7.448452` | `0.000000` (`0.00%`) | **Harmless Dead Code (Not a Bug):** `random` is never called anywhere in `fertility.py` and BPE tokenization is deterministic. Flagging this as a bug would be a false positive. |
| **E6b** | `unicodedata.normalize("NFC", line)` (`fertility.py:L49`) looks suspicious for Indic text. | Remove `NFC` normalization (and test `NFD` counterfactual) | `eng` fert: `1.265206`<br>`hin` fert: `7.448452`<br>lines altered: `0/10` | `eng` fert: `1.265206`<br>`hin` fert: `7.448452`<br>lines altered: `0/10` | `0.000000` (`0.00%`) | **Harmless / Good Practice (Not a Bug):** Both sample files are already in NFC form (`0` characters changed; even under NFD, Devanagari nukta words in `hin_sample.txt` already use decomposed `\u093c` nukta which is canonical in both NFC and NFD). Moreover, NFC normalization is standard defensive preprocessing. |
| **E6c** | `add_special_tokens=False` (`fertility.py:L33`) looks suspicious compared to HF defaults. | Code inspection & analytical effect of `add_special_tokens=True` | Adds `0` special tokens per line (`tiktoken` also adds `0`). | `True` adds $+1$ or $+2$ tokens (`<bos>`/`<eos>` or `[CLS]`/`[SEP]`) per line. | `0` on `gpt2`; would add $+10$ or $+20$ tokens across 10 lines on HF models. | **Correct Design Choice (Not a Bug):** Disabling special tokens in line-by-line evaluation is **correct**—otherwise short sentences get penalized by fixed per-line framing overhead, and HF models would be inconsistent with `tiktoken`. |

---

## 2. Detailed Experiment Analyses

### E1 — Word-Counting Behavior (`line.split(" ")` vs. `line.split()`)
- **What Happens in `fertility.py`:** Line 62 uses `words = line.split(" ")`. In Python, `str.split(" ")` splits on every single space character rather than collapsing runs of contiguous whitespace.
- **Exact Lines Affected in `corpus_sample/`:**
  - `eng_sample.txt` Line 7: `"Please keep the books  in the cupboard."` $\rightarrow$ `line.split(" ")` produces `['please', 'keep', 'the', 'books', '', 'in', 'the', 'cupboard.']` (`8` elements instead of `7` words).
    - Line 7 fertility changes from $10 / 8 = 1.250000$ to $10 / 7 = 1.428571$ ($\Delta = +0.178571$).
    - Macro-average `eng` fertility increases by $+0.178571 / 10 = \mathbf{+0.017857}$ (`1.265206` $\rightarrow$ `1.283063`, rounded `1.27` $\rightarrow$ `1.28`).
  - `hin_sample.txt` Line 10: `"किताबें  अलमारी में रखी हैं।"` $\rightarrow$ `line.split(" ")` produces `['किताबें', '', 'अलमारी', 'में', 'रखी', 'हैं।']` (`6` elements instead of `5` words).
    - Line 10 fertility changes from $45 / 6 = 7.500000$ to $45 / 5 = 9.000000$ ($\Delta = +1.500000$).
    - Macro-average `hin` fertility increases by $+1.500000 / 10 = \mathbf{+0.150000}$ (`7.448452` $\rightarrow$ `7.598452`, rounded `7.45` $\rightarrow$ `7.60`).
- **Distortion:** Because `gpt2` still emits an extra token for the second space (`' '`, token ID `220`), `split(" ")` both leaves the extra space token in the numerator and inflates the denominator by $+1$ word, shifting the `hin/eng` ratio from `5.89x` (`5.887148`) to `5.92x` (`5.922121`).
- **Verdict:** **Implementation Bug.**

---

### E2 — Lowercasing (`line = line.lower()` vs. Cased Text)
- **What Happens in `fertility.py`:** Line 60 forces `line = line.lower()` with the comment `# lowercase so casing doesn't add noise to the comparison`.
- **Why This Is Flawed:**
  1. **Asymmetric Script Impact:** English uses the bicameral Latin script (capital and lowercase letters have distinct ASCII bytes), whereas Hindi uses the unicameral Devanagari script (where `.lower()` is a strict no-op: `459` tokens before and after).
  2. **Subword Vocabulary Mismatch:** Modern BPE tokenizers (`gpt2`, `o200k_base`, LLM tokenizers) are trained on cased text and contain single-token merges for capitalized words and acronyms. Lowercasing forces capitalized words onto rarer lowercase subword paths:
     - **Line 2 (`eng`):** `" Thursday"` (`1` token) $\xrightarrow{\text{.lower()}}$ `" th"`, `"ursday"` (`2` tokens, $\Delta = +1$).
     - **Line 6 (`eng`):** `"NASA"` (`1` token) $\xrightarrow{\text{.lower()}}$ `"n"`, `"asa"` (`2` tokens, $\Delta = +1$).
     - **Line 10 (`eng`):** `" GPU"` (`1` token) $\xrightarrow{\text{.lower()}}$ `" g"`, `"pu"` (`2` tokens, $\Delta = +1$).
     - *(Note: On Line 8, `" Mysuru"` splits into 3 tokens in both cased `[' M', 'ys', 'uru']` and lowercased `[' my', 'sur', 'u']`, so token count stays `9`.)*
- **Measured Effect:** Removing `.lower()` reduces total English tokens from `99` to `96` (`-3.03%`), lowers `eng` fertility from `1.265206` (`1.27`) to `1.229329` (`1.23`), lowers `eng` `tok/char` from `0.225636` (`0.226`) to `0.219562` (`0.220`), and increases the `hin/eng` fertility ratio from `5.89x` (`5.887148`) to `6.06x` (`6.058958`, $+2.92\%$).
- **Verdict:** **Methodological / Code Bug** when benchmarking production tokenizers for serving cost, since real user traffic is cased and `.lower()` selectively penalizes bicameral scripts.

---

### E3 — Macro-Average vs. Micro-Average
- **Mathematical Difference:**
  - **Macro-average (implemented in `fertility.py`):**
    $$\text{Fertility}_{\text{macro}} = \frac{1}{N} \sum_{i=1}^{N} \frac{T_i}{W_i}$$
    This answers: *"What is the unweighted expected fertility of a randomly sampled sentence, treating a 4-word sentence and a 12-word sentence as equally important?"*
  - **Micro-average (standard corpus definition):**
    $$\text{Fertility}_{\text{micro}} = \frac{\sum_{i=1}^{N} T_i}{\sum_{i=1}^{N} W_i} = \sum_{i=1}^{N} \left(\frac{W_i}{\sum_j W_j}\right) \frac{T_i}{W_i}$$
    This answers: *"Across an entire corpus or traffic stream, how many tokens are produced per word overall?"*
- **Measured Effect on `corpus_sample/`:**
  - In `eng`, short lines happen to have slightly higher fertility than the 12-word Line 3 (`1.083`), so micro-averaging drops `eng` fertility from `1.265206` (`1.27`) to `99 / 79 = 1.253165` (`1.25`, $-0.95\%$).
  - In `hin`, the shortest 4-word line (Line 8: `"मुझे थोड़ा पानी चाहिए।"`, fertility `8.75`) pulls up the macro-average; micro-averaging drops `hin` fertility from `7.448452` (`7.45`) to `459 / 62 = 7.403226` (`7.40`, $-0.61\%$).
  - The `hin/eng` fertility ratio shifts from `5.887148` (`5.89x`) to `5.907625` (`5.91x`), and the `tok/char` ratio shifts from `6.998478` (`7.00x`) to `7.162382` (`7.16x`).
- **Verdict:** **Metric-Design Choice (Inappropriate for Capacity/Cost Estimation).** Because GPU prefill/decode compute and KV-cache memory scale with the **sum of tokens across requests** ($\sum T_i$), micro-averaging is the right estimator for serving cost.

---

### E4 — Character-Counting Method (`len(line)` vs. UTF-8 Bytes vs. Grapheme Clusters)
*(Note: Because `tiktoken` requires `regex` as a dependency, `regex` is already present in `.venv-x64`, allowing us to measure exact Unicode Extended Grapheme Clusters `\X` alongside Unicode codepoints and UTF-8 bytes without installing any additional packages!)*

| Language | Total `gpt2` Tokens | UTF-8 Bytes (`len(s.encode('utf-8'))`) | Unicode Codepoints (`len(s)`) | Extended Grapheme Clusters (`regex.findall(r'\X', s)`) | Non-Space Codepoints |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **English (`eng`)** | `99` | `448` (`0.226` tok/B) | `448` (`0.226` tok/char) | `448` (`0.226` tok/grapheme) | `379` (`0.268` tok/non-space-char) |
| **Hindi (`hin`)** | `459` | `764` (`0.599` tok/B) | `290` (`1.579` tok/char) | `188` (`2.450` tok/grapheme) | `238` (`1.922` tok/non-space-char) |
| **`hin / eng` Ratio (Macro)** | — | **`2.65x`** (`2.654683`) | **`7.00x`** (`6.998478`) | **`10.86x`** (`10.857013`) | **`7.16x`** (`7.159222`) |

- **Why These Three "Character" Units Diverge Massively in Indic Scripts:**
  1. **Unicode Codepoints (`len(line)` = `290` in Hindi):** Counts every consonant (`क`), dependent vowel sign / *matra* (`ि`, `ा`, `े`), nasalization sign (`ं`, `ँ`), virama/halant (`्`), and nukta (`़`) as a separate "character".
  2. **Extended Grapheme Clusters (`\X` = `188` in Hindi):** Counts user-perceived orthographic characters (*aksharas*), grouping a base consonant + nukta + virama + consonant + matra + anusvara into a single visual unit. For example, in `hin` L04, `"हफ़्ते"` is `6` Unicode codepoints (`ह` + `फ` + `़` + `्` + `त` + `े`) and `18` UTF-8 bytes, but only `2` grapheme clusters (`ह` and `फ़्ते`)!
  3. **UTF-8 Bytes (`764` in Hindi):** Every Devanagari codepoint (`U+0900`–`U+097F`) plus the danda `।` (`U+0964`) encodes to **3 bytes** in UTF-8 (`238` non-space codepoints $\times 3 + 50$ ASCII spaces/question-mark $= 764$ bytes).
- **Direct Refutation of `REPORT_v0.md` Finding #2 and Finding #3:**
  - `REPORT_v0.md` Finding #3 claims: *"Root cause: Hindi simply has more Unicode characters per word, so any tokenizer will struggle. This is a property of the script, not the tokenizer."*
  - **Mathematical Refutation:** Look at the corpus counts:
    - English has $448\text{ codepoints} / 79\text{ words} = \mathbf{5.671\text{ Unicode characters/word}}$ ($5.744$ with `split()`).
    - Hindi has $290\text{ codepoints} / 62\text{ words} = \mathbf{4.677\text{ Unicode characters/word}}$ ($4.754$ with `split()`).
    - **Hindi actually has $17.5\%$ FEWER Unicode characters per word than English!**
    - Even simple algebra from the report's own table proves Finding #3 is backwards: since $\text{tok/word} = \text{tok/char} \times \text{char/word}$, a `5.89x` ratio in `tok/word` divided by a `7.00x` ratio in `tok/char` implies Hindi has $5.89 / 7.00 = 0.84\times$ as many characters per word as English!
    - The real reason `gpt2` emits `1.579` tokens per Unicode character in Hindi ($>1$ token per character!) is that `gpt2` is an English-centric **byte-level** BPE tokenizer that lacks Devanagari merges and therefore fragments each 3-byte Devanagari codepoint into $1\text{--}3$ raw byte tokens (`0.599` tokens per UTF-8 byte).

---

### E5 — Corpus Comparability (`eng_sample.txt` vs. `hin_sample.txt`)
- **Line-by-Line Inspection:** The assignment scenario notes that `corpus_sample/` was supposed to be *"English + Hindi, parallel line-by-line"*. Direct translation of all 10 lines reveals that **not a single line number `L01`–`L10` matches between the two files**, and **5 out of 10 sentences in each file do not exist in the other file at all**:

| Line | `eng_sample.txt` | `hin_sample.txt` (with English translation) | Relationship |
| :---: | :--- | :--- | :--- |
| **L01** | Bengaluru International Airport handled record traffic in March. | मुझे सुबह की चाय बहुत पसंद है। *(I like morning tea very much.)* | **Unrelated** (`eng` L01 has no Hindi match; `hin` L01 has no English match) |
| **L02** | The Quarterly Review meeting moved to Thursday. | बेंगलुरु में आज हल्की बारिश हो रही है। *(It is raining lightly in Bengaluru today.)* | **Unrelated** (neither has a match) |
| **L03** | I bought this book yesterday from a small shop near MG Road. *(12 words)* | यह किताब मैंने कल ही खरीदी थी। *(I bought this book just yesterday. — 7 words)* | **Partial semantic match (`eng` L03 $\leftrightarrow$ `hin` L03)**, but English has an extra prepositional phrase *"from a small shop near MG Road"*! |
| **L04** | Children are playing cricket on the ground. | हम अगले हफ़्ते मैसूर जा रहे हैं। *(We are going to Mysuru next week.)* | **Out of order** (`eng` L04 matches `hin` L07; `hin` L04 matches `eng` L08) |
| **L05** | The train arrived exactly on time. | क्या तुमने खाना खा लिया? *(Have you eaten food?)* | **Out of order / Unrelated** (`eng` L05 matches `hin` L06; `hin` L05 has no match) |
| **L06** | NASA and ISRO announced a joint mission update. | ट्रेन ठीक समय पर पहुँची। *(The train arrived right on time.)* | **Out of order / Unrelated** (`eng` L06 has no match; `hin` L06 matches `eng` L05) |
| **L07** | Please keep the books  in the cupboard. | बच्चे मैदान में क्रिकेट खेल रहे हैं। *(Children are playing cricket on the ground.)* | **Out of order** (`eng` L07 matches `hin` L10; `hin` L07 matches `eng` L04) |
| **L08** | We are visiting Mysuru next week. | मुझे थोड़ा पानी चाहिए। *(I want some water.)* | **Out of order / Unrelated** (`eng` L08 matches `hin` L04; `hin` L08 has no match) |
| **L09** | Do you want tea or coffee? | आज दफ़्तर में बहुत काम था। *(There was a lot of work at the office today.)* | **Unrelated** (neither has a match) |
| **L10** | The GPU cluster ran out of memory during the night job. | किताबें  अलमारी में रखी हैं। *(The books are kept in the cupboard.)* | **Out of order / Unrelated** (`eng` L10 has no match; `hin` L10 matches `eng` L07) |

- **Quantitative Impact of Non-Parallelism:**
  - Because `eng_sample.txt` contains longer, technical/proper-noun sentences (`Bengaluru International Airport`, `Quarterly Review`, `NASA and ISRO`, `GPU cluster`, `from a small shop near MG Road`), the English sample has **27% more words** (`78` vs `61`) and **54% more Unicode characters** (`448` vs `290`) than the Hindi sample!
  - Restricting to the 5 loosely matched sentence pairs (`eng` L03, L04, L05, L07, L08 vs `hin` L03, L07, L06, L10, L04) shifts the macro fertility ratio from **`5.89x` (`5.887148`) to `6.17x` (`6.174419`, $+4.88\%$)** and increases the total token ratio (`hin_tokens / eng_tokens`) from **`4.64x` (`459 / 99`) to `5.15x` (`242 / 47`, $+11.06\%$)**—and even within those 5 pairs, `eng` L03 (`"from a small shop near MG Road"`) and `eng` L07 (`"Please keep..."` vs declarative `"are kept"`) are not exact translations!
  - **Why Non-Parallel Text Invalidates Cross-Language Cost Ratios:** To know whether serving a user query in Hindi costs $K\times$ more tokens than serving the *same query* in English, the underlying semantic content must be held constant (a parallel corpus).

---

### E6 — Harmless-Looking Code Verification (Avoiding False-Positive Bug Claims)
Per the A2 prompt warning (*"And at least one thing in the script looks suspicious but is actually fine... Flagging the harmless thing as a bug — without evidence — costs you points"*), we tested all three suspicious-looking constructs in `fertility.py`:

1. **`random.seed(1337)` ([fertility.py:L21, L25](file:///c:/Users/TANNU/OneDrive/Desktop/starter_kit/fertility.py#L21-L25)):**
   - **Suspicion:** Why is `random` imported and seeded at the top of a deterministic tokenizer script?
   - **Experiment & Result:** Changing or removing `random.seed(1337)` produces a delta of **`0.0000000000`** across all metrics because `random` is never referenced anywhere else in `fertility.py`. It is harmless dead code.
2. **`unicodedata.normalize("NFC", line)` ([fertility.py:L49](file:///c:/Users/TANNU/OneDrive/Desktop/starter_kit/fertility.py#L49)):**
   - **Suspicion:** Does `NFC` normalization corrupt or alter Indic characters/tokens in `corpus_sample/`?
   - **Experiment & Result:** Comparing raw lines (`read_lines_raw`) vs. NFC-normalized lines (`read_lines_nfc`) shows **`0/10` lines changed in `eng_sample.txt` and `0/10` lines changed in `hin_sample.txt`** ($\Delta = \mathbf{0.0000000000}$). Even under a counterfactual `NFD` normalization test on `hin_sample.txt`, the 4 nukta words (`हफ़्ते`, `पहुँची`, `थोड़ा`, `दफ़्तर`) already use the canonical decomposed nukta (`\u093c`) in both NFC and NFD (because Unicode excludes precomposed Devanagari nukta letters from canonical composition), so `NFD` also produces `290` codepoints and `459` tokens! Furthermore, normalizing messy web corpora to `NFC` before tokenization is standard, good engineering practice.
3. **`tok.encode(s, add_special_tokens=False)` ([fertility.py:L33](file:///c:/Users/TANNU/OneDrive/Desktop/starter_kit/fertility.py#L33)):**
   - **Suspicion:** When using HuggingFace tokenizers (`hf:...`), disabling special tokens deviates from `tok.encode(s)`'s default (`add_special_tokens=True`).
   - **Analysis & Result:** This is **methodologically correct**, not a bug:
     - First, `tiktoken`'s `enc.encode(s)` does not add BOS/EOS tokens, so `add_special_tokens=False` ensures apples-to-apples parity between `tiktoken` and HuggingFace tokenizers.
     - Second, when evaluating corpus lines one sentence at a time, adding `[CLS]` + `[SEP]` ($+2$ tokens per line) would inject $+20$ artificial framing tokens across 10 lines, inflating short-sentence fertility by $15\text{--}50\%$ for reasons unrelated to subword segmentation.
4. **Bonus Observation — Unrounded Ratio Division ([fertility.py:L93, L100](file:///c:/Users/TANNU/OneDrive/Desktop/starter_kit/fertility.py#L93-L100)):**
   - Readers checking `REPORT_v0.md` by calculator notice `7.45 / 1.27 = 5.866` ($\neq 5.89$). That is because `fertility.py` prints `fert` formatted to `.2f` (`1.265206` $\rightarrow$ `1.27`, `7.448452` $\rightarrow$ `7.45`), but computes `ratio = results[lang][0] / results[base][0]` using full IEEE-754 double precision (`7.448452 / 1.265206 = 5.887148` $\rightarrow$ `5.89x`). Computing ratios in full precision before display rounding is **mathematically right** (avoiding intermediate rounding error).
