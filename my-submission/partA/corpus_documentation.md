# Part A1 — Multilingual Evaluation Corpus Documentation

## 1. Corpus Overview & Design Rationale

To replace the 10-sentence non-parallel smoke-test files in `corpus_sample/`, we constructed a **6-language, 1,012-sentence multi-way parallel evaluation corpus** from the official **FLORES-200 (`devtest`)** benchmark dataset released by Meta AI / the NLLB Team (Costa-jussà et al., 2022).

### Languages Included (6 Languages: English, Hindi, and 4 Dravidian Languages)
| Short Code | FLORES-200 ID | Language | Language Family | Script (ISO 15924) | Morphological Typology |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`eng`** | `eng_Latn` | English | Indo-European (Germanic) | Latin (`Latn`) | Analytic |
| **`hin`** | `hin_Deva` | Hindi | Indo-European (Indo-Aryan) | Devanagari (`Deva`) | Analytic / Postpositional (spaces before postpositions & auxiliaries) |
| **`tam`** | `tam_Taml` | Tamil | Dravidian (Southern) | Tamil (`Taml`) | Agglutinative |
| **`tel`** | `tel_Telu` | Telugu | Dravidian (South-Central) | Telugu (`Telu`) | Agglutinative |
| **`kan`** | `kan_Knda` | Kannada | Dravidian (Southern) | Kannada (`Knda`) | Agglutinative |
| **`mal`** | `mal_Mlym` | Malayalam | Dravidian (Southern) | Malayalam (`Mlym`) | Highly Agglutinative |

---

## 2. Provenance, Licensing, and Domain Metadata

- **Source Archive URL:** `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`
- **Archive Size:** `25,585,843` bytes
- **Archive SHA-256:** `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6`
- **Split Used:** `devtest` (`./flores200_dataset/devtest/<lang_script>.devtest` and `./flores200_dataset/metadata_devtest.tsv`)
- **License & Usage Terms:** **CC-BY-SA 4.0** (Creative Commons Attribution-ShareAlike 4.0 International), matching the Wikimedia source articles from which FLORES-200 was professionally translated.
- **Collection Method:** Automated, deterministic download and extraction via `your-submission/partA/build_corpus.py` using Python's standard `urllib.request` and `tarfile` modules.
- **Corpus Size & Document Provenance (from `metadata_devtest.tsv`):**
  - **Sentences per language:** `1,012` (strictly aligned 1-to-1 line-by-line across all 6 languages; `6,072` total sentences).
  - **Unique source documents (`URL`s):** `281` distinct English Wikimedia articles (approx. 3–4 contiguous sentences extracted per article).
  - **Domain Breakdown:**
    - **WikiBooks (`wikibooks` / WikiJunior):** `351` sentences (`34.68%`) — factual non-fiction and educational prose.
    - **WikiNews (`wikinews`):** `341` sentences (`33.70%`) — international journalism and news reporting.
    - **WikiVoyage (`wikivoyage`):** `320` sentences (`31.62%`) — travel guides, geography, and practical advice.
  - **Top Topic Tags (case-normalized):** `travel` (`91`), `politics` (`51`), `sports` (`39`), `crime and law` (`32`), `entertainment` (`25`), `travel, safety` (`25`), `health` (`23`), `science` (`20`).

---

## 3. Directory Structure & Preprocessing Pipeline

Raw and processed text are stored in separate directories under `your-submission/partA/corpus/`:

- **Raw Text (Unmodified Upstream Bytes):**
  - `your-submission/partA/corpus/raw/{eng,hin,tam,tel,kan,mal}.txt`
  - `your-submission/partA/corpus/raw/metadata_devtest.tsv`
  - `your-submission/partA/corpus/raw/README`
- **Processed / Normalized Text (Ready for Tokenizer Evaluation):**
  - `your-submission/partA/corpus/processed/{eng,hin,tam,tel,kan,mal}.txt`

### Preprocessing Rules Applied (`raw` $\rightarrow$ `processed`)
1. **UTF-8 Decoding (`strict`):** Verified that every raw file is valid UTF-8 with `0` replacement characters (`U+FFFD`) and `0` C0/C1 control characters.
2. **Line Trimming & Empty-Line Check:** Each line is stripped of leading/trailing whitespace (`line.strip()`). All `1,012` lines in every language are non-empty, preserving exact 1-to-1 line index alignment.
3. **Unicode NFC Normalization (`unicodedata.normalize("NFC", line)`):**
   - In raw `hin_Deva.devtest`, **93 lines** contain legacy precomposed Devanagari nukta singletons (`U+0958`–`U+095F`, such as `U+095B` `ज़` and `U+095C` `ड़`), which are Unicode canonical composition exclusions that expand under `NFC` into base consonant + `U+093C` nukta (`+122` codepoints across the 93 lines).
   - In raw Dravidian files, **14 lines** total (`7` in `kan`, `4` in `mal`, `2` in `tam`, `1` in `tel`) contain decomposed two-part vowel signs that canonically compose under `NFC` (`-13` codepoints in `kan`, `-5` in `tel`, `-4` in `mal`, `-2` in `tam`).
4. **Consecutive Whitespace Collapsing (`" ".join(line.split())`):**
   - Raw FLORES-200 contains consecutive internal spaces (`"  "`) in **213 Kannada lines** (`+477` extra spaces), **136 Telugu lines** (`+274` extra spaces), **75 Tamil lines** (`+100` extra spaces), **54 Malayalam lines** (`+71` extra spaces), and **6 Hindi lines** (`+27` extra spaces), vs. **0 English lines**. Collapsing internal whitespace runs to a single ASCII space `' '` prevents extra space tokens and `split(" ")` empty-word inflation.
5. **Original Casing Preserved:** Text is **never** lowercased (`lowercase=False`), avoiding the bicameral BPE distortion identified in Experiment E2.
6. **What Was Intentionally Not Altered:** Upstream punctuation (including CSV-escaped double quotes `""` present in 60 `tam`, 68 `tel`, and 58 `mal` lines in raw FLORES-200) was not hand-edited so that corpus construction remains 100% deterministic and faithful to the official benchmark text; this upstream artifact is documented below in our caveats.

---

## 4. Corpus Validation Results

### 4.1 Starter-Kit Sample Corpus (`corpus_sample/`)
| Lang | File Path | Lines | Empty | Dups | Multi-Space Lines | Non-NFC Lines | Words (`split(" ")`) | Words (`split()`) | Unicode Codepoints | Grapheme Clusters (`\X`) | UTF-8 Bytes (Lines / File) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`eng`** | `corpus_sample/eng_sample.txt` | `10` | `0` | `0` | `1` (`L07`) | `0` | `79` | `78` | `448` | `448` | `448` / `458` |
| **`hin`** | `corpus_sample/hin_sample.txt` | `10` | `0` | `0` | `1` (`L10`) | `0` | `62` | `61` | `290` | `188` | `764` / `774` |

### 4.2 Raw FLORES-200 `devtest` Corpus (`your-submission/partA/corpus/raw/`)
| Lang | FLORES ID | Lines | Empty | Dups | UTF-8 Valid | Non-NFC Lines ($\Delta$ CP) | Multi-Space Lines (Phantom Words) | Words (`split(" ")`) | Words (`split()`) | Unicode Codepoints | Grapheme Clusters (`\X`) | UTF-8 Bytes (Lines / File) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`eng`** | `eng_Latn` | `1,012` | `0` | `0` | Yes | `0` (`+0`) | `0` (`+0`) | `21,901` | `21,901` | `131,966` | `131,966` | `132,096` / `133,108` |
| **`hin`** | `hin_Deva` | `1,012` | `0` | `0` | Yes | `93` (`+122`) | `6` (`+27`) | `25,670` | `25,643` | `131,079` | `85,978` | `337,094` / `338,106` |
| **`tam`** | `tam_Taml` | `1,012` | `0` | `0` | Yes | `2` (`-2`) | `75` (`+100`) | `16,875` | `16,775` | `154,133` | `99,724` | `421,641` / `422,653` |
| **`tel`** | `tel_Telu` | `1,012` | `0` | `0` | Yes | `1` (`-5`) | `136` (`+274`) | `17,212` | `16,938` | `132,505` | `76,602` | `353,711` / `354,723` |
| **`kan`** | `kan_Knda` | `1,012` | `0` | `0` | Yes | `7` (`-13`) | `213` (`+477`) | `16,577` | `16,100` | `138,140` | `90,471` | `375,480` / `376,492` |
| **`mal`** | `mal_Mlym` | `1,012` | `0` | `0` | Yes | `4` (`-4`) | `54` (`+71`) | `15,001` | `14,930` | `149,336` | `79,428` | `411,753` / `412,765` |

### 4.3 Processed FLORES-200 `devtest` Corpus (`your-submission/partA/corpus/processed/`)
| Lang | Script | Lines | Empty | Dups | Non-NFC | Multi-Space | Words (`split()`) | Unicode Codepoints | Non-Space Codepoints | Grapheme Clusters (`\X`) | UTF-8 Bytes (Lines / File) | Codepoints / Word | Graphemes / Word | UTF-8 Bytes / Word |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`eng`** | Latin | `1,012` | `0` | `0` | `0` | `0` | `21,901` | `131,966` | `111,077` | `131,966` | `132,096` / `133,108` | `6.03` | `6.03` | `6.03` |
| **`hin`** | Devanagari | `1,012` | `0` | `0` | `0` | `0` | `25,643` | `131,174` | `106,543` | `85,951` | `337,433` / `338,445` | `5.12` | `3.35` | `13.16` |
| **`tam`** | Tamil | `1,012` | `0` | `0` | `0` | `0` | `16,775` | `154,031` | `138,268` | `99,624` | `421,535` / `422,547` | `9.18` | `5.94` | `25.13` |
| **`tel`** | Telugu | `1,012` | `0` | `0` | `0` | `0` | `16,938` | `132,226` | `116,300` | `76,328` | `353,422` / `354,434` | `7.81` | `4.51` | `20.87` |
| **`kan`** | Kannada | `1,012` | `0` | `0` | `0` | `0` | `16,100` | `137,650` | `122,562` | `89,994` | `374,964` / `375,976` | `8.55` | `5.59` | `23.29` |
| **`mal`** | Malayalam | `1,012` | `0` | `0` | `0` | `0` | `14,930` | `149,261` | `135,343` | `79,357` | `411,670` / `412,682` | `10.00` | `5.32` | `27.57` |

---

## 5. What This Corpus Cannot Tell Us (Sample-Size, Domain, and Translation Caveats)

While FLORES-200 `devtest` (`1,012` 6-way parallel sentences across `281` articles) eliminates the fatal non-parallelism of `corpus_sample/`, it has four concrete limitations that must be accounted for when making production routing and capacity decisions:

1. **Formal Encyclopedic Register vs. Conversational Assistant Traffic:** FLORES-200 is drawn entirely from edited Wikimedia prose (`WikiBooks`, `WikiNews`, `WikiVoyage`). Real user-assistant traffic in India includes casual imperatives, colloquial slang, multi-turn chat formatting, markdown/JSON/code snippets, and **Romanized Indic / Hinglish / Tanglish code-switching** (e.g., Hindi or Tamil written in Latin script), where byte-level BPE tokenizers behave very differently than on pure native-script prose.
2. **Translationese and English-Centric Source Bias:** All `1,012` sentences were authored in English and translated into Hindi and Dravidian languages. Translated text ("translationese") systematically mirrors English clause structure, named entities (`Dr. Moll`, `Toronto`), and loanwords (e.g., Kannada L0001 transliterates *"diabetic"* as `ಡಯಾಬೆಟಿಕ್` whereas Hindi uses `मधुमेह`), which may under- or over-estimate subword fragmentation relative to natively authored Indic requests.
3. **Upstream FLORES-200 Formatting & Translation Artifacts:** Line-level inspection revealed that raw FLORES-200 contains CSV-escaped quote marks (`""` / `"""`) in `60` Tamil, `68` Telugu, and `58` Malayalam sentences (e.g., Sentence #0001), as well as occasional clause omissions by human translators (e.g., Sentence #1012 in `tel_Telu` omits the second clause *"and are expected to obey their superiors' instructions without question"*). Although minor across `1,012` sentences, this means alignment is human-translation parallel rather than synthetically exact down to every clause.
4. **Input vs. Output Token Asymmetry:** Measuring tokenizer fertility on static text tells us how many tokens a prompt or response of equivalent meaning consumes, but cannot tell us whether a model generates *more verbose explanations* or *hallucinates/repeats* more frequently in lower-resource Dravidian languages during autoregressive decoding.

---

## 6. Reproducible Commands

```powershell
# Rebuild raw and processed corpora and run full validation
.\.venv-x64\Scripts\python.exe your-submission/partA/build_corpus.py
```
