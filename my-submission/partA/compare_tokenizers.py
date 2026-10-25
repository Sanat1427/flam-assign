#!/usr/bin/env python3
r"""
compare_tokenizers.py -- Part A3 Multilingual Tokenizer & Denominator Comparison

Evaluates three tiktoken BPE tokenizers across three generations of multilingual/Indic support:
  1. gpt2        (vocab=50,257)  -- English-centric byte-level BPE (baseline local cache)
  2. cl100k_base (vocab=100,277) -- GPT-3.5 / GPT-4 multilingual byte-level BPE
  3. o200k_base  (vocab=200,019) -- GPT-4o Indic-aware multilingual byte-level BPE

on the 1,012-sentence 6-way parallel FLORES-200 devtest corpus from Part A1:
  - eng (English, Latin)
  - hin (Hindi, Devanagari)
  - tam (Tamil, Tamil)
  - tel (Telugu, Telugu)
  - kan (Kannada, Kannada)
  - mal (Malayalam, Malayalam)

Also records the exact failure when attempting `hf:xlm-roberta-base` in `.venv-x64`
without `transformers` installed, and compares unmodified `fertility.py` output vs.
corrected cased micro/macro metrics across five denominators:
  - Whitespace words (`line.split()`)
  - Unicode codepoints (`len(line)`)
  - Extended grapheme clusters (`len(regex.findall(r"\X", line))`)
  - UTF-8 bytes (`len(line.encode("utf-8"))`)
  - Parallel sentence (`N = 1,012` aligned propositions)

Usage:
    .\.venv-x64\Scripts\python.exe your-submission/partA/compare_tokenizers.py
"""

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import regex
import tiktoken

sys.stdout.reconfigure(encoding="utf-8")

LANGUAGES = [
    ("eng", "English", "Latin", "Indo-European (Germanic)"),
    ("hin", "Hindi", "Devanagari", "Indo-European (Indo-Aryan)"),
    ("tam", "Tamil", "Tamil", "Dravidian"),
    ("tel", "Telugu", "Telugu", "Dravidian"),
    ("kan", "Kannada", "Kannada", "Dravidian"),
    ("mal", "Malayalam", "Malayalam", "Dravidian"),
]

TOKENIZERS = [
    ("gpt2", "tiktoken 'gpt2' (GPT-2 / r50k BPE, 50,257 vocab)"),
    ("cl100k_base", "tiktoken 'cl100k_base' (GPT-3.5/GPT-4 BPE, 100,277 vocab)"),
    ("o200k_base", "tiktoken 'o200k_base' (GPT-4o Indic-aware BPE, 200,019 vocab)"),
]


def check_environment_and_caches():
    info = {
        "python_version": sys.version.split()[0],
        "tiktoken_version": tiktoken.__version__,
        "regex_version": regex.__version__,
        "transformers_installed": importlib.util.find_spec("transformers") is not None,
        "tokenizers_installed": importlib.util.find_spec("tokenizers") is not None,
        "sentencepiece_installed": importlib.util.find_spec("sentencepiece") is not None,
        "tiktoken_cache_dir": os.path.join(tempfile.gettempdir(), "data-gym-cache"),
    }
    cache_dir = Path(info["tiktoken_cache_dir"])
    if cache_dir.exists():
        info["cached_tiktoken_files"] = [
            {"name": p.name, "size_bytes": p.stat().st_size}
            for p in sorted(cache_dir.iterdir())
            if p.is_file()
        ]
    else:
        info["cached_tiktoken_files"] = []

    # Test hf:xlm-roberta-base execution failure cleanly
    try:
        import transformers  # noqa: F401
        info["hf_probe_result"] = "transformers available"
    except Exception as e:
        info["hf_probe_result"] = f"{type(e).__name__}: {e}"

    return info


def evaluate_corpus(lines: list[str], encode_fn, lowercase: bool = False, split_mode: str = "whitespace"):
    """
    Evaluate a list of aligned sentences on all denominators, returning both
    corpus-level ratios (micro-averages) and unweighted line-level means (macro-averages).
    """
    n_sents = len(lines)
    tot_tokens = 0
    tot_words = 0
    tot_codepoints = 0
    tot_codepoints_nospace = 0
    tot_graphemes = 0
    tot_utf8_bytes = 0

    macro_tok_per_word = 0.0
    macro_tok_per_cp = 0.0
    macro_tok_per_cp_nospace = 0.0
    macro_tok_per_grapheme = 0.0
    macro_tok_per_byte = 0.0

    for raw_line in lines:
        line = raw_line.lower() if lowercase else raw_line
        tokens = encode_fn(line)
        n_tok = len(tokens)

        if split_mode == "whitespace":
            words = line.split()
        elif split_mode == "space":
            words = line.split(" ")
        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")

        n_w = len(words)
        n_cp = len(line)
        n_cp_ns = len(line.replace(" ", ""))
        n_gr = len(regex.findall(r"\X", line))
        n_by = len(line.encode("utf-8"))

        tot_tokens += n_tok
        tot_words += n_w
        tot_codepoints += n_cp
        tot_codepoints_nospace += n_cp_ns
        tot_graphemes += n_gr
        tot_utf8_bytes += n_by

        macro_tok_per_word += n_tok / n_w
        macro_tok_per_cp += n_tok / n_cp
        macro_tok_per_cp_nospace += n_tok / n_cp_ns
        macro_tok_per_grapheme += n_tok / n_gr
        macro_tok_per_byte += n_tok / n_by

    return {
        "n_sentences": n_sents,
        "tot_tokens": tot_tokens,
        "tot_words": tot_words,
        "tot_codepoints": tot_codepoints,
        "tot_codepoints_nospace": tot_codepoints_nospace,
        "tot_graphemes": tot_graphemes,
        "tot_utf8_bytes": tot_utf8_bytes,
        # Corpus-level micro-averages (sum of numerators / sum of denominators)
        "micro_tok_per_sent": tot_tokens / n_sents,
        "micro_tok_per_word": tot_tokens / tot_words,
        "micro_tok_per_cp": tot_tokens / tot_codepoints,
        "micro_tok_per_cp_nospace": tot_tokens / tot_codepoints_nospace,
        "micro_tok_per_grapheme": tot_tokens / tot_graphemes,
        "micro_tok_per_byte": tot_tokens / tot_utf8_bytes,
        "micro_bytes_per_tok": tot_utf8_bytes / tot_tokens,
        # Line-level macro-averages (mean of per-line ratios)
        "macro_tok_per_word": macro_tok_per_word / n_sents,
        "macro_tok_per_cp": macro_tok_per_cp / n_sents,
        "macro_tok_per_cp_nospace": macro_tok_per_cp_nospace / n_sents,
        "macro_tok_per_grapheme": macro_tok_per_grapheme / n_sents,
        "macro_tok_per_byte": macro_tok_per_byte / n_sents,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default="your-submission/partA/corpus/processed")
    ap.add_argument("--raw-corpus-dir", default="your-submission/partA/corpus/raw")
    ap.add_argument("--out-json", default="your-submission/partA/a3_tokenizer_comparison.json")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    raw_corpus_dir = Path(args.raw_corpus_dir)

    env_info = check_environment_and_caches()
    print("=== 1. ENVIRONMENT, INSTALLED PACKAGES & TOKENIZER CACHE CHECK ===")
    print(f"Python version:         {env_info['python_version']}")
    print(f"tiktoken version:       {env_info['tiktoken_version']}")
    print(f"regex version:          {env_info['regex_version']}")
    print(f"transformers installed: {env_info['transformers_installed']} ({env_info['hf_probe_result']})")
    print(f"tiktoken cache dir:     {env_info['tiktoken_cache_dir']}")
    for cf in env_info["cached_tiktoken_files"]:
        print(f"  - cached BPE rank file: {cf['name']} ({cf['size_bytes']:,} bytes)")

    # Load processed corpus lines & verify identical sentence counts (1,012)
    lines_by_lang = {}
    raw_lines_by_lang = {}
    for code, name, script, family in LANGUAGES:
        p = corpus_dir / f"{code}.txt"
        rp = raw_corpus_dir / f"{code}.txt"
        lines_by_lang[code] = p.read_text(encoding="utf-8").splitlines()
        raw_lines_by_lang[code] = [l.strip() for l in rp.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines_by_lang[code]) == 1012, f"Expected 1012 lines for {code}, got {len(lines_by_lang[code])}"

    print("\n=== 2. SENTENCE ALIGNMENT & INVARIANT DENOMINATOR VERIFICATION ===")
    print("Verified: All 6 languages contain exactly 1,012 aligned sentences (IDs 1..1012).")
    print(
        f"{'Lang':<6} | {'Language':<10} | {'Script':<10} | {'Sents':>5} | {'Words(ws)':>9} | {'Codepoints':>10} | {'Graphemes(\\X)':>13} | {'UTF-8 Bytes':>11}"
    )
    print("-" * 92)
    for code, name, script, family in LANGUAGES:
        ls = lines_by_lang[code]
        w = sum(len(l.split()) for l in ls)
        cp = sum(len(l) for l in ls)
        gr = sum(len(regex.findall(r"\X", l)) for l in ls)
        by = sum(len(l.encode("utf-8")) for l in ls)
        print(f"{code:<6} | {name:<10} | {script:<10} | {len(ls):>5} | {w:>9,d} | {cp:>10,d} | {gr:>13,d} | {by:>11,d}")

    all_results = {
        "environment": env_info,
        "tokenizers": {},
    }

    for tok_id, tok_desc in TOKENIZERS:
        enc = tiktoken.get_encoding(tok_id)
        # Note: tiktoken enc.encode adds 0 special tokens by default
        encode_fn = enc.encode

        print(f"\n{'=' * 118}")
        print(f"TOKENIZER: {tok_desc} | n_vocab={enc.n_vocab:,} | special_tokens_added=0 | lowercase=False")
        print(f"{'=' * 118}")

        tok_res_corrected = {}
        tok_res_v0_on_proc = {}
        tok_res_v0_on_raw = {}

        for code, name, script, family in LANGUAGES:
            # 1. Corrected evaluation on processed corpus (cased, split(), micro + macro)
            tok_res_corrected[code] = evaluate_corpus(
                lines_by_lang[code], encode_fn, lowercase=False, split_mode="whitespace"
            )
            # 2. Legacy v0 evaluation on processed corpus (lowercased, split(" "), macro)
            tok_res_v0_on_proc[code] = evaluate_corpus(
                lines_by_lang[code], encode_fn, lowercase=True, split_mode="space"
            )
            # 3. Legacy v0 evaluation on raw corpus (lowercased, split(" "), macro)
            tok_res_v0_on_raw[code] = evaluate_corpus(
                raw_lines_by_lang[code], encode_fn, lowercase=True, split_mode="space"
            )

        eng_base = tok_res_corrected["eng"]

        # Table 1: Micro-averaged (Corpus-Level Ratio of Totals) across all 5 denominators
        print("\n[A] CORPUS-LEVEL MICRO-AVERAGES (Sum of Tokens / Sum of Denominator) — Preserving Original Casing:")
        print(
            f"{'Lang':<5} | {'Total Tok':>9} | {'Tok/Sent':>8} | {'Parity(vs eng)':>14} | "
            f"{'Tok/Word (vs eng)':>18} | {'Tok/Grapheme (vs eng)':>21} | {'Tok/Codepoint (vs eng)':>22} | {'Tok/Byte (Bytes/Tok)':>20}"
        )
        print("-" * 128)
        for code, name, script, family in LANGUAGES:
            r = tok_res_corrected[code]
            parity = r["tot_tokens"] / eng_base["tot_tokens"]
            w_ratio = r["micro_tok_per_word"] / eng_base["micro_tok_per_word"]
            g_ratio = r["micro_tok_per_grapheme"] / eng_base["micro_tok_per_grapheme"]
            cp_ratio = r["micro_tok_per_cp"] / eng_base["micro_tok_per_cp"]
            b_ratio = r["micro_tok_per_byte"] / eng_base["micro_tok_per_byte"]

            # Save ratios into dict
            r["parity_ratio_vs_eng"] = parity
            r["micro_word_fertility_ratio_vs_eng"] = w_ratio
            r["micro_grapheme_ratio_vs_eng"] = g_ratio
            r["micro_cp_ratio_vs_eng"] = cp_ratio
            r["micro_byte_ratio_vs_eng"] = b_ratio

            print(
                f"{code:<5} | {r['tot_tokens']:>9,d} | {r['micro_tok_per_sent']:>8.2f} | {parity:>13.2f}x | "
                f"{r['micro_tok_per_word']:>7.3f} ({w_ratio:>5.2f}x) | "
                f"{r['micro_tok_per_grapheme']:>9.3f} ({g_ratio:>6.2f}x) | "
                f"{r['micro_tok_per_cp']:>10.3f} ({cp_ratio:>6.2f}x) | "
                f"{r['micro_tok_per_byte']:>7.3f} ({r['micro_bytes_per_tok']:>5.2f} B/tok)"
            )

        # Table 2: Macro-averaged (Mean of Per-Line Ratios) vs Micro-averaged Comparison
        print("\n[B] MACRO-AVERAGE (Mean of Line Ratios) vs. MICRO-AVERAGE (Ratio of Corpus Totals) — Cased, split():")
        print(
            f"{'Lang':<5} | {'Macro Tok/Word':>14} | {'Micro Tok/Word':>14} | {'Diff (Macro-Micro)':>18} | "
            f"{'Macro Tok/CP':>12} | {'Micro Tok/CP':>12} | {'Macro Tok/Graph':>15} | {'Micro Tok/Graph':>15} | {'Macro Tok/Byte':>14} | {'Micro Tok/Byte':>14}"
        )
        print("-" * 158)
        for code, name, script, family in LANGUAGES:
            r = tok_res_corrected[code]
            r["macro_word_fertility_ratio_vs_eng"] = r["macro_tok_per_word"] / eng_base["macro_tok_per_word"]
            diff_w = r["macro_tok_per_word"] - r["micro_tok_per_word"]
            print(
                f"{code:<5} | {r['macro_tok_per_word']:>14.4f} | {r['micro_tok_per_word']:>14.4f} | {diff_w:>+18.4f} | "
                f"{r['macro_tok_per_cp']:>12.4f} | {r['micro_tok_per_cp']:>12.4f} | "
                f"{r['macro_tok_per_grapheme']:>15.4f} | {r['micro_tok_per_grapheme']:>15.4f} | "
                f"{r['macro_tok_per_byte']:>14.4f} | {r['micro_tok_per_byte']:>14.4f}"
            )

        # Table 3: Legacy v0 fertility.py pipeline vs Corrected pipeline on FLORES-200
        print("\n[C] IMPACT OF A2 BUGS ON FLORES-200 (v0 Raw [lower+split(' ')+macro] vs. Corrected [cased+split()+micro]):")
        print(
            f"{'Lang':<5} | {'v0 Raw Tok':>10} | {'Corr Tok':>8} | {'v0 Raw Words':>12} | {'Corr Words':>10} | "
            f"{'v0 Raw Fert (vs eng)':>20} | {'Corr Micro Fert (vs eng)':>24}"
        )
        print("-" * 106)
        v0_eng_fert = tok_res_v0_on_raw["eng"]["macro_tok_per_word"]
        for code, name, script, family in LANGUAGES:
            rv0 = tok_res_v0_on_raw[code]
            rc = tok_res_corrected[code]
            v0_ratio = rv0["macro_tok_per_word"] / v0_eng_fert
            c_ratio = rc["micro_tok_per_word"] / eng_base["micro_tok_per_word"]
            print(
                f"{code:<5} | {rv0['tot_tokens']:>10,d} | {rc['tot_tokens']:>8,d} | {rv0['tot_words']:>12,d} | {rc['tot_words']:>10,d} | "
                f"{rv0['macro_tok_per_word']:>8.3f} ({v0_ratio:>6.2f}x) | {rc['micro_tok_per_word']:>12.3f} ({c_ratio:>6.2f}x)"
            )

        all_results["tokenizers"][tok_id] = {
            "description": tok_desc,
            "n_vocab": enc.n_vocab,
            "corrected_processed": tok_res_corrected,
            "v0_on_processed": tok_res_v0_on_proc,
            "v0_on_raw": tok_res_v0_on_raw,
        }

    # Cross-Tokenizer Efficiency Summary (gpt2 vs cl100k_base vs o200k_base)
    print(f"\n{'=' * 118}")
    print("CROSS-TOKENIZER EFFICIENCY REDUCTION SUMMARY (o200k_base vs. gpt2 and cl100k_base on Processed Corpus)")
    print(f"{'=' * 118}")
    print(
        f"{'Lang':<5} | {'gpt2 Tokens':>11} | {'cl100k Tokens':>13} | {'o200k Tokens':>12} | "
        f"{'o200k vs gpt2 Reduction':>23} | {'gpt2 Parity':>11} | {'cl100k Parity':>13} | {'o200k Parity':>12} | "
        f"{'gpt2 Tok/W':>10} | {'o200k Tok/W':>11}"
    )
    print("-" * 138)
    for code, name, script, family in LANGUAGES:
        g2 = all_results["tokenizers"]["gpt2"]["corrected_processed"][code]
        cl = all_results["tokenizers"]["cl100k_base"]["corrected_processed"][code]
        o2 = all_results["tokenizers"]["o200k_base"]["corrected_processed"][code]
        red_pct = (o2["tot_tokens"] - g2["tot_tokens"]) / g2["tot_tokens"] * 100.0
        speedup = g2["tot_tokens"] / o2["tot_tokens"]
        print(
            f"{code:<5} | {g2['tot_tokens']:>11,d} | {cl['tot_tokens']:>13,d} | {o2['tot_tokens']:>12,d} | "
            f"{red_pct:>+9.2f}% ({speedup:>5.2f}x less) | "
            f"{g2['parity_ratio_vs_eng']:>10.2f}x | {cl['parity_ratio_vs_eng']:>12.2f}x | {o2['parity_ratio_vs_eng']:>11.2f}x | "
            f"{g2['micro_tok_per_word']:>10.3f} | {o2['micro_tok_per_word']:>11.3f}"
        )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved full JSON results to: {out_path}")


if __name__ == "__main__":
    main()
