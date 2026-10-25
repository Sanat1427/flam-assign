#!/usr/bin/env python3
r"""
audit_experiments.py -- Part A2 Controlled Tokenizer Audit (E1-E6)

Runs baseline and single-variable controlled experiments on fertility.py
and corpus_sample/ (eng_sample.txt, hin_sample.txt) using tiktoken 'gpt2'.
Does not modify any original starter-kit files.

Usage:
    .\.venv-x64\Scripts\python.exe my-submission/partA/audit_experiments.py
"""

import argparse
import importlib.util
import random
import sys
import unicodedata

import regex
import tiktoken

sys.stdout.reconfigure(encoding="utf-8")


def read_lines_raw(path: str):
    """Read stripped non-empty lines WITHOUT NFC normalization."""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            lines.append(line)
    return lines


def read_lines_nfc(path: str):
    """Exact read_lines() from fertility.py (with NFC normalization)."""
    return [unicodedata.normalize("NFC", line) for line in read_lines_raw(path)]


def compute_metrics(
    lines,
    encode,
    lowercase: bool = True,
    split_mode: str = "space",  # "space" -> line.split(" "), "whitespace" -> line.split()
    agg_mode: str = "macro",  # "macro" -> mean of line ratios, "micro" -> ratio of totals
    char_mode: str = "codepoints",  # "codepoints", "codepoints_no_space", "utf8_bytes", "graphemes", "graphemes_no_space"
):
    per_line_fert = []
    per_line_tpc = []
    tot_tokens = 0
    tot_words = 0
    tot_chars = 0
    line_details = []

    for idx, orig_line in enumerate(lines, 1):
        line = orig_line.lower() if lowercase else orig_line
        tokens = encode(line)
        if split_mode == "space":
            words = line.split(" ")
        elif split_mode == "whitespace":
            words = line.split()
        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")

        if char_mode == "codepoints":
            chars = len(line)
        elif char_mode == "codepoints_no_space":
            chars = len(line.replace(" ", ""))
        elif char_mode == "utf8_bytes":
            chars = len(line.encode("utf-8"))
        elif char_mode == "graphemes":
            chars = len(regex.findall(r"\X", line))
        elif char_mode == "graphemes_no_space":
            chars = len(regex.findall(r"\X", line.replace(" ", "")))
        else:
            raise ValueError(f"Unknown char_mode: {char_mode}")

        n_tok = len(tokens)
        n_w = len(words)
        f_line = n_tok / n_w
        c_line = n_tok / chars

        per_line_fert.append(f_line)
        per_line_tpc.append(c_line)
        tot_tokens += n_tok
        tot_words += n_w
        tot_chars += chars

        line_details.append(
            {
                "line_num": idx,
                "orig": orig_line,
                "proc": line,
                "n_tokens": n_tok,
                "n_words": n_w,
                "n_chars": chars,
                "fert": f_line,
                "tpc": c_line,
                "tokens": tokens,
                "words_list": words,
            }
        )

    n = len(lines)
    if agg_mode == "macro":
        fert = sum(per_line_fert) / n
        tpc = sum(per_line_tpc) / n
    elif agg_mode == "micro":
        fert = tot_tokens / tot_words
        tpc = tot_tokens / tot_chars
    else:
        raise ValueError(f"Unknown agg_mode: {agg_mode}")

    return {
        "fert": fert,
        "tpc": tpc,
        "micro_fert": tot_tokens / tot_words,
        "micro_tpc": tot_tokens / tot_chars,
        "tot_tokens": tot_tokens,
        "tot_words": tot_words,
        "tot_chars": tot_chars,
        "lines": line_details,
    }


def print_comparison(title: str, base_eng, base_hin, exp_eng, exp_hin):
    base_f_ratio = base_hin["fert"] / base_eng["fert"]
    base_c_ratio = base_hin["tpc"] / base_eng["tpc"]
    exp_f_ratio = exp_hin["fert"] / exp_eng["fert"]
    exp_c_ratio = exp_hin["tpc"] / exp_eng["tpc"]

    print(f"\n{'=' * 78}")
    print(f"{title}")
    print(f"{'=' * 78}")
    print(
        f"{'Metric':<28} | {'Baseline (unrounded / fmt)':<26} | {'Experiment (unrounded / fmt)':<26} | {'Delta (Exp - Base)':<18}"
    )
    print("-" * 106)
    rows = [
        ("eng fertility (tok/word)", base_eng["fert"], exp_eng["fert"], ".2f"),
        ("hin fertility (tok/word)", base_hin["fert"], exp_hin["fert"], ".2f"),
        ("hin/eng fertility ratio", base_f_ratio, exp_f_ratio, ".2f"),
        ("eng tok/char (or denom)", base_eng["tpc"], exp_eng["tpc"], ".3f"),
        ("hin tok/char (or denom)", base_hin["tpc"], exp_hin["tpc"], ".3f"),
        ("hin/eng tok/char ratio", base_c_ratio, exp_c_ratio, ".2f"),
    ]
    for name, b_val, e_val, fmt in rows:
        delta = e_val - b_val
        pct = (delta / b_val) * 100 if b_val != 0 else 0.0
        b_str = f"{b_val:.6f} ({format(b_val, fmt)})"
        e_str = f"{e_val:.6f} ({format(e_val, fmt)})"
        d_str = f"{delta:+.6f} ({pct:+.2f}%)"
        print(f"{name:<28} | {b_str:<26} | {e_str:<26} | {d_str:<18}")
    print(
        f"Totals (Base -> Exp): eng tokens={base_eng['tot_tokens']}->{exp_eng['tot_tokens']}, "
        f"eng words={base_eng['tot_words']}->{exp_eng['tot_words']}, eng chars={base_eng['tot_chars']}->{exp_eng['tot_chars']} | "
        f"hin tokens={base_hin['tot_tokens']}->{exp_hin['tot_tokens']}, "
        f"hin words={base_hin['tot_words']}->{exp_hin['tot_words']}, hin chars={base_hin['tot_chars']}->{exp_hin['tot_chars']}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eng", default="corpus_sample/eng_sample.txt")
    ap.add_argument("--hin", default="corpus_sample/hin_sample.txt")
    ap.add_argument("--tokenizer", default="gpt2")
    args = ap.parse_args()

    enc = tiktoken.get_encoding(args.tokenizer)
    encode = enc.encode

    eng_lines = read_lines_nfc(args.eng)
    hin_lines = read_lines_nfc(args.hin)

    # 0. Baseline
    base_eng = compute_metrics(eng_lines, encode, lowercase=True, split_mode="space", agg_mode="macro", char_mode="codepoints")
    base_hin = compute_metrics(hin_lines, encode, lowercase=True, split_mode="space", agg_mode="macro", char_mode="codepoints")

    print("=== BASELINE PER-LINE BREAKDOWN ===")
    for lang, data in [("eng", base_eng), ("hin", base_hin)]:
        print(f"\n--- {lang} (tot_tokens={data['tot_tokens']}, tot_words={data['tot_words']}, tot_chars={data['tot_chars']}) ---")
        for d in data["lines"]:
            print(
                f"L{d['line_num']:02d}: tok={d['n_tokens']:2d} words={d['n_words']:2d} chars={d['n_chars']:2d} "
                f"fert={d['fert']:.6f} tpc={d['tpc']:.6f} | {d['orig']}"
            )

    # E1: Word-counting behavior: line.split(" ") vs line.split()
    e1_eng = compute_metrics(eng_lines, encode, lowercase=True, split_mode="whitespace", agg_mode="macro", char_mode="codepoints")
    e1_hin = compute_metrics(hin_lines, encode, lowercase=True, split_mode="whitespace", agg_mode="macro", char_mode="codepoints")
    print_comparison("E1: Word-counting behavior — line.split(' ') [Base] vs line.split() [Exp]", base_eng, base_hin, e1_eng, e1_hin)
    print("\nE1 Line-level word count differences (split(' ') vs split()):")
    for lang, b_data, e_data in [("eng", base_eng, e1_eng), ("hin", base_hin, e1_hin)]:
        for b_line, e_line in zip(b_data["lines"], e_data["lines"]):
            if b_line["n_words"] != e_line["n_words"]:
                print(
                    f"  [{lang} L{b_line['line_num']:02d}] split(' ')={b_line['n_words']} words {b_line['words_list']} "
                    f"(fert={b_line['fert']:.6f}) vs split()={e_line['n_words']} words {e_line['words_list']} (fert={e_line['fert']:.6f})"
                )

    # E2: Lowercasing: line.lower() vs original casing
    e2_eng = compute_metrics(eng_lines, encode, lowercase=False, split_mode="space", agg_mode="macro", char_mode="codepoints")
    e2_hin = compute_metrics(hin_lines, encode, lowercase=False, split_mode="space", agg_mode="macro", char_mode="codepoints")
    print_comparison("E2: Lowercasing — line.lower() [Base] vs Cased/Original [Exp]", base_eng, base_hin, e2_eng, e2_hin)
    print("\nE2 Line-level token differences (Cased vs Lowercased):")
    for lang, b_data, e_data in [("eng", base_eng, e2_eng), ("hin", base_hin, e2_hin)]:
        diffs_found = 0
        for b_line, e_line in zip(b_data["lines"], e_data["lines"]):
            if b_line["n_tokens"] != e_line["n_tokens"] or b_line["tokens"] != e_line["tokens"]:
                diffs_found += 1
                b_dec = [enc.decode_single_token_bytes(t).decode("utf-8", errors="replace") for t in b_line["tokens"]]
                e_dec = [enc.decode_single_token_bytes(t).decode("utf-8", errors="replace") for t in e_line["tokens"]]
                print(
                    f"  [{lang} L{b_line['line_num']:02d}] lower_tok={b_line['n_tokens']} vs cased_tok={e_line['n_tokens']} "
                    f"(delta={e_line['n_tokens'] - b_line['n_tokens']:+d})"
                )
                print(f"    lower tokens: {b_dec}")
                print(f"    cased tokens: {e_dec}")
        if diffs_found == 0:
            print(f"  [{lang}] 0/10 lines changed token IDs or token counts.")

    # E3: Macro-average vs Micro-average
    e3_eng = compute_metrics(eng_lines, encode, lowercase=True, split_mode="space", agg_mode="micro", char_mode="codepoints")
    e3_hin = compute_metrics(hin_lines, encode, lowercase=True, split_mode="space", agg_mode="micro", char_mode="codepoints")
    print_comparison("E3: Aggregation — Macro-average [Base] vs Micro-average (tot_tok/tot_words) [Exp]", base_eng, base_hin, e3_eng, e3_hin)

    # E4: Character-counting method
    # E4a: UTF-8 bytes
    e4_bytes_eng = compute_metrics(eng_lines, encode, lowercase=True, split_mode="space", agg_mode="macro", char_mode="utf8_bytes")
    e4_bytes_hin = compute_metrics(hin_lines, encode, lowercase=True, split_mode="space", agg_mode="macro", char_mode="utf8_bytes")
    print_comparison("E4a: Character denominator — Unicode codepoints len(line) [Base] vs UTF-8 bytes [Exp]", base_eng, base_hin, e4_bytes_eng, e4_bytes_hin)
    print(
        f"  E4a Micro-averaged tok/byte: eng={e4_bytes_eng['micro_tpc']:.6f} ({e4_bytes_eng['tot_tokens']}/{e4_bytes_eng['tot_chars']}), "
        f"hin={e4_bytes_hin['micro_tpc']:.6f} ({e4_bytes_hin['tot_tokens']}/{e4_bytes_hin['tot_chars']}), "
        f"ratio={e4_bytes_hin['micro_tpc'] / e4_bytes_eng['micro_tpc']:.6f} ({e4_bytes_hin['micro_tpc'] / e4_bytes_eng['micro_tpc']:.2f}x)"
    )

    # E4b: Extended Grapheme Clusters (\X via regex)
    e4_graph_eng = compute_metrics(eng_lines, encode, lowercase=True, split_mode="space", agg_mode="macro", char_mode="graphemes")
    e4_graph_hin = compute_metrics(hin_lines, encode, lowercase=True, split_mode="space", agg_mode="macro", char_mode="graphemes")
    print_comparison("E4b: Character denominator — Unicode codepoints len(line) [Base] vs Grapheme clusters (\\X) [Exp]", base_eng, base_hin, e4_graph_eng, e4_graph_hin)
    print(
        f"  E4b Micro-averaged tok/grapheme: eng={e4_graph_eng['micro_tpc']:.6f} ({e4_graph_eng['tot_tokens']}/{e4_graph_eng['tot_chars']}), "
        f"hin={e4_graph_hin['micro_tpc']:.6f} ({e4_graph_hin['tot_tokens']}/{e4_graph_hin['tot_chars']}), "
        f"ratio={e4_graph_hin['micro_tpc'] / e4_graph_eng['micro_tpc']:.6f} ({e4_graph_hin['micro_tpc'] / e4_graph_eng['micro_tpc']:.2f}x)"
    )

    # E4c: Non-space Unicode codepoints
    e4_nospace_eng = compute_metrics(eng_lines, encode, lowercase=True, split_mode="space", agg_mode="macro", char_mode="codepoints_no_space")
    e4_nospace_hin = compute_metrics(hin_lines, encode, lowercase=True, split_mode="space", agg_mode="macro", char_mode="codepoints_no_space")
    print_comparison("E4c: Character denominator — len(line) with spaces [Base] vs non-space codepoints [Exp]", base_eng, base_hin, e4_nospace_eng, e4_nospace_hin)

    print("\nE4 Hindi Line-by-Line Codepoints vs Graphemes vs UTF-8 Bytes:")
    for i, l in enumerate(hin_lines, 1):
        ll = l.lower()
        cp = len(ll)
        gr = len(regex.findall(r"\X", ll))
        by = len(ll.encode("utf-8"))
        toks = len(encode(ll))
        print(f"  hin L{i:02d}: tokens={toks:2d} | codepoints={cp:2d} | graphemes={gr:2d} | utf8_bytes={by:3d} | {l}")

    print("\nE4 Characters-per-word Check (Refuting REPORT_v0.md Finding #3):")
    print(
        f"  Using split(' '): eng chars/word = {base_eng['tot_chars']}/{base_eng['tot_words']} = {base_eng['tot_chars']/base_eng['tot_words']:.3f} | "
        f"hin chars/word = {base_hin['tot_chars']}/{base_hin['tot_words']} = {base_hin['tot_chars']/base_hin['tot_words']:.3f}"
    )
    print(
        f"  Using split():    eng chars/word = {e1_eng['tot_chars']}/{e1_eng['tot_words']} = {e1_eng['tot_chars']/e1_eng['tot_words']:.3f} | "
        f"hin chars/word = {e1_hin['tot_chars']}/{e1_hin['tot_words']} = {e1_hin['tot_chars']/e1_hin['tot_words']:.3f}"
    )

    # E5: Corpus comparability (parallel vs non-parallel subset analysis)
    matched_pairs = [
        (3, 3, "Buying book yesterday"),
        (4, 7, "Children playing cricket on ground"),
        (5, 6, "Train arrived on time"),
        (7, 10, "Keep books in cupboard"),
        (8, 4, "Visiting Mysuru next week"),
    ]
    print(f"\n{'=' * 78}")
    print("E5: Corpus Comparability — Semantic Alignment Analysis")
    print(f"{'=' * 78}")
    print("Matched semantic pairs between eng_sample.txt and hin_sample.txt (out of order!):")
    eng_matched_lines = []
    hin_matched_lines = []
    for eng_idx, hin_idx, desc in matched_pairs:
        el = eng_lines[eng_idx - 1]
        hl = hin_lines[hin_idx - 1]
        eng_matched_lines.append(el)
        hin_matched_lines.append(hl)
        et = len(encode(el.lower()))
        ht = len(encode(hl.lower()))
        print(f"  Pair ({desc}):")
        print(f"    eng L{eng_idx:02d} (tok={et:2d}): {el}")
        print(f"    hin L{hin_idx:02d} (tok={ht:2d}): {hl}")

    e5_eng = compute_metrics(eng_matched_lines, encode, lowercase=True, split_mode="space", agg_mode="macro", char_mode="codepoints")
    e5_hin = compute_metrics(hin_matched_lines, encode, lowercase=True, split_mode="space", agg_mode="macro", char_mode="codepoints")
    print_comparison("E5: Full 10-line non-parallel corpus [Base] vs 5-line semantically matched subset [Exp]", base_eng, base_hin, e5_eng, e5_hin)

    # E6: Harmless-looking code checks
    print(f"\n{'=' * 78}")
    print("E6: Harmless-Looking Code Checks (random.seed, NFC normalization, add_special_tokens)")
    print(f"{'=' * 78}")
    # E6a: random.seed
    random.seed(999999)
    e6_seed_eng = compute_metrics(eng_lines, encode, lowercase=True, split_mode="space", agg_mode="macro", char_mode="codepoints")
    e6_seed_hin = compute_metrics(hin_lines, encode, lowercase=True, split_mode="space", agg_mode="macro", char_mode="codepoints")
    print(f"E6a (random.seed removed/changed): eng_fert_diff={e6_seed_eng['fert'] - base_eng['fert']:.10f}, hin_fert_diff={e6_seed_hin['fert'] - base_hin['fert']:.10f}")

    # E6b: NFC normalization on sample corpora
    eng_raw = read_lines_raw(args.eng)
    hin_raw = read_lines_raw(args.hin)
    e6_raw_eng = compute_metrics(eng_raw, encode, lowercase=True, split_mode="space", agg_mode="macro", char_mode="codepoints")
    e6_raw_hin = compute_metrics(hin_raw, encode, lowercase=True, split_mode="space", agg_mode="macro", char_mode="codepoints")
    eng_nfc_diffs = sum(1 for a, b in zip(eng_raw, eng_lines) if a != b)
    hin_nfc_diffs = sum(1 for a, b in zip(hin_raw, hin_lines) if a != b)
    print(f"E6b (NFC vs Raw on corpus_sample): eng lines changed={eng_nfc_diffs}/10, hin lines changed={hin_nfc_diffs}/10")
    print(f"    eng_fert_diff={e6_raw_eng['fert'] - base_eng['fert']:.10f}, hin_fert_diff={e6_raw_hin['fert'] - base_hin['fert']:.10f}")

    # Demonstrate NFC vs NFD on Hindi nukta words (हफ़्ते, दफ़्तर, थोड़ा, पहुँची)
    hin_nfd = [unicodedata.normalize("NFD", l) for l in hin_raw]
    e6_nfd_hin = compute_metrics(hin_nfd, encode, lowercase=True, split_mode="space", agg_mode="macro", char_mode="codepoints")
    print(f"    (Counterfactual NFD vs NFC on hin_sample): NFD tot_chars={e6_nfd_hin['tot_chars']} vs NFC tot_chars={base_hin['tot_chars']}, "
          f"NFD tot_tokens={e6_nfd_hin['tot_tokens']} vs NFC tot_tokens={base_hin['tot_tokens']}")

    # E6c: Special token handling check
    has_transformers = importlib.util.find_spec("transformers") is not None
    print(
        f"E6c (add_special_tokens=False): spec='{args.tokenizer}' starts with 'hf:'={args.tokenizer.startswith('hf:')}; "
        f"transformers installed={has_transformers}; tiktoken enc.encode adds 0 special tokens by default."
    )

    # Bonus: Combined clean code fixes (whitespace split + cased + micro-avg) on full 10-line
    comb_eng = compute_metrics(eng_lines, encode, lowercase=False, split_mode="whitespace", agg_mode="micro", char_mode="codepoints")
    comb_hin = compute_metrics(hin_lines, encode, lowercase=False, split_mode="whitespace", agg_mode="micro", char_mode="codepoints")
    print_comparison("Combined Code Fixes (split() + cased + micro-avg) on 10-line sample", base_eng, base_hin, comb_eng, comb_hin)


if __name__ == "__main__":
    main()
