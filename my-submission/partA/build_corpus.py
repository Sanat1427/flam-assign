#!/usr/bin/env python3
r"""
build_corpus.py -- Part A1 Multilingual Parallel Corpus Construction & Validation

Downloads (or uses cached) official FLORES-200 benchmark archive from Meta AI / NLLB:
    https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
Extracts raw devtest files for 6 languages:
    - eng (eng_Latn): English (Latin script, Indo-European / Germanic)
    - hin (hin_Deva): Hindi (Devanagari script, Indo-European / Indo-Aryan)
    - tam (tam_Taml): Tamil (Tamil script, Dravidian)
    - tel (tel_Telu): Telugu (Telugu script, Dravidian)
    - kan (kan_Knda): Kannada (Kannada script, Dravidian)
    - mal (mal_Mlym): Malayalam (Malayalam script, Dravidian)

Saves unmodified raw files to:
    my-submission/partA/corpus/raw/<lang>.txt
Saves normalized/processed files (UTF-8, NFC normalized, consecutive whitespace
collapsed to single space, original casing preserved) to:
    my-submission/partA/corpus/processed/<lang>.txt

Also validates starter_kit/corpus_sample/ alongside the raw and processed FLORES-200 corpora.

Usage:
    .\.venv-x64\Scripts\python.exe my-submission/partA/build_corpus.py
"""

import argparse
import collections
import hashlib
import json
import shutil
import sys
import tarfile
import unicodedata
import urllib.request
from pathlib import Path

import regex

sys.stdout.reconfigure(encoding="utf-8")

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_EXPECTED_BYTES = 25585843

LANGUAGES = [
    ("eng", "eng_Latn", "English", "Indo-European (Germanic)", "Latin"),
    ("hin", "hin_Deva", "Hindi", "Indo-European (Indo-Aryan)", "Devanagari"),
    ("tam", "tam_Taml", "Tamil", "Dravidian", "Tamil"),
    ("tel", "tel_Telu", "Telugu", "Dravidian", "Telugu"),
    ("kan", "kan_Knda", "Kannada", "Dravidian", "Kannada"),
    ("mal", "mal_Mlym", "Malayalam", "Dravidian", "Malayalam"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_archive(archive_path: Path) -> dict:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path = archive_path.parent.parent / "flores200_dataset.tar.gz"
    if not archive_path.exists() and legacy_path.exists():
        shutil.move(str(legacy_path), str(archive_path))

    downloaded = False
    if not archive_path.exists():
        print(f"Downloading FLORES-200 archive from {FLORES_URL} ...")
        urllib.request.urlretrieve(FLORES_URL, str(archive_path))
        downloaded = True

    size_bytes = archive_path.stat().st_size
    digest = sha256_file(archive_path)
    return {
        "url": FLORES_URL,
        "archive_path": str(archive_path),
        "downloaded_now": downloaded,
        "size_bytes": size_bytes,
        "sha256": digest,
    }


def inspect_corpus_file(path: Path) -> dict:
    raw_bytes = path.read_bytes()
    file_byte_size = len(raw_bytes)

    # Check UTF-8 validity
    try:
        text = raw_bytes.decode("utf-8", errors="strict")
        utf8_valid = True
    except UnicodeDecodeError:
        text = raw_bytes.decode("utf-8", errors="replace")
        utf8_valid = False

    raw_lines = text.splitlines()
    n_lines = len(raw_lines)
    empty_lines = sum(1 for l in raw_lines if not l.strip())

    # Duplicate check
    line_counts = collections.Counter(raw_lines)
    duplicate_excess = sum(c - 1 for c in line_counts.values() if c > 1)
    duplicate_examples = []
    seen = {}
    for idx, l in enumerate(raw_lines, 1):
        if l in seen:
            duplicate_examples.append({"line": l, "first_line_num": seen[l], "dup_line_num": idx})
        else:
            seen[l] = idx

    # Encoding / character hygiene checks
    replacement_char_lines = sum(1 for l in raw_lines if "\ufffd" in l)
    control_char_lines = sum(
        1
        for l in raw_lines
        if any(unicodedata.category(c).startswith("C") and c not in ("\t",) for c in l)
    )
    non_nfc_lines = 0
    nfc_codepoint_delta = 0
    non_nfc_examples = []
    for idx, l in enumerate(raw_lines, 1):
        nfc_l = unicodedata.normalize("NFC", l)
        if nfc_l != l:
            non_nfc_lines += 1
            nfc_codepoint_delta += len(nfc_l) - len(l)
            if len(non_nfc_examples) < 3:
                bad_chars = []
                for c in l:
                    if unicodedata.normalize("NFC", c) != c:
                        bad_chars.append(f"U+{ord(c):04X} ({unicodedata.name(c, 'UNKNOWN')})")
                non_nfc_examples.append(
                    {
                        "line_num": idx,
                        "cp_delta": len(nfc_l) - len(l),
                        "singleton_chars": sorted(set(bad_chars)),
                    }
                )

    multi_space_lines = 0
    multi_space_examples = []
    leading_trailing_space_lines = 0
    escaped_quote_lines = 0
    for idx, l in enumerate(raw_lines, 1):
        if l != l.strip():
            leading_trailing_space_lines += 1
        if "  " in l or "\t" in l:
            multi_space_lines += 1
            if len(multi_space_examples) < 3:
                multi_space_examples.append(idx)
        if '""' in l:
            escaped_quote_lines += 1

    # Counts across lines
    words_split_space = sum(len(l.split(" ")) for l in raw_lines if l.strip())
    words_split_ws = sum(len(l.split()) for l in raw_lines if l.strip())
    codepoints_with_spaces = sum(len(l) for l in raw_lines)
    codepoints_no_spaces = sum(len(l.replace(" ", "")) for l in raw_lines)
    utf8_bytes_lines = sum(len(l.encode("utf-8")) for l in raw_lines)
    graphemes_with_spaces = sum(len(regex.findall(r"\X", l)) for l in raw_lines)
    graphemes_no_spaces = sum(len(regex.findall(r"\X", l.replace(" ", ""))) for l in raw_lines)

    return {
        "path": str(path),
        "file_byte_size": file_byte_size,
        "utf8_valid": utf8_valid,
        "n_lines": n_lines,
        "empty_lines": empty_lines,
        "duplicate_lines": duplicate_excess,
        "duplicate_examples": duplicate_examples,
        "replacement_char_lines": replacement_char_lines,
        "control_char_lines": control_char_lines,
        "non_nfc_lines": non_nfc_lines,
        "nfc_codepoint_delta": nfc_codepoint_delta,
        "non_nfc_examples": non_nfc_examples,
        "leading_trailing_space_lines": leading_trailing_space_lines,
        "multi_space_lines": multi_space_lines,
        "multi_space_example_lines": multi_space_examples,
        "escaped_quote_lines": escaped_quote_lines,
        "words_split_space": words_split_space,
        "words_split_ws": words_split_ws,
        "phantom_empty_words": words_split_space - words_split_ws,
        "codepoints": codepoints_with_spaces,
        "codepoints_no_space": codepoints_no_spaces,
        "graphemes": graphemes_with_spaces,
        "graphemes_no_space": graphemes_no_spaces,
        "utf8_bytes_lines": utf8_bytes_lines,
        "chars_per_word_ws": codepoints_with_spaces / words_split_ws if words_split_ws else 0.0,
        "graphemes_per_word_ws": graphemes_with_spaces / words_split_ws if words_split_ws else 0.0,
        "bytes_per_word_ws": utf8_bytes_lines / words_split_ws if words_split_ws else 0.0,
        "bytes_per_codepoint": utf8_bytes_lines / codepoints_with_spaces if codepoints_with_spaces else 0.0,
    }


def preprocess_line(line: str) -> str:
    """
    Deterministic, lossless-for-tokenization corpus normalization:
    1. Strip leading/trailing whitespace.
    2. Normalize Unicode to Canonical Composition (NFC).
    3. Collapse runs of consecutive internal whitespace to a single ASCII space ' '.
    4. Preserve original casing (do NOT lowercase).
    """
    s = line.strip()
    s = unicodedata.normalize("NFC", s)
    s = " ".join(s.split())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="my-submission/partA/corpus")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    raw_dir = base_dir / "raw"
    proc_dir = base_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    archive_path = base_dir / "flores200_dataset.tar.gz"
    archive_info = ensure_archive(archive_path)

    print("=== 1. SOURCE ARCHIVE VERIFICATION ===")
    print(f"URL:          {archive_info['url']}")
    print(f"Archive Path: {archive_info['archive_path']}")
    print(f"Size (bytes): {archive_info['size_bytes']} (expected {FLORES_EXPECTED_BYTES})")
    print(f"SHA-256:      {archive_info['sha256']}")
    print("License:      CC-BY-SA-4.0 (Creative Commons Attribution-ShareAlike 4.0 International)")

    # Extract raw devtest files and metadata
    with tarfile.open(archive_path, "r:gz") as tf:
        meta_bytes = tf.extractfile("./flores200_dataset/metadata_devtest.tsv").read()
        (raw_dir / "metadata_devtest.tsv").write_bytes(meta_bytes)
        readme_bytes = tf.extractfile("./flores200_dataset/README").read()
        (raw_dir / "README").write_bytes(readme_bytes)

        for short_code, flores_code, lang_name, family, script_name in LANGUAGES:
            member_name = f"./flores200_dataset/devtest/{flores_code}.devtest"
            raw_content = tf.extractfile(member_name).read()
            raw_file_path = raw_dir / f"{short_code}.txt"
            raw_file_path.write_bytes(raw_content)

            # Build processed version
            raw_text = raw_content.decode("utf-8", errors="strict")
            raw_lines = raw_text.splitlines()
            proc_lines = [preprocess_line(l) for l in raw_lines if l.strip()]
            proc_file_path = proc_dir / f"{short_code}.txt"
            proc_file_path.write_text("\n".join(proc_lines) + "\n", encoding="utf-8", newline="\n")

    # Parse metadata_devtest.tsv
    meta_lines = (raw_dir / "metadata_devtest.tsv").read_text(encoding="utf-8").splitlines()
    meta_rows = [r.split("\t") for r in meta_lines[1:] if r.strip()]
    unique_urls = set(r[0] for r in meta_rows)
    domain_counts = collections.Counter(r[1] for r in meta_rows)
    topic_counts = collections.Counter(r[2].lower() for r in meta_rows)

    print("\n=== 2. FLORES-200 DEVTEST METADATA & PROVENANCE ===")
    print(f"Split:                  devtest (multi-way parallel across all languages)")
    print(f"Total sentences/lang:   {len(meta_rows)}")
    print(f"Unique source articles: {len(unique_urls)} English Wikimedia articles")
    print(f"Domains:                {dict(domain_counts)}")
    print(f"Top 8 topics (norm):    {topic_counts.most_common(8)}")

    # Validate original starter_kit/corpus_sample/
    print("\n=== 3. STARTER-KIT SAMPLE CORPUS VALIDATION (corpus_sample/) ===")
    sample_stats = {}
    for lang_code, sample_rel in [("eng", "corpus_sample/eng_sample.txt"), ("hin", "corpus_sample/hin_sample.txt")]:
        st = inspect_corpus_file(Path(sample_rel))
        sample_stats[lang_code] = st
        print(
            f"  {lang_code} ({sample_rel}): lines={st['n_lines']}, empty={st['empty_lines']}, dups={st['duplicate_lines']}, "
            f"multi_space_lines={st['multi_space_lines']} (lines {st['multi_space_example_lines']}), "
            f"non_nfc_lines={st['non_nfc_lines']}, words_split(' ')={st['words_split_space']}, words_split()={st['words_split_ws']}, "
            f"codepoints={st['codepoints']}, graphemes={st['graphemes']}, utf8_bytes={st['utf8_bytes_lines']} (file={st['file_byte_size']} B)"
        )

    # Validate Raw FLORES-200 files
    print("\n=== 4. RAW FLORES-200 DEVTEST VALIDATION (my-submission/partA/corpus/raw/) ===")
    print(
        f"{'Lang':<6} | {'FLORES Code':<10} | {'Lines':>5} | {'Empty':>5} | {'Dups':>4} | {'NonNFC':>6} | {'MultiSpc':>8} | "
        f"{'Words(sp)':>9} | {'Words(ws)':>9} | {'Codepoints':>10} | {'Graphemes':>9} | {'UTF8 Bytes':>10}"
    )
    print("-" * 118)
    raw_stats = {}
    for short_code, flores_code, lang_name, family, script_name in LANGUAGES:
        st = inspect_corpus_file(raw_dir / f"{short_code}.txt")
        raw_stats[short_code] = st
        print(
            f"{short_code:<6} | {flores_code:<10} | {st['n_lines']:>5} | {st['empty_lines']:>5} | {st['duplicate_lines']:>4} | "
            f"{st['non_nfc_lines']:>6} | {st['multi_space_lines']:>8} | {st['words_split_space']:>9} | {st['words_split_ws']:>9} | "
            f"{st['codepoints']:>10} | {st['graphemes']:>9} | {st['utf8_bytes_lines']:>10}"
        )

    print("\nDetailed Raw Anomalies Detected Before Preprocessing:")
    for short_code, flores_code, lang_name, family, script_name in LANGUAGES:
        st = raw_stats[short_code]
        print(
            f"  - {short_code} ({lang_name}): phantom_words_from_double_spaces=+{st['phantom_empty_words']} "
            f"(across {st['multi_space_lines']} lines, e.g. L{st['multi_space_example_lines']}); "
            f"non_nfc_lines={st['non_nfc_lines']} (nfc_cp_delta={st['nfc_codepoint_delta']:+d}) {st['non_nfc_examples'][:2]}; "
            f"csv_escaped_quote_lines={st['escaped_quote_lines']}"
        )

    # Validate Processed FLORES-200 files
    print("\n=== 5. PROCESSED FLORES-200 DEVTEST VALIDATION (my-submission/partA/corpus/processed/) ===")
    print(
        f"{'Lang':<6} | {'Script':<10} | {'Lines':>5} | {'Empty':>5} | {'Dups':>4} | {'NonNFC':>6} | {'MultiSpc':>8} | "
        f"{'Words':>8} | {'Codepoints':>10} | {'Graphemes':>9} | {'UTF8 Bytes':>10} | {'CP/Word':>7} | {'Grapheme/W':>10}"
    )
    print("-" * 126)
    proc_stats = {}
    for short_code, flores_code, lang_name, family, script_name in LANGUAGES:
        st = inspect_corpus_file(proc_dir / f"{short_code}.txt")
        proc_stats[short_code] = st
        print(
            f"{short_code:<6} | {script_name:<10} | {st['n_lines']:>5} | {st['empty_lines']:>5} | {st['duplicate_lines']:>4} | "
            f"{st['non_nfc_lines']:>6} | {st['multi_space_lines']:>8} | {st['words_split_ws']:>8} | "
            f"{st['codepoints']:>10} | {st['graphemes']:>9} | {st['utf8_bytes_lines']:>10} | "
            f"{st['chars_per_word_ws']:>7.2f} | {st['graphemes_per_word_ws']:>10.2f}"
        )

    # Show sample parallel alignment (Line 1, 100, 500, 1012 across all 6 languages)
    print("\n=== 6. PARALLEL ALIGNMENT SPOT-CHECK (Processed Corpus) ===")
    proc_lines_by_lang = {
        short_code: (proc_dir / f"{short_code}.txt").read_text(encoding="utf-8").splitlines()
        for short_code, *_ in LANGUAGES
    }
    for sample_idx in [1, 100, 500, 1012]:
        row_meta = meta_rows[sample_idx - 1]
        print(f"\nSentence #{sample_idx:04d} [Domain: {row_meta[1]} | Topic: {row_meta[2]} | URL: {row_meta[0]}]:")
        for short_code, flores_code, lang_name, *_ in LANGUAGES:
            print(f"  [{short_code}] {proc_lines_by_lang[short_code][sample_idx - 1]}")

    # Save structured summary JSON
    summary = {
        "archive": archive_info,
        "metadata": {
            "split": "devtest",
            "n_sentences": len(meta_rows),
            "n_unique_documents": len(unique_urls),
            "domains": dict(domain_counts),
        },
        "starter_kit_sample": sample_stats,
        "raw_flores200": raw_stats,
        "processed_flores200": proc_stats,
    }
    summary_path = base_dir / "corpus_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved validation summary JSON to: {summary_path}")


if __name__ == "__main__":
    main()
