#!/usr/bin/env python3
"""
Part B — Capacity Reconciliation Script
AI Team Intern Assignment — The Audit

Reads supplied specifications from `bench/model_spec.md` and `bench/bench_log.csv`
(without modifying any starter-kit files) and computes:
1. Extracted model, hardware, precision, KV-cache, and workload parameters + missing parameters.
2. Per-token, per-sequence, and per-batch KV-cache memory footprints (in bytes, decimal MB/GB,
   and binary KiB/MiB/GiB) for both GQA (n_kv=8) and MHA pitfall (n_q=24).
3. Available KV-cache memory budget and reconciliation with `kv_cache_util` and `preempted_seqs`
   across all 13 rows of `bench_log.csv`.
4. Throughput anomaly decomposition: Total-Token Throughput (`reported_tok_s`) vs.
   Generated-Output-Token Throughput (`num_requests * gen_len / wall_clock_s`) vs.
   Instantaneous Decode Rate (`batch_size / (itl_ms_p50 / 1000)`).
5. Decode memory-bandwidth utilization analysis (`(weights + KV_read) / itl_ms_p50` vs. 300 GB/s peak).
6. Sensitivity analysis across sequence lengths, non-KV overhead assumptions, unit conventions,
   KV-cache quantization (fp16 vs. fp8/int8), and Part A3 multilingual token-count ratios.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH_CSV = ROOT / "bench" / "bench_log.csv"
MODEL_SPEC = ROOT / "bench" / "model_spec.md"
OUT_DIR = Path(__file__).resolve().parent
OUT_JSON = OUT_DIR / "partB_reconciliation.json"
OUT_TXT = OUT_DIR / "partB_calculation_output.txt"

# ---------------------------------------------------------------------------
# 1. Exact Supplied Parameters from `bench/model_spec.md`
# ---------------------------------------------------------------------------
SUPPLIED_SPEC = {
    "model_name": "FLM-4B-Instruct (dense)",
    "parameters_b": 4.2,                  # 4.2 B parameters (`model_spec.md:L7`)
    "layers": 28,                         # L = 28 (`model_spec.md:L8`)
    "d_model": 3072,                      # d_model = 3072 (`model_spec.md:L9`)
    "n_heads_q": 24,                      # attention heads (Q) = 24 (`model_spec.md:L10`)
    "n_heads_kv": 8,                      # KV heads (GQA) = 8 (`model_spec.md:L11`)
    "head_dim": 128,                      # head_dim = 128 (`model_spec.md:L12`)
    "vocab_size": 128_000,                # vocab = 128k (`model_spec.md:L13`)
    "weights_precision": "fp16",          # 2 bytes/param (`model_spec.md:L14`)
    "weights_bytes_per_param": 2,
    "kv_cache_precision": "fp16",         # 2 bytes/element (`model_spec.md:L15`)
    "kv_bytes_per_elem": 2,
    "gpu_model": "1x NVIDIA L4 (24 GB)",  # (`model_spec.md:L21`)
    "gpu_memory_gb": 24.0,                # (`model_spec.md:L21`)
    "peak_mem_bw_gb_s": 300.0,            # 300 GB/s (`model_spec.md:L22`)
    "peak_fp16_tflops": 121.0,            # ~121 TFLOPS (`model_spec.md:L23`)
    "max_model_len": 4096,                # 4096 tokens (`model_spec.md:L24`)
    "gpu_memory_utilization": 0.92,       # 0.92 (`model_spec.md:L25`)
    "non_kv_overhead_gb": 1.6,            # assume ~1.6 GB (`model_spec.md:L26`)
}

MISSING_PARAMETERS = [
    "Serving framework/engine identifier and version (e.g., vLLM, TensorRT-LLM, SGLang) and PagedAttention block size (e.g., 16 or 32 tokens/block) and block-allocator metadata overhead.",
    "Scheduler preemption policy (`recompute` via KV-block eviction + re-prefill vs. `swap` to CPU RAM over PCIe) and admission/waiting-queue policy.",
    "Whether chunked prefill or prefix caching (`enable_prefix_caching`) is active: not specified in `model_spec.md`; observed `kv_cache_util` values are consistent with the simple linear per-request allocation model (B * S).",
    "Exact MLP intermediate size (`d_ff`), activation function (e.g., SwiGLU), and whether input/output embeddings are tied (only rounded `4.2 B` total parameters is given).",
    "Whether `GB` in `model_spec.md` (`24 GB`, `1.6 GB`) is specified in decimal gigabytes (10^9 B) or binary gibibytes (2^30 B); both conventions are tracked and compared against `kv_cache_util`.",
    "Hardware profiler telemetry (e.g., Nsight Compute / DCGM DRAM read bytes/s) and per-request TTFT/ITL distributions (mean, p95, p99) and total preemption event counts.",
]


def compute_kv_fundamentals(spec: dict) -> dict:
    # GQA KV bytes per token: 2 (K,V) * L * n_kv * head_dim * bytes_per_elem
    kv_bytes_per_token_gqa = (
        2
        * spec["layers"]
        * spec["n_heads_kv"]
        * spec["head_dim"]
        * spec["kv_bytes_per_elem"]
    )
    # MHA pitfall (if using n_heads_q=24 instead of n_heads_kv=8)
    kv_bytes_per_token_mha = (
        2
        * spec["layers"]
        * spec["n_heads_q"]
        * spec["head_dim"]
        * spec["kv_bytes_per_elem"]
    )

    # Decimal GB convention (1 GB = 10^9 B)
    usable_gpu_gb = spec["gpu_memory_gb"] * spec["gpu_memory_utilization"]  # 24 * 0.92 = 22.08 GB
    weights_gb = spec["parameters_b"] * spec["weights_bytes_per_param"]      # 4.2 * 2 = 8.40 GB
    overhead_gb = spec["non_kv_overhead_gb"]                                 # 1.60 GB
    kv_pool_gb = usable_gpu_gb - weights_gb - overhead_gb                    # 12.08 GB
    kv_pool_bytes_decimal = int(round(kv_pool_gb * 1_000_000_000))           # 12,080,000,000 B

    # Binary GiB convention if 12.08 is treated as GiB (12.08 * 2^30 B)
    kv_pool_bytes_binary_gib = int(round(kv_pool_gb * (1024 ** 3)))          # 12,970,801,234 B

    # Mixed hardware convention: L4 has 24,159,191,040 B (22.5 GiB / 24.16 GB) or 24e9 B,
    # while 12.08e9 B = 11.25037 GiB (notice 11.25 GiB / 0.4375 GiB per 4096-tok seq = 25.7143 seqs!)
    kv_pool_gib_from_decimal = kv_pool_bytes_decimal / (1024 ** 3)           # 11.25037 GiB

    max_tokens_decimal_pool = kv_pool_bytes_decimal // kv_bytes_per_token_gqa
    max_tokens_binary_pool = kv_pool_bytes_binary_gib // kv_bytes_per_token_gqa

    # Per-sequence KV cache at short (512+256=768) and long (3584+512=4096) lengths
    seq_768_bytes = 768 * kv_bytes_per_token_gqa
    seq_4096_bytes = 4096 * kv_bytes_per_token_gqa

    max_seqs_768_dec = kv_pool_bytes_decimal // seq_768_bytes
    max_seqs_4096_dec = kv_pool_bytes_decimal // seq_4096_bytes
    max_seqs_4096_bin = kv_pool_bytes_binary_pool = kv_pool_bytes_binary_gib // seq_4096_bytes

    return {
        "kv_bytes_per_token_gqa": kv_bytes_per_token_gqa,
        "kv_kib_per_token_gqa": kv_bytes_per_token_gqa / 1024.0,
        "kv_mb_per_token_gqa_decimal": kv_bytes_per_token_gqa / 1e6,
        "kv_mib_per_token_gqa_binary": kv_bytes_per_token_gqa / (1024.0 ** 2),
        "kv_bytes_per_token_mha_pitfall": kv_bytes_per_token_mha,
        "kv_kib_per_token_mha_pitfall": kv_bytes_per_token_mha / 1024.0,
        "gqa_compression_factor": spec["n_heads_q"] // spec["n_heads_kv"],
        "usable_gpu_gb": round(usable_gpu_gb, 4),
        "weights_gb": round(weights_gb, 4),
        "non_kv_overhead_gb": round(overhead_gb, 4),
        "kv_pool_gb_decimal": round(kv_pool_gb, 4),
        "kv_pool_bytes_decimal": kv_pool_bytes_decimal,
        "kv_pool_gib_equivalent": round(kv_pool_gib_from_decimal, 6),
        "kv_pool_bytes_if_12_08_gib": kv_pool_bytes_binary_gib,
        "max_tokens_in_pool_decimal": max_tokens_decimal_pool,
        "max_tokens_in_pool_if_12_08_gib": max_tokens_binary_pool,
        "seq_768_kv_bytes": seq_768_bytes,
        "seq_768_kv_mb_decimal": round(seq_768_bytes / 1e6, 6),
        "seq_768_kv_mib_binary": round(seq_768_bytes / (1024 ** 2), 6),
        "seq_4096_kv_bytes": seq_4096_bytes,
        "seq_4096_kv_mb_decimal": round(seq_4096_bytes / 1e6, 6),
        "seq_4096_kv_mib_binary": round(seq_4096_bytes / (1024 ** 2), 6),
        "seq_4096_kv_gib_binary": round(seq_4096_bytes / (1024 ** 3), 6),
        "max_concurrent_seqs_768_decimal_pool": max_seqs_768_dec,
        "max_concurrent_seqs_4096_decimal_pool_exact": kv_pool_bytes_decimal / seq_4096_bytes,
        "max_concurrent_seqs_4096_decimal_pool_floor": max_seqs_4096_dec,
        "util_at_24_seqs_4096_decimal": (24 * seq_4096_bytes) / kv_pool_bytes_decimal,
        "util_at_25_seqs_4096_decimal": (25 * seq_4096_bytes) / kv_pool_bytes_decimal,
        "util_at_26_seqs_4096_decimal": (26 * seq_4096_bytes) / kv_pool_bytes_decimal,
        "max_concurrent_seqs_4096_if_12_08_gib_exact": kv_pool_bytes_binary_gib / seq_4096_bytes,
        "max_concurrent_seqs_4096_if_12_08_gib_floor": max_seqs_4096_bin,
    }


def analyze_bench_log(kv_info: dict) -> list[dict]:
    kv_b_tok = kv_info["kv_bytes_per_token_gqa"]
    pool_b_dec = kv_info["kv_pool_bytes_decimal"]
    pool_b_bin = kv_info["kv_pool_bytes_if_12_08_gib"]
    weights_gb = kv_info["weights_gb"]

    results = []
    with open(BENCH_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, r in enumerate(reader, 1):
            b = int(r["batch_size"])
            n = int(r["num_requests"])
            p = int(r["prompt_len"])
            g = int(r["gen_len"])
            wall_s = float(r["wall_clock_s"])
            rep_tok_s = float(r["reported_tok_s"])
            ttft_ms = float(r["ttft_ms_p50"])
            itl_ms = float(r["itl_ms_p50"])
            e2e_ms = float(r["e2e_ms_p95"])
            preempted = int(r["preempted_seqs"])
            logged_kvu = float(r["kv_cache_util"])

            seq_len = p + g
            tot_prompt_tokens = n * p
            tot_gen_tokens = n * g
            tot_all_tokens = n * seq_len

            # Unthrottled KV demand if all `n` requests reside in KV cache at full length `p + g`
            kv_demand_bytes = tot_all_tokens * kv_b_tok
            kv_demand_gb_dec = kv_demand_bytes / 1e9
            kv_demand_gib_bin = kv_demand_bytes / (1024 ** 3)
            kv_demand_ratio_dec = kv_demand_bytes / pool_b_dec
            kv_demand_ratio_bin = kv_demand_bytes / pool_b_bin

            # Active sequences capped by physical KV pool capacity (25 for 4096-token seqs)
            max_fit_seqs = pool_b_dec // (seq_len * kv_b_tok)
            active_seqs_est = min(n, max_fit_seqs)
            expected_preempted_seqs = max(0, n - max_fit_seqs)
            capped_kv_util_dec = (active_seqs_est * seq_len * kv_b_tok) / pool_b_dec

            # Throughput metrics
            calc_total_tok_s = tot_all_tokens / wall_s
            calc_gen_tok_s = tot_gen_tokens / wall_s
            calc_prompt_tok_s_over_wall = tot_prompt_tokens / wall_s
            total_to_gen_multiplier = seq_len / g  # (p + g) / g

            # Instantaneous decode throughput (if all `b` sequences decoded in parallel without preemption)
            inst_decode_tok_s_nominal = b / (itl_ms / 1000.0)
            # Instantaneous decode throughput of active (non-evicted) batch
            inst_decode_tok_s_active = active_seqs_est / (itl_ms / 1000.0)

            # Time breakdown & effective HBM bandwidth during decode
            # Average tokens per sequence during decode = p + g/2
            avg_seq_during_decode = p + (g / 2.0)
            avg_kv_gb_active = (active_seqs_est * avg_seq_during_decode * kv_b_tok) / 1e9
            peak_kv_gb_active = (active_seqs_est * seq_len * kv_b_tok) / 1e9
            eff_hbm_bw_avg_gb_s = (weights_gb + avg_kv_gb_active) / (itl_ms / 1000.0)
            eff_hbm_bw_peak_gb_s = (weights_gb + peak_kv_gb_active) / (itl_ms / 1000.0)

            # Single-wave median latency estimate: ttft_ms_p50 + g * itl_ms_p50
            single_wave_p50_ms = ttft_ms + g * itl_ms

            results.append({
                "row": idx,
                "workload": "short_512_256" if p == 512 else "long_3584_512",
                "batch_size": b,
                "num_requests": n,
                "prompt_len": p,
                "gen_len": g,
                "seq_len": seq_len,
                "wall_clock_s": wall_s,
                "reported_tok_s": rep_tok_s,
                "calc_total_tok_s": round(calc_total_tok_s, 2),
                "calc_gen_tok_s": round(calc_gen_tok_s, 2),
                "calc_prompt_tok_s_over_wall": round(calc_prompt_tok_s_over_wall, 2),
                "total_to_gen_ratio": round(total_to_gen_multiplier, 2),
                "inst_decode_tok_s_nominal": round(inst_decode_tok_s_nominal, 2),
                "inst_decode_tok_s_active": round(inst_decode_tok_s_active, 2),
                "ttft_ms_p50": ttft_ms,
                "itl_ms_p50": itl_ms,
                "single_wave_p50_ms": round(single_wave_p50_ms, 1),
                "e2e_ms_p95": e2e_ms,
                "tot_prompt_tokens": tot_prompt_tokens,
                "tot_gen_tokens": tot_gen_tokens,
                "tot_all_tokens": tot_all_tokens,
                "kv_demand_bytes": kv_demand_bytes,
                "kv_demand_gb_decimal": round(kv_demand_gb_dec, 4),
                "kv_demand_gib_binary": round(kv_demand_gib_bin, 4),
                "kv_demand_ratio_decimal": round(kv_demand_ratio_dec, 4),
                "kv_demand_ratio_if_12_08_gib": round(kv_demand_ratio_bin, 4),
                "capped_kv_util_decimal": round(capped_kv_util_dec, 4),
                "logged_kv_cache_util": logged_kvu,
                "kv_util_matches_rounded_decimal": round(capped_kv_util_dec, 2) == logged_kvu,
                "max_fit_seqs_in_pool": max_fit_seqs,
                "active_seqs_est": active_seqs_est,
                "expected_preempted_seqs": expected_preempted_seqs,
                "logged_preempted_seqs": preempted,
                "eff_hbm_bw_avg_gb_s": round(eff_hbm_bw_avg_gb_s, 2),
                "eff_hbm_bw_peak_gb_s": round(eff_hbm_bw_peak_gb_s, 2),
                "eff_hbm_bw_pct_of_300gbs": round((eff_hbm_bw_avg_gb_s / 300.0) * 100.0, 2),
            })
    return results


def run_sensitivity_analyses(kv_info: dict) -> dict:
    kv_b_fp16 = kv_info["kv_bytes_per_token_gqa"]        # 114,688 B/tok
    kv_b_fp8 = kv_b_fp16 // 2                            # 57,344 B/tok (fp8 / int8 KV cache)
    pool_b_dec = kv_info["kv_pool_bytes_decimal"]        # 12,080,000,000 B

    # 1. Sequence-length & KV-precision sensitivity
    seq_lengths = [512, 768, 1024, 1536, 2048, 3072, 4096]
    seq_sensitivity = []
    for s in seq_lengths:
        seq_b_fp16 = s * kv_b_fp16
        seq_b_fp8 = s * kv_b_fp8
        seq_sensitivity.append({
            "seq_len": s,
            "fp16_mb_per_seq_dec": round(seq_b_fp16 / 1e6, 2),
            "fp16_mib_per_seq_bin": round(seq_b_fp16 / (1024 ** 2), 2),
            "fp16_max_seqs_12_08gb": pool_b_dec // seq_b_fp16,
            "fp8_mb_per_seq_dec": round(seq_b_fp8 / 1e6, 2),
            "fp8_max_seqs_12_08gb": pool_b_dec // seq_b_fp8,
        })

    # 2. Non-KV runtime overhead sensitivity (model_spec.md says "assume ~1.6 GB")
    overheads_gb = [1.0, 1.2, 1.6, 2.0, 2.4, 3.0]
    overhead_sensitivity = []
    usable_gb = 24.0 * 0.92  # 22.08 GB
    weights_gb = 8.4         # 8.40 GB
    seq_4096_b = 4096 * kv_b_fp16
    for ov in overheads_gb:
        pool_gb = usable_gb - weights_gb - ov
        pool_b = int(round(pool_gb * 1e9))
        overhead_sensitivity.append({
            "non_kv_overhead_gb": ov,
            "kv_pool_gb": round(pool_gb, 2),
            "max_tokens_fp16": pool_b // kv_b_fp16,
            "max_seqs_at_4096_fp16": pool_b // seq_4096_b,
            "max_seqs_at_768_fp16": pool_b // (768 * kv_b_fp16),
        })

    # 3. Multilingual token-inflation sensitivity (connecting Part A3 sentence-parity ratios
    #    to effective concurrent requests when an English 768-token turn is translated)
    #    Using exact Part A3 token-count-per-aligned-sentence ratios from `a3_tokenizer_comparison.json`
    a3_ratios = {
        "eng (Baseline)": {"o200k_ratio": 1.0000, "cl100k_ratio": 1.0000},
        "hin (Hindi)": {"o200k_ratio": 42241 / 26873, "cl100k_ratio": 129567 / 27182},
        "tel (Telugu)": {"o200k_ratio": 51610 / 26873, "cl100k_ratio": 225022 / 27182},
        "mal (Malayalam)": {"o200k_ratio": 52515 / 26873, "cl100k_ratio": 242859 / 27182},
        "kan (Kannada)": {"o200k_ratio": 52534 / 26873, "cl100k_ratio": 240364 / 27182},
        "tam (Tamil)": {"o200k_ratio": 53114 / 26873, "cl100k_ratio": 207687 / 27182},
    }
    multilingual_sensitivity = []
    base_eng_seq = 768  # 512 prompt + 256 gen
    for lang, rdict in a3_ratios.items():
        s_o200k = int(round(base_eng_seq * rdict["o200k_ratio"]))
        s_cl100k = int(round(base_eng_seq * rdict["cl100k_ratio"]))
        multilingual_sensitivity.append({
            "language": lang,
            "o200k_sentence_token_ratio": round(rdict["o200k_ratio"], 4),
            "equiv_seq_len_o200k": s_o200k,
            "fits_max_model_len_4096_o200k": s_o200k <= 4096,
            "max_concurrent_seqs_o200k": pool_b_dec // (s_o200k * kv_b_fp16),
            "cl100k_sentence_token_ratio": round(rdict["cl100k_ratio"], 4),
            "equiv_seq_len_cl100k": s_cl100k,
            "fits_max_model_len_4096_cl100k": s_cl100k <= 4096,
            "max_concurrent_seqs_cl100k": (
                pool_b_dec // (s_cl100k * kv_b_fp16) if s_cl100k <= 4096 else 0
            ),
        })

    return {
        "sequence_length_and_precision": seq_sensitivity,
        "non_kv_overhead_gb": overhead_sensitivity,
        "multilingual_token_ratio_impact_from_base_768": multilingual_sensitivity,
    }


def format_report_text(kv_info: dict, rows: list[dict], sens: dict) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append("PART B -- CAPACITY & THROUGHPUT RECONCILIATION (VERIFIED OUTPUT)")
    lines.append("=" * 100)
    lines.append("")
    lines.append("1. KV-CACHE FUNDAMENTALS (FROM bench/model_spec.md)")
    lines.append("-" * 100)
    lines.append(
        f"  Formula (GQA): 2 (K,V) * L ({SUPPLIED_SPEC['layers']}) * n_kv ({SUPPLIED_SPEC['n_heads_kv']}) "
        f"* head_dim ({SUPPLIED_SPEC['head_dim']}) * fp16 ({SUPPLIED_SPEC['kv_bytes_per_elem']} B)"
    )
    lines.append(
        f"  KV Bytes / Token (GQA, n_kv=8) : {kv_info['kv_bytes_per_token_gqa']:,} B "
        f"= {kv_info['kv_kib_per_token_gqa']:.1f} KiB = {kv_info['kv_mb_per_token_gqa_decimal']:.6f} MB (dec) "
        f"= {kv_info['kv_mib_per_token_gqa_binary']:.6f} MiB (bin)"
    )
    lines.append(
        f"  MHA Pitfall (if n_q=24 used)   : {kv_info['kv_bytes_per_token_mha_pitfall']:,} B "
        f"= {kv_info['kv_kib_per_token_mha_pitfall']:.1f} KiB (3x overestimate)"
    )
    lines.append(
        f"  GPU Memory Budget (Decimal GB) : 24.0 GB * 0.92 = {kv_info['usable_gpu_gb']:.2f} GB usable"
    )
    lines.append(
        f"  Less Model Weights (4.2B fp16) : -{kv_info['weights_gb']:.2f} GB"
    )
    lines.append(
        f"  Less Non-KV Runtime Overhead   : -{kv_info['non_kv_overhead_gb']:.2f} GB"
    )
    lines.append(
        f"  Available KV-Cache Pool        : {kv_info['kv_pool_gb_decimal']:.2f} GB "
        f"({kv_info['kv_pool_bytes_decimal']:,} B decimal = {kv_info['kv_pool_gib_equivalent']:.4f} GiB binary)"
    )
    lines.append(
        f"  Max Tokens in 12.08 GB Pool    : {kv_info['max_tokens_in_pool_decimal']:,} tokens "
        f"(or {kv_info['max_tokens_in_pool_if_12_08_gib']:,} tokens if 12.08 GiB = 12.97e9 B)"
    )
    lines.append(
        f"  Short Seq (512+256 = 768 tok)  : {kv_info['seq_768_kv_bytes']:,} B = {kv_info['seq_768_kv_mb_decimal']:.2f} MB "
        f"({kv_info['seq_768_kv_mib_binary']:.2f} MiB) -> Idealized max {kv_info['max_concurrent_seqs_768_decimal_pool']} concurrent seqs"
    )
    lines.append(
        f"  Long Seq (3584+512 = 4096 tok) : {kv_info['seq_4096_kv_bytes']:,} B = {kv_info['seq_4096_kv_mb_decimal']:.2f} MB "
        f"({kv_info['seq_4096_kv_mib_binary']:.2f} MiB = {kv_info['seq_4096_kv_gib_binary']:.4f} GiB) -> "
        f"Idealized max {kv_info['max_concurrent_seqs_4096_decimal_pool_floor']} concurrent seqs "
        f"(exact {kv_info['max_concurrent_seqs_4096_decimal_pool_exact']:.3f}; allocator/block overhead may reduce actual capacity)"
    )
    lines.append(
        f"  Idealized KV Util @ 4096 tok   : 24 seqs = {kv_info['util_at_24_seqs_4096_decimal']:.4f} (rounds to 0.93) | "
        f"25 seqs = {kv_info['util_at_25_seqs_4096_decimal']:.4f} (rounds to 0.97) | "
        f"26 seqs = {kv_info['util_at_26_seqs_4096_decimal']:.4f} (>1.00, exceeds idealized pool)"
    )
    lines.append("")
    lines.append("2. ROW-BY-ROW RECONCILIATION OF bench/bench_log.csv")
    lines.append("   (Note: Model_BW_GB/s is an analytical traffic-model estimate = (weights + avg_active_KV)/itl_p50, not hardware-measured bandwidth)")
    lines.append("-" * 100)
    hdr = (
        f"{'Row':>3} | {'B':>2} | {'p':>4} | {'g':>3} | {'wall_s':>6} | "
        f"{'rep_tok/s':>9} | {'tot_tok/s':>9} | {'gen_tok/s':>9} | {'(p+g)/g':>7} | "
        f"{'KV_Dem_GB':>9} | {'Dem_Rat':>7} | {'Cap_Rat':>7} | {'Log_KVU':>7} | "
        f"{'Pre(Idl/Log)':>12} | {'ITL_ms':>6} | {'Model_BW':>8}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rows:
        pre_str = f"{r['expected_preempted_seqs']:2d} / {r['logged_preempted_seqs']:2d}"
        lines.append(
            f"{r['row']:3d} | {r['batch_size']:2d} | {r['prompt_len']:4d} | {r['gen_len']:3d} | "
            f"{r['wall_clock_s']:6.2f} | {r['reported_tok_s']:9.1f} | {r['calc_total_tok_s']:9.2f} | "
            f"{r['calc_gen_tok_s']:9.2f} | {r['total_to_gen_ratio']:6.1f}x | "
            f"{r['kv_demand_gb_decimal']:9.3f} | {r['kv_demand_ratio_decimal']:7.4f} | "
            f"{r['capped_kv_util_decimal']:7.4f} | {r['logged_kv_cache_util']:7.2f} | "
            f"{pre_str:>12} | {r['itl_ms_p50']:6.2f} | {r['eff_hbm_bw_avg_gb_s']:8.1f}"
        )
    lines.append("")
    lines.append("3. SENSITIVITY ANALYSES")
    lines.append("-" * 100)
    lines.append("3a. Sequence Length & KV Precision (fp16 vs. fp8/int8 on 12.08 GB Pool):")
    for s in sens["sequence_length_and_precision"]:
        lines.append(
            f"  S = {s['seq_len']:4d} tok | fp16: {s['fp16_mb_per_seq_dec']:6.2f} MB ({s['fp16_mib_per_seq_bin']:6.2f} MiB) "
            f"-> B_max = {s['fp16_max_seqs_12_08gb']:3d} seqs | fp8/int8: {s['fp8_mb_per_seq_dec']:6.2f} MB "
            f"-> B_max = {s['fp8_max_seqs_12_08gb']:3d} seqs"
        )
    lines.append("")
    lines.append("3b. Non-KV Runtime Overhead Assumption (`model_spec.md` assumes ~1.6 GB):")
    for ov in sens["non_kv_overhead_gb"]:
        lines.append(
            f"  Overhead = {ov['non_kv_overhead_gb']:4.1f} GB -> KV Pool = {ov['kv_pool_gb']:5.2f} GB "
            f"({ov['max_tokens_fp16']:,} max tok) -> B_max(4096) = {ov['max_seqs_at_4096_fp16']:2d} seqs | "
            f"B_max(768) = {ov['max_seqs_at_768_fp16']:3d} seqs"
        )
    lines.append("")
    lines.append("3c. Cross-Part Sensitivity: Effect of Part A3 Multilingual Token Ratios on Base 768-Token Turn:")
    for m in sens["multilingual_token_ratio_impact_from_base_768"]:
        lines.append(
            f"  {m['language']:16s} | o200k ({m['o200k_sentence_token_ratio']:.2f}x): S={m['equiv_seq_len_o200k']:4d} tok "
            f"-> B_max={m['max_concurrent_seqs_o200k']:3d} seqs | "
            f"cl100k ({m['cl100k_sentence_token_ratio']:.2f}x): S={m['equiv_seq_len_cl100k']:4d} tok "
            f"(fits 4096? {str(m['fits_max_model_len_4096_cl100k']):5s}) -> B_max={m['max_concurrent_seqs_cl100k']:3d} seqs"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kv_info = compute_kv_fundamentals(SUPPLIED_SPEC)
    rows = analyze_bench_log(kv_info)
    sens = run_sensitivity_analyses(kv_info)

    payload = {
        "supplied_spec": SUPPLIED_SPEC,
        "missing_parameters": MISSING_PARAMETERS,
        "kv_fundamentals": kv_info,
        "bench_log_reconciliation": rows,
        "sensitivity_analyses": sens,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_txt = format_report_text(kv_info, rows, sens)
    OUT_TXT.write_text(report_txt, encoding="utf-8")
    print(report_txt)


if __name__ == "__main__":
    main()
