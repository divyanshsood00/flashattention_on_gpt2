"""Export one CSV table per figure from ``presentation_report.json``.

This makes every plot in ``figures/`` trivially reproducible in other tools
(LaTeX/TikZ, pgfplots, Observable, Excel, Google Sheets, Python/pandas).
Each CSV is named exactly like the PNG/PDF it backs.

Usage
-----
python -m project.export_report_tables \\
    --report_json presentation_report_v2.json \\
    --output_dir figures/presentation_report_v2/data
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Any, Dict, List, Optional

import fire


def _unwrap(payload: Any) -> Any:
    if isinstance(payload, dict) and payload.get("status") == "ok" and "result" in payload:
        return payload["result"]
    return payload


def _rows(section: Any) -> List[Dict[str, Any]]:
    s = _unwrap(section)
    if not isinstance(s, dict):
        return []
    rows = s.get("results", [])
    good = [r for r in rows if isinstance(r, dict) and r.get("status", "ok") == "ok"]
    good.sort(key=lambda r: r.get("shape", {}).get("N", 0))
    return good


def _write_csv(path: str, header: List[str], rows: List[List[Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])


def _fmt(x: Optional[float], prec: int = 6) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(f"{float(x):.{prec}g}")
    except (TypeError, ValueError):
        return x


def export(
    report_json: str = "presentation_report_v2.json",
    output_dir: str = "figures/presentation_report_v2/data",
) -> Dict[str, Any]:
    with open(report_json, "r", encoding="utf-8") as f:
        report = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    written: Dict[str, int] = {}

    # Performance sections ---------------------------------------------------
    perf = _rows(report.get("performance_standard") or report.get("performance"))
    scaling = _rows(report.get("scaling_curves"))
    primary = scaling if scaling else perf  # scaling_curves is the intended source for N-sweeps

    # 1) forward latency vs N -----------------------------------------------
    h = ["N", "d", "B", "H",
         "baseline_forward_ms", "flash_forward_ms", "forward_speedup"]
    rows = [
        [r["shape"].get("N"), r["shape"].get("d"), r["shape"].get("B"), r["shape"].get("H"),
         _fmt(r.get("baseline_forward_ms")),
         _fmt(r.get("flash_forward_ms")),
         _fmt(r.get("forward_speedup"), 4)]
        for r in primary
    ]
    _write_csv(os.path.join(output_dir, "latency_forward_comparison.csv"), h, rows)
    written["latency_forward_comparison.csv"] = len(rows)

    # 2) forward+backward latency vs N --------------------------------------
    h = ["N", "d", "B", "H",
         "baseline_fw_bw_ms", "flash_fw_bw_ms", "fw_bw_speedup"]
    rows = [
        [r["shape"].get("N"), r["shape"].get("d"), r["shape"].get("B"), r["shape"].get("H"),
         _fmt(r.get("baseline_fw_bw_ms")),
         _fmt(r.get("flash_fw_bw_ms")),
         _fmt(r.get("fw_bw_speedup"), 4)]
        for r in primary
    ]
    _write_csv(os.path.join(output_dir, "latency_fw_bw_comparison.csv"), h, rows)
    written["latency_fw_bw_comparison.csv"] = len(rows)

    # 3) speedup vs N (dual line) -------------------------------------------
    h = ["N", "forward_speedup", "fw_bw_speedup"]
    rows = [
        [r["shape"].get("N"),
         _fmt(r.get("forward_speedup"), 4),
         _fmt(r.get("fw_bw_speedup"), 4)]
        for r in primary
    ]
    _write_csv(os.path.join(output_dir, "speedup_vs_seq.csv"), h, rows)
    written["speedup_vs_seq.csv"] = len(rows)

    # 4) memory savings vs N (measured) -------------------------------------
    h = ["N", "B", "H", "d",
         "baseline_forward_peak_mem_delta_mb", "flash_forward_peak_mem_delta_mb",
         "forward_peak_mem_savings_mb_measured", "peak_mem_reduction_ratio_est"]
    rows = [
        [r["shape"].get("N"), r["shape"].get("B"), r["shape"].get("H"), r["shape"].get("d"),
         _fmt(r.get("baseline_forward_peak_mem_delta_mb"), 6),
         _fmt(r.get("flash_forward_peak_mem_delta_mb"), 6),
         _fmt(r.get("forward_peak_mem_savings_mb_measured"), 6),
         _fmt(r.get("peak_mem_reduction_ratio_est"), 4)]
        for r in primary
    ]
    _write_csv(os.path.join(output_dir, "memory_savings_vs_seq.csv"), h, rows)
    written["memory_savings_vs_seq.csv"] = len(rows)

    # 5) theoretical memory scaling ----------------------------------------
    # Reproduces the model used by plot_presentation_results.py:
    #   standard:   B*H * 4 * N^2  bytes (fp32 scores) -> MB
    #   flash:      B*H * 4 * N    bytes (per-row scalars) -> MB
    def _mb(bytes_: float) -> float:
        return bytes_ / (1024.0 * 1024.0)

    h = ["N", "B", "H", "d",
         "standard_mem_quadratic_mb", "flash_mem_linear_mb", "theoretical_ratio"]
    rows = []
    for r in primary:
        N = r["shape"].get("N", 0)
        B = r["shape"].get("B", 0)
        H = r["shape"].get("H", 0)
        d = r["shape"].get("d", 0)
        std_mb = _mb(B * H * 4.0 * N * N)
        fl_mb = _mb(B * H * 4.0 * N)
        ratio = (std_mb / fl_mb) if fl_mb > 0 else None
        rows.append([N, B, H, d, _fmt(std_mb, 6), _fmt(fl_mb, 6), _fmt(ratio, 4)])
    _write_csv(os.path.join(output_dir, "memory_scaling_theory.csv"), h, rows)
    written["memory_scaling_theory.csv"] = len(rows)

    # 6) TFLOPs comparison (per config, from standard perf block) -----------
    h = ["config", "N", "B", "H", "d", "baseline_tflops", "flash_tflops"]
    rows = [
        [r.get("config"),
         r["shape"].get("N"), r["shape"].get("B"), r["shape"].get("H"), r["shape"].get("d"),
         _fmt(r.get("baseline_tflops"), 6),
         _fmt(r.get("flash_tflops"), 6)]
        for r in perf or scaling
    ]
    _write_csv(os.path.join(output_dir, "tflops_comparison.csv"), h, rows)
    written["tflops_comparison.csv"] = len(rows)

    # 7) Tokens/sec comparison ----------------------------------------------
    h = ["config", "N", "B", "H", "d", "baseline_tokens_per_sec", "flash_tokens_per_sec"]
    rows = [
        [r.get("config"),
         r["shape"].get("N"), r["shape"].get("B"), r["shape"].get("H"), r["shape"].get("d"),
         _fmt(r.get("baseline_tokens_per_sec"), 6),
         _fmt(r.get("flash_tokens_per_sec"), 6)]
        for r in perf or scaling
    ]
    _write_csv(os.path.join(output_dir, "tokens_per_sec_comparison.csv"), h, rows)
    written["tokens_per_sec_comparison.csv"] = len(rows)

    # 8) numerical gradient error (finite diff) -----------------------------
    numerical = _unwrap(report.get("numerical_correctness")) or {}
    fd = (numerical.get("finite_difference") or {}).get("per_tensor", {})
    h = ["tensor", "max_abs_error", "mean_abs_error", "n_samples",
         "threshold", "pass_global", "eps"]
    global_thr = (numerical.get("finite_difference") or {}).get("threshold")
    global_pass = (numerical.get("finite_difference") or {}).get("pass")
    eps = (numerical.get("finite_difference") or {}).get("eps")
    rows = []
    for tname in ("q", "k_t", "v"):
        stats = fd.get(tname, {})
        rows.append([tname,
                     _fmt(stats.get("max_abs_error"), 6),
                     _fmt(stats.get("mean_abs_error"), 6),
                     stats.get("n_samples"),
                     _fmt(global_thr), global_pass, _fmt(eps)])
    _write_csv(os.path.join(output_dir, "numerical_grad_error.csv"), h, rows)
    written["numerical_grad_error.csv"] = len(rows)

    # 8b) forward parity (single number) ------------------------------------
    h = ["metric", "value", "threshold", "pass"]
    rows = [
        ["forward_max_abs_error",
         _fmt(numerical.get("forward_max_abs_error"), 6),
         _fmt(numerical.get("forward_threshold")),
         numerical.get("forward_pass")],
    ]
    _write_csv(os.path.join(output_dir, "numerical_forward_parity.csv"), h, rows)
    written["numerical_forward_parity.csv"] = len(rows)

    # 9) BLEU comparison ----------------------------------------------------
    def _bleu(section: Any) -> Optional[float]:
        s = _unwrap(section)
        if not isinstance(s, dict):
            return None
        b = s.get("bleu") or {}
        return b.get("bleu")

    flash = _bleu(report.get("bleu_flash") or report.get("bleu"))
    base = _bleu(report.get("bleu_baseline"))
    h = ["variant", "BLEU", "n_samples",
         "attention_dropout", "use_flash_attention", "use_fused_kernel",
         "workdir"]
    rows = []

    def _row(variant_label: str, section: Any):
        s = _unwrap(section) or {}
        bleu_obj = s.get("bleu") or {}
        rows.append([
            variant_label,
            _fmt(bleu_obj.get("bleu"), 6),
            bleu_obj.get("n_samples"),
            _fmt(s.get("attention_dropout"), 3),
            s.get("use_flash_attention"),
            s.get("use_fused_kernel"),
            s.get("workdir"),
        ])

    _row("FlashAttention", report.get("bleu_flash") or report.get("bleu"))
    if report.get("bleu_baseline"):
        _row("Standard", report.get("bleu_baseline"))
    _write_csv(os.path.join(output_dir, "bleu_comparison.csv"), h, rows)
    written["bleu_comparison.csv"] = len(rows)

    # 10) quality vs throughput tradeoff -----------------------------------
    tr = report.get("throughput_tradeoff") or {}
    grid = tr.get("grid") or []
    h = ["epochs", "attention_dropout", "use_flash_attention",
         "BLEU", "tokens_per_sec", "bleu_duration_sec", "perf_duration_sec"]
    rows = []
    for p in grid:
        par = p.get("params", {})
        bleu_duration = (p.get("bleu") or {}).get("duration_sec")
        perf_duration = (p.get("performance") or {}).get("duration_sec")
        rows.append([
            par.get("epochs"),
            _fmt(par.get("dropout"), 3),
            par.get("use_flash_attention"),
            _fmt(p.get("bleu_value"), 6),
            _fmt(p.get("tokens_per_sec"), 6),
            _fmt(bleu_duration, 4),
            _fmt(perf_duration, 4),
        ])
    _write_csv(os.path.join(output_dir, "quality_throughput_tradeoff.csv"), h, rows)
    written["quality_throughput_tradeoff.csv"] = len(rows)

    # 11) ablation summary --------------------------------------------------
    ab = report.get("ablation_comparison") or {}
    h = ["metric", "flash", "baseline", "delta_flash_minus_baseline"]
    rows = [["BLEU",
             _fmt(ab.get("flash_bleu"), 6),
             _fmt(ab.get("baseline_bleu"), 6),
             _fmt(ab.get("bleu_delta_flash_minus_baseline"), 6)]]
    _write_csv(os.path.join(output_dir, "ablation_bleu.csv"), h, rows)
    written["ablation_bleu.csv"] = len(rows)

    # Manifest --------------------------------------------------------------
    manifest = {
        "report_json": report_json,
        "output_dir": output_dir,
        "tables": [
            {"file": name, "rows": n,
             "png": name.replace(".csv", ".png"),
             "pdf": name.replace(".csv", ".pdf")}
            for name, n in sorted(written.items())
        ],
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(written)} CSV table(s) to {output_dir}/")
    for name, n in sorted(written.items()):
        print(f"  {name}  ({n} row{'s' if n != 1 else ''})")
    return manifest


if __name__ == "__main__":
    fire.Fire(export)
