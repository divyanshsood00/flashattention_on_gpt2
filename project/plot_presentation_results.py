"""Publication-quality figure generator for the FlashAttention report.

Produces research-paper-grade matplotlib figures (PDF + PNG) from the JSON
report emitted by :mod:`project.presentation_report`. Every figure is designed
to stand alone: self-contained title, clearly labeled axes, annotated data
points, and a consistent visual identity.

Usage
-----
python -m project.plot_presentation_results \\
    --report_json presentation_report.json \\
    --output_dir figures
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fire
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

COLOR_BASELINE = "#D1495B"
COLOR_FLASH = "#2E86AB"
COLOR_ACCENT = "#F2A900"
COLOR_NEUTRAL = "#4A4A4A"
COLOR_Q = "#2E86AB"
COLOR_K = "#D1495B"
COLOR_V = "#66A182"

PALETTE_BARS = [COLOR_FLASH, COLOR_BASELINE, COLOR_ACCENT, COLOR_NEUTRAL]


def _apply_paper_style() -> None:
    """Set a consistent publication style for all figures."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Nimbus Roman"],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "regular",
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#CCCCCC",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.85,
            "legend.edgecolor": "#BFBFBF",
            "legend.fancybox": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.5,
            "lines.markeredgewidth": 0.8,
            "errorbar.capsize": 3,
        }
    )


def _save(fig, out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{stem}.png"))
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data access helpers
# ---------------------------------------------------------------------------


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _unwrap_step(payload: Any) -> Any:
    if isinstance(payload, dict) and payload.get("status") == "ok" and "result" in payload:
        return payload["result"]
    return payload


def _safe_perf_rows(section: Any) -> List[Dict[str, Any]]:
    perf = _unwrap_step(section)
    if not isinstance(perf, dict):
        return []
    rows = perf.get("results", [])
    if not isinstance(rows, list):
        return []
    good = [r for r in rows if isinstance(r, dict) and r.get("status", "ok") == "ok"]
    good.sort(key=lambda r: r.get("shape", {}).get("N", 0))
    return good


def _bleu_value(section: Any) -> Optional[float]:
    step = _unwrap_step(section)
    if not isinstance(step, dict):
        return None
    bleu = step.get("bleu")
    if isinstance(bleu, dict):
        return bleu.get("bleu")
    return None


def _gpu_name(payload: Dict[str, Any]) -> str:
    for key in ("performance_standard", "scaling_curves", "performance"):
        perf = _unwrap_step(payload.get(key))
        if isinstance(perf, dict):
            name = (perf.get("meta") or {}).get("gpu_name")
            if name:
                return str(name)
    return "GPU"


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------


def _format_int_k(value: float, _pos: Any = None) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def _set_log2_x(ax, xs: Sequence[float]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(int(x)) for x in xs])


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------


def _plot_latency_generic(
    rows: List[Dict[str, Any]],
    out_dir: str,
    gpu_name: str,
    *,
    baseline_key: str,
    flash_key: str,
    ylabel: str,
    title: str,
    stem: str,
    speedup_fmt: str = r"$\times${speedup:.1f}",
) -> None:
    rows = [r for r in rows if r.get(baseline_key) and r.get(flash_key)]
    if not rows:
        return

    xs = [r["shape"]["N"] for r in rows]
    baseline = [r[baseline_key] for r in rows]
    flash = [r[flash_key] for r in rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(xs, baseline, marker="o", color=COLOR_BASELINE, label="Standard attention")
    ax.plot(xs, flash, marker="s", color=COLOR_FLASH, label="FlashAttention (ours)")

    _set_log2_x(ax, xs)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Sequence length $N$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left")

    y_min = min(flash) if flash else 1.0
    y_max = max(baseline) if baseline else y_min
    if y_min > 0 and y_max > 0:
        ax.set_ylim(y_min * 0.55, y_max * 1.9)

    for x, b, f in zip(xs, baseline, flash):
        speedup = b / f if f > 0 else 0.0
        mid_y = math.sqrt(max(b, 1e-9) * max(f, 1e-9))
        ax.annotate(
            speedup_fmt.format(speedup=speedup),
            xy=(x, mid_y),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color=COLOR_FLASH,
            fontweight="bold",
        )

    fig.tight_layout()
    _save(fig, out_dir, stem)


def _plot_latency(rows: List[Dict[str, Any]], out_dir: str, gpu_name: str) -> None:
    _plot_latency_generic(
        rows,
        out_dir,
        gpu_name,
        baseline_key="baseline_forward_ms",
        flash_key="flash_forward_ms",
        ylabel="Forward latency (ms, log scale)",
        title=f"Forward latency vs. sequence length — {gpu_name}",
        stem="latency_forward_comparison",
    )


def _plot_fw_bw_latency(rows: List[Dict[str, Any]], out_dir: str, gpu_name: str) -> None:
    _plot_latency_generic(
        rows,
        out_dir,
        gpu_name,
        baseline_key="baseline_fw_bw_ms",
        flash_key="flash_fw_bw_ms",
        ylabel="Forward+Backward latency (ms, log scale)",
        title=f"Training-step latency vs. sequence length — {gpu_name}",
        stem="latency_fw_bw_comparison",
        speedup_fmt=r"$\times${speedup:.2f}",
    )


def _plot_speedup(rows: List[Dict[str, Any]], out_dir: str, gpu_name: str) -> None:
    fw = [(r["shape"]["N"], r.get("forward_speedup")) for r in rows if r.get("forward_speedup") is not None]
    fwbw = [(r["shape"]["N"], r.get("fw_bw_speedup")) for r in rows if r.get("fw_bw_speedup") is not None]
    if not fw and not fwbw:
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if fw:
        xs_fw, ys_fw = zip(*fw)
        ax.plot(xs_fw, ys_fw, marker="o", color=COLOR_FLASH, label="Forward")
        for x, y in zip(xs_fw, ys_fw):
            ax.annotate(
                rf"$\times${y:.2f}",
                xy=(x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=COLOR_FLASH,
                fontweight="bold",
            )
    if fwbw:
        xs_fb, ys_fb = zip(*fwbw)
        ax.plot(xs_fb, ys_fb, marker="s", color=COLOR_ACCENT, label="Forward + Backward")
        for x, y in zip(xs_fb, ys_fb):
            ax.annotate(
                rf"$\times${y:.2f}",
                xy=(x, y),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=COLOR_ACCENT,
                fontweight="bold",
            )

    ax.axhline(1.0, linestyle="--", linewidth=1.0, color=COLOR_NEUTRAL, alpha=0.7)

    xs_all = sorted({n for n, _ in fw} | {n for n, _ in fwbw})
    _set_log2_x(ax, xs_all)
    ax.set_xlabel("Sequence length $N$")
    ax.set_ylabel("Speedup (baseline / FlashAttention)")
    ax.set_title(f"FlashAttention speedup vs. sequence length — {gpu_name}")
    ax.legend(loc="center right")
    ax.set_ylim(bottom=0.0)
    ax.text(
        0.02,
        0.06,
        "parity",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        color=COLOR_NEUTRAL,
        fontsize=9,
        alpha=0.85,
    )

    fig.tight_layout()
    _save(fig, out_dir, "speedup_vs_seq")


def _plot_memory(rows: List[Dict[str, Any]], out_dir: str, gpu_name: str) -> None:
    rows = [
        r
        for r in rows
        if r.get("baseline_forward_peak_mem_delta_mb") is not None
        and r.get("flash_forward_peak_mem_delta_mb") is not None
    ]
    if not rows:
        return

    xs = [r["shape"]["N"] for r in rows]
    baseline = [r["baseline_forward_peak_mem_delta_mb"] for r in rows]
    flash = [r["flash_forward_peak_mem_delta_mb"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    width = 0.38
    idx = np.arange(len(xs))
    b1 = ax.bar(idx - width / 2, baseline, width, color=COLOR_BASELINE, label="Standard attention")
    b2 = ax.bar(idx + width / 2, flash, width, color=COLOR_FLASH, label="FlashAttention (ours)")

    for rect in list(b1) + list(b2):
        h = rect.get_height()
        if h <= 0:
            continue
        ax.annotate(
            f"{h:.0f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=COLOR_NEUTRAL,
        )

    for i, (b, f) in enumerate(zip(baseline, flash)):
        if b > 0 and f >= 0:
            ratio = b / max(f, 1e-9)
            ax.annotate(
                rf"$\div${ratio:.1f}",
                xy=(i, max(b, f)),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=COLOR_FLASH,
                fontweight="bold",
            )

    ax.set_xticks(idx)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("Sequence length $N$")
    ax.set_ylabel("Forward peak-memory $\\Delta$ (MB)")
    ax.set_title(f"GPU peak-memory usage: standard vs. FlashAttention — {gpu_name}")
    ax.legend(loc="upper left")
    ax.margins(y=0.18)

    fig.tight_layout()
    _save(fig, out_dir, "memory_savings_vs_seq")


def _plot_memory_scaling(rows: List[Dict[str, Any]], out_dir: str, gpu_name: str) -> None:
    rows = [
        r
        for r in rows
        if r.get("baseline_peak_mem_est_mb") is not None
        and r.get("flash_peak_mem_est_mb") is not None
    ]
    if not rows:
        return

    xs = [r["shape"]["N"] for r in rows]
    base_theory = [r["baseline_peak_mem_est_mb"] for r in rows]
    flash_theory = [r["flash_peak_mem_est_mb"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(xs, base_theory, marker="o", color=COLOR_BASELINE, label=r"Standard $O(N^2)$")
    ax.plot(xs, flash_theory, marker="s", color=COLOR_FLASH, label=r"FlashAttention $O(N)$")

    meas_x, meas_y = [], []
    for r in rows:
        m = r.get("forward_peak_mem_savings_mb_measured")
        if m is None:
            continue
        meas_x.append(r["shape"]["N"])
        meas_y.append(m)

    _set_log2_x(ax, xs)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Sequence length $N$")
    ax.set_ylabel("Peak activation memory (MB, log scale)")
    ax.set_title(f"Attention memory scaling — {gpu_name}")
    ax.legend(loc="upper left")

    fig.tight_layout()
    _save(fig, out_dir, "memory_scaling_theory")


def _plot_tflops(rows: List[Dict[str, Any]], out_dir: str, gpu_name: str) -> None:
    rows = [
        r
        for r in rows
        if r.get("baseline_tflops") is not None and r.get("flash_tflops") is not None
    ]
    if not rows:
        return

    xs = [r["shape"]["N"] for r in rows]
    base = [r["baseline_tflops"] for r in rows]
    flash = [r["flash_tflops"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(xs, base, marker="o", color=COLOR_BASELINE, label="Standard attention")
    ax.plot(xs, flash, marker="s", color=COLOR_FLASH, label="FlashAttention (ours)")

    _set_log2_x(ax, xs)
    ax.set_xlabel("Sequence length $N$")
    ax.set_ylabel("Achieved throughput (TFLOP/s)")
    ax.set_title(f"Arithmetic throughput of attention kernels — {gpu_name}")
    ax.legend(loc="upper left")

    fig.tight_layout()
    _save(fig, out_dir, "tflops_comparison")


def _plot_tokens_per_sec(rows: List[Dict[str, Any]], out_dir: str, gpu_name: str) -> None:
    rows = [
        r
        for r in rows
        if r.get("baseline_tokens_per_sec") is not None and r.get("flash_tokens_per_sec") is not None
    ]
    if not rows:
        return

    xs = [r["shape"]["N"] for r in rows]
    base = [r["baseline_tokens_per_sec"] for r in rows]
    flash = [r["flash_tokens_per_sec"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    width = 0.38
    idx = np.arange(len(xs))
    ax.bar(idx - width / 2, base, width, color=COLOR_BASELINE, label="Standard")
    ax.bar(idx + width / 2, flash, width, color=COLOR_FLASH, label="FlashAttention")

    for i, (b, f) in enumerate(zip(base, flash)):
        ax.annotate(
            _format_int_k(b),
            xy=(i - width / 2, b),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=COLOR_NEUTRAL,
        )
        ax.annotate(
            _format_int_k(f),
            xy=(i + width / 2, f),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=COLOR_NEUTRAL,
        )

    ax.set_xticks(idx)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(_format_int_k))
    ax.set_xlabel("Sequence length $N$")
    ax.set_ylabel("Attention throughput (tokens/s, log scale)")
    ax.set_title(f"Effective attention throughput — {gpu_name}")
    ax.legend(loc="upper right")

    fig.tight_layout()
    _save(fig, out_dir, "tokens_per_sec_comparison")


def _plot_numerical(numerical_section: Any, out_dir: str) -> None:
    numerical = _unwrap_step(numerical_section)
    if not isinstance(numerical, dict):
        return
    per_tensor = numerical.get("finite_difference", {}).get("per_tensor", {})
    if not per_tensor:
        return

    keys = [k for k in ("q", "k_t", "v") if k in per_tensor]
    labels = {"q": r"$\mathbf{Q}$", "k_t": r"$\mathbf{K}^{\top}$", "v": r"$\mathbf{V}$"}
    max_vals = [per_tensor[k].get("max_abs_error", 0.0) for k in keys]
    mean_vals = [per_tensor[k].get("mean_abs_error", 0.0) for k in keys]

    threshold = numerical.get("finite_difference", {}).get("threshold")
    fwd_threshold = numerical.get("forward_threshold")
    fwd_error = numerical.get("forward_max_abs_error")

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    idx = np.arange(len(keys))
    width = 0.38
    ax.bar(
        idx - width / 2,
        max_vals,
        width,
        color=COLOR_BASELINE,
        label="Max |analytic − finite diff|",
    )
    ax.bar(
        idx + width / 2,
        mean_vals,
        width,
        color=COLOR_FLASH,
        label="Mean |analytic − finite diff|",
    )

    for i, (mx, mn) in enumerate(zip(max_vals, mean_vals)):
        ax.annotate(
            f"{mx:.1e}",
            xy=(i - width / 2, mx),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=COLOR_NEUTRAL,
        )
        ax.annotate(
            f"{mn:.1e}",
            xy=(i + width / 2, mn),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=COLOR_NEUTRAL,
        )

    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels([labels.get(k, k) for k in keys])
    ax.set_xlabel("Tensor")
    ax.set_ylabel("Gradient error (log scale)")

    subtitle = "Backward gradient parity against finite differences"
    if fwd_error is not None and fwd_threshold is not None:
        subtitle += (
            f"\nforward max error = {fwd_error:.1e} (threshold {fwd_threshold:g})"
        )
    ax.set_title(subtitle)
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    ax.margins(y=0.45)

    if threshold is not None and threshold > 0:
        ax.axhline(threshold, linestyle="--", color=COLOR_ACCENT, linewidth=1.3)
        xmin, xmax = ax.get_xlim()
        ax.text(
            xmax - 0.02 * (xmax - xmin),
            threshold,
            f"pass threshold {threshold:g}",
            va="bottom",
            ha="right",
            color=COLOR_ACCENT,
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    _save(fig, out_dir, "numerical_grad_error")


def _plot_bleu(flash_bleu: Optional[float], baseline_bleu: Optional[float], out_dir: str) -> None:
    if flash_bleu is None and baseline_bleu is None:
        return

    items: List[Tuple[str, float, str]] = []
    if baseline_bleu is not None:
        items.append(("Standard", float(baseline_bleu), COLOR_BASELINE))
    if flash_bleu is not None:
        items.append(("FlashAttention", float(flash_bleu), COLOR_FLASH))

    labels = [x[0] for x in items]
    vals = [x[1] for x in items]
    colors = [x[2] for x in items]

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    width = 0.55 if len(items) > 1 else 0.35
    bars = ax.bar(labels, vals, color=colors, width=width)
    if len(items) == 1:
        ax.set_xlim(-1, 1)

    for rect, v in zip(bars, vals):
        ax.annotate(
            f"{v:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    delta = None
    if flash_bleu is not None and baseline_bleu is not None:
        delta = flash_bleu - baseline_bleu

    ax.set_ylabel("BLEU")
    title = "End-to-end quality: IWSLT14 de$\\rightarrow$en"
    if delta is not None:
        sign = "+" if delta >= 0 else ""
        title += f" ($\\Delta$BLEU = {sign}{delta:.2f})"
    elif len(items) == 1:
        title += " (single-configuration run)"
    ax.set_title(title)
    ax.margins(y=0.2)

    fig.tight_layout()
    _save(fig, out_dir, "bleu_comparison")


def _plot_tradeoff(tradeoff_section: Any, out_dir: str) -> None:
    tradeoff = _unwrap_step(tradeoff_section)
    if not isinstance(tradeoff, dict):
        return
    grid = tradeoff.get("grid", [])
    pts = [g for g in grid if g.get("bleu_value") is not None and g.get("tokens_per_sec") is not None]
    if not pts:
        return

    flash_pts = [p for p in pts if p.get("params", {}).get("use_flash_attention")]
    base_pts = [p for p in pts if not p.get("params", {}).get("use_flash_attention")]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if base_pts:
        ax.scatter(
            [p["tokens_per_sec"] for p in base_pts],
            [p["bleu_value"] for p in base_pts],
            c=COLOR_BASELINE,
            s=70,
            label="Standard",
            edgecolor="white",
            linewidth=0.8,
        )
    if flash_pts:
        ax.scatter(
            [p["tokens_per_sec"] for p in flash_pts],
            [p["bleu_value"] for p in flash_pts],
            c=COLOR_FLASH,
            s=70,
            marker="s",
            label="FlashAttention",
            edgecolor="white",
            linewidth=0.8,
        )

    for p in pts:
        prm = p.get("params", {})
        tag = f"e{prm.get('epochs')} d{prm.get('dropout')}"
        ax.annotate(
            tag,
            (p["tokens_per_sec"], p["bleu_value"]),
            fontsize=8,
            xytext=(6, 4),
            textcoords="offset points",
            color=COLOR_NEUTRAL,
        )

    ax.xaxis.set_major_formatter(FuncFormatter(_format_int_k))
    ax.set_xlabel("Attention throughput (tokens/s)")
    ax.set_ylabel("BLEU")
    ax.set_title("Quality-throughput tradeoff")
    ax.legend(loc="best")

    fig.tight_layout()
    _save(fig, out_dir, "quality_throughput_tradeoff")


def _plot_summary(
    perf_rows: List[Dict[str, Any]],
    scaling_rows: List[Dict[str, Any]],
    numerical_section: Any,
    flash_bleu: Optional[float],
    baseline_bleu: Optional[float],
    out_dir: str,
    gpu_name: str,
) -> None:
    """Single 'figure 1' style multi-panel overview suitable for a paper intro."""
    rows = scaling_rows if scaling_rows else perf_rows
    if not rows:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel 1: forward speedup
    ax = axes[0]
    xs = [r["shape"]["N"] for r in rows if r.get("forward_speedup") is not None]
    ys = [r["forward_speedup"] for r in rows if r.get("forward_speedup") is not None]
    if xs:
        ax.plot(xs, ys, marker="o", color=COLOR_FLASH, label="Forward")
        ax.axhline(1.0, linestyle="--", color=COLOR_NEUTRAL, linewidth=1.0, alpha=0.7)
        for x, y in zip(xs, ys):
            ax.annotate(
                rf"$\times${y:.1f}",
                xy=(x, y),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=COLOR_FLASH,
                fontweight="bold",
            )
        _set_log2_x(ax, xs)
    ax.set_xlabel("Sequence length $N$")
    ax.set_ylabel("Forward speedup")
    ax.set_title("(a) Forward speedup")

    # Panel 2: peak memory (measured)
    ax = axes[1]
    bx = [r["shape"]["N"] for r in rows if r.get("baseline_forward_peak_mem_delta_mb") is not None]
    bb = [r["baseline_forward_peak_mem_delta_mb"] for r in rows if r.get("baseline_forward_peak_mem_delta_mb") is not None]
    bf = [r["flash_forward_peak_mem_delta_mb"] for r in rows if r.get("flash_forward_peak_mem_delta_mb") is not None]
    if bx:
        idx = np.arange(len(bx))
        w = 0.38
        ax.bar(idx - w / 2, bb, w, color=COLOR_BASELINE, label="Standard")
        ax.bar(idx + w / 2, bf, w, color=COLOR_FLASH, label="Flash")
        ax.set_xticks(idx)
        ax.set_xticklabels([str(n) for n in bx])
        ax.legend(loc="upper left")
    ax.set_xlabel("Sequence length $N$")
    ax.set_ylabel("Peak memory $\\Delta$ (MB)")
    ax.set_title("(b) Peak GPU memory")

    # Panel 3: BLEU comparison or numerical correctness fallback
    ax = axes[2]
    panel_c_title = "(c) Quality / correctness"
    if flash_bleu is not None and baseline_bleu is not None:
        vals = [baseline_bleu, flash_bleu]
        labs = ["Standard", "Flash"]
        bars = ax.bar(labs, vals, color=[COLOR_BASELINE, COLOR_FLASH], width=0.55)
        for rect, v in zip(bars, vals):
            ax.annotate(
                f"{v:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_ylabel("BLEU (IWSLT14 de$\\rightarrow$en)")
        ax.margins(y=0.2)
        panel_c_title = "(c) End-to-end BLEU"
    else:
        numerical = _unwrap_step(numerical_section)
        per_tensor = (numerical or {}).get("finite_difference", {}).get("per_tensor", {})
        keys = [k for k in ("q", "k_t", "v") if k in per_tensor]
        labels = {"q": r"$\mathbf{Q}$", "k_t": r"$\mathbf{K}^{\top}$", "v": r"$\mathbf{V}$"}
        max_vals = [per_tensor[k].get("max_abs_error", 0.0) for k in keys]
        bars = ax.bar([labels.get(k, k) for k in keys], max_vals, color=COLOR_FLASH, width=0.55)
        for rect, v in zip(bars, max_vals):
            ax.annotate(
                f"{v:.1e}",
                xy=(rect.get_x() + rect.get_width() / 2, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=COLOR_NEUTRAL,
            )
        threshold = (numerical or {}).get("finite_difference", {}).get("threshold")
        if threshold:
            ax.axhline(threshold, linestyle="--", color=COLOR_ACCENT, linewidth=1.2)
            ax.text(
                len(keys) - 0.5,
                threshold,
                f" threshold {threshold:g}",
                va="bottom",
                ha="right",
                color=COLOR_ACCENT,
                fontsize=9,
                fontweight="bold",
            )
        ax.set_yscale("log")
        ax.set_ylabel("Max |finite-diff $-$ analytic|")
        if flash_bleu is not None:
            panel_c_title = f"(c) Gradient parity (Flash BLEU = {flash_bleu:.2f})"
        else:
            panel_c_title = "(c) Gradient parity"

    ax.set_title(panel_c_title)

    fig.suptitle(
        f"FlashAttention on GPT-2-style transformer — {gpu_name}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, out_dir, "overview_summary")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    report_json: str = "presentation_report.json",
    output_dir: str = "figures",
) -> Dict[str, Any]:
    _apply_paper_style()
    os.makedirs(output_dir, exist_ok=True)

    payload = _load_json(report_json)

    perf_rows = _safe_perf_rows(payload.get("performance_standard") or payload.get("performance"))
    scaling_rows = _safe_perf_rows(payload.get("scaling_curves"))
    plot_rows = scaling_rows if scaling_rows else perf_rows
    gpu_name = _gpu_name(payload)

    _plot_latency(plot_rows, output_dir, gpu_name)
    _plot_fw_bw_latency(plot_rows, output_dir, gpu_name)
    _plot_speedup(plot_rows, output_dir, gpu_name)
    _plot_memory(plot_rows, output_dir, gpu_name)
    _plot_memory_scaling(plot_rows, output_dir, gpu_name)
    _plot_tflops(plot_rows, output_dir, gpu_name)
    _plot_tokens_per_sec(plot_rows, output_dir, gpu_name)

    _plot_numerical(payload.get("numerical_correctness"), output_dir)

    flash_bleu = _bleu_value(payload.get("bleu_flash") or payload.get("bleu"))
    baseline_bleu = _bleu_value(payload.get("bleu_baseline"))
    _plot_bleu(flash_bleu, baseline_bleu, output_dir)

    _plot_tradeoff(payload.get("throughput_tradeoff"), output_dir)

    _plot_summary(
        perf_rows,
        scaling_rows,
        payload.get("numerical_correctness"),
        flash_bleu,
        baseline_bleu,
        output_dir,
        gpu_name,
    )

    generated = sorted(
        name
        for name in os.listdir(output_dir)
        if name.endswith(".png") or name.endswith(".pdf")
    )

    summary = {
        "report_json": report_json,
        "output_dir": output_dir,
        "gpu_name": gpu_name,
        "plots": generated,
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    fire.Fire(main)
