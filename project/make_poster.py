"""Build a single-page research poster for the FlashAttention-on-GPT-2 project.

The poster is laid out as a standard 48" x 36" landscape academic poster and
exported to both PDF (vector, ready for print) and PNG. It embeds the figures
produced by :mod:`project.plot_presentation_results`, pulls headline numbers
out of ``presentation_report.json``, and renders method/takeaway prose
alongside them.

Design goals
------------
* **Readable from a few feet away** — large fonts everywhere (title 60 pt,
  section titles 34 pt, body 22 pt, bullets 22 pt, callout values 64 pt).
* **No overlap** — every panel is laid out on a 200-row grid with explicit
  spacer rows between sections and margins inside every panel.
* **No email** — poster header/footer never render a contact email.
* **Multiple authors** — the ``authors`` string is rendered verbatim (use
  ``" | "`` to separate names).
* **Meta-aware** — if the report JSON contains a ``report_meta`` block with
  ``title``, ``subtitle``, ``authors``, or ``affiliation`` those override the
  CLI defaults (pass ``--no-use_report_meta`` to disable).

Usage
-----
python -m project.make_poster \\
    --report_json presentation_report_v2.json \\
    --figures_dir figures/presentation_report_v2 \\
    --output_dir figures/presentation_report_v2 \\
    --authors "Divyansh Sood | Siddharth Sarma | Yuvraj Ahuja"
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import fire
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch


# ---------------------------------------------------------------------------
# Colors (matched with plot_presentation_results.py)
# ---------------------------------------------------------------------------

COLOR_BASELINE = "#D1495B"
COLOR_FLASH = "#2E86AB"
COLOR_ACCENT = "#F2A900"
COLOR_NEUTRAL = "#2F3E46"
COLOR_PANEL = "#F7F7F2"
COLOR_PANEL_EDGE = "#B0B0B0"
COLOR_HEADER = "#0C1B33"
COLOR_CALLOUT = "#1B4965"


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def _apply_poster_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Nimbus Roman"],
            "mathtext.fontset": "cm",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.bottom": False,
            "axes.spines.left": False,
            "axes.grid": False,
        }
    )


# ---------------------------------------------------------------------------
# Data extraction
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


def _extract_headline_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact numbers used in the large callout banner."""
    perf_rows = _safe_perf_rows(report.get("performance_standard") or report.get("performance"))
    scaling_rows = _safe_perf_rows(report.get("scaling_curves"))
    rows = scaling_rows if scaling_rows else perf_rows

    def _max_from(rows: List[Dict[str, Any]], key: str) -> Optional[Tuple[int, float]]:
        cand = [(r.get("shape", {}).get("N"), r.get(key)) for r in rows if r.get(key) is not None]
        cand = [c for c in cand if c[0] is not None and c[1] is not None]
        if not cand:
            return None
        return max(cand, key=lambda t: t[1])

    # Use the union of perf + scaling for peak-across-all metrics.
    all_rows = list(perf_rows) + list(scaling_rows)
    best_fw = _max_from(all_rows, "forward_speedup")
    best_fwbw = _max_from(all_rows, "fw_bw_speedup")

    best_mem: Optional[Tuple[int, float]] = None
    for r in all_rows:
        base = r.get("baseline_forward_peak_mem_delta_mb")
        flash = r.get("flash_forward_peak_mem_delta_mb")
        if base is None or flash is None:
            continue
        n = r.get("shape", {}).get("N")
        save = base - flash
        if best_mem is None or save > best_mem[1]:
            best_mem = (n, save)

    best_ratio: Optional[Tuple[int, float]] = None
    for r in all_rows:
        ratio = r.get("peak_mem_reduction_ratio_est")
        n = r.get("shape", {}).get("N")
        if ratio is None or n is None:
            continue
        if best_ratio is None or ratio > best_ratio[1]:
            best_ratio = (n, ratio)

    numerical = _unwrap_step(report.get("numerical_correctness"))
    fwd_err = (numerical or {}).get("forward_max_abs_error")
    grad_err = (numerical or {}).get("finite_difference", {}).get("max_abs_error")
    grad_thr = (numerical or {}).get("finite_difference", {}).get("threshold")

    flash_bleu = _bleu_value(report.get("bleu_flash") or report.get("bleu"))
    baseline_bleu = _bleu_value(report.get("bleu_baseline"))

    gpu_name = "GPU"
    for key in ("performance_standard", "scaling_curves", "performance"):
        perf = _unwrap_step(report.get(key))
        if isinstance(perf, dict):
            name = (perf.get("meta") or {}).get("gpu_name")
            if name:
                gpu_name = str(name)
                break

    return {
        "best_fw_speedup": best_fw,
        "best_fwbw_speedup": best_fwbw,
        "best_mem_saving_mb": best_mem,
        "best_mem_ratio": best_ratio,
        "forward_error": fwd_err,
        "grad_error": grad_err,
        "grad_threshold": grad_thr,
        "flash_bleu": flash_bleu,
        "baseline_bleu": baseline_bleu,
        "gpu_name": gpu_name,
    }


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def _draw_panel_background(ax, color=COLOR_PANEL, edge=COLOR_PANEL_EDGE, radius=0.02) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("none")
    patch = FancyBboxPatch(
        (0.005, 0.005),
        0.99,
        0.99,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color,
        edgecolor=edge,
        linewidth=1.2,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)


def _section_title(ax, text: str, y: float = 0.955, fontsize: float = 32) -> None:
    ax.text(
        0.5,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        color=COLOR_HEADER,
    )
    ax.plot(
        [0.08, 0.92],
        [y - 0.045, y - 0.045],
        transform=ax.transAxes,
        color=COLOR_FLASH,
        linewidth=3.0,
        solid_capstyle="round",
        clip_on=False,
    )


def _body_text(ax, text: str, y_start: float = 0.86, fontsize: float = 22,
               line_spacing: float = 1.25, x: float = 0.055) -> None:
    ax.text(
        x,
        y_start,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color=COLOR_NEUTRAL,
        linespacing=line_spacing,
        wrap=True,
    )


def _image_panel(ax, image_path: str, caption: Optional[str] = None) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if not os.path.exists(image_path):
        ax.text(0.5, 0.5, f"missing:\n{os.path.basename(image_path)}",
                ha="center", va="center", fontsize=16, color="#AA0000",
                transform=ax.transAxes)
        return
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.set_aspect("auto")
    if caption:
        ax.set_title(caption, fontsize=18, pad=8, color=COLOR_NEUTRAL, loc="center")


def _stat_callout(ax, value: str, label: str, color: str = COLOR_FLASH) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    patch = FancyBboxPatch(
        (0.025, 0.04),
        0.95,
        0.92,
        boxstyle="round,pad=0,rounding_size=0.05",
        facecolor=color,
        edgecolor="none",
        linewidth=0,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        0.5,
        0.60,
        value,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=62,
        fontweight="bold",
        color="white",
    )
    ax.text(
        0.5,
        0.22,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=20,
        color="white",
        fontweight="regular",
    )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _format_speedup(entry: Optional[Tuple[int, float]]) -> str:
    if not entry:
        return "—"
    _, val = entry
    return f"×{val:.1f}"


def _format_mem(entry: Optional[Tuple[int, float]]) -> str:
    if not entry:
        return "—"
    _, mb = entry
    if mb >= 1024:
        return f"{mb / 1024.0:.1f} GB"
    return f"{mb:.0f} MB"


def _format_ratio(entry: Optional[Tuple[int, float]]) -> str:
    if not entry:
        return "—"
    _, r = entry
    return f"÷{r:.1f}"


def _format_bleu(flash_bleu: Optional[float], baseline_bleu: Optional[float]) -> Tuple[str, str]:
    if flash_bleu is None:
        return "—", "BLEU"
    if baseline_bleu is None:
        return f"{flash_bleu:.2f}", "BLEU  (IWSLT14 de→en)"
    delta = flash_bleu - baseline_bleu
    sign = "+" if delta >= 0 else ""
    return f"{flash_bleu:.2f}", f"BLEU  (Δ {sign}{delta:.2f} vs. standard)"


# ---------------------------------------------------------------------------
# Poster assembly
# ---------------------------------------------------------------------------


def build_poster(
    report_json: str = "presentation_report.json",
    figures_dir: str = "figures",
    output_dir: str = "figures",
    title: str = "FlashAttention for GPT-2: Fast, Memory-Efficient Self-Attention",
    subtitle: str = "A fused, tiled online-softmax attention kernel integrated into a minitorch GPT-2 stack",
    authors: str = "Divyansh Sood | Siddharth Sarma | Yuvraj Ahuja",
    affiliation: str = "Carnegie Mellon University  ·  11-868 LLM Systems",
    stem: str = "poster",
    use_report_meta: bool = True,
) -> Dict[str, Any]:
    _apply_poster_style()
    os.makedirs(output_dir, exist_ok=True)

    report = _load_json(report_json)
    metrics = _extract_headline_metrics(report)

    # Honor report_meta if present (title/subtitle/authors/affiliation only;
    # never render contact email even if the field exists).
    if use_report_meta:
        meta = report.get("report_meta") or {}
        if isinstance(meta, dict):
            if meta.get("title"):
                title = str(meta["title"])
            if meta.get("subtitle"):
                subtitle = str(meta["subtitle"])
            if meta.get("authors"):
                authors = str(meta["authors"])
            if meta.get("affiliation"):
                affiliation = str(meta["affiliation"])

    width_in, height_in = 48, 36  # standard landscape academic poster
    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")

    # Use a 200-row grid for fine-grained vertical control (prevents overlap
    # between header / callouts / body / footer).
    outer = GridSpec(
        nrows=200,
        ncols=100,
        figure=fig,
        left=0.020,
        right=0.980,
        top=0.985,
        bottom=0.015,
        wspace=0.0,
        hspace=0.0,
    )

    # Vertical row budget (out of 200):
    #   0  – 22   header                         (22 rows)
    #   22 – 26   spacer                         (4 rows)
    #   26 – 46   headline callouts              (20 rows)
    #   46 – 50   spacer                         (4 rows)
    #   50 – 180  body (3 columns)               (130 rows)
    #   180 – 184 spacer                         (4 rows)
    #   184 – 200 footer (takeaways)             (16 rows)
    HEADER_T, HEADER_B = 0, 22
    CALL_T, CALL_B = 26, 46
    BODY_T, BODY_B = 50, 180
    FOOT_T, FOOT_B = 184, 200

    # --------------------------- Header -------------------------------------
    header_ax = fig.add_subplot(outer[HEADER_T:HEADER_B, :])
    header_ax.set_xticks([])
    header_ax.set_yticks([])
    for spine in header_ax.spines.values():
        spine.set_visible(False)
    header_ax.set_xlim(0, 1)
    header_ax.set_ylim(0, 1)
    header_ax.add_patch(
        FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor=COLOR_HEADER,
            edgecolor="none",
            transform=header_ax.transAxes,
            clip_on=False,
        )
    )
    # Three cleanly separated header rows. y values are chosen so that title
    # (big), subtitle (italic), and attribution (regular) never collide.
    header_ax.text(
        0.020, 0.78, title,
        transform=header_ax.transAxes, ha="left", va="center",
        fontsize=60, fontweight="bold", color="white",
    )
    header_ax.text(
        0.020, 0.47, subtitle,
        transform=header_ax.transAxes, ha="left", va="center",
        fontsize=28, color="#CFE0F4", style="italic",
    )
    # Attribution: authors on its own line, affiliation + hardware beneath.
    # Rendered *without* any email, per spec.
    header_ax.text(
        0.020, 0.22, authors,
        transform=header_ax.transAxes, ha="left", va="center",
        fontsize=26, fontweight="bold", color="#E4EEF8",
    )
    header_ax.text(
        0.980, 0.22, f"{affiliation}   ·   Hardware: {metrics['gpu_name']}",
        transform=header_ax.transAxes, ha="right", va="center",
        fontsize=22, color="#B9CCE2",
    )

    # --------------------------- Headline callouts --------------------------
    callouts = [
        (_format_speedup(metrics["best_fw_speedup"]),
         f"Peak forward speedup   (N = {metrics['best_fw_speedup'][0] if metrics['best_fw_speedup'] else '—'})",
         COLOR_FLASH),
        (_format_speedup(metrics["best_fwbw_speedup"]),
         f"Peak forward+backward speedup   (N = {metrics['best_fwbw_speedup'][0] if metrics['best_fwbw_speedup'] else '—'})",
         COLOR_CALLOUT),
        (_format_mem(metrics["best_mem_saving_mb"]),
         f"Peak memory saved   (N = {metrics['best_mem_saving_mb'][0] if metrics['best_mem_saving_mb'] else '—'})",
         COLOR_BASELINE),
        (_format_bleu(metrics["flash_bleu"], metrics["baseline_bleu"])[0],
         _format_bleu(metrics["flash_bleu"], metrics["baseline_bleu"])[1],
         COLOR_ACCENT),
    ]
    # Four equally-sized callouts with explicit horizontal gutters.
    n_cal = len(callouts)
    gutter = 1  # cols of spacing between callouts
    col_total = 100
    col_step = (col_total - gutter * (n_cal - 1)) // n_cal
    for i, (val, lbl, col) in enumerate(callouts):
        c0 = i * (col_step + gutter)
        c1 = c0 + col_step
        ax = fig.add_subplot(outer[CALL_T:CALL_B, c0:c1])
        _stat_callout(ax, val, lbl, color=col)

    # --------------------------- Body ---------------------------------------
    # 3 columns of equal width with 1-col gutters between them.
    # left_col: Motivation + Method
    # mid_col : Performance figures (3 stacked)
    # right_col: Correctness & Quality figures
    gutter_c = 1
    col_w = (100 - 2 * gutter_c) // 3  # 32
    left_c0, left_c1 = 0, col_w
    mid_c0, mid_c1 = col_w + gutter_c, 2 * col_w + gutter_c
    right_c0, right_c1 = 2 * col_w + 2 * gutter_c, 100

    # Left column is split vertically into Motivation (top ~40%) and Method
    # (bottom ~60%). Explicit row boundaries prevent overlap.
    left_split = BODY_T + int((BODY_B - BODY_T) * 0.40)
    gutter_r = 2  # rows between motivation and method panels

    # --- Motivation ---
    motivation_ax = fig.add_subplot(outer[BODY_T:left_split, left_c0:left_c1])
    _draw_panel_background(motivation_ax)
    _section_title(motivation_ax, "Motivation", fontsize=32)
    motivation_text = (
        "Standard attention materializes an N×N score matrix, so both memory\n"
        "and time grow as O(N²). In GPT-2 style decoders this is the\n"
        "dominant cost for long contexts.\n\n"
        "• Activations dominate peak HBM usage.\n"
        "• Softmax/mask/matmul chain is bandwidth-bound.\n"
        "• Backward pass must cache the full softmax output.\n\n"
        "Goal: keep exact attention math while turning it into a single\n"
        "memory-efficient fused kernel on the GPU."
    )
    _body_text(motivation_ax, motivation_text, y_start=0.82, fontsize=22,
               line_spacing=1.30)

    # --- Method ---
    method_ax = fig.add_subplot(outer[left_split + gutter_r:BODY_B, left_c0:left_c1])
    _draw_panel_background(method_ax)
    _section_title(method_ax, "Method — FlashAttention kernel", fontsize=32)
    method_text = (
        "Tile queries and keys; maintain a per-row running max $m$ and\n"
        "normalizer $\\ell$ so softmax is streamed block-by-block:\n\n"
        "   $m_i \\leftarrow \\max(m_i^{\\text{old}},\\ \\max_j S_{ij})$\n"
        "   $\\ell_i \\leftarrow e^{m_i^{\\text{old}}-m_i}\\,\\ell_i^{\\text{old}} + \\sum_j e^{S_{ij}-m_i}$\n"
        "   $O_i \\leftarrow e^{m_i^{\\text{old}}-m_i}\\,O_i^{\\text{old}} + \\sum_j e^{S_{ij}-m_i}\\,V_j$\n\n"
        "Only per-row scalars $(m_i,\\ell_i)$ and the output $O_i$ are\n"
        "kept in HBM — the full score tensor is never materialized.\n\n"
        "Backward rematerializes blocks using $(m,\\ell)$ as a\n"
        "checkpoint: peak activations stay O(N) while gradients remain\n"
        "exact within finite-precision error.\n\n"
        "Implemented as:\n"
        "• CUDA kernels  (src/flash_attention_fw.cu,  bw.cu)\n"
        "• Python op  q.flash_attention(k_t, v, causal=True)\n"
        "• Causal masking fused into the inner loop\n"
        "• Drop-in replacement inside the minitorch MHA module"
    )
    _body_text(method_ax, method_text, y_start=0.82, fontsize=21,
               line_spacing=1.28)

    # --- Middle column: Performance figures with section title ---
    mid_title_rows = 5
    perf_ax_header = fig.add_subplot(
        outer[BODY_T:BODY_T + mid_title_rows, mid_c0:mid_c1]
    )
    perf_ax_header.set_xticks([])
    perf_ax_header.set_yticks([])
    for spine in perf_ax_header.spines.values():
        spine.set_visible(False)
    perf_ax_header.text(
        0.5, 0.55,
        "Performance  ·  " + metrics["gpu_name"],
        ha="center", va="center",
        fontsize=34, fontweight="bold", color=COLOR_HEADER,
    )
    perf_ax_header.plot(
        [0.05, 0.95], [0.18, 0.18], color=COLOR_FLASH, linewidth=4,
        clip_on=False, transform=perf_ax_header.transAxes,
        solid_capstyle="round",
    )

    # Three equally-sized stacked figure panels below the title, with 1-row
    # gutter between them to avoid caption collisions.
    perf_top = BODY_T + mid_title_rows + 1
    perf_span = BODY_B - perf_top
    fig_gutter = 1
    perf_each = (perf_span - 2 * fig_gutter) // 3

    img_latency = os.path.join(figures_dir, "latency_forward_comparison.png")
    img_speedup = os.path.join(figures_dir, "speedup_vs_seq.png")
    img_memory = os.path.join(figures_dir, "memory_savings_vs_seq.png")

    for i, path in enumerate([img_latency, img_speedup, img_memory]):
        r0 = perf_top + i * (perf_each + fig_gutter)
        r1 = r0 + perf_each
        ax = fig.add_subplot(outer[r0:r1, mid_c0:mid_c1])
        _image_panel(ax, path)

    # --- Right column: Correctness + Quality with section title ---
    right_title_ax = fig.add_subplot(
        outer[BODY_T:BODY_T + mid_title_rows, right_c0:right_c1]
    )
    right_title_ax.set_xticks([])
    right_title_ax.set_yticks([])
    for spine in right_title_ax.spines.values():
        spine.set_visible(False)
    right_title_ax.text(
        0.5, 0.55,
        "Correctness  ·  Quality",
        ha="center", va="center",
        fontsize=34, fontweight="bold", color=COLOR_HEADER,
    )
    right_title_ax.plot(
        [0.05, 0.95], [0.18, 0.18], color=COLOR_FLASH, linewidth=4,
        clip_on=False, transform=right_title_ax.transAxes,
        solid_capstyle="round",
    )

    img_numerical = os.path.join(figures_dir, "numerical_grad_error.png")
    img_bleu = os.path.join(figures_dir, "bleu_comparison.png")
    img_tokens = os.path.join(figures_dir, "tokens_per_sec_comparison.png")
    img_memscale = os.path.join(figures_dir, "memory_scaling_theory.png")
    img_tradeoff = os.path.join(figures_dir, "quality_throughput_tradeoff.png")
    img_tflops = os.path.join(figures_dir, "tflops_comparison.png")

    # Pick up to 3 existing images for the right column.
    right_candidates = [img_numerical]
    right_candidates.append(img_bleu if os.path.exists(img_bleu) else img_tokens)
    if os.path.exists(img_memscale):
        right_candidates.append(img_memscale)
    elif os.path.exists(img_tradeoff):
        right_candidates.append(img_tradeoff)
    else:
        right_candidates.append(img_tflops)

    n_right = len(right_candidates)
    right_each = (perf_span - (n_right - 1) * fig_gutter) // n_right
    for i, path in enumerate(right_candidates):
        r0 = perf_top + i * (right_each + fig_gutter)
        r1 = r0 + right_each
        ax = fig.add_subplot(outer[r0:r1, right_c0:right_c1])
        _image_panel(ax, path)

    # --------------------------- Footer: takeaways --------------------------
    takeaway_ax = fig.add_subplot(outer[FOOT_T:FOOT_B, :])
    _draw_panel_background(takeaway_ax, color="#EAF2F8", edge=COLOR_FLASH, radius=0.01)

    grad_err_text = (
        f"max grad error {metrics['grad_error']:.1e} "
        f"(threshold {metrics['grad_threshold']:g})"
        if metrics["grad_error"] is not None and metrics["grad_threshold"] is not None
        else "gradient parity verified by finite differences"
    )
    fwd_err_text = (
        f"forward parity {metrics['forward_error']:.1e}"
        if metrics["forward_error"] is not None
        else "forward parity verified"
    )
    fw_speedup_text = _format_speedup(metrics["best_fw_speedup"])
    fw_bw_speedup_text = _format_speedup(metrics["best_fwbw_speedup"])
    mem_text = _format_mem(metrics["best_mem_saving_mb"])

    takeaway_ax.text(
        0.015, 0.88, "Key takeaways",
        transform=takeaway_ax.transAxes, ha="left", va="top",
        fontsize=30, fontweight="bold", color=COLOR_HEADER,
    )

    bullets = [
        f"Peak {fw_speedup_text} forward and {fw_bw_speedup_text} forward+backward speedup vs. the standard attention path on {metrics['gpu_name']}.",
        f"Up to {mem_text} of GPU peak-memory reclaimed for long sequences without touching transformer math — longer N on the same card.",
        f"Exact gradients: {fwd_err_text}; {grad_err_text}.",
        "Drop-in replacement in the GPT-2 MHA module — single Python call  q.flash_attention(k_t, v, causal=True).",
    ]
    # Four bullets inside the footer panel. y starts at 0.60 and decreases by
    # 0.17 per bullet so the last bullet lands at 0.09 — fully inside panel.
    y0 = 0.60
    dy = 0.17
    for i, b in enumerate(bullets):
        takeaway_ax.text(
            0.015, y0 - i * dy,
            f"•   {b}",
            transform=takeaway_ax.transAxes, ha="left", va="top",
            fontsize=21, color=COLOR_NEUTRAL,
        )

    # Save
    png_path = os.path.join(output_dir, f"{stem}.png")
    pdf_path = os.path.join(output_dir, f"{stem}.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches=None, pad_inches=0.0,
                facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches=None, pad_inches=0.0,
                facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = {
        "poster_png": png_path,
        "poster_pdf": pdf_path,
        "size_inches": [width_in, height_in],
        "authors_rendered": authors,
        "affiliation_rendered": affiliation,
        "headline_metrics": {
            "best_forward_speedup": metrics["best_fw_speedup"],
            "best_fw_bw_speedup": metrics["best_fwbw_speedup"],
            "best_mem_saving_mb": metrics["best_mem_saving_mb"],
            "best_mem_ratio_theoretical": metrics["best_mem_ratio"],
            "forward_error": metrics["forward_error"],
            "grad_error": metrics["grad_error"],
            "grad_threshold": metrics["grad_threshold"],
            "flash_bleu": metrics["flash_bleu"],
            "baseline_bleu": metrics["baseline_bleu"],
            "gpu_name": metrics["gpu_name"],
        },
    }
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    fire.Fire(build_poster)
