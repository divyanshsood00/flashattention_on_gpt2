"""Collect reproducible FlashAttention metrics (correctness, GPU benchmarks, BLEU).

The ``all`` subcommand writes structured JSON with ``report_schema`` and
``executive_summary``, and optionally a Markdown sibling (``write_markdown``).
Enable ``print_summary`` (default) for a readable ASCII overview on stdout.
"""

import json
import os
import textwrap
import time
import traceback
from ast import literal_eval
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import fire
import numpy as np

import minitorch

# NOTE: CUDA / pycuda initialization can fail in environments without a working GPU.
# To keep `python -m project.presentation_report --help` and other non-GPU tasks usable,
# we import CUDA-backed ops lazily inside the functions that actually need them.


datatype = np.float32

REPORT_SCHEMA_NAME = "presentation_report"
REPORT_SCHEMA_VERSION = "1.2"


@dataclass
class NumericalConfig:
    batch_size: int = 1
    nhead: int = 2
    seq_len: int = 8
    head_dim: int = 16
    finite_diff_samples: int = 12
    finite_diff_eps: float = 1e-3
    seed: int = 7


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _as_grid_values(value: Any) -> List[Any]:
    """Accept Fire's tuple coercion, JSON-ish lists, or comma-separated strings."""
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    except (SyntaxError, ValueError):
        pass
    text = text.strip("[]()")
    return [x.strip() for x in text.split(",") if x.strip()]


def _int_grid(value: Any) -> List[int]:
    return [int(x) for x in _as_grid_values(value)]


def _float_grid(value: Any) -> List[float]:
    return [float(x) for x in _as_grid_values(value)]


def _bool_grid(value: Any) -> List[bool]:
    return [_to_bool(x) for x in _as_grid_values(value)]


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)
            f.write("\n")


def _unwrap_step_result(payload: Any) -> Any:
    """If payload is a _safe_call envelope, return inner result; else payload."""
    if isinstance(payload, dict) and payload.get("status") == "ok" and "result" in payload:
        return payload["result"]
    return payload


def _step_status(payload: Any) -> str:
    if isinstance(payload, dict) and "status" in payload:
        return str(payload.get("status", "unknown"))
    return "ok" if payload is not None else "missing"


def _step_duration(payload: Any) -> Optional[float]:
    if isinstance(payload, dict) and "duration_sec" in payload:
        return float(payload["duration_sec"])
    return None


def _format_duration(sec: Optional[float]) -> str:
    if sec is None:
        return "—"
    if sec < 60:
        return f"{sec:.1f}s"
    return f"{int(sec // 60)}m {sec % 60:.0f}s"


def _iso_utc(epoch_sec: float) -> str:
    return datetime.fromtimestamp(epoch_sec, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _numerical_one_liner(inner: Any) -> str:
    inner = _unwrap_step_result(inner) if inner is not None else None
    if not isinstance(inner, dict):
        return "No numerical result."
    fwd_ok = inner.get("forward_pass")
    fd = inner.get("finite_difference") or {}
    g_ok = fd.get("pass")
    fe = inner.get("forward_max_abs_error")
    ge = fd.get("max_abs_error")
    parts = []
    if fe is not None:
        parts.append(f"forward max |err| = {fe:.3e} ({'PASS' if fwd_ok else 'FAIL'})")
    if ge is not None:
        parts.append(f"grad max |err| = {ge:.3e} ({'PASS' if g_ok else 'FAIL'})")
    return "; ".join(parts) if parts else "Numerical section present."


def _performance_one_liner(inner: Any) -> str:
    inner = _unwrap_step_result(inner) if inner is not None else None
    if not isinstance(inner, dict):
        return "No performance result."
    rows = inner.get("results") or []
    ok_rows = [r for r in rows if isinstance(r, dict) and r.get("status", "ok") == "ok"]
    if not ok_rows:
        return "No successful benchmark rows."
    best = max(
        (r for r in ok_rows if r.get("forward_speedup") is not None),
        key=lambda r: float(r["forward_speedup"]),
        default=None,
    )
    if best is None:
        return f"{len(ok_rows)} workload(s) reported."
    n = best.get("shape", {}).get("N")
    sp = best.get("forward_speedup")
    return f"Peak forward speedup ×{sp:.2f} at N={n} ({len(ok_rows)} configs)."


def _bleu_one_liner(step: Any, label: str) -> str:
    inner = _unwrap_step_result(step) if step is not None else None
    if not isinstance(inner, dict):
        return f"{label}: no result."
    b = inner.get("bleu") or {}
    score = b.get("bleu")
    if score is None:
        return f"{label}: run finished (no BLEU score)."
    return f"{label}: BLEU = {float(score):.4f} (n={b.get('n_samples', '?')})."


def _build_executive_summary(master: Dict[str, Any]) -> Dict[str, Any]:
    """Compact, human-oriented view of the full report (also drives Markdown)."""
    gen = master.get("generated_at_epoch_sec")
    gen_iso = _iso_utc(float(gen)) if isinstance(gen, (int, float)) else None

    at_a_glance: List[str] = []

    num = master.get("numerical_correctness")
    num_st = _step_status(num)
    at_a_glance.append(f"Numerical correctness ({num_st}): {_numerical_one_liner(num)}")

    parity = master.get("parity_suite")
    if parity is not None:
        p_in = _unwrap_step_result(parity)
        if isinstance(p_in, dict):
            fwd = p_in.get("forward_max_abs_error_overall")
            grd = p_in.get("grad_max_abs_error_overall")
            at_a_glance.append(
                "Parity suite "
                f"({_step_status(parity)}): "
                f"forward max |err| = {float(fwd):.3e}, grad max |err| = {float(grd):.3e}"
                if (fwd is not None and grd is not None)
                else f"Parity suite ({_step_status(parity)})."
            )

    perf = master.get("performance_standard") or master.get("performance")
    perf_key = "performance_standard" if master.get("performance_standard") is not None else "performance"
    perf_st = _step_status(perf)
    at_a_glance.append(f"Performance ({perf_st}): {_performance_one_liner(perf)}")

    scaling = master.get("scaling_curves")
    if scaling is not None:
        sc_st = _step_status(scaling)
        at_a_glance.append(f"Scaling curves ({sc_st}): {_performance_one_liner(scaling)}")

    bf = master.get("bleu_flash")
    bb = master.get("bleu_baseline")
    legacy_bleu = master.get("bleu")
    if bf is not None or bb is not None:
        at_a_glance.append(
            f"BLEU flash ({_step_status(bf)}): {_bleu_one_liner(bf, 'Flash')}"
        )
        at_a_glance.append(
            f"BLEU baseline ({_step_status(bb)}): {_bleu_one_liner(bb, 'Standard')}"
        )
    elif legacy_bleu is not None:
        at_a_glance.append(
            f"BLEU ({_step_status(legacy_bleu)}): {_bleu_one_liner(legacy_bleu, 'Run')}"
        )

    ab = master.get("ablation_comparison")
    if isinstance(ab, dict) and ab.get("bleu_delta_flash_minus_baseline") is not None:
        d = ab["bleu_delta_flash_minus_baseline"]
        at_a_glance.append(f"Ablation: ΔBLEU (flash − baseline) = {float(d):+.4f}")

    tr = master.get("throughput_tradeoff")
    if tr is not None:
        tr_in = _unwrap_step_result(tr)
        grid = (tr_in or {}).get("grid") if isinstance(tr_in, dict) else None
        n = len(grid) if isinstance(grid, list) else 0
        at_a_glance.append(f"Throughput tradeoff ({_step_status(tr)}): {n} grid point(s).")

    section_defs: List[Tuple[str, str, Any]] = [
        ("numerical_correctness", "Numerical correctness (FP32 + finite differences)", num),
        ("parity_suite", "Parity suite (flash vs baseline; causal + non-causal)", parity),
        (perf_key, "Microbenchmarks (latency, memory, TFLOP/s)", perf),
        ("scaling_curves", "Sequence-length scaling sweep", scaling),
        ("bleu_flash", "End-to-end BLEU — FlashAttention", bf),
        ("bleu_baseline", "End-to-end BLEU — standard attention", bb),
        ("bleu", "End-to-end BLEU (single run)", legacy_bleu if bf is None and bb is None else None),
        ("ablation_comparison", "Matched ablation (flash vs baseline)", ab),
        ("throughput_tradeoff", "Quality vs throughput grid", tr),
    ]

    sections: List[Dict[str, Any]] = []
    for sec_id, title, payload in section_defs:
        if payload is None:
            continue
        dur = _step_duration(payload)
        headline = ""
        if sec_id == "numerical_correctness":
            headline = _numerical_one_liner(payload)
        elif sec_id == "parity_suite":
            p_in = _unwrap_step_result(payload)
            if isinstance(p_in, dict):
                fwd = p_in.get("forward_max_abs_error_overall")
                grd = p_in.get("grad_max_abs_error_overall")
                headline = (
                    f"forward max |err|={float(fwd):.3e}, grad max |err|={float(grd):.3e}"
                    if (fwd is not None and grd is not None)
                    else "Forward+grad parity summary"
                )
        elif sec_id in ("performance_standard", "performance", "scaling_curves"):
            headline = _performance_one_liner(payload)
        elif sec_id == "bleu_flash":
            headline = _bleu_one_liner(payload, "Flash")
        elif sec_id == "bleu_baseline":
            headline = _bleu_one_liner(payload, "Standard")
        elif sec_id == "bleu":
            headline = _bleu_one_liner(payload, "Run")
        elif sec_id == "ablation_comparison" and isinstance(payload, dict):
            headline = (
                f"ΔBLEU = {payload.get('bleu_delta_flash_minus_baseline')}"
                if payload.get("bleu_delta_flash_minus_baseline") is not None
                else "Ablation summary"
            )
        elif sec_id == "throughput_tradeoff":
            tr_in = _unwrap_step_result(payload)
            grid = (tr_in or {}).get("grid") if isinstance(tr_in, dict) else []
            headline = f"{len(grid)} sweep configurations" if isinstance(grid, list) else "Tradeoff sweep"

        sections.append(
            {
                "id": sec_id,
                "title": title,
                "status": _step_status(payload),
                "duration_sec": dur,
                "duration_human": _format_duration(dur),
                "headline": headline,
            }
        )

    return {
        "document_title": "FlashAttention integration — presentation report",
        "generated_at_epoch_sec": gen,
        "generated_at_utc": gen_iso,
        "how_to_read": (
            "In the JSON file, `report_schema` and `executive_summary` appear first. "
            "Use `at_a_glance` and `sections` for a one-screen overview. "
            "Payload keys (`numerical_correctness`, `performance_standard`, …) may hold "
            "either a raw result dict or a {status, duration_sec, result|error} envelope; "
            "plotting tools unwrap envelopes automatically."
        ),
        "at_a_glance": at_a_glance,
        "sections": sections,
    }


def _write_markdown_report(path: str, master: Dict[str, Any], executive: Dict[str, Any]) -> None:
    if not path:
        return
    lines: List[str] = []
    lines.append("# FlashAttention — presentation report\n")
    if executive.get("generated_at_utc"):
        lines.append(f"**Generated (UTC):** {executive['generated_at_utc']}\n")
    lines.append("## Executive summary\n")
    for item in executive.get("at_a_glance") or []:
        lines.append(f"- {item}\n")
    htr = executive.get("how_to_read")
    if htr:
        lines.append("\n> **How to read this report:** " + str(htr).replace("\n", " ") + "\n")
    lines.append("\n## Section index\n\n")
    lines.append("| Section | Status | Duration | Summary |\n")
    lines.append("|---------|--------|----------|--------|\n")
    for s in executive.get("sections") or []:
        hid = str(s.get("id", "")).replace("|", "\\|")
        st = str(s.get("status", ""))
        du = str(s.get("duration_human", "—"))
        hl = str(s.get("headline", "")).replace("|", "\\|")[:120]
        lines.append(f"| {hid} | {st} | {du} | {hl} |\n")
    notes = master.get("notes") or {}
    if notes:
        lines.append("\n## Notes\n\n")
        for k, v in notes.items():
            lines.append(f"- **{k}:** {v}\n")
    lines.append(
        "\n## Full JSON\n\n"
        "Machine-readable data (numerical tensors, benchmark tables, BLEU payloads) "
        "is in the companion `.json` file with the same basename.\n"
    )
    body = "".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


def _preserve_report_meta(output_json: str, master: Dict[str, Any]) -> None:
    """If an on-disk report already has ``report_meta``, keep it (poster / print metadata)."""
    if not output_json or not os.path.isfile(output_json):
        return
    try:
        with open(output_json, "r", encoding="utf-8") as f:
            prior = json.load(f)
        meta = prior.get("report_meta")
        if isinstance(meta, dict) and meta:
            master["report_meta"] = meta
    except (OSError, json.JSONDecodeError, TypeError):
        pass


def _print_executive_banner(executive: Dict[str, Any], json_path: str) -> None:
    """Readable stdout summary after `all` completes."""
    width = 76
    bar = "=" * width
    print(bar)
    print(executive.get("document_title", "Presentation report").center(width))
    print(bar)
    if executive.get("generated_at_utc"):
        print(f"  UTC: {executive['generated_at_utc']}")
    if json_path:
        print(f"  JSON: {json_path}")
    print()
    print("  At a glance")
    print("  " + "-" * (width - 4))
    for item in executive.get("at_a_glance") or []:
        wrapped = textwrap.wrap(str(item), width=width - 6)
        for i, line in enumerate(wrapped):
            prefix = "  • " if i == 0 else "    "
            print(prefix + line)
    print()
    print("  Sections")
    print("  " + "-" * (width - 4))
    for s in executive.get("sections") or []:
        st = s.get("status", "?")
        du = s.get("duration_human", "—")
        sid = s.get("id", "")
        print(f"    [{st:5}] {du:>8}  {sid}")
        hl = s.get("headline", "")
        if hl:
            for line in textwrap.wrap(
                str(hl),
                width=width - 10,
                initial_indent="          ",
                subsequent_indent="          ",
            ):
                print(line)
    print(bar)


def _safe_call(step_name: str, fn, **kwargs) -> Dict[str, Any]:
    start = time.time()
    try:
        result = fn(**kwargs)
        return {
            "step": step_name,
            "status": "ok",
            "duration_sec": time.time() - start,
            "result": result,
        }
    except Exception as exc:
        return {
            "step": step_name,
            "status": "error",
            "duration_sec": time.time() - start,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _causal_mask(batch_size: int, nhead: int, seq_len: int, backend):
    mask = -np.finfo(datatype).max * np.triu(
        np.ones((batch_size, nhead, seq_len, seq_len), dtype=datatype),
        1,
    )
    return minitorch.tensor_from_numpy(mask, backend=backend, requires_grad=False)


def _maybe_causal_mask(batch_size: int, nhead: int, seq_len: int, backend, causal: bool):
    return _causal_mask(batch_size, nhead, seq_len, backend) if causal else None


def _baseline_attention(q, k_t, v, head_dim: int, backend):
    mask = _causal_mask(q.shape[0], q.shape[1], q.shape[2], backend)
    scores = (q @ k_t) / np.sqrt(head_dim)
    probs = minitorch.nn.softmax(scores + mask, dim=3)
    return probs @ v


def _baseline_attention_mode(q, k_t, v, head_dim: int, backend, causal: bool):
    scores = (q @ k_t) / np.sqrt(head_dim)
    mask = _maybe_causal_mask(q.shape[0], q.shape[1], q.shape[2], backend, causal=causal)
    if mask is not None:
        scores = scores + mask
    probs = minitorch.nn.softmax(scores, dim=3)
    return probs @ v


def _build_tensors(q_np, k_t_np, v_np, backend, requires_grad: bool):
    q = minitorch.tensor_from_numpy(q_np.astype(datatype), backend=backend, requires_grad=requires_grad)
    k_t = minitorch.tensor_from_numpy(k_t_np.astype(datatype), backend=backend, requires_grad=requires_grad)
    v = minitorch.tensor_from_numpy(v_np.astype(datatype), backend=backend, requires_grad=requires_grad)
    return q, k_t, v


def _flash_scalar_loss(q_np, k_t_np, v_np, upstream_np, backend, causal: bool):
    q, k_t, v = _build_tensors(q_np, k_t_np, v_np, backend=backend, requires_grad=True)
    upstream = minitorch.tensor_from_numpy(upstream_np.astype(datatype), backend=backend, requires_grad=False)
    out = q.flash_attention(k_t, v, causal=causal)
    loss = (out * upstream).sum()
    return loss, q, k_t, v


def _baseline_scalar_loss(q_np, k_t_np, v_np, upstream_np, backend, head_dim: int, causal: bool):
    q, k_t, v = _build_tensors(q_np, k_t_np, v_np, backend=backend, requires_grad=True)
    upstream = minitorch.tensor_from_numpy(upstream_np.astype(datatype), backend=backend, requires_grad=False)
    out = _baseline_attention_mode(q, k_t, v, head_dim=head_dim, backend=backend, causal=causal)
    loss = (out * upstream).sum()
    return loss, q, k_t, v


def _finite_diff_component(
    q_np,
    k_t_np,
    v_np,
    upstream_np,
    backend,
    which: str,
    flat_index: int,
    eps: float,
):
    q_pos = q_np.copy()
    q_neg = q_np.copy()
    k_pos = k_t_np.copy()
    k_neg = k_t_np.copy()
    v_pos = v_np.copy()
    v_neg = v_np.copy()

    if which == "q":
        q_pos.flat[flat_index] += eps
        q_neg.flat[flat_index] -= eps
    elif which == "k_t":
        k_pos.flat[flat_index] += eps
        k_neg.flat[flat_index] -= eps
    elif which == "v":
        v_pos.flat[flat_index] += eps
        v_neg.flat[flat_index] -= eps
    else:
        raise ValueError(f"Unknown tensor key '{which}'")

    loss_pos, _, _, _ = _flash_scalar_loss(q_pos, k_pos, v_pos, upstream_np, backend, causal=True)
    loss_neg, _, _, _ = _flash_scalar_loss(q_neg, k_neg, v_neg, upstream_np, backend, causal=True)
    return (loss_pos.item() - loss_neg.item()) / (2.0 * eps)


def run_numerical_correctness(
    batch_size: int = 1,
    nhead: int = 2,
    seq_len: int = 8,
    head_dim: int = 16,
    finite_diff_samples: int = 12,
    finite_diff_eps: float = 1e-3,
    seed: int = 7,
    forward_abs_error_threshold: float = 1e-3,
    grad_abs_error_threshold: float = 2e-2,
):
    from minitorch.cuda_kernel_ops import CudaKernelOps

    backend = minitorch.TensorBackend(CudaKernelOps)
    rng = np.random.default_rng(seed)

    q_np = rng.standard_normal((batch_size, nhead, seq_len, head_dim), dtype=datatype)
    k_np = rng.standard_normal((batch_size, nhead, seq_len, head_dim), dtype=datatype)
    v_np = rng.standard_normal((batch_size, nhead, seq_len, head_dim), dtype=datatype)
    k_t_np = np.transpose(k_np, (0, 1, 3, 2)).copy()

    q_fw, k_t_fw, v_fw = _build_tensors(q_np, k_t_np, v_np, backend=backend, requires_grad=True)
    out_flash = q_fw.flash_attention(k_t_fw, v_fw, causal=True)
    out_ref = _baseline_attention(q_fw, k_t_fw, v_fw, head_dim=head_dim, backend=backend)

    max_abs_error = float(np.max(np.abs(out_flash.to_numpy() - out_ref.to_numpy())))

    upstream_np = rng.standard_normal((batch_size, nhead, seq_len, head_dim), dtype=datatype)
    loss, q, k_t, v = _flash_scalar_loss(q_np, k_t_np, v_np, upstream_np, backend, causal=True)
    loss.backward()

    analytic = {
        "q": q.grad.to_numpy(),
        "k_t": k_t.grad.to_numpy(),
        "v": v.grad.to_numpy(),
    }

    finite_diff_stats: Dict[str, Dict[str, float]] = {}
    for key, arr in [("q", q_np), ("k_t", k_t_np), ("v", v_np)]:
        total = arr.size
        n_samp = min(total, int(finite_diff_samples))
        sample_idx = rng.choice(total, size=n_samp, replace=False)

        errors = []
        for idx in sample_idx:
            grad_num = _finite_diff_component(
                q_np=q_np,
                k_t_np=k_t_np,
                v_np=v_np,
                upstream_np=upstream_np,
                backend=backend,
                which=key,
                flat_index=int(idx),
                eps=float(finite_diff_eps),
            )
            grad_ana = float(analytic[key].flat[int(idx)])
            errors.append(abs(grad_num - grad_ana))

        finite_diff_stats[key] = {
            "max_abs_error": float(np.max(errors)),
            "mean_abs_error": float(np.mean(errors)),
            "n_samples": int(n_samp),
        }

    max_grad_error = max(v["max_abs_error"] for v in finite_diff_stats.values())

    result = {
        "task": "numerical_correctness",
        "dtype": "fp32",
        "shape": {
            "batch_size": int(batch_size),
            "nhead": int(nhead),
            "seq_len": int(seq_len),
            "head_dim": int(head_dim),
        },
        "forward_max_abs_error": max_abs_error,
        "forward_threshold": float(forward_abs_error_threshold),
        "forward_pass": bool(max_abs_error <= forward_abs_error_threshold),
        "finite_difference": {
            "eps": float(finite_diff_eps),
            "per_tensor": finite_diff_stats,
            "max_abs_error": float(max_grad_error),
            "threshold": float(grad_abs_error_threshold),
            "pass": bool(max_grad_error <= grad_abs_error_threshold),
        },
    }
    return result


def run_parity_suite(
    batch_size: int = 1,
    nhead: int = 2,
    head_dim: int = 16,
    seq_lens: str = "8,16,32",
    causal_modes: str = "true,false",
    seeds: str = "7,11",
    forward_abs_error_threshold: float = 1e-3,
    grad_abs_error_threshold: float = 2e-2,
):
    """
    Extended correctness checks beyond finite differences:

    - Forward parity: FlashAttention vs baseline attention (causal and non-causal)
    - Gradient parity: dL/dq, dL/dk_t, dL/dv for the same scalar loss
    """
    from minitorch.cuda_kernel_ops import CudaKernelOps

    backend = minitorch.TensorBackend(CudaKernelOps)
    seq_list = _int_grid(seq_lens)
    seed_list = _int_grid(seeds)
    causal_list = _bool_grid(causal_modes)

    rows: List[Dict[str, Any]] = []
    for seed in seed_list:
        rng = np.random.default_rng(seed)
        for seq_len in seq_list:
            q_np = rng.standard_normal((batch_size, nhead, seq_len, head_dim), dtype=datatype)
            k_np = rng.standard_normal((batch_size, nhead, seq_len, head_dim), dtype=datatype)
            v_np = rng.standard_normal((batch_size, nhead, seq_len, head_dim), dtype=datatype)
            k_t_np = np.transpose(k_np, (0, 1, 3, 2)).copy()
            upstream_np = rng.standard_normal((batch_size, nhead, seq_len, head_dim), dtype=datatype)

            for causal in causal_list:
                q_fw, k_t_fw, v_fw = _build_tensors(q_np, k_t_np, v_np, backend=backend, requires_grad=False)
                out_flash = q_fw.flash_attention(k_t_fw, v_fw, causal=causal)
                out_ref = _baseline_attention_mode(
                    q_fw, k_t_fw, v_fw, head_dim=head_dim, backend=backend, causal=causal
                )
                fwd_err = float(np.max(np.abs(out_flash.to_numpy() - out_ref.to_numpy())))

                loss_f, q_f, k_f, v_f = _flash_scalar_loss(
                    q_np, k_t_np, v_np, upstream_np, backend=backend, causal=causal
                )
                loss_b, q_b, k_b, v_b = _baseline_scalar_loss(
                    q_np, k_t_np, v_np, upstream_np, backend=backend, head_dim=head_dim, causal=causal
                )
                loss_f.backward()
                loss_b.backward()

                dq_err = float(np.max(np.abs(q_f.grad.to_numpy() - q_b.grad.to_numpy())))
                dk_err = float(np.max(np.abs(k_f.grad.to_numpy() - k_b.grad.to_numpy())))
                dv_err = float(np.max(np.abs(v_f.grad.to_numpy() - v_b.grad.to_numpy())))
                grad_err = float(max(dq_err, dk_err, dv_err))

                rows.append(
                    {
                        "seed": int(seed),
                        "causal": bool(causal),
                        "shape": {
                            "batch_size": int(batch_size),
                            "nhead": int(nhead),
                            "seq_len": int(seq_len),
                            "head_dim": int(head_dim),
                        },
                        "forward_max_abs_error": fwd_err,
                        "forward_threshold": float(forward_abs_error_threshold),
                        "forward_pass": bool(fwd_err <= forward_abs_error_threshold),
                        "grad_max_abs_error": grad_err,
                        "grad_threshold": float(grad_abs_error_threshold),
                        "grad_pass": bool(grad_err <= grad_abs_error_threshold),
                        "per_tensor_grad_max_abs_error": {
                            "q": dq_err,
                            "k_t": dk_err,
                            "v": dv_err,
                        },
                    }
                )

    report = {
        "task": "parity_suite",
        "dtype": "fp32",
        "n_rows": int(len(rows)),
        "forward_max_abs_error_overall": float(max(r["forward_max_abs_error"] for r in rows)) if rows else None,
        "grad_max_abs_error_overall": float(max(r["grad_max_abs_error"] for r in rows)) if rows else None,
        "n_forward_pass": int(sum(1 for r in rows if r.get("forward_pass"))),
        "n_grad_pass": int(sum(1 for r in rows if r.get("grad_pass"))),
        "rows": rows,
    }
    return report


class PresentationReport:
    def numerical(self, **kwargs):
        """Runs numerical correctness tests."""
        output_json = kwargs.pop("output_json", "")
        result = run_numerical_correctness(**kwargs)
        print(json.dumps(result, indent=2))
        _write_json(output_json, result)
        return result

    def parity_suite(self, **kwargs):
        """Runs forward+gradient parity (flash vs baseline), causal + non-causal."""
        output_json = kwargs.pop("output_json", "")
        result = run_parity_suite(**kwargs)
        print(json.dumps(result, indent=2))
        _write_json(output_json, result)
        return result

    def performance(self, configs: str = "hw3_default,medium,long,xlong", **kwargs):
        """Runs performance benchmarks with memory-budget aware behavior."""
        output_json = kwargs.pop("output_json", "")
        warmup = kwargs.get("warmup", 5)
        iters = kwargs.get("iters", 30)
        seed = kwargs.get("seed", 7)
        gpu_index = kwargs.get("gpu_index", 0)
        measure_gpu_memory = kwargs.get("measure_gpu_memory", True)
        memory_sample_interval_ms = kwargs.get("memory_sample_interval_ms", 20)
        memory_budget_mb = kwargs.get("memory_budget_mb", 3900.0)
        auto_shrink_to_budget = kwargs.get("auto_shrink_to_budget", True)
        skip_on_budget_exceed = kwargs.get("skip_on_budget_exceed", True)
        strict_configs = kwargs.get("strict_configs", False)

        from project import benchmark_flash_attention as flash_benchmark

        result = flash_benchmark.main(
            warmup=warmup,
            iters=iters,
            seed=seed,
            configs=configs,
            gpu_index=gpu_index,
            measure_gpu_memory=measure_gpu_memory,
            memory_sample_interval_ms=memory_sample_interval_ms,
            memory_budget_mb=memory_budget_mb,
            auto_shrink_to_budget=auto_shrink_to_budget,
            skip_on_budget_exceed=skip_on_budget_exceed,
            strict_configs=strict_configs,
            output_json=output_json,
        )
        return result

    def bleu(self, **kwargs):
        """Runs the machine translation pipeline and evaluates BLEU."""
        output_json = kwargs.pop("output_json", "")
        from project import run_machine_translation as mt

        result = mt.main(
            dataset_name=kwargs.get("dataset_name", "bbaaaa/iwslt14-de-en-preprocess"),
            model_max_length=kwargs.get("model_max_length", 40),
            n_epochs=kwargs.get("n_epochs", 1),
            batch_size=kwargs.get("batch_size", 128),
            learning_rate=kwargs.get("learning_rate", 0.002),
            samples_per_epoch=kwargs.get("samples_per_epoch", 20000),
            n_vocab=kwargs.get("n_vocab", 10000),
            n_embd=kwargs.get("n_embd", 256),
            seed=kwargs.get("seed", 11111),
            use_fused_kernel=kwargs.get("use_fused_kernel", False),
            use_flash_attention=kwargs.get("use_flash_attention", None),
            attention_dropout=kwargs.get("attention_dropout", 0.1),
            eval_bleu=True,
            bleu_split=kwargs.get("bleu_split", "test"),
            bleu_samples=kwargs.get("bleu_samples", 100),
        )
        _write_json(output_json, result)
        return result

    def baseline_bleu(self, **kwargs):
        """Runs the exact same BLEU pipeline as `bleu` but with standard attention."""
        kwargs["use_flash_attention"] = False
        return self.bleu(**kwargs)

    def scaling_curves(self, output_json: str = "scaling_curves.json", **kwargs):
        configs_str = kwargs.pop("configs", "seq_128,seq_256,seq_512,seq_1024,seq_2048")
        print(f"Running scaling curves: {configs_str}")
        return self.performance(configs=configs_str, output_json=output_json, **kwargs)

    def ablation(self, output_json: str = "ablation_table.json", **kwargs):
        print("\n--- Running with Flash Attention ---")
        flash_payload = _safe_call("bleu_flash", self.bleu, use_flash_attention=True, **kwargs)

        print("\n--- Running Baseline (Standard Attention) ---")
        base_payload = _safe_call("bleu_baseline", self.baseline_bleu, **kwargs)

        flash_bleu = None
        base_bleu = None
        if flash_payload["status"] == "ok":
            flash_bleu = flash_payload["result"].get("bleu", {}).get("bleu")
        if base_payload["status"] == "ok":
            base_bleu = base_payload["result"].get("bleu", {}).get("bleu")

        report = {
            "task": "ablation",
            "flash": flash_payload,
            "baseline": base_payload,
            "bleu_delta_flash_minus_baseline": (
                float(flash_bleu - base_bleu)
                if (flash_bleu is not None and base_bleu is not None)
                else None
            ),
        }
        _write_json(output_json, report)
        return report

    def throughput_tradeoff(self, output_json: str = "throughput_tradeoff.json", **kwargs):
        """Runs BLEU + throughput sweeps over epochs/dropout/flash settings."""
        epochs_grid = _int_grid(kwargs.get("epochs_grid", "1,2"))
        dropout_grid = _float_grid(kwargs.get("dropout_grid", "0.0,0.1"))
        flash_grid = _bool_grid(kwargs.get("flash_grid", "true,false"))

        perf_config = kwargs.get("performance_config", "hw3_default")
        perf_warmup = int(kwargs.get("perf_warmup", 1))
        perf_iters = int(kwargs.get("perf_iters", 3))
        memory_budget_mb = float(kwargs.get("memory_budget_mb", 3900.0))

        run_params = {
            "dataset_name": kwargs.get("dataset_name", "bbaaaa/iwslt14-de-en-preprocess"),
            "model_max_length": kwargs.get("model_max_length", 40),
            "batch_size": kwargs.get("batch_size", 128),
            "learning_rate": kwargs.get("learning_rate", 0.002),
            "samples_per_epoch": kwargs.get("samples_per_epoch", 20000),
            "n_vocab": kwargs.get("n_vocab", 10000),
            "n_embd": kwargs.get("n_embd", 256),
            "seed": kwargs.get("seed", 11111),
            "use_fused_kernel": kwargs.get("use_fused_kernel", True),
            "bleu_split": kwargs.get("bleu_split", "test"),
            "bleu_samples": kwargs.get("bleu_samples", 100),
        }

        rows = []
        for epochs in epochs_grid:
            for dropout in dropout_grid:
                for use_flash in flash_grid:
                    print(
                        f"Tradeoff grid: epochs={epochs}, dropout={dropout}, "
                        f"use_flash_attention={use_flash}"
                    )
                    bleu_payload = _safe_call(
                        "bleu",
                        self.bleu,
                        n_epochs=epochs,
                        attention_dropout=dropout,
                        use_flash_attention=use_flash,
                        **run_params,
                    )
                    perf_payload = _safe_call(
                        "performance",
                        self.performance,
                        configs=perf_config,
                        warmup=perf_warmup,
                        iters=perf_iters,
                        memory_budget_mb=memory_budget_mb,
                        output_json="",
                    )

                    bleu_value = None
                    tokens_per_sec = None
                    if bleu_payload["status"] == "ok":
                        bleu_value = bleu_payload["result"].get("bleu", {}).get("bleu")

                    if perf_payload["status"] == "ok":
                        perf_results = perf_payload["result"].get("results", [])
                        perf_ok = next((r for r in perf_results if r.get("status") == "ok"), None)
                        if perf_ok is not None:
                            tokens_per_sec = (
                                perf_ok.get("flash_tokens_per_sec")
                                if use_flash
                                else perf_ok.get("baseline_tokens_per_sec")
                            )

                    rows.append(
                        {
                            "params": {
                                "epochs": epochs,
                                "dropout": dropout,
                                "use_flash_attention": use_flash,
                            },
                            "bleu_value": bleu_value,
                            "tokens_per_sec": tokens_per_sec,
                            "bleu": bleu_payload,
                            "performance": perf_payload,
                        }
                    )

        report = {
            "task": "throughput_tradeoff",
            "grid": rows,
        }
        _write_json(output_json, report)
        return report

    def all(
        self,
        output_json: str = "presentation_report.json",
        numerical_seq_len: int = 8,
        numerical_head_dim: int = 16,
        finite_diff_samples: int = 12,
        include_parity_suite: bool = True,
        parity_seq_lens: str = "8,16,32",
        parity_causal_modes: str = "true,false",
        parity_seeds: str = "7,11",
        perf_configs: str = "hw3_default,medium,long,xlong",
        perf_warmup: int = 5,
        perf_iters: int = 30,
        perf_memory_budget_mb: float = 3900.0,
        scaling_configs: str = "seq_128,seq_256,seq_512,seq_1024,seq_2048",
        gpu_index: int = 0,
        bleu_samples: int = 100,
        bleu_epochs: int = 1,
        bleu_samples_per_epoch: int = 20000,
        use_fused_kernel: bool = True,
        use_flash_attention: bool = True,
        attention_dropout: float = 0.0,
        include_scaling: bool = True,
        include_tradeoff: bool = True,
        include_ablation: bool = True,
        print_summary: bool = True,
        write_markdown: bool = True,
        markdown_path: str = "",
        preserve_report_meta: bool = True,
    ):
        common_bleu_kwargs = {
            "n_epochs": bleu_epochs,
            "samples_per_epoch": bleu_samples_per_epoch,
            "bleu_samples": bleu_samples,
            "use_fused_kernel": use_fused_kernel,
            "attention_dropout": attention_dropout,
        }

        numerical = _safe_call(
            "numerical_correctness",
            self.numerical,
            seq_len=numerical_seq_len,
            head_dim=numerical_head_dim,
            finite_diff_samples=finite_diff_samples,
            output_json="",
        )

        parity_suite = None
        if _to_bool(include_parity_suite):
            parity_suite = _safe_call(
                "parity_suite",
                self.parity_suite,
                seq_lens=parity_seq_lens,
                causal_modes=parity_causal_modes,
                seeds=parity_seeds,
                head_dim=numerical_head_dim,
                output_json="",
            )

        performance = _safe_call(
            "performance_standard",
            self.performance,
            configs=perf_configs,
            warmup=perf_warmup,
            iters=perf_iters,
            gpu_index=gpu_index,
            memory_budget_mb=perf_memory_budget_mb,
            output_json="",
        )

        scaling = None
        if _to_bool(include_scaling):
            scaling = _safe_call(
                "scaling_curves",
                self.scaling_curves,
                configs=scaling_configs,
                warmup=perf_warmup,
                iters=perf_iters,
                gpu_index=gpu_index,
                memory_budget_mb=perf_memory_budget_mb,
                output_json="",
            )

        # Flash BLEU run (configured mode)
        bleu_flash = _safe_call(
            "bleu_flash",
            self.bleu,
            use_flash_attention=use_flash_attention,
            **common_bleu_kwargs,
        )

        # Exact-matched baseline BLEU run
        bleu_baseline = _safe_call(
            "bleu_baseline",
            self.baseline_bleu,
            **common_bleu_kwargs,
        )

        ablation = None
        if _to_bool(include_ablation):
            flash_bleu = None
            base_bleu = None
            if bleu_flash["status"] == "ok":
                flash_bleu = bleu_flash["result"].get("bleu", {}).get("bleu")
            if bleu_baseline["status"] == "ok":
                base_bleu = bleu_baseline["result"].get("bleu", {}).get("bleu")
            ablation = {
                "task": "ablation",
                "flash": bleu_flash,
                "baseline": bleu_baseline,
                "bleu_delta_flash_minus_baseline": (
                    float(flash_bleu - base_bleu)
                    if (flash_bleu is not None and base_bleu is not None)
                    else None
                ),
            }

        tradeoff = None
        if _to_bool(include_tradeoff):
            tradeoff = _safe_call(
                "throughput_tradeoff",
                self.throughput_tradeoff,
                output_json="",
                samples_per_epoch=max(2000, int(bleu_samples_per_epoch // 4)),
                bleu_samples=max(20, min(bleu_samples, 100)),
                memory_budget_mb=perf_memory_budget_mb,
            )

        generated_ts = time.time()
        master_report: Dict[str, Any] = {
            "generated_at_epoch_sec": generated_ts,
            "numerical_correctness": numerical,
            "parity_suite": parity_suite,
            "performance_standard": performance,
            "scaling_curves": scaling,
            "bleu_flash": bleu_flash,
            "bleu_baseline": bleu_baseline,
            "ablation_comparison": ablation,
            "throughput_tradeoff": tradeoff,
            "notes": {
                "memory_budget_mb": perf_memory_budget_mb,
                "standard_benchmark_configs": perf_configs,
                "scaling_configs": scaling_configs if _to_bool(include_scaling) else "(not run)",
                "graceful_failures": "Each section is isolated; errors are captured and remaining sections continue.",
            },
        }

        if _to_bool(preserve_report_meta):
            _preserve_report_meta(output_json, master_report)

        executive = _build_executive_summary(master_report)
        schema_block = {
            "name": REPORT_SCHEMA_NAME,
            "version": REPORT_SCHEMA_VERSION,
            "description": "Structured metrics for FlashAttention correctness, speed, memory, and BLEU.",
        }
        # Re-order so human readers see overview first in the JSON file.
        master_report = {
            "report_schema": schema_block,
            "executive_summary": executive,
            **master_report,
        }

        _write_json(output_json, master_report)

        md_out = markdown_path.strip()
        if not md_out and _to_bool(write_markdown) and output_json.endswith(".json"):
            md_out = output_json[:-5] + ".md"
        if md_out:
            _write_markdown_report(md_out, master_report, executive)

        if _to_bool(print_summary):
            _print_executive_banner(executive, output_json or "")
        elif output_json:
            print(f"Master report saved to {output_json}")
        if output_json and md_out:
            print(f"Markdown summary: {md_out}")
        return master_report


if __name__ == "__main__":
    fire.Fire(PresentationReport)