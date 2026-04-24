import json
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import fire
import numpy as np
import torch

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps


datatype = np.float32


@dataclass
class Workload:
    batch_size: int
    nhead: int
    seq_len: int
    head_dim: int


DEFAULT_WORKLOADS: Dict[str, Workload] = {
    "hw3_default": Workload(batch_size=64, nhead=8, seq_len=128, head_dim=32),
    "medium": Workload(batch_size=32, nhead=8, seq_len=512, head_dim=32),
    "long": Workload(batch_size=16, nhead=8, seq_len=1024, head_dim=64),
    "xlong": Workload(batch_size=8, nhead=8, seq_len=2048, head_dim=64),
}


def _to_tensor(np_arr, backend, requires_grad=False):
    return minitorch.tensor_from_numpy(np_arr.astype(datatype), backend=backend, requires_grad=requires_grad)


def _build_inputs(w: Workload, backend, seed: int, requires_grad: bool):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((w.batch_size, w.nhead, w.seq_len, w.head_dim), dtype=datatype)
    k = rng.standard_normal((w.batch_size, w.nhead, w.seq_len, w.head_dim), dtype=datatype)
    v = rng.standard_normal((w.batch_size, w.nhead, w.seq_len, w.head_dim), dtype=datatype)

    q_t = _to_tensor(q, backend, requires_grad=requires_grad)
    k_t = _to_tensor(np.transpose(k, (0, 1, 3, 2)).copy(), backend, requires_grad=requires_grad)
    v_t = _to_tensor(v, backend, requires_grad=requires_grad)
    return q_t, k_t, v_t


def _causal_mask(w: Workload, backend):
    mask = -np.finfo(datatype).max * np.triu(
        np.ones((w.batch_size, w.nhead, w.seq_len, w.seq_len), dtype=datatype),
        1,
    )
    return _to_tensor(mask, backend, requires_grad=False)


def _baseline_attention(q, k_t, v, mask, head_dim: int):
    scores = (q @ k_t) / np.sqrt(head_dim)
    probs = minitorch.nn.softmax(scores + mask, dim=3)
    return probs @ v


def _flash_attention(q, k_t, v):
    return q.flash_attention(k_t, v, causal=True)


def _bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _query_gpu_memory_used_mb(gpu_index: int) -> float | None:
    """Read current used GPU memory in MB via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if gpu_index < 0 or gpu_index >= len(lines):
            return None
        return float(lines[gpu_index])
    except Exception:
        return None


def _query_gpu_name(gpu_index: int) -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if gpu_index < 0 or gpu_index >= len(lines):
            return None
        return lines[gpu_index]
    except Exception:
        return None


def _bench_with_memory(
    fn,
    warmup: int,
    iters: int,
    gpu_index: int,
    measure_gpu_memory: bool,
    memory_sample_interval_ms: int,
):
    for _ in range(warmup):
        fn()

    times = []
    peak_used_mb = None
    baseline_used_mb = None

    stop_event = threading.Event()
    sample_lock = threading.Lock()

    def _sampler():
        nonlocal peak_used_mb
        interval = max(1, memory_sample_interval_ms) / 1000.0
        while not stop_event.is_set():
            cur = _query_gpu_memory_used_mb(gpu_index)
            if cur is not None:
                with sample_lock:
                    peak_used_mb = cur if peak_used_mb is None else max(peak_used_mb, cur)
            time.sleep(interval)

    sampler_thread = None
    if measure_gpu_memory:
        baseline_used_mb = _query_gpu_memory_used_mb(gpu_index)
        sampler_thread = threading.Thread(target=_sampler, daemon=True)
        sampler_thread.start()

    try:
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    finally:
        if sampler_thread is not None:
            stop_event.set()
            sampler_thread.join(timeout=2.0)

    median_time = float(np.median(times))

    peak_delta_mb = None
    if baseline_used_mb is not None and peak_used_mb is not None:
        peak_delta_mb = max(0.0, float(peak_used_mb - baseline_used_mb))

    return median_time, peak_delta_mb


def _estimate_memory_bytes(w: Workload) -> Tuple[int, int]:
    qkv_o = 4 * w.batch_size * w.nhead * w.seq_len * w.head_dim
    baseline_scores_probs = 2 * w.batch_size * w.nhead * w.seq_len * w.seq_len
    flash_lse = w.batch_size * w.nhead * w.seq_len

    baseline_bytes = (qkv_o + baseline_scores_probs) * 4
    flash_bytes = (qkv_o + flash_lse) * 4
    return baseline_bytes, flash_bytes


def _run_one(
    name: str,
    w: Workload,
    backend,
    warmup: int,
    iters: int,
    seed: int,
    gpu_index: int,
    measure_gpu_memory: bool,
    memory_sample_interval_ms: int,
):
    result = {
        "config": name,
        "shape": {
            "B": w.batch_size,
            "H": w.nhead,
            "N": w.seq_len,
            "d": w.head_dim,
        },
    }

    mask = _causal_mask(w, backend)

    # Forward-only benchmark
    q_b, k_b, v_b = _build_inputs(w, backend, seed=seed, requires_grad=False)
    q_f, k_f, v_f = _build_inputs(w, backend, seed=seed, requires_grad=False)

    try:
        baseline_fw_s, baseline_fw_mem = _bench_with_memory(
            lambda: _baseline_attention(q_b, k_b, v_b, mask, w.head_dim),
            warmup=warmup,
            iters=iters,
            gpu_index=gpu_index,
            measure_gpu_memory=measure_gpu_memory,
            memory_sample_interval_ms=memory_sample_interval_ms,
        )
        result["baseline_forward_ms"] = baseline_fw_s * 1000.0
        if baseline_fw_mem is not None:
            result["baseline_forward_peak_mem_delta_mb"] = baseline_fw_mem
    except Exception as exc:
        result["baseline_forward_error"] = str(exc)
        baseline_fw_s = None

    try:
        flash_fw_s, flash_fw_mem = _bench_with_memory(
            lambda: _flash_attention(q_f, k_f, v_f),
            warmup=warmup,
            iters=iters,
            gpu_index=gpu_index,
            measure_gpu_memory=measure_gpu_memory,
            memory_sample_interval_ms=memory_sample_interval_ms,
        )
        result["flash_forward_ms"] = flash_fw_s * 1000.0
        if flash_fw_mem is not None:
            result["flash_forward_peak_mem_delta_mb"] = flash_fw_mem
    except Exception as exc:
        result["flash_forward_error"] = str(exc)
        flash_fw_s = None

    if baseline_fw_s and flash_fw_s:
        result["forward_speedup"] = baseline_fw_s / flash_fw_s

    # Forward+backward benchmark
    q_b_bw, k_b_bw, v_b_bw = _build_inputs(w, backend, seed=seed + 1, requires_grad=True)
    q_f_bw, k_f_bw, v_f_bw = _build_inputs(w, backend, seed=seed + 1, requires_grad=True)

    def baseline_fw_bw():
        q_b_bw.zero_grad_()
        k_b_bw.zero_grad_()
        v_b_bw.zero_grad_()
        out = _baseline_attention(q_b_bw, k_b_bw, v_b_bw, mask, w.head_dim)
        out.sum().backward()

    def flash_fw_bw():
        q_f_bw.zero_grad_()
        k_f_bw.zero_grad_()
        v_f_bw.zero_grad_()
        out = _flash_attention(q_f_bw, k_f_bw, v_f_bw)
        out.sum().backward()

    try:
        baseline_fw_bw_s, baseline_fw_bw_mem = _bench_with_memory(
            baseline_fw_bw,
            warmup=warmup,
            iters=iters,
            gpu_index=gpu_index,
            measure_gpu_memory=measure_gpu_memory,
            memory_sample_interval_ms=memory_sample_interval_ms,
        )
        result["baseline_fw_bw_ms"] = baseline_fw_bw_s * 1000.0
        if baseline_fw_bw_mem is not None:
            result["baseline_fw_bw_peak_mem_delta_mb"] = baseline_fw_bw_mem
    except Exception as exc:
        result["baseline_fw_bw_error"] = str(exc)
        baseline_fw_bw_s = None

    try:
        flash_fw_bw_s, flash_fw_bw_mem = _bench_with_memory(
            flash_fw_bw,
            warmup=warmup,
            iters=iters,
            gpu_index=gpu_index,
            measure_gpu_memory=measure_gpu_memory,
            memory_sample_interval_ms=memory_sample_interval_ms,
        )
        result["flash_fw_bw_ms"] = flash_fw_bw_s * 1000.0
        if flash_fw_bw_mem is not None:
            result["flash_fw_bw_peak_mem_delta_mb"] = flash_fw_bw_mem
    except Exception as exc:
        result["flash_fw_bw_error"] = str(exc)
        flash_fw_bw_s = None

    if baseline_fw_bw_s and flash_fw_bw_s:
        result["fw_bw_speedup"] = baseline_fw_bw_s / flash_fw_bw_s

    if flash_fw_s:
        flops = 4.0 * w.batch_size * w.nhead * (w.seq_len ** 2) * w.head_dim
        result["flash_tflops"] = flops / flash_fw_s / 1e12
        result["flash_tokens_per_sec"] = (w.batch_size * w.seq_len) / flash_fw_s

    if baseline_fw_s:
        flops = 4.0 * w.batch_size * w.nhead * (w.seq_len ** 2) * w.head_dim
        result["baseline_tflops"] = flops / baseline_fw_s / 1e12
        result["baseline_tokens_per_sec"] = (w.batch_size * w.seq_len) / baseline_fw_s

    baseline_mem, flash_mem = _estimate_memory_bytes(w)
    result["baseline_peak_mem_est_mb"] = baseline_mem / (1024.0 ** 2)
    result["flash_peak_mem_est_mb"] = flash_mem / (1024.0 ** 2)
    if flash_mem > 0:
        result["peak_mem_reduction_ratio_est"] = baseline_mem / flash_mem

    if (
        "baseline_forward_peak_mem_delta_mb" in result
        and "flash_forward_peak_mem_delta_mb" in result
    ):
        result["forward_peak_mem_savings_mb_measured"] = (
            result["baseline_forward_peak_mem_delta_mb"]
            - result["flash_forward_peak_mem_delta_mb"]
        )

    return result


def main(
    warmup: int = 5,
    iters: int = 30,
    seed: int = 7,
    configs: str = "hw3_default,medium,long,xlong",
    output_json: str = "",
    gpu_index: int = 0,
    measure_gpu_memory: bool = True,
    memory_sample_interval_ms: int = 20,
):
    backend = minitorch.TensorBackend(CudaKernelOps)
    selected = [cfg.strip() for cfg in configs.split(",") if cfg.strip()]

    meta = {
        "gpu_index": int(gpu_index),
        "gpu_name": _query_gpu_name(gpu_index),
        "measure_gpu_memory": bool(measure_gpu_memory),
        "memory_sample_interval_ms": int(memory_sample_interval_ms),
    }

    results: List[Dict] = []
    for cfg in selected:
        if cfg not in DEFAULT_WORKLOADS:
            raise ValueError(f"Unknown config '{cfg}'. Available: {list(DEFAULT_WORKLOADS.keys())}")
        print(f"Running benchmark for config={cfg}...")
        results.append(
            _run_one(
                cfg,
                DEFAULT_WORKLOADS[cfg],
                backend,
                warmup=warmup,
                iters=iters,
                seed=seed,
                gpu_index=gpu_index,
                measure_gpu_memory=bool(measure_gpu_memory),
                memory_sample_interval_ms=memory_sample_interval_ms,
            )
        )

    payload = {
        "meta": meta,
        "results": results,
    }

    print(json.dumps(payload, indent=2))

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return payload


if __name__ == "__main__":
    fire.Fire(main)
