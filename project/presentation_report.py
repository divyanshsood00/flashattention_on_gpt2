import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import fire
import numpy as np

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

from project import benchmark_flash_attention as flash_benchmark
from project import run_machine_translation as mt


datatype = np.float32


@dataclass
class NumericalConfig:
    batch_size: int = 1
    nhead: int = 2
    seq_len: int = 8
    head_dim: int = 16
    finite_diff_samples: int = 12
    finite_diff_eps: float = 1e-3
    seed: int = 7


def _causal_mask(batch_size: int, nhead: int, seq_len: int, backend):
    mask = -np.finfo(datatype).max * np.triu(
        np.ones((batch_size, nhead, seq_len, seq_len), dtype=datatype),
        1,
    )
    return minitorch.tensor_from_numpy(mask, backend=backend, requires_grad=False)


def _baseline_attention(q, k_t, v, head_dim: int, backend):
    mask = _causal_mask(q.shape[0], q.shape[1], q.shape[2], backend)
    scores = (q @ k_t) / np.sqrt(head_dim)
    probs = minitorch.nn.softmax(scores + mask, dim=3)
    return probs @ v


def _build_tensors(q_np, k_t_np, v_np, backend, requires_grad: bool):
    q = minitorch.tensor_from_numpy(q_np.astype(datatype), backend=backend, requires_grad=requires_grad)
    k_t = minitorch.tensor_from_numpy(k_t_np.astype(datatype), backend=backend, requires_grad=requires_grad)
    v = minitorch.tensor_from_numpy(v_np.astype(datatype), backend=backend, requires_grad=requires_grad)
    return q, k_t, v


def _flash_scalar_loss(q_np, k_t_np, v_np, upstream_np, backend):
    q, k_t, v = _build_tensors(q_np, k_t_np, v_np, backend=backend, requires_grad=True)
    upstream = minitorch.tensor_from_numpy(upstream_np.astype(datatype), backend=backend, requires_grad=False)
    out = q.flash_attention(k_t, v, causal=True)
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

    loss_pos, _, _, _ = _flash_scalar_loss(q_pos, k_pos, v_pos, upstream_np, backend)
    loss_neg, _, _, _ = _flash_scalar_loss(q_neg, k_neg, v_neg, upstream_np, backend)
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
    loss, q, k_t, v = _flash_scalar_loss(q_np, k_t_np, v_np, upstream_np, backend)
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


class PresentationReport:
    def numerical(
        self,
        batch_size: int = 1,
        nhead: int = 2,
        seq_len: int = 8,
        head_dim: int = 16,
        finite_diff_samples: int = 12,
        finite_diff_eps: float = 1e-3,
        seed: int = 7,
        output_json: str = "",
    ):
        result = run_numerical_correctness(
            batch_size=batch_size,
            nhead=nhead,
            seq_len=seq_len,
            head_dim=head_dim,
            finite_diff_samples=finite_diff_samples,
            finite_diff_eps=finite_diff_eps,
            seed=seed,
        )
        print(json.dumps(result, indent=2))
        if output_json:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        return result

    def performance(
        self,
        warmup: int = 5,
        iters: int = 30,
        seed: int = 7,
        configs: str = "hw3_default,medium,long,xlong",
        gpu_index: int = 0,
        measure_gpu_memory: bool = True,
        memory_sample_interval_ms: int = 20,
        output_json: str = "",
    ):
        return flash_benchmark.main(
            warmup=warmup,
            iters=iters,
            seed=seed,
            configs=configs,
            gpu_index=gpu_index,
            measure_gpu_memory=measure_gpu_memory,
            memory_sample_interval_ms=memory_sample_interval_ms,
            output_json=output_json,
        )

    def bleu(
        self,
        dataset_name: str = "bbaaaa/iwslt14-de-en-preprocess",
        model_max_length: int = 40,
        n_epochs: int = 1,
        batch_size: int = 128,
        learning_rate: float = 0.002,
        samples_per_epoch: int = 20000,
        n_vocab: int = 10000,
        n_embd: int = 256,
        seed: int = 11111,
        use_fused_kernel: bool = False,
        use_flash_attention=None,
        attention_dropout: float = 0.1,
        bleu_split: str = "test",
        bleu_samples: int = 100,
    ):
        return mt.main(
            dataset_name=dataset_name,
            model_max_length=model_max_length,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            samples_per_epoch=samples_per_epoch,
            n_vocab=n_vocab,
            n_embd=n_embd,
            seed=seed,
            use_fused_kernel=use_fused_kernel,
            use_flash_attention=use_flash_attention,
            attention_dropout=attention_dropout,
            eval_bleu=True,
            bleu_split=bleu_split,
            bleu_samples=bleu_samples,
        )

    def all(
        self,
        output_json: str = "presentation_report.json",
        numerical_seq_len: int = 8,
        numerical_head_dim: int = 16,
        finite_diff_samples: int = 12,
        perf_configs: str = "hw3_default,medium,long,xlong",
        perf_warmup: int = 5,
        perf_iters: int = 30,
        gpu_index: int = 0,
        bleu_samples: int = 100,
        bleu_epochs: int = 1,
        bleu_samples_per_epoch: int = 20000,
        use_fused_kernel: bool = True,
        use_flash_attention: bool = True,
        attention_dropout: float = 0.0,
    ):
        numerical = run_numerical_correctness(
            seq_len=numerical_seq_len,
            head_dim=numerical_head_dim,
            finite_diff_samples=finite_diff_samples,
        )

        performance = flash_benchmark.main(
            warmup=perf_warmup,
            iters=perf_iters,
            configs=perf_configs,
            gpu_index=gpu_index,
            measure_gpu_memory=True,
            memory_sample_interval_ms=20,
            output_json="",
        )

        bleu = mt.main(
            n_epochs=bleu_epochs,
            samples_per_epoch=bleu_samples_per_epoch,
            eval_bleu=True,
            bleu_split="test",
            bleu_samples=bleu_samples,
            use_fused_kernel=use_fused_kernel,
            use_flash_attention=use_flash_attention,
            attention_dropout=attention_dropout,
        )

        report = {
            "numerical_correctness": numerical,
            "performance": performance,
            "bleu": bleu,
        }

        print(json.dumps(report, indent=2))
        if output_json:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        return report


if __name__ == "__main__":
    fire.Fire(PresentationReport)
