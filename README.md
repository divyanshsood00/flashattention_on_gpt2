# FlashAttention for MiniTorch

**Fused, IO-aware self-attention CUDA kernel for decoder-only transformers -- built inside the MiniTorch educational framework.**

> 11-868 LLM Systems - Carnegie Mellon University - Spring 2026
> Siddharth Sarma - Divyansh Sood - Yuvraj Ahuja

---

## What this is

A drop-in replacement for MiniTorch's standard self-attention path, implemented as a single fused CUDA kernel using **online softmax** and **row-streaming tiling**. It computes mathematically exact attention while avoiding the materialization of the `[B, H, N, N]` score and probability tensors that make standard attention quadratic in memory.

Integrated end-to-end with MiniTorch's autograd and a 4-layer GPT-2-style decoder trained on IWSLT14 German->English.

## Headline results

Measured on an **NVIDIA RTX A1000 Laptop GPU** (3.9 GB working budget):

| Metric                                   | Value                                          |
| ---------------------------------------- | ---------------------------------------------- |
| Peak forward speedup (`N=512`)           | **7.3x**                                       |
| Peak forward+backward speedup (`N=2048`) | **1.9x**                                       |
| Peak attention memory saved              | **2.9 GB**                                     |
| Peak activation-memory reduction         | **~48x**                                       |
| Forward parity (FP32, vs reference)      | `3.6e-7` max abs error                         |
| Gradient parity (finite differences)     | `5.3e-3` max abs error (passes at 2e-2 threshold) |
| IWSLT14 de->en BLEU (FlashAttention)     | **8.16**                                       |
| IWSLT14 de->en BLEU (Standard)           | 5.46                                           |

## What we built

- **Fused CUDA forward kernel** (`src/flash_attention_kernel.cu`) -- online softmax with causal masking, one CUDA block per query row, keys/values streamed through SRAM.
- **MiniTorch autograd integration** -- exposed as `Tensor.flash_attention(k_t, v, causal=True)`, with a recomputation-based backward and a safe fallback to the standard attention path when shapes/dtypes/dropout are not supported.
- **`ctypes`/`nvcc` build integration** -- kernel compiles into the existing `combine.so` shared library, no extra linker plumbing required.
- **Benchmark harness** -- latency, throughput (tokens/s), TFLOP/s, and `nvidia-smi`-sampled peak memory across `N in {128, 256, 512, 1024, 2048}`.
- **Correctness suite** -- FP32 forward parity tests and finite-difference gradient checks.
- **End-to-end IWSLT14 training and BLEU evaluation** with matched ablations across epochs x dropout x flash-on/off.

## Repository layout

```
.
├── src/
│   ├── flash_attention_kernel.cu       # Fused CUDA forward kernel
│   └── combine.cu                      # MiniTorch CUDA backend (extended)
├── minitorch/
│   ├── tensor_functions.py             # FlashAttentionFn autograd Function
│   ├── cuda_kernel_ops.py              # CudaKernelOps.flash_attention_fw dispatch
│   └── modules_transformer.py          # MultiHeadAttention with flash path
├── kernel_tests/
│   ├── test_flash_attention_fw.py      # Kernel forward parity
│   └── test_flash_attention_bw.py      # Kernel backward parity
├── tests/
│   └── test_flash_attention.py         # Pytest suite
├── project/
│   ├── benchmark_flash_attention.py    # Latency / memory / throughput sweep
│   ├── presentation_report.py          # End-to-end report generator
│   ├── plot_presentation_results.py    # Figures from report JSON
│   ├── make_poster.py                  # Poster generator
│   └── run_machine_translation.py      # IWSLT14 training + BLEU eval
├── report/
│   └── LLMSYS_Final_Report_Team5.pdf   # Final report (PDF)
└── README.md
```

## Quick start

### Prerequisites

- Linux with CUDA toolkit (`nvcc` >= 11.0)
- Python >= 3.10
- An NVIDIA GPU with compute capability >= 7.0

### Build

```bash
git clone https://github.com/<TBD>/flashattention-minitorch.git
cd flashattention-minitorch
pip install -e .
bash compile_cuda.sh
```

### Verify correctness

```bash
python kernel_tests/test_flash_attention_fw.py
python kernel_tests/test_flash_attention_bw.py
pytest tests/test_flash_attention.py -v
```

All kernel and op-level tests should pass. Forward parity should be at FP32 round-off (~`4e-7`).

### Run benchmarks

```bash
# Single-shape microbenchmark
python -m project.benchmark_flash_attention --configs hw3_default

# Full sequence-length sweep
python -m project.benchmark_flash_attention --configs seq_128,seq_256,seq_512,seq_1024,seq_2048

# Full presentation report (correctness + benchmarks + BLEU)
python -m project.presentation_report all --output_json presentation_report.json
```

### Train a model with FlashAttention

```python
from minitorch.modules_transformer import MultiHeadAttention

# Drop-in replacement: just set use_flash_attention=True
attn = MultiHeadAttention(
    n_embd=256,
    n_head=8,
    causal=True,
    use_flash_attention=True,    # <-- this is the only change
    attention_dropout=0.0,       # required for the fused path
)
```

## How it works (one paragraph)

Standard attention dispatches 18+ separate MiniTorch tensor ops (`tensorMap`, `tensorZip`, `tensorReduce`, `MatrixMultiply`) and materializes the full `[B, H, N, N]` score and probability tensors in HBM. Our kernel replaces all of that with a single CUDA launch. Each CUDA block owns one query row; it loads `Q[b, h, i, :]` into shared memory, then streams keys/values one column at a time, computing `s = q . k_j / sqrt(d)`, updating the running pair `(m, l)` via the online-softmax recurrence, and accumulating into the output vector `O`. Causal masking is hard-coded as `if (j > i) continue;` inside the inner loop. The kernel writes back only the output `O` and the per-row log-sum-exp `Lse = m + log l`. Peak attention activation memory drops from `O(N^2)` to `O(N)`.

## Honest scope

This is a faithful educational implementation of the central FlashAttention-1 forward idea, not a full reproduction of the kernel. Specifically:

- **Forward is fused** (single CUDA kernel, online softmax, no `N x N` materialization).
- **Backward is not fused.** It recomputes attention through standard MiniTorch ops, which is correct but caps forward+backward speedup at ~1.9x even when forward speedup hits 7.3x. A fused backward using the saved `Lse` is the most impactful next step.
- **Row-streaming, not blockwise tiling.** We use one CUDA block per query row rather than the paper's `B_r x B_c` tile schedule. Simpler, but lower arithmetic intensity at large `N`.
- **FP32 only.** No FP16/BF16, no tensor-core path. Correctness-first.
- **Targeted at memory-constrained Ampere (A1000), not Hopper.** No TMA, no async warp groups, no FP8. The kernel would not be a strong baseline on an A100/H100.
- **BLEU is an integration sanity check**, not a quality claim. The +2.7 delta BLEU on a 100-sentence eval slice is well within run-to-run variance -- the right reading is "the fused path does not break translation quality."

See `report/LLMSYS_Final_Report_Team5.pdf` Section 5 for the full discussion.

## Reproducing the headline numbers

```bash
# Forward / forward+backward speedup curves
python -m project.benchmark_flash_attention \
    --configs seq_128,seq_256,seq_512,seq_1024,seq_2048 \
    --output_json bench.json

# Memory scaling
python -m project.benchmark_flash_attention \
    --configs seq_128,seq_256,seq_512,seq_1024,seq_2048 \
    --measure_gpu_memory True --output_json mem.json

# IWSLT14 BLEU (~30 min/run on the A1000)
python -m project.run_machine_translation --use_flash_attention True --epochs 1 --eval_bleu True
python -m project.run_machine_translation --use_flash_attention False --epochs 1 --eval_bleu True
```

## Documentation

- `report/LLMSYS_Final_Report_Team5.pdf` -- Full final report. Methodology, A1000-specific design choices, complete experimental results, analysis of forward vs. forward+backward gap, limitations.
- `flashattention_results_deep_dive.md` -- Detailed results analysis and parity discussion.
- `flashattention_results_comparison.md` -- Side-by-side metrics and baselines.
- `flashattention_vs_original_paper.md` -- Summary of similarities and deltas vs. the FlashAttention paper.
- `presentation_report.md` -- Auto-generated report from `project.presentation_report`.

## Citation

If this work is useful to you, the paper we re-implement is:

> Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re.
> **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.**
> NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

## License

MIT (see LICENSE if present).

## Acknowledgements

Built on top of [MiniTorch](https://github.com/minitorch/minitorch) by Sasha Rush. Thanks to the course staff of CMU 11-868 LLM Systems (Spring 2026) for guidance and infrastructure.
