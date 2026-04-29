# FlashAttention for MiniTorch

**Fused, IO-aware self-attention CUDA kernel for decoder-only transformers, built inside the MiniTorch educational framework.**

> 11-868 LLM Systems - Carnegie Mellon University - Spring 2026
> Siddharth Sarma - Divyansh Sood - Yuvraj Ahuja

---

## At a glance

- Single fused CUDA forward kernel with online softmax and causal masking
- Autograd-integrated FlashAttention path for `MultiHeadAttention`
- End-to-end GPT-2 style decoder trained on IWSLT14 de->en
- Full correctness suite, microbenchmarks, scaling sweeps, and report pipeline
- Reproducible figures, CSV tables, and poster generation

## Headline results

Measured on an **NVIDIA RTX A1000 Laptop GPU** (3.9 GB working budget):

| Metric                                   | Value                                           |
| ---------------------------------------- | ----------------------------------------------- |
| Peak forward speedup (`N=512`)           | **7.3x**                                        |
| Peak forward+backward speedup (`N=2048`) | **1.9x**                                        |
| Peak attention memory saved              | **2.9 GB**                                      |
| Peak activation-memory reduction         | **~48x**                                        |
| Forward parity (FP32, vs reference)      | `3.6e-7` max abs error                          |
| Gradient parity (finite differences)     | `5.3e-3` max abs error (passes at 2e-2 threshold) |
| IWSLT14 de->en BLEU (FlashAttention)     | **8.16**                                        |
| IWSLT14 de->en BLEU (Standard)           | 5.46                                            |

## Repository map

```
.
├── src/                              # CUDA kernels
│   ├── flash_attention_kernel.cu     # FlashAttention forward kernel
│   ├── combine.cu                    # MiniTorch CUDA backend (extended)
│   ├── softmax_kernel.cu             # Baseline softmax kernel
│   └── layernorm_kernel.cu           # Baseline layernorm kernel
├── minitorch/                        # Core MiniTorch modules + autograd glue
├── kernel_tests/                     # Kernel-level parity tests
├── tests/                            # Pytest suite (op + module coverage)
├── project/                          # Benchmarks, reports, training, plots
├── report/                           # Final PDF report
├── figures/                          # Generated plots and posters
├── workdir_*/                        # Tokenizers + training artifacts
├── compile_cuda.sh                   # Build kernels -> minitorch/cuda_kernels/
├── run_all.sh                        # One-command report + figures + poster
├── requirements.txt                  # Python dependencies
├── setup.py                          # Editable install entry point
└── README.md
```

## Environment and setup

All commands are expected to run inside the **`cuda121`** conda environment.

```bash
conda activate cuda121
```

If you hit `ImportError: Numba needs NumPy 2.3 or less. Got NumPy 2.4.`, pin NumPy in the environment (the repo already specifies `numpy==2.3.5` in [requirements.txt](requirements.txt)):

```bash
python -m pip install --force-reinstall "numpy==2.3.5"
```

Install dependencies and the package in editable mode:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Build kernels

Use [compile_cuda.sh](compile_cuda.sh) to compile all CUDA kernels into `minitorch/cuda_kernels/`:

```bash
bash compile_cuda.sh
```

Default arch is `sm_86`. Override with `CUDA_ARCH_FLAGS` if needed:

```bash
CUDA_ARCH_FLAGS="-gencode arch=compute_80,code=sm_80" bash compile_cuda.sh
```

## Correctness tests

Kernel-level parity tests:

```bash
python kernel_tests/test_flash_attention_fw.py
python kernel_tests/test_flash_attention_bw.py
```

Optional baseline kernel checks (softmax and layernorm):

```bash
python kernel_tests/test_softmax_fw.py
python kernel_tests/test_softmax_bw.py
python kernel_tests/test_layernorm_fw.py
python kernel_tests/test_layernorm_bw.py
```

Pytest coverage for op-level and module-level integration:

```bash
pytest tests/test_flash_attention.py -q
```

## Benchmarks and reports

Single-shape microbenchmark:

```bash
python -m project.benchmark_flash_attention --configs hw3_default
```

Sequence-length sweep:

```bash
python -m project.benchmark_flash_attention --configs seq_128,seq_256,seq_512,seq_1024,seq_2048
```

Full presentation report (correctness + benchmarks + BLEU):

```bash
python -m project.presentation_report all --output_json presentation_report.json
```

Export per-figure CSV tables:

```bash
python -m project.export_report_tables --report_json presentation_report.json --output_dir figures/data
```

### One-command full pipeline

The full report + figures + poster pipeline lives in [run_all.sh](run_all.sh):

```bash
bash run_all.sh
```

Useful overrides:

```bash
SKIP_BLEU=1 bash run_all.sh
OUTPUT=my_report.json bash run_all.sh
```

## Figures and poster

Generate figures (PDF + PNG):

```bash
python -m project.plot_presentation_results --report_json presentation_report.json --output_dir figures
```

Build the poster:

```bash
python -m project.make_poster --report_json presentation_report.json --figures_dir figures --output_dir figures
```

## Training and BLEU

Run IWSLT14 de->en with FlashAttention enabled:

```bash
python -m project.run_machine_translation --use_fused_kernel True --use_flash_attention True --attention_dropout 0.0 --eval_bleu True --bleu_split test --bleu_samples 100
```

Baseline run (FlashAttention disabled):

```bash
python -m project.run_machine_translation --use_fused_kernel True --use_flash_attention False --attention_dropout 0.0 --eval_bleu True --bleu_split test --bleu_samples 100
```

Drop-in usage from code:

```python
from minitorch.modules_transformer import MultiHeadAttention

attn = MultiHeadAttention(
    n_embd=256,
    n_head=8,
    causal=True,
    use_flash_attention=True,
    attention_dropout=0.0,
)
```

## Artifacts and outputs

- Final report PDF: [report/LLMSYS_Final_Report_Team5.pdf](report/LLMSYS_Final_Report_Team5.pdf)
- Auto-generated report: [presentation_report.json](presentation_report.json) and [presentation_report.md](presentation_report.md)
- Additional report variants: [presentation_report_v2.json](presentation_report_v2.json), [presentation_report_haggu.json](presentation_report_haggu.json)
- Numerical correctness snapshot: [numerical_report.json](numerical_report.json)
- Parity suite output: [parity_suite.json](parity_suite.json) and [parity_suite_smoke.stdout](parity_suite_smoke.stdout)
- Throughput tradeoff runs: [throughput_tradeoff.json](throughput_tradeoff.json) and [throughput_tradeoff_remaining.json](throughput_tradeoff_remaining.json)
- BLEU baseline snapshot: [bleu_baseline.json](bleu_baseline.json)
- Figures and poster output: [figures](figures) and [figures.zip](figures.zip)
- Paper comparison notes: [flashattention_paper_excerpt_results_comparison.md](flashattention_paper_excerpt_results_comparison.md), [flashattention_results_comparison.md](flashattention_results_comparison.md), [flashattention_results_deep_dive.md](flashattention_results_deep_dive.md), [flashattention_vs_original_paper.md](flashattention_vs_original_paper.md), [flashattention_poster_presentation_notes.md](flashattention_poster_presentation_notes.md)

## How it works (one paragraph)

Standard attention dispatches many MiniTorch tensor ops and materializes the full `[B, H, N, N]` score and probability tensors in HBM. Our kernel replaces that with a single CUDA launch. Each CUDA block owns one query row, loads `Q[b, h, i, :]` into shared memory, then streams keys and values, computing `s = q . k_j / sqrt(d)`, updating the running pair `(m, l)` via the online-softmax recurrence, and accumulating the output vector `O`. Causal masking is hard-coded as `if (j > i) continue;` in the inner loop. The kernel writes back only the output `O` and per-row log-sum-exp `Lse = m + log l`. Peak activation memory drops from `O(N^2)` to `O(N)`.

## Scope

This is a faithful educational implementation of the central FlashAttention-1 forward idea, not a full reproduction of the kernel:

- **Forward is fused** (single CUDA kernel, online softmax, no `N x N` materialization)
- **Backward is not fused**; it recomputes attention through standard MiniTorch ops
- **Row-streaming, not blockwise tiling**; one CUDA block per query row
- **FP32 only**; no FP16 or BF16 path
- **Targeted at memory-constrained Ampere (A1000), not Hopper**
- **BLEU is an integration sanity check**, not a quality claim

See [report/LLMSYS_Final_Report_Team5.pdf](report/LLMSYS_Final_Report_Team5.pdf) Section 5 for details.

## Citation

If this work is useful to you, the paper we re-implement is:

> Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re.
> **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.**
> NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)


## Acknowledgements

Built on top of [MiniTorch](https://github.com/minitorch/minitorch) by Sasha Rush. Thanks to the CMU 11-868 LLM Systems course staff for guidance and infrastructure.
