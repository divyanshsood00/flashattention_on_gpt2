# llmsys_f25_hw4

Public repository for Assignment 4 of 11-868 LLM Systems.

## FlashAttention integration

This repo now includes a fused FlashAttention forward kernel with online softmax.

### Build kernels

```bash
bash compile_cuda.sh
```

By default this compiles for both `sm_70` (V100) and `sm_86`.
You can override with:

```bash
CUDA_ARCH_FLAGS="-gencode arch=compute_80,code=sm_80" bash compile_cuda.sh
```

### Correctness tests

Kernel-level parity tests:

```bash
python kernel_tests/test_flash_attention_fw.py
python kernel_tests/test_flash_attention_bw.py
```

Pytest coverage for op-level and module-level integration:

```bash
pytest tests/test_flash_attention.py -q
```

### Benchmarking and metrics

Run workload sweep and collect latency, speedup, TFLOPs/s, tokens/s, and peak-memory estimates:

```bash
python -m project.benchmark_flash_attention --warmup 5 --iters 30 --configs hw3_default,medium,long,xlong --output_json flash_bench.json
```

### Training with fused attention

Use fused kernels with:

```bash
python -m project.run_machine_translation --use_fused_kernel True --attention_dropout 0.0
```

Enable/disable FlashAttention explicitly (independent of other fused kernels):

```bash
python -m project.run_machine_translation --use_fused_kernel True --use_flash_attention True
python -m project.run_machine_translation --use_fused_kernel True --use_flash_attention False
```

`attention_dropout=0.0` is recommended for direct FlashAttention execution in training mode.

## Presentation Metrics

### 1) Numerical correctness (FP32 + finite differences)

Run:

```bash
python -m project.presentation_report numerical --output_json numerical_report.json
```

The output includes:

- `forward_max_abs_error` (target: <= 1e-3 for FP32)
- finite-difference gradient parity (`finite_difference.max_abs_error`)

### 2) Performance and memory on NVIDIA A1000

Run:

```bash
python -m project.benchmark_flash_attention --warmup 5 --iters 30 --configs hw3_default,medium,long,xlong --gpu_index 0 --measure_gpu_memory True --output_json performance_report.json
```

This reports:

- forward / fw+bw latency and speedup
- measured GPU memory deltas (via `nvidia-smi` sampling)
- estimated peak-memory reduction ratio

### 3) End-to-end quality (BLEU on IWSLT de->en)

Run:

```bash
python -m project.run_machine_translation --use_fused_kernel True --use_flash_attention True --attention_dropout 0.0 --eval_bleu True --bleu_split test --bleu_samples 100
```

This prints a JSON summary with BLEU score (`bleu.bleu`) and sample count.

### One-command full report

```bash
python -m project.presentation_report all --output_json presentation_report.json
```
