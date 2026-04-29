# llmsys_f25_hw4

Public repository for Assignment 4 of 11-868 LLM Systems.

## Environment (conda `cuda121`)

Every command in this README is expected to run inside the **`cuda121`** conda
environment. Activate it once at the start of the session:

```bash
conda activate cuda121
```

If you hit `ImportError: Numba needs NumPy 2.3 or less. Got NumPy 2.4.`, pin
NumPy to a Numba-compatible version inside `cuda121` (the repo's
`requirements.txt` already specifies `numpy==2.3.5`):

```bash
conda activate cuda121
python -m pip install --force-reinstall "numpy==2.3.5"
```

Then verify:

```bash
python -c "import numpy, numba; print('numpy', numpy.__version__, 'numba', numba.__version__)"
```

For the plotting / poster steps you also need `fire` and `matplotlib`; they
are pulled in by `requirements.txt`, but if they are missing in `cuda121`
install them directly:

```bash
python -m pip install fire matplotlib
```

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

This includes:

- numerical correctness
- performance benchmark
- sequence scaling curves
- Flash BLEU run
- exact matched baseline BLEU run (standard attention)
- quality-throughput tradeoff sweep

The written JSON is structured for both tools and people:

- **`report_schema`** — name and version of the report format.
- **`executive_summary`** — UTC timestamp, “at a glance” bullets (PASS/FAIL and key numbers), and a **section index** table (id, status, duration, one-line headline). In the saved file **`report_schema` and `executive_summary` are written before** the heavy payload keys so the overview is easy to spot in an editor or with `jq`.
- **`notes`** — e.g. `memory_budget_mb`, benchmark config strings, graceful-failure reminder.
- **`preserve_report_meta`** (default **true**) — if `presentation_report.json` already exists and contains a top-level **`report_meta`** object (poster / print metadata), it is copied into the new file so poster fields are not lost on re-run.

By default, `all` also prints a **readable ASCII summary** to the terminal (`print_summary=True`) and writes a **Markdown** companion next to the JSON (`presentation_report.md` when the output path ends in `.json` and `write_markdown=True`). Disable either if you want JSON only:

```bash
python -m project.presentation_report all --output_json presentation_report.json --write_markdown=False --print_summary=False
```

Use an explicit Markdown path:

```bash
python -m project.presentation_report all --output_json out/report.json --markdown_path out/REPORT.md
```

### Explicit baseline BLEU run (matched settings)

Use the same training hyperparameters as FlashAttention and only disable flash attention:

```bash
python -m project.run_machine_translation --use_fused_kernel True --use_flash_attention False --attention_dropout 0.0 --eval_bleu True --bleu_split test --bleu_samples 100
```

### Memory-budget-aware report runs

To keep workloads within a 4GB device budget and continue when a config cannot run:

```bash
python -m project.presentation_report all --perf_memory_budget_mb 3900 --output_json presentation_report.json
```

### Plot generation

Generate publication-quality figures (PDF + PNG) from `presentation_report.json`:

```bash
python -m project.plot_presentation_results --report_json presentation_report.json --output_dir figures
```

This writes the following research-paper-ready artifacts to `figures/`:

| File (stem) | Contents |
| --- | --- |
| `overview_summary` | 3-panel figure-1 overview: forward speedup, peak memory, gradient parity / BLEU |
| `speedup_vs_seq` | Forward and forward+backward speedup vs. sequence length, with parity reference |
| `latency_forward_comparison` | Forward latency vs. $N$ on log scale, with $\times$speedup annotations |
| `latency_fw_bw_comparison` | Training-step (fw+bw) latency vs. $N$ on log scale |
| `memory_savings_vs_seq` | Measured peak-memory deltas (MB), grouped bars with $\div$ratio labels |
| `memory_scaling_theory` | $O(N^2)$ vs. $O(N)$ theoretical activation memory on log axes |
| `tflops_comparison` | Achieved TFLOP/s per attention kernel |
| `tokens_per_sec_comparison` | Effective attention tokens/s on log scale |
| `numerical_grad_error` | Max / mean finite-difference gradient error per tensor with pass threshold |
| `bleu_comparison` | End-to-end BLEU on IWSLT14 de$\rightarrow$en (shows $\Delta$BLEU when baseline is also run) |
| `quality_throughput_tradeoff` | Pareto scatter of BLEU vs. tokens/s across the hyperparameter grid |

Each figure is emitted both as a `.png` (300 dpi) and a `.pdf` (vector) with embedded
Type 42 fonts, so the `.pdf` outputs can be dropped directly into a LaTeX paper.

### Poster generation

Build a single-page, print-ready research poster (48" × 36" landscape) from the
same report and figure set. The poster embeds the already-generated figures and
pulls headline numbers (peak speedup, memory saved, BLEU, gradient error)
directly out of `presentation_report.json`:

```bash
conda activate cuda121
python -m project.make_poster \
    --report_json presentation_report.json \
    --figures_dir figures \
    --output_dir figures \
    --title "FlashAttention for GPT-2: Fast, Memory-Efficient Self-Attention" \
    --authors "Divyansh Sood" \
    --affiliation "11-868 LLM Systems, Carnegie Mellon University"
```

This writes `figures/poster.pdf` (vector, for printing) and `figures/poster.png`
(200 dpi, for web / screen). Sections included: title banner, 4 headline stat
callouts (peak forward / forward+backward speedup, peak memory saved, BLEU),
Motivation, Method (with the online-softmax update rules), three Performance
panels (latency, speedup, peak memory), three Correctness / Quality panels
(gradient parity, BLEU, memory scaling), and a footer with key takeaways.

Regenerate the underlying figures first if `presentation_report.json` has
changed:

```bash
python -m project.plot_presentation_results --report_json presentation_report.json --output_dir figures
python -m project.make_poster              --report_json presentation_report.json --figures_dir figures --output_dir figures
```

### End-to-end (report → figures → poster) on `cuda121`

```bash
conda activate cuda121
python -m project.presentation_report all --perf_memory_budget_mb 3900 --output_json presentation_report.json
python -m project.plot_presentation_results --report_json presentation_report.json --output_dir figures
python -m project.make_poster              --report_json presentation_report.json --figures_dir figures --output_dir figures
```

### `report_meta` in `presentation_report.json` (poster / print metadata)

The repo’s `presentation_report.json` may include an optional top-level object
`report_meta` with human-readable fields (title, subtitle, authors, course,
contact email, dataset description, benchmark notes, etc.). When present,
`project.make_poster` reads it automatically (`use_report_meta=True` by default)
and uses it for the poster header and footer detail lines. Edit those strings
to match your poster (especially `contact_email`).

### Figures + a second poster from the same JSON (isolated folder)

To keep one set of plots and a dedicated poster next to them (without
overwriting `figures/poster.*`):

```bash
conda activate cuda121
mkdir -p figures/presentation_report
python -m project.plot_presentation_results \
    --report_json presentation_report.json \
    --output_dir figures/presentation_report
python -m project.make_poster \
    --report_json presentation_report.json \
    --figures_dir figures/presentation_report \
    --output_dir figures/presentation_report \
    --stem poster_presentation_report
```

Outputs: `figures/presentation_report/*.pdf` / `*.png` plus
`figures/presentation_report/poster_presentation_report.pdf` (and `.png`).
