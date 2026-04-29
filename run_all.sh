#!/usr/bin/env bash
# Run every stage of the FlashAttention-for-GPT2 presentation pipeline:
#   1) correctness (forward, finite-diff grads)
#   2) parity suite (flash-vs-baseline forward + grads, causal + non-causal, multi-seed/shape)
#   3) performance microbenchmarks
#   4) sequence-length scaling sweep
#   5) end-to-end BLEU (flash and baseline) + matched ablation
#   6) throughput-vs-quality tradeoff grid
#   7) publication-quality figures (PNG + PDF)
#   8) per-figure CSV tables (for external plotting)
#   9) research poster (PNG + PDF)
#
# Everything is driven from ``project/presentation_report.py`` (single JSON report),
# then the plotting / export / poster scripts consume that JSON.
#
# Usage:
#   bash run_all.sh                         # full default run
#   SKIP_BLEU=1 bash run_all.sh             # skip BLEU + ablation + tradeoff (fast dev loop)
#   OUTPUT=my_report.json bash run_all.sh   # custom output filename

set -euo pipefail

PY="${PY:-/home/divyanshsood/.conda/envs/cuda121/bin/python}"
OUTPUT="${OUTPUT:-presentation_report.json}"
FIG_DIR="${FIG_DIR:-figures/$(basename "${OUTPUT%.json}")}"
DATA_DIR="${DATA_DIR:-${FIG_DIR}/data}"

SKIP_BLEU="${SKIP_BLEU:-0}"

echo "Python:    ${PY}"
echo "Report:    ${OUTPUT}"
echo "Figures:   ${FIG_DIR}"
echo "Tables:    ${DATA_DIR}"
echo "Skip BLEU: ${SKIP_BLEU}"
mkdir -p "${FIG_DIR}" "${DATA_DIR}"

if [[ "${SKIP_BLEU}" == "1" ]]; then
    INCLUDE_ABL="false"
    INCLUDE_TRADE="false"
    BLEU_EPOCHS="0"
    echo "[1/4] Running all correctness + perf (BLEU/tradeoff disabled)..."
else
    INCLUDE_ABL="true"
    INCLUDE_TRADE="true"
    BLEU_EPOCHS="${BLEU_EPOCHS:-1}"
    echo "[1/4] Running full correctness + perf + BLEU + tradeoff..."
fi

"${PY}" -m project.presentation_report all \
    --output_json            "${OUTPUT}" \
    --include_parity_suite   true \
    --parity_seq_lens        "8,16,32" \
    --parity_causal_modes    "true,false" \
    --parity_seeds           "7,11" \
    --perf_configs           "hw3_default,medium,long,xlong" \
    --scaling_configs        "seq_128,seq_256,seq_512,seq_1024,seq_2048" \
    --bleu_samples           100 \
    --bleu_epochs            "${BLEU_EPOCHS}" \
    --bleu_samples_per_epoch 20000 \
    --use_fused_kernel       true \
    --use_flash_attention    true \
    --attention_dropout      0.0 \
    --include_scaling        true \
    --include_ablation       "${INCLUDE_ABL}" \
    --include_tradeoff       "${INCLUDE_TRADE}" \
    --print_summary          true \
    --write_markdown         true

echo "[2/4] Rendering figures into ${FIG_DIR}..."
"${PY}" -m project.plot_presentation_results \
    --report_json "${OUTPUT}" \
    --output_dir  "${FIG_DIR}"

echo "[3/4] Exporting per-figure CSV tables into ${DATA_DIR}..."
"${PY}" -m project.export_report_tables \
    --report_json "${OUTPUT}" \
    --output_dir  "${DATA_DIR}"

echo "[4/4] Building poster into ${FIG_DIR}/poster.{png,pdf}..."
"${PY}" -m project.make_poster \
    --report_json "${OUTPUT}" \
    --figures_dir "${FIG_DIR}" \
    --output_png  "${FIG_DIR}/poster.png" \
    --output_pdf  "${FIG_DIR}/poster.pdf"

echo
echo "Done."
echo "  JSON   : ${OUTPUT}"
echo "  MD     : ${OUTPUT%.json}.md"
echo "  Figures: ${FIG_DIR}/"
echo "  Tables : ${DATA_DIR}/"
echo "  Poster : ${FIG_DIR}/poster.png  ${FIG_DIR}/poster.pdf"
