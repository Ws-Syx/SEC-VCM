#!/bin/bash
# ============================================================
# SEC-VCM Demo: Simple PNG -> PNG + JSON Pipeline
# Usage: bash test.sh
# ============================================================

set -e

INPUT_DIR="${1:-./input-sequence}"
OUTPUT_DIR="${2:-./output-sequence}"
INTRA_MODEL="${3:-./checkpoint/intra_model.pth.tar}"
INTER_MODEL="${4:-./checkpoint/inter_model.model}"
RATE_NUM="${5:-4}"
GOP="${6:-32}"
CUDA_DEVICE="${7:-0}"

echo "============================================"
echo "  SEC-VCM Demo: Video Compression Pipeline"
echo "============================================"
echo "  Input:       ${INPUT_DIR}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  Intra model: ${INTRA_MODEL}"
echo "  Inter model: ${INTER_MODEL}"
echo "  Rate points: ${RATE_NUM}"
echo "  GOP size:    ${GOP}"
echo "  GPU:         ${CUDA_DEVICE}"
echo "============================================"

export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}

python test_video.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --intra_model "${INTRA_MODEL}" \
    --inter_model "${INTER_MODEL}" \
    --rate_num ${RATE_NUM} \
    --gop ${GOP} \
    --cuda

echo ""
echo "============================================"
echo "  Demo Complete!"
echo "  Check output: ${OUTPUT_DIR}/rate_*/"
echo "============================================"
