#!/bin/bash
set -euo pipefail

CONFIG="${1:-configs/config.yaml}"
PT_PATH="/tmp/model_mlp_best.pt"
ONNX_PATH="/tmp/model_mlp_best.onnx"

echo "========================================"
echo " Step 1: Retrain MLP"
echo "========================================"
python3 -m scripts.retrain --config "$CONFIG"

echo ""
echo "========================================"
echo " Step 2: Export .pt -> .onnx & upload"
echo "========================================"
python3 scripts/export_to_onnx.py \
    --config    "$CONFIG"   \
    --pt-path   "$PT_PATH"  \
    --onnx-path "$ONNX_PATH"

echo ""
echo "========================================"
echo " Pipeline complete!"
echo "========================================"