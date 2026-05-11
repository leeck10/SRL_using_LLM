#!/bin/bash
# =============================================================================
# Run the full SRL-ICL pipeline (with ConE ordering optimization)
#
# This script runs all 5 steps sequentially:
#   Step 1: Train BERT-CRF encoder
#   Step 2: Build retrieval database
#   Step 3: Select in-context examples
#   Step 4: Optimize example ordering via ConE
#   Step 5: Evaluate with LLM (using optimized order from Step 4)
#
# Usage:
#   bash scripts/run_all.sh --config configs/en_config.yaml
#   bash scripts/run_all.sh --config configs/ko_config.yaml
#   bash scripts/run_all.sh --config configs/en_config.yaml --gpu 1
#   bash scripts/run_all.sh --config configs/en_config.yaml --strategy mmr --lambda 0.9
# =============================================================================

set -e

CONFIG=""
STRATEGY="topk"
LAMBDA=0.7
GPU=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift ;;
        --strategy) STRATEGY="$2"; shift ;;
        --lambda) LAMBDA="$2"; shift ;;
        --gpu) GPU="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$CONFIG" ]; then
    echo "Usage: bash scripts/run_all.sh --config <config.yaml> [--gpu <id>] [--strategy topk|mmr]"
    exit 1
fi

GPU_FLAG=""
if [ -n "$GPU" ]; then
    GPU_FLAG="--gpu $GPU"
fi

echo "=========================================="
echo "SRL-ICL Full Pipeline (with ConE)"
echo "Config: $CONFIG"
echo "Strategy: $STRATEGY"
if [ -n "$GPU" ]; then
    echo "GPU: $GPU"
fi
echo "=========================================="

echo ""
echo "[Step 1/5] Training BERT-CRF encoder..."
python scripts/01_train_crf.py --config "$CONFIG" $GPU_FLAG

echo ""
echo "[Step 2/5] Building retrieval database..."
python scripts/02_build_retrieval_db.py --config "$CONFIG" $GPU_FLAG

echo ""
echo "[Step 3/5] Selecting in-context examples ($STRATEGY)..."
python scripts/03_select_examples.py --config "$CONFIG" --strategy "$STRATEGY" --lambda_param "$LAMBDA" $GPU_FLAG

echo ""
echo "[Step 4/5] Optimizing example ordering (ConE)..."
python scripts/04_optimize_order.py --config "$CONFIG" $GPU_FLAG

echo ""
echo "[Step 5/5] Evaluating with LLM (using optimized order)..."
python scripts/05_evaluate.py --config "$CONFIG" $GPU_FLAG

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
