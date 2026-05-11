#!/bin/bash
# =============================================================================
# Run the SRL-ICL pipeline WITHOUT ConE ordering optimization
#
# Skips Step 4 (ConE) and uses a fixed sequential order [0,1,2,3,4] instead.
# This is useful when:
#   - ConE ordering has already been computed separately
#   - You want a quick baseline with default ordering
#   - You want to test with a custom fixed order
#
# Pipeline:
#   Step 1: Train BERT-CRF encoder
#   Step 2: Build retrieval database
#   Step 3: Select in-context examples
#   Step 5: Evaluate with LLM (using fixed order)
#
# Usage:
#   bash scripts/run_without_cone.sh --config configs/en_config.yaml
#   bash scripts/run_without_cone.sh --config configs/en_config.yaml --order 0,1,2,3,4
#   bash scripts/run_without_cone.sh --config configs/en_config.yaml --order 2,0,4,1,3 --gpu 1
#   bash scripts/run_without_cone.sh --config configs/en_config.yaml --strategy mmr
# =============================================================================

set -e

CONFIG=""
STRATEGY="topk"
LAMBDA=0.7
GPU=""
ORDER="0,1,2,3,4"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift ;;
        --strategy) STRATEGY="$2"; shift ;;
        --lambda) LAMBDA="$2"; shift ;;
        --gpu) GPU="$2"; shift ;;
        --order) ORDER="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$CONFIG" ]; then
    echo "Usage: bash scripts/run_without_cone.sh --config <config.yaml> [--order 0,1,2,3,4] [--gpu <id>]"
    exit 1
fi

GPU_FLAG=""
if [ -n "$GPU" ]; then
    GPU_FLAG="--gpu $GPU"
fi

echo "=========================================="
echo "SRL-ICL Pipeline (without ConE)"
echo "Config: $CONFIG"
echo "Strategy: $STRATEGY"
echo "Fixed order: $ORDER"
if [ -n "$GPU" ]; then
    echo "GPU: $GPU"
fi
echo "=========================================="

echo ""
echo "[Step 1/4] Training BERT-CRF encoder..."
python scripts/01_train_crf.py --config "$CONFIG" $GPU_FLAG

echo ""
echo "[Step 2/4] Building retrieval database..."
python scripts/02_build_retrieval_db.py --config "$CONFIG" $GPU_FLAG

echo ""
echo "[Step 3/4] Selecting in-context examples ($STRATEGY)..."
python scripts/03_select_examples.py --config "$CONFIG" --strategy "$STRATEGY" --lambda_param "$LAMBDA" $GPU_FLAG

echo ""
echo "[Step 4/4] Evaluating with LLM (fixed order: $ORDER)..."
python scripts/05_evaluate.py --config "$CONFIG" --order "$ORDER" $GPU_FLAG

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
