#!/bin/bash
# Chained scoring of all 4 cells. Run from /workspace/repo on the pod.
set -e
source /opt/venvs/research/bin/activate
export PYTHONUNBUFFERED=1
cd /workspace/repo

GEMMA_BS="${GEMMA_BS:-16}"
QWEN_BS="${QWEN_BS:-4}"

echo "=== Gemma user-turn (batch=$GEMMA_BS) ==="
python -m experiments.eot_discrimination_v2.scripts.run_scoring \
  --model gemma-3-27b --turn user --device cuda:0 --batch-size "$GEMMA_BS"

echo "=== Gemma assistant-turn (batch=$GEMMA_BS) ==="
python -m experiments.eot_discrimination_v2.scripts.run_scoring \
  --model gemma-3-27b --turn assistant --device cuda:0 --batch-size "$GEMMA_BS"

echo "=== Qwen user-turn (batch=$QWEN_BS) ==="
python -m experiments.eot_discrimination_v2.scripts.run_scoring \
  --model qwen3.5-122b-nothink --turn user --device auto --batch-size "$QWEN_BS"

echo "=== Qwen assistant-turn (batch=$QWEN_BS) ==="
python -m experiments.eot_discrimination_v2.scripts.run_scoring \
  --model qwen3.5-122b-nothink --turn assistant --device auto --batch-size "$QWEN_BS"

echo "=== ALL DONE ==="
