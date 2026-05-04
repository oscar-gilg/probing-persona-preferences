#!/bin/bash
# Wrapper that loads HF_TOKEN from .env safely and runs extraction.
set -e
cd /workspace/repo
TOKEN=$(grep -E '^HF_TOKEN=' .env | head -1 | cut -d= -f2-)
export HF_TOKEN="$TOKEN"
export HUGGING_FACE_HUB_TOKEN="$TOKEN"
export HF_HOME=/opt/hf_cache
echo "[run_extraction] HF_TOKEN length: ${#HF_TOKEN}"
exec /opt/extract_venv/bin/python -m src.probes.extraction.run \
    configs/extraction/gemma3_27b_default_8layer.yaml
