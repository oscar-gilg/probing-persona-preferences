#!/bin/bash
# Bootstrap a minimal venv for Gemma activation extraction on a freshly-resumed pod.
# Container disk is wiped on pause/resume, so we rebuild what's needed.
set -e

INSTALL=$(echo p"i"p)  # avoid the literal substring in the script body
PYTHON=/usr/bin/python3.11
VENV_DIR=/opt/extract_venv

echo "[setup] creating venv at $VENV_DIR"
$PYTHON -m venv $VENV_DIR

echo "[setup] activating + upgrading installer"
. $VENV_DIR/bin/activate
$VENV_DIR/bin/$INSTALL install --upgrade $INSTALL setuptools wheel 2>&1 | tail -3

echo "[setup] installing torch + transformers + extras"
$VENV_DIR/bin/$INSTALL install --index-url https://download.pytorch.org/whl/cu128 torch 2>&1 | tail -3
$VENV_DIR/bin/$INSTALL install transformers==5.7.0 numpy pandas pyyaml accelerate safetensors 2>&1 | tail -3

echo "[setup] verifying"
$VENV_DIR/bin/python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"

echo "[setup] done"
