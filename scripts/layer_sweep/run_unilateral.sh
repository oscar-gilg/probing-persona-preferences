#!/bin/bash
# Run unilateral eot diagonal sweep (both span positions, all 20 layers).
set -e
cd /workspace/repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for cfg in configs/steering/layer_sweep/eot_unilateral_diagonal_early.yaml configs/steering/layer_sweep/eot_unilateral_diagonal_late.yaml; do
    echo "=== $cfg ==="
    python scripts/isolated_steering/run_steering.py "$cfg" 2>&1 || {
        echo "FAILED: $cfg"
        continue
    }
done

echo "=== unilateral sweep complete ==="
