#!/bin/bash
# Run all eot steering configs. Uses expandable_segments to reduce OOM.
set -e
cd /workspace/repo
mkdir -p experiments/layer_sweep/checkpoints
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for cfg in configs/steering/layer_sweep/eot_probe_L*.yaml; do
    echo "=== $cfg ==="
    python scripts/isolated_steering/run_steering.py "$cfg" 2>&1 || {
        echo "FAILED: $cfg"
        continue
    }
done

echo "=== eot sweep complete ==="
