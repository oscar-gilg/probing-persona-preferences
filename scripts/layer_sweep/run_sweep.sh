#!/bin/bash
# Run all layer_sweep steering configs sequentially. Resume-safe (each config's checkpoint is self-contained).
set -e
cd /workspace/repo
mkdir -p experiments/layer_sweep/checkpoints

for cfg in configs/steering/layer_sweep/tb-2_probe_L*.yaml configs/steering/layer_sweep/eot_probe_L*.yaml; do
    echo "=== $cfg ==="
    python scripts/isolated_steering/run_steering.py "$cfg" 2>&1 || {
        echo "FAILED: $cfg"
        continue
    }
done

echo "=== sweep complete ==="
