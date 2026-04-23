#!/bin/bash
# Finish the remaining tb-2 configs with OOM mitigation.
set -e
cd /workspace/repo
mkdir -p experiments/layer_sweep/checkpoints
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for L in L53 L56 L59; do
    cfg="configs/steering/layer_sweep/tb-2_probe_${L}.yaml"
    echo "=== $cfg ==="
    python scripts/isolated_steering/run_steering.py "$cfg" 2>&1 || {
        echo "FAILED: $cfg"
        continue
    }
done

echo "=== tb-2 tail complete ==="
