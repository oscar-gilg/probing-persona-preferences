#!/bin/bash
# Run extraction for the 8 sweep-recommended personas on the canonical 6000-task set.
# Output lands at /workspace/activations/gemma-3-27b_it/pref_<persona>_sweep/
# and stays on the pod — transfer to storage pod is done from laptop (two-hop).

set -u
set -o pipefail

REPO=/workspace/repo
LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"

PERSONAS=(
    sadist mathematician poet strategist
    contrarian slacker therapist entrepreneur
)

cd "$REPO"

for persona in "${PERSONAS[@]}"; do
    echo "=== $(date -u +%F' '%T) UTC :: $persona ==="
    log="$LOG_DIR/extract_${persona}_sweep.log"

    python -m src.probes.extraction.run "configs/extraction/pref_${persona}_sweep.yaml" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "FAIL: $persona extraction exited $rc — stopping"
        exit "$rc"
    fi

    echo "=== done: $persona ==="
done
