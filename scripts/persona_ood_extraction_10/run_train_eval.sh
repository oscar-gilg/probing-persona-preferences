#!/bin/bash
# Run train+eval extraction for all 10 personas sequentially.
# Uses configs/extraction/pref_<persona>_train_eval.yaml (5000 tasks each).
# Output lands at /workspace/activations/gemma-3-27b_it/pref_<persona>_train_eval/
# and stays on the pod — transfer to storage pod is done from laptop (two-hop).

set -u
set -o pipefail

REPO=/workspace/repo
LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"

PERSONAS=(
    evil_genius chaos_agent obsessive_perfectionist lazy_minimalist
    nationalist_ideologue conspiracy_theorist contrarian_intellectual
    whimsical_poet depressed_nihilist people_pleaser
)

cd "$REPO"

for persona in "${PERSONAS[@]}"; do
    echo "=== $(date -u +%F' '%T) UTC :: $persona (train+eval) ==="
    log="$LOG_DIR/extract_${persona}_train_eval.log"

    python -m src.probes.extraction.run "configs/extraction/pref_${persona}_train_eval.yaml" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "FAIL: $persona extraction exited $rc — stopping"
        exit "$rc"
    fi

    echo "=== done: $persona ==="
done
