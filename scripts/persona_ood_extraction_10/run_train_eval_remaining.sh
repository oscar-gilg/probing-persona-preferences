#!/bin/bash
# Run train+eval extraction for the 9 remaining personas (evil_genius already done).
# Same as run_train_eval.sh but starting from chaos_agent.

set -u
set -o pipefail

REPO=/workspace/repo
LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"

PERSONAS=(
    chaos_agent obsessive_perfectionist lazy_minimalist
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
