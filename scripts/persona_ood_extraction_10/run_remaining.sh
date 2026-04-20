#!/bin/bash
# Run extraction for personas that haven't been processed yet.
# Each persona: extract -> verify -> continue (no rsync; see running_log.md).

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
    echo "=== $(date -u +%F' '%T) UTC :: $persona ==="
    log="$LOG_DIR/extract_${persona}.log"

    python -m src.probes.extraction.run "configs/extraction/pref_${persona}.yaml" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "FAIL: $persona extraction exited $rc"
        exit "$rc"
    fi

    python scripts/persona_ood_extraction_10/verify_extraction.py "$persona"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "FAIL: $persona verification"
        exit "$rc"
    fi

    echo "=== $(date -u +%F' '%T) UTC :: $persona done ==="
done

echo "ALL 9 PERSONAS DONE"
