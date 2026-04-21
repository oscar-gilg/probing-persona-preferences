#!/bin/bash
# Run train+eval extraction for the 9 remaining personas (evil_genius already done).
# Retries each persona up to 3 times on failure (mfs I/O errors have been observed);
# before each retry, wipes any partial output directory so the run starts clean.

set -u
set -o pipefail

REPO=/workspace/repo
LOG_DIR=/workspace/logs
ACT_ROOT=/workspace/activations/gemma-3-27b_it
mkdir -p "$LOG_DIR"

PERSONAS=(
    chaos_agent obsessive_perfectionist lazy_minimalist
    nationalist_ideologue conspiracy_theorist contrarian_intellectual
    whimsical_poet depressed_nihilist people_pleaser
)

cd "$REPO"

for persona in "${PERSONAS[@]}"; do
    out_dir="$ACT_ROOT/pref_${persona}_train_eval"
    log="$LOG_DIR/extract_${persona}_train_eval.log"

    attempt=1
    max_attempts=3
    success=0
    while [ "$attempt" -le "$max_attempts" ]; do
        echo "=== $(date -u +%F' '%T) UTC :: $persona attempt $attempt/$max_attempts ==="
        # Clean any partial output from a prior failed attempt
        if [ -d "$out_dir" ] && [ "$attempt" -gt 1 ]; then
            echo "Wiping partial $out_dir before retry"
            rm -f "$out_dir"/*.npz "$out_dir"/*.tmp.npz "$out_dir"/*.json
            rmdir "$out_dir" || true
        fi

        python -m src.probes.extraction.run "configs/extraction/pref_${persona}_train_eval.yaml" 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
        if [ "$rc" -eq 0 ]; then
            echo "=== done: $persona (attempt $attempt) ==="
            success=1
            break
        fi
        echo "WARN: $persona attempt $attempt exited $rc — retrying"
        sleep 10
        attempt=$((attempt + 1))
    done

    if [ "$success" -ne 1 ]; then
        echo "FAIL: $persona exhausted $max_attempts attempts — stopping"
        exit 1
    fi
done
