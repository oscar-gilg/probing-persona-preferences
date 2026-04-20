#!/bin/bash
# Run extraction for all 10 personas sequentially.
#
# Per persona: extract → verify NPZ → attempt rsync to storage pod →
# (on successful rsync) delete GPU-pod copy; else leave in place.
#
# NOTE (2026-04-20 session): storage-pod SSH is blocked on this pod (encrypted
# private key, no passphrase available). rsync + delete are skipped; outputs
# are left at /workspace/activations/gemma-3-27b_it/pref_<persona>/ for manual
# transfer. See experiments/persona_ood_extraction_10/running_log.md.

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
    echo "=== $(date -u +%F' '%T) UTC :: $persona ==="
    log="$LOG_DIR/extract_${persona}.log"

    python -m src.probes.extraction.run "configs/extraction/pref_${persona}.yaml" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "FAIL: $persona extraction exited $rc — stopping"
        exit "$rc"
    fi

    echo "--- verify $persona ---"
    python scripts/persona_ood_extraction_10/verify_extraction.py "$persona" || {
        echo "FAIL: $persona verification — stopping"
        exit 1
    }

    echo "=== done: $persona ==="
done
