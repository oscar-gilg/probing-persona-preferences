#!/bin/bash
# Run train+eval extraction for the 6 remaining personas (evil_genius, chaos_agent,
# obsessive_perfectionist already done on mfs).
#
# Strategy: write to local overlay disk (/root/activations_local/...) to avoid mfs
# I/O errors during large atomic checkpoint saves, then copy each completed persona
# to mfs and clean the local dir.
#
# Retries each persona up to 3× on failure.

set -u
set -o pipefail

REPO=/workspace/repo
LOG_DIR=/workspace/logs
LOCAL_ROOT=/root/activations_local/gemma-3-27b_it
MFS_ROOT=/workspace/activations/gemma-3-27b_it

mkdir -p "$LOG_DIR" "$LOCAL_ROOT" "$MFS_ROOT"

PERSONAS=(
    lazy_minimalist
    nationalist_ideologue conspiracy_theorist contrarian_intellectual
    whimsical_poet depressed_nihilist people_pleaser
)

cd "$REPO"

for persona in "${PERSONAS[@]}"; do
    local_dir="$LOCAL_ROOT/pref_${persona}_train_eval"
    mfs_dir="$MFS_ROOT/pref_${persona}_train_eval"
    log="$LOG_DIR/extract_${persona}_train_eval.log"

    attempt=1
    max_attempts=3
    success=0
    while [ "$attempt" -le "$max_attempts" ]; do
        echo "=== $(date -u +%F' '%T) UTC :: $persona attempt $attempt/$max_attempts ==="
        # Clean local partial output from prior failed attempt
        if [ -d "$local_dir" ]; then
            echo "Wiping $local_dir before attempt"
            rm -f "$local_dir"/*.npz "$local_dir"/*.tmp.npz "$local_dir"/*.json
            rmdir "$local_dir" || true
        fi

        python -m src.probes.extraction.run "configs/extraction/pref_${persona}_train_eval_local.yaml" 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
        if [ "$rc" -eq 0 ]; then
            echo "=== extraction done: $persona (attempt $attempt) ==="
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

    # Sync local → mfs. cp -r works around mfs atomic-rename flakiness because
    # cp writes each file independently (small writes) rather than one big np.savez.
    echo "=== syncing $local_dir -> $mfs_dir ==="
    mkdir -p "$mfs_dir"
    sync_attempt=1
    sync_max=5
    sync_ok=0
    while [ "$sync_attempt" -le "$sync_max" ]; do
        cp -r "$local_dir"/. "$mfs_dir"/ && sync_ok=1 && break
        echo "WARN: sync attempt $sync_attempt failed — retrying"
        sleep 10
        sync_attempt=$((sync_attempt + 1))
    done
    if [ "$sync_ok" -ne 1 ]; then
        echo "FAIL: could not sync $persona to mfs after $sync_max attempts — stopping"
        exit 1
    fi

    # Verify count of files on mfs
    n=$(ls "$mfs_dir" | wc -l)
    echo "mfs dir has $n files"

    # Free local disk
    rm -f "$local_dir"/*.npz "$local_dir"/*.json
    rmdir "$local_dir" || true
    echo "=== done: $persona ==="
done
