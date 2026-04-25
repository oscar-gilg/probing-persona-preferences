#!/usr/bin/env bash
# Robust sweep: each persona gets up to 3 attempts with a 40-min timeout.
# Both gen and parse are resumable (gen via checkpoint_counts, parse via existing_keys),
# so retries pick up where the last attempt left off. This handles the known
# OpenRouter socket-hang issue observed during judge parsing.
set -uo pipefail   # no -e: tolerate timeout/non-zero exits
cd /workspace/repo

PERSONAS=(aura contrarian mathematician sadist slacker strategist)
# Allow skipping personas via: SKIP="aura contrarian" bash run_all_robust.sh
SKIP="${SKIP:-}"

for P in "${PERSONAS[@]}"; do
    if echo " $SKIP " | grep -q " $P "; then
        echo "=== Skipping persona: $P"
        continue
    fi
    echo "================================================================"
    echo "=== Persona: $P   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "================================================================"
    for attempt in 1 2 3 4; do
        echo "  attempt $attempt for $P"
        timeout 2400 python -u -m src.steering.runner "configs/steering/cross_persona_differential/$P.yaml"
        rc=$?
        if [ "$rc" -eq 0 ]; then
            echo "  $P attempt $attempt OK"
            break
        fi
        echo "  $P attempt $attempt exit=$rc (timeout 124, sigkill 137, etc.); retry"
    done
    echo
done

echo "=== All personas done   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
