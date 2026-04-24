#!/usr/bin/env bash
# Run all 6 personas under differential steering, sequentially.
# Resumable: the runner's checkpoint-count logic skips completed (pair, ordering, mult).
# Load/unload model once per persona — ~1 min overhead per switch.
set -euo pipefail
cd /workspace/repo

PERSONAS=(aura contrarian mathematician sadist slacker strategist)

for P in "${PERSONAS[@]}"; do
    echo "================================================================"
    echo "=== Persona: $P   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "================================================================"
    python -u -m src.steering.runner "configs/steering/cross_persona_differential/$P.yaml"
    echo
done

echo "=== All personas done   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
