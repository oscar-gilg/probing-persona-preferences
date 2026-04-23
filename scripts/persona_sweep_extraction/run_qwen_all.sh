#!/bin/bash
# Run extraction for default + final-six personas on the canonical 6000-task set,
# Qwen-3.5-122B. Output lands at /workspace/activations/qwen35_122b/pref_<persona>_sweep/
# and stays on the pod — transfer to storage pod is done from laptop (two-hop).

set -u
set -o pipefail

REPO=/workspace/repo
LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"

# default + metadata.final_six from experiments/persona_sweep/sweep_personas.json
PERSONAS=(default sadist mathematician aura strategist contrarian slacker)

cd "$REPO"

for persona in "${PERSONAS[@]}"; do
    echo "=== $(date -u +%F' '%T) UTC :: $persona ==="
    log="$LOG_DIR/extract_qwen_${persona}_sweep.log"

    python -m src.probes.extraction.run "configs/extraction/qwen35_pref_${persona}_sweep.yaml" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "FAIL: $persona extraction exited $rc — stopping"
        exit "$rc"
    fi

    echo "=== done: $persona ==="
done
