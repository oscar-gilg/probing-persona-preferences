#!/usr/bin/env bash
# Run all 5 unilateral random-direction seeds in sequence.
# Logs to scripts/random_direction_l23_unilateral/logs/seed{s}.log.
#
# Designed for `tmux new-session -d -s zombuul-l23-uni 'bash run_all_seeds.sh'`.

set -u

SEEDS=(0 1 2 3 42)
LOG_DIR="scripts/random_direction_l23_unilateral/logs"
CONFIG_DIR="configs/steering/random_direction_l23_unilateral"

mkdir -p "$LOG_DIR"

for seed in "${SEEDS[@]}"; do
    config="$CONFIG_DIR/random_single_task_seed${seed}.yaml"
    log="$LOG_DIR/seed${seed}.log"
    echo "============================================================"
    echo "[$(date)] Seed $seed: $config"
    echo "============================================================"
    # Stream to log; do NOT abort the whole loop on a single seed failure.
    python -u -m scripts.isolated_steering.run_steering "$config" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    echo "[$(date)] Seed $seed done (rc=$rc)"
    if [ "$rc" -ne 0 ]; then
        echo "WARNING: seed $seed exited non-zero — continuing to next seed"
    fi
done
echo "[$(date)] All seeds processed."
