#!/bin/bash
# Overnight monitor: waits for measurements, syncs results, runs probes
# Log: /tmp/overnight_monitor.log

set -e
LOG="/tmp/overnight_monitor.log"
REPO="/Users/oscargilg/Dev/MATS/Preferences"
POD="root@213.192.2.99"
PORT=41560
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS="-o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
SSH_CMD="ssh $POD -p $PORT -i $KEY $SSH_OPTS"
SCP_CMD="scp -P $PORT -i $KEY $SSH_OPTS"
RSYNC_SSH="ssh -p $PORT -i $KEY $SSH_OPTS"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

log "=== Overnight monitor started ==="
log "Watching PIDs: 14331 (gptoss_qwen_mra2), 23154 (qwen35_10k)"

GPTOSS_DONE=0
QWEN10K_DONE=0

while true; do
    # Check gptoss_qwen_mra2 (PID 14331)
    if [ "$GPTOSS_DONE" -eq 0 ] && ! kill -0 14331 2>/dev/null; then
        log "gptoss_qwen_mra2 finished!"

        log "Syncing GPT-OSS + Qwen MRA results to pod..."
        $SSH_CMD "mkdir -p /workspace/results/experiments/gptoss_qwen_mra2/pre_task_active_learning"
        rsync -avz --progress -e "$RSYNC_SSH" \
            "$REPO/results/experiments/gptoss_qwen_mra2/pre_task_active_learning/" \
            "$POD:/workspace/results/experiments/gptoss_qwen_mra2/pre_task_active_learning/"
        log "Results synced."

        log "Generating cross-model probe configs..."
        $SSH_CMD "cd /workspace/repo && python -m scripts.cross_model_probes.generate_configs" 2>&1 | tee -a "$LOG"
        log "Configs generated."

        log "Running cross-model probe training..."
        $SSH_CMD "cd /workspace/repo && python -m scripts.cross_model_probes.run_all_probes" 2>&1 | tee -a "$LOG"
        log "Cross-model probe training complete."

        log "Running cross-eval..."
        $SSH_CMD "cd /workspace/repo && python -m scripts.cross_model_probes.cross_eval" 2>&1 | tee -a "$LOG"
        log "Cross-eval complete."

        log "Running analysis..."
        $SSH_CMD "cd /workspace/repo && python -m scripts.cross_model_probes.analyze" 2>&1 | tee -a "$LOG"
        log "Analysis complete."

        GPTOSS_DONE=1
    fi

    # Check qwen35_10k (PID 23154)
    if [ "$QWEN10K_DONE" -eq 0 ] && ! kill -0 23154 2>/dev/null; then
        log "qwen35_10k_active_learning finished!"

        log "Syncing Qwen 10k results to pod..."
        $SSH_CMD "mkdir -p /workspace/results/experiments/qwen35_10k_active_learning/pre_task_active_learning"
        rsync -avz --progress -e "$RSYNC_SSH" \
            "$REPO/results/experiments/qwen35_10k_active_learning/pre_task_active_learning/" \
            "$POD:/workspace/results/experiments/qwen35_10k_active_learning/pre_task_active_learning/"
        log "Qwen 10k results synced."

        log "Running Qwen probe training (5 selectors)..."
        for cfg in "$REPO/configs/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m"*.yaml; do
            name=$(basename "$cfg")
            log "  Training: $name"
            $SSH_CMD "cd /workspace/repo && python -m src.probes.experiments.run_dir_probes --config configs/probes/qwen35_122b/$name" 2>&1 | tee -a "$LOG"
        done
        log "Qwen probe training complete."

        QWEN10K_DONE=1
    fi

    # Both done?
    if [ "$GPTOSS_DONE" -eq 1 ] && [ "$QWEN10K_DONE" -eq 1 ]; then
        log "=== All tasks complete ==="
        break
    fi

    # Status update
    if [ "$GPTOSS_DONE" -eq 0 ]; then
        log "Waiting: gptoss_qwen_mra2 (PID 14331) still running"
    fi
    if [ "$QWEN10K_DONE" -eq 0 ]; then
        log "Waiting: qwen35_10k (PID 23154) still running"
    fi

    sleep 3600  # Check every hour
done
