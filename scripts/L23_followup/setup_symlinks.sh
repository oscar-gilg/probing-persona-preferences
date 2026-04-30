#!/usr/bin/env bash
set -euo pipefail
export MAIN_REPO=/Users/oscargilg/Dev/MATS/Preferences

# Activations dir (whole tree)
rm -rf activations
ln -s "$MAIN_REPO/activations" activations

# Source pair set
mkdir -p results/experiments/uniform_eval_gemma3_27b_v3/pre_task_revealed/completion_preference_gemma-3-27b_completion_canonical_seed0_uniform_eval
ln -sf "$MAIN_REPO/results/experiments/uniform_eval_gemma3_27b_v3/pre_task_revealed/completion_preference_gemma-3-27b_completion_canonical_seed0_uniform_eval/measurements.yaml" \
       results/experiments/uniform_eval_gemma3_27b_v3/pre_task_revealed/completion_preference_gemma-3-27b_completion_canonical_seed0_uniform_eval/measurements.yaml

# Probe-train task IDs source
mkdir -p results/experiments/persona_sweep_final_six/pre_task_active_learning/default_train
ln -sf "$MAIN_REPO/results/experiments/persona_sweep_final_six/pre_task_active_learning/default_train/measurements.yaml" \
       results/experiments/persona_sweep_final_six/pre_task_active_learning/default_train/measurements.yaml

# L23 probe + manifest (from layer_sweep worktree)
mkdir -p results/probes/layer_sweep_eot/probes
ln -sf "$MAIN_REPO/.claude/worktrees/layer_sweep/results/probes/layer_sweep/eot/probes/probe_ridge_L23.npy" \
       results/probes/layer_sweep_eot/probes/probe_ridge_L23.npy
ln -sf "$MAIN_REPO/.claude/worktrees/layer_sweep/results/probes/layer_sweep/eot/manifest.json" \
       results/probes/layer_sweep_eot/manifest.json

# Parent B0 measurements
mkdir -p experiments/preference_direction_ablation/results/B0
ln -sf "$MAIN_REPO/experiments/preference_direction_ablation/results/B0/measurements.jsonl" \
       experiments/preference_direction_ablation/results/B0/measurements.jsonl

echo "SYMLINKS_DONE"
