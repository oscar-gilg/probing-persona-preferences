#!/bin/bash
# Rename pod /workspace/activations/ dir to new scheme.
# Safe to re-run. Executed via a single SSH session so mkdir+mv stay atomic.
set -eu

SSH='ssh root@213.192.2.99 -p 41560 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no'

$SSH 'bash -s' << 'REMOTE'
set -eu
ROOT=/workspace/activations

mkdir -p "$ROOT/gemma-3-27b_it"
mkdir -p "$ROOT/gemma-3-27b_pt"
mkdir -p "$ROOT/gpt-oss_120b"
mkdir -p "$ROOT/qwen35_122b"
mkdir -p "$ROOT/qwen3-emb_8b"
mkdir -p "$ROOT/character"

# gemma-3-27b it
[ -d "$ROOT/gemma_3_27b_turn_boundary_sweep" ] && mv "$ROOT/gemma_3_27b_turn_boundary_sweep" "$ROOT/gemma-3-27b_it/pref_main"
[ -d "$ROOT/gemma_3_27b_creak_raw"            ] && mv "$ROOT/gemma_3_27b_creak_raw"            "$ROOT/gemma-3-27b_it/truth_creak_raw"
[ -d "$ROOT/gemma_3_27b_creak_repeat"         ] && mv "$ROOT/gemma_3_27b_creak_repeat"         "$ROOT/gemma-3-27b_it/truth_creak_repeat"

# gemma-3-27b pt: two different extraction runs
[ -d "$ROOT/gemma_3_27b_pt_task_mean" ] && mv "$ROOT/gemma_3_27b_pt_task_mean" "$ROOT/gemma-3-27b_pt/pref_main"
[ -d "$ROOT/gemma_3_27b_pt"           ] && mv "$ROOT/gemma_3_27b_pt"           "$ROOT/gemma-3-27b_pt/pref_main_prompt_last"

# gpt-oss: two different extraction runs
[ -d "$ROOT/gptoss_120b_turn_boundary_sweep" ] && mv "$ROOT/gptoss_120b_turn_boundary_sweep" "$ROOT/gpt-oss_120b/pref_main"
[ -d "$ROOT/gpt_oss_120b"                    ] && mv "$ROOT/gpt_oss_120b"                    "$ROOT/gpt-oss_120b/pref_main_prompt_last"

# qwen35_122b
[ -d "$ROOT/qwen35_122b_turn_boundary_sweep" ] && mv "$ROOT/qwen35_122b_turn_boundary_sweep" "$ROOT/qwen35_122b/pref_main"
[ -d "$ROOT/qwen35_122b_mra_2500"             ] && mv "$ROOT/qwen35_122b_mra_2500"             "$ROOT/qwen35_122b/pref_mra_2500"

# character
[ -d "$ROOT/character_probes" ] && mv "$ROOT/character_probes" "$ROOT/character/gemma-3-27b_it"

# ood → gemma-3-27b_it/pref_ood_*
if [ -d "$ROOT/ood/exp1_prompts" ]; then
  mkdir -p "$ROOT/gemma-3-27b_it/pref_ood_prompts"
  mv "$ROOT/ood/exp1_prompts"/* "$ROOT/gemma-3-27b_it/pref_ood_prompts/"
  rmdir "$ROOT/ood/exp1_prompts"
fi
[ -d "$ROOT/ood/exp1_category"       ] && mv "$ROOT/ood/exp1_category"       "$ROOT/gemma-3-27b_it/pref_ood_category"
[ -d "$ROOT/ood/exp2_roles"          ] && mv "$ROOT/ood/exp2_roles"          "$ROOT/gemma-3-27b_it/pref_ood_roles"
[ -d "$ROOT/ood/exp3_minimal_pairs"  ] && mv "$ROOT/ood/exp3_minimal_pairs"  "$ROOT/gemma-3-27b_it/pref_ood_minimal_pairs"
[ -d "$ROOT/ood/exp3v8_minimal_pairs" ] && mv "$ROOT/ood/exp3v8_minimal_pairs" "$ROOT/gemma-3-27b_it/pref_ood_minimal_pairs_v8"
rmdir "$ROOT/ood" 2>/dev/null || true

echo "Pod rename complete. Top-level:"
ls "$ROOT"
echo "--- gemma-3-27b_it ---"
ls "$ROOT/gemma-3-27b_it"
REMOTE
