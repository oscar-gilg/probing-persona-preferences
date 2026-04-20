#!/bin/bash
# Rename local activations/ dir to new scheme: activations/<model>/<task_set>[/<condition>]/
# Safe to re-run: each mv is conditional on source existing.
set -eu

ROOT=/Users/oscargilg/Dev/MATS/Preferences/activations

# Create model top-level dirs
mkdir -p "$ROOT/gemma-3-27b_it"
mkdir -p "$ROOT/gemma-3-27b_pt"
mkdir -p "$ROOT/qwen35_122b"
mkdir -p "$ROOT/qwen3-emb_8b"

# Gemma-3-27B IT — preferences
if [ -d "$ROOT/gemma_3_27b_turn_boundary_sweep" ]; then
  mv "$ROOT/gemma_3_27b_turn_boundary_sweep" "$ROOT/gemma-3-27b_it/pref_main"
fi
if [ -d "$ROOT/gemma_3_27b_villain_tb" ]; then
  mv "$ROOT/gemma_3_27b_villain_tb" "$ROOT/gemma-3-27b_it/pref_villain"
fi
if [ -d "$ROOT/gemma_3_27b_midwest_tb" ]; then
  mv "$ROOT/gemma_3_27b_midwest_tb" "$ROOT/gemma-3-27b_it/pref_midwest"
fi
if [ -d "$ROOT/gemma_3_27b_sadist_tb" ]; then
  mv "$ROOT/gemma_3_27b_sadist_tb" "$ROOT/gemma-3-27b_it/pref_sadist"
fi

# ood_tb/exp1_prompts/<cond>/ → gemma-3-27b_it/pref_ood_prompts/<cond>/
if [ -d "$ROOT/ood_tb/exp1_prompts" ]; then
  mkdir -p "$ROOT/gemma-3-27b_it/pref_ood_prompts"
  mv "$ROOT/ood_tb/exp1_prompts"/* "$ROOT/gemma-3-27b_it/pref_ood_prompts/"
  rmdir "$ROOT/ood_tb/exp1_prompts"
  rmdir "$ROOT/ood_tb" 2>/dev/null || true
fi

# Gemma-3-27B IT — truth/honesty
if [ -d "$ROOT/gemma_3_27b_creak_raw" ]; then
  mv "$ROOT/gemma_3_27b_creak_raw" "$ROOT/gemma-3-27b_it/truth_creak_raw"
fi
if [ -d "$ROOT/gemma_3_27b_creak_repeat" ]; then
  mv "$ROOT/gemma_3_27b_creak_repeat" "$ROOT/gemma-3-27b_it/truth_creak_repeat"
fi
if [ -d "$ROOT/gemma_3_27b_error_prefill" ]; then
  mv "$ROOT/gemma_3_27b_error_prefill" "$ROOT/gemma-3-27b_it/truth_error_prefill"
fi
if [ -d "$ROOT/gemma_3_27b_lying_prefill" ]; then
  mv "$ROOT/gemma_3_27b_lying_prefill" "$ROOT/gemma-3-27b_it/truth_lying_prefill"
fi

# Gemma-3-27B PT
if [ -d "$ROOT/gemma_3_27b_pt_task_mean" ]; then
  mv "$ROOT/gemma_3_27b_pt_task_mean" "$ROOT/gemma-3-27b_pt/pref_main"
fi

# Qwen-3 235B A22B (aka "qwen35_122b" — keep short name)
if [ -d "$ROOT/qwen35_122b_turn_boundary_sweep" ]; then
  mv "$ROOT/qwen35_122b_turn_boundary_sweep" "$ROOT/qwen35_122b/pref_main"
fi

# Qwen-3 Embedding 8B
if [ -d "$ROOT/qwen3_emb_8b" ]; then
  # Old dir contains files directly at its root; move inside a pref_main subdir.
  TMP="$ROOT/qwen3_emb_8b_tmp"
  mv "$ROOT/qwen3_emb_8b" "$TMP"
  mv "$TMP" "$ROOT/qwen3-emb_8b/pref_main"
fi

echo "Local rename complete."
ls "$ROOT"
