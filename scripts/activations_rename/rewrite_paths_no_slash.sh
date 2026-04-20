#!/bin/bash
# Second pass: rewrite activations/<name> (no trailing slash) — picks up output_dir in configs.
set -eu

cd "$(git rev-parse --show-toplevel)"

TARGETS=(configs scripts src)

# Use a less-specific-match-last pattern: use word boundary ($) or quote/end-of-line to anchor.
# Simple approach: substitute exact string when followed by end-of-line, quote, whitespace, comma, or colon.
SUBSTITUTIONS=(
  's|activations/gemma_3_27b_turn_boundary_sweep\([^a-zA-Z0-9_/]\)|activations/gemma-3-27b_it/pref_main\1|g'
  's|activations/gemma_3_27b_villain_tb\([^a-zA-Z0-9_/]\)|activations/gemma-3-27b_it/pref_villain\1|g'
  's|activations/gemma_3_27b_midwest_tb\([^a-zA-Z0-9_/]\)|activations/gemma-3-27b_it/pref_midwest\1|g'
  's|activations/gemma_3_27b_sadist_tb\([^a-zA-Z0-9_/]\)|activations/gemma-3-27b_it/pref_sadist\1|g'
  's|activations/gemma_3_27b_creak_raw\([^a-zA-Z0-9_/]\)|activations/gemma-3-27b_it/truth_creak_raw\1|g'
  's|activations/gemma_3_27b_creak_repeat\([^a-zA-Z0-9_/]\)|activations/gemma-3-27b_it/truth_creak_repeat\1|g'
  's|activations/gemma_3_27b_error_prefill\([^a-zA-Z0-9_/]\)|activations/gemma-3-27b_it/truth_error_prefill\1|g'
  's|activations/gemma_3_27b_lying_prefill\([^a-zA-Z0-9_/]\)|activations/gemma-3-27b_it/truth_lying_prefill\1|g'
  's|activations/gemma_3_27b_pt_task_mean\([^a-zA-Z0-9_/]\)|activations/gemma-3-27b_pt/pref_main\1|g'
  's|activations/qwen35_122b_turn_boundary_sweep\([^a-zA-Z0-9_/]\)|activations/qwen35_122b/pref_main\1|g'
  's|activations/qwen3_emb_8b\([^a-zA-Z0-9_/]\)|activations/qwen3-emb_8b/pref_main\1|g'
  's|activations/ood_tb\([^a-zA-Z0-9_/]\)|activations/gemma-3-27b_it/pref_ood_prompts_parent_removed\1|g'
  # Also handle end-of-line (no trailing char). sed macOS needs $ separately.
  's|activations/gemma_3_27b_turn_boundary_sweep$|activations/gemma-3-27b_it/pref_main|g'
  's|activations/gemma_3_27b_villain_tb$|activations/gemma-3-27b_it/pref_villain|g'
  's|activations/gemma_3_27b_midwest_tb$|activations/gemma-3-27b_it/pref_midwest|g'
  's|activations/gemma_3_27b_sadist_tb$|activations/gemma-3-27b_it/pref_sadist|g'
  's|activations/gemma_3_27b_creak_raw$|activations/gemma-3-27b_it/truth_creak_raw|g'
  's|activations/gemma_3_27b_creak_repeat$|activations/gemma-3-27b_it/truth_creak_repeat|g'
  's|activations/gemma_3_27b_error_prefill$|activations/gemma-3-27b_it/truth_error_prefill|g'
  's|activations/gemma_3_27b_lying_prefill$|activations/gemma-3-27b_it/truth_lying_prefill|g'
  's|activations/gemma_3_27b_pt_task_mean$|activations/gemma-3-27b_pt/pref_main|g'
  's|activations/qwen35_122b_turn_boundary_sweep$|activations/qwen35_122b/pref_main|g'
  's|activations/qwen3_emb_8b$|activations/qwen3-emb_8b/pref_main|g'
)

for target in "${TARGETS[@]}"; do
  while IFS= read -r file; do
    # Skip the rename scripts themselves so they stay reversible.
    case "$file" in
      scripts/activations_rename/*) continue ;;
    esac
    for sub in "${SUBSTITUTIONS[@]}"; do
      sed -i '' "$sub" "$file"
    done
  done < <(find "$target" -type f \( -name "*.yaml" -o -name "*.py" -o -name "*.sh" \))
done

echo "Second-pass rewrite complete. Diff summary:"
git diff --stat | tail -30
