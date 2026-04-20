#!/bin/bash
# Rewrite hardcoded activations/* paths in configs/, scripts/, and src/ to the new scheme.
# Skip: experiments/ (markdown + running logs — historical), results/**/manifest.json (historical).
set -eu

cd "$(git rev-parse --show-toplevel)"

# List of directories to rewrite in.
TARGETS=(configs scripts src)

# sed invocations: one per path rename. In-place on macOS with empty -i ''.
# Order: most-specific first (e.g. "ood_tb/exp1_prompts/" before "ood_tb/") to avoid partial matches.
SUBSTITUTIONS=(
  "s|activations/gemma-3-27b_it/pref_ood_prompts/|activations/gemma-3-27b_it/pref_ood_prompts/|g"
  "s|activations/gemma-3-27b_it/pref_ood_category/|activations/gemma-3-27b_it/pref_ood_category/|g"
  "s|activations/gemma-3-27b_it/pref_main/|activations/gemma-3-27b_it/pref_main/|g"
  "s|activations/gemma-3-27b_it/pref_villain/|activations/gemma-3-27b_it/pref_villain/|g"
  "s|activations/gemma-3-27b_it/pref_midwest/|activations/gemma-3-27b_it/pref_midwest/|g"
  "s|activations/gemma-3-27b_it/pref_sadist/|activations/gemma-3-27b_it/pref_sadist/|g"
  "s|activations/gemma-3-27b_it/truth_creak_raw/|activations/gemma-3-27b_it/truth_creak_raw/|g"
  "s|activations/gemma-3-27b_it/truth_creak_repeat/|activations/gemma-3-27b_it/truth_creak_repeat/|g"
  "s|activations/gemma-3-27b_it/truth_error_prefill/|activations/gemma-3-27b_it/truth_error_prefill/|g"
  "s|activations/gemma-3-27b_it/truth_lying_prefill/|activations/gemma-3-27b_it/truth_lying_prefill/|g"
  "s|activations/gemma-3-27b_pt/pref_main/|activations/gemma-3-27b_pt/pref_main/|g"
  "s|activations/gemma-3-27b_pt/pref_main_prompt_last/|activations/gemma-3-27b_pt/pref_main_prompt_last/|g"
  "s|activations/gpt-oss_120b/pref_main/|activations/gpt-oss_120b/pref_main/|g"
  "s|activations/gpt-oss_120b/pref_main_prompt_last/|activations/gpt-oss_120b/pref_main_prompt_last/|g"
  "s|activations/qwen35_122b/pref_main/|activations/qwen35_122b/pref_main/|g"
  "s|activations/qwen35_122b/pref_mra_2500/|activations/qwen35_122b/pref_mra_2500/|g"
  "s|activations/qwen3-emb_8b/pref_main/|activations/qwen3-emb_8b/pref_main/|g"
  "s|activations/character/gemma-3-27b_it/|activations/character/gemma-3-27b_it/|g"
)

for target in "${TARGETS[@]}"; do
  # Find all text files under the target (yaml, py, sh, md, json)
  while IFS= read -r file; do
    for sub in "${SUBSTITUTIONS[@]}"; do
      sed -i '' "$sub" "$file"
    done
  done < <(find "$target" -type f \( -name "*.yaml" -o -name "*.py" -o -name "*.sh" \))
done

echo "Path rewrite complete. Changes:"
git diff --stat | tail -20
