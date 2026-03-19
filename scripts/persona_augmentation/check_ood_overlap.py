"""Check OOD task overlap with training data only (not eval/test)."""
import numpy as np
from src.probes.data_loading import load_thurstonian_scores
from pathlib import Path

# Training data = 10k noprompt scores + MRA persona split A (1000 tasks)
scores_10k = load_thurstonian_scores(Path("results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0"))
train_10k = set(scores_10k.keys())

with open("configs/measurement/active_learning/mra_exp2_split_a_1000_task_ids.txt") as f:
    mra_train = {l.strip() for l in f if l.strip()}

all_train = train_10k | mra_train

# Eval data (OK to overlap)
scores_4k = load_thurstonian_scores(Path("results/experiments/main_probes/gemma3_4k_pre_task/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0"))
eval_4k = set(scores_4k.keys())

with open("configs/measurement/active_learning/mra_exp2_split_b_500_task_ids.txt") as f:
    mra_sweep = {l.strip() for l in f if l.strip()}
with open("configs/measurement/active_learning/mra_exp2_split_c_1000_task_ids.txt") as f:
    mra_test = {l.strip() for l in f if l.strip()}

print(f"Training set: {len(all_train)} tasks (10k noprompt + 1k MRA split A)")
print(f"Eval/test (OK): {len(eval_4k | mra_sweep | mra_test)} tasks\n")

for exp_dir in ["exp1_category", "exp1_prompts", "exp2_roles", "exp3_minimal_pairs", "exp3v8_minimal_pairs"]:
    p = Path(f"activations/ood/{exp_dir}/baseline/activations_prompt_last.npz")
    if not p.exists():
        continue
    ids = set(np.load(p, allow_pickle=True)["task_ids"])
    in_train = ids & all_train
    in_eval = ids & (eval_4k | mra_sweep | mra_test) - all_train
    clean = ids - all_train
    print(f"{exp_dir} ({len(ids)} tasks): {len(in_train)} in train (EXCLUDE), {len(clean)} clean")

n_conditions = 0
for exp_dir in ["exp1_category", "exp1_prompts", "exp2_roles", "exp3_minimal_pairs", "exp3v8_minimal_pairs"]:
    p = Path(f"activations/ood/{exp_dir}")
    if p.exists():
        n = len([d for d in p.iterdir() if d.is_dir()])
        n_conditions += n
        print(f"  {exp_dir}: {n} conditions")
print(f"\nTotal conditions to re-extract: {n_conditions}")
