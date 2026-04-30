"""Compare default-persona Thurstonian utilities between the two runs:
  - previous: qwen35_10k_active_learning (reasoning silently ON, pre-fix)
  - current:  qwen_persona_sweep_final_six default_train (reasoning explicitly OFF)

Test hypothesis: did the reasoning-mode bug fix materially change Qwen's
preferences, lowering current probe r vs previous?

Reports:
  - shared task ID count
  - Pearson + Spearman correlation of μ on the shared subset
  - per-task |Δμ| distribution
  - top-10 tasks where μ disagrees most
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.probes.data_loading import load_thurstonian_scores

REPO = Path(__file__).resolve().parents[2]
PREV_DIR = REPO / "results/experiments/qwen35_10k_active_learning/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d"
CURR_DIR = REPO / "results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/default_train"


def main() -> None:
    prev_mu = load_thurstonian_scores(PREV_DIR)
    curr_mu = load_thurstonian_scores(CURR_DIR)

    print(f"previous run μ: {len(prev_mu)} tasks")
    print(f"current  run μ: {len(curr_mu)} tasks")

    shared = sorted(set(prev_mu) & set(curr_mu))
    print(f"shared task IDs: {len(shared)}")
    if not shared:
        print("no overlap — cannot compare")
        return

    prev_arr = np.array([prev_mu[tid] for tid in shared])
    curr_arr = np.array([curr_mu[tid] for tid in shared])

    r_p, _ = pearsonr(prev_arr, curr_arr)
    r_s, _ = spearmanr(prev_arr, curr_arr)
    print(f"\nPearson  r(prev μ, curr μ) = {r_p:+.3f}")
    print(f"Spearman r(prev μ, curr μ) = {r_s:+.3f}")

    diffs = np.abs(prev_arr - curr_arr)
    print(f"\n|Δμ| distribution:")
    print(f"  mean = {diffs.mean():.3f}")
    print(f"  median = {np.median(diffs):.3f}")
    print(f"  p90 = {np.quantile(diffs, 0.9):.3f}")
    print(f"  max = {diffs.max():.3f}")

    print(f"\nμ ranges:")
    print(f"  prev μ: [{prev_arr.min():+.2f}, {prev_arr.max():+.2f}]  std = {prev_arr.std():.2f}")
    print(f"  curr μ: [{curr_arr.min():+.2f}, {curr_arr.max():+.2f}]  std = {curr_arr.std():.2f}")

    print(f"\ntop 10 disagreement tasks (|prev μ - curr μ|):")
    top_idx = np.argsort(-diffs)[:10]
    for idx in top_idx:
        print(f"  {shared[idx]:>40s}  prev={prev_arr[idx]:+.2f}  curr={curr_arr[idx]:+.2f}  Δ={diffs[idx]:.2f}")


if __name__ == "__main__":
    main()
