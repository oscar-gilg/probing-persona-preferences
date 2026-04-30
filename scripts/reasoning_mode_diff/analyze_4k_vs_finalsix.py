"""Compare default-persona Qwen utilities from qwen35_4k_active_learning
(18 Mar 2026) vs qwen_persona_sweep_final_six default split (25 Apr 2026).

Both ran in 'no-think' mode (effectively thinking due to the reasoning bug,
both pre-fix). Question: do the utilities agree on overlapping tasks?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[2]


def origin_from_id(tid: str) -> str:
    if tid.startswith("competition_math_") or tid.startswith("math_"):
        return "math"
    if tid.startswith("stresstest_"):
        return "stress_test"
    for tag in ("wildchat", "alpaca", "bailbench"):
        if tid.startswith(tag + "_"):
            return tag
    return "other"


def load_utils(d: Path) -> dict[str, float]:
    csvs = sorted(d.glob("thurstonian_*.csv"))
    if not csvs:
        return {}
    df = pd.read_csv(csvs[0])
    return dict(zip(df["task_id"].astype(str), df["mu"].astype(float)))


def main() -> None:
    fk = load_utils(REPO / "results/experiments/qwen35_4k_active_learning/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d_qwen35_4k_task_ids")
    fs_eval = load_utils(REPO / "results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/default_eval")
    fs_test = load_utils(REPO / "results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/default_test")
    fs_train = load_utils(REPO / "results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/default_train")
    fs_all = {**fs_eval, **fs_test, **fs_train}

    print(f"4k AL:           n={len(fk)}  range=[{min(fk.values()):+.2f}, {max(fk.values()):+.2f}]  std={np.std(list(fk.values())):.2f}")
    print(f"final_six total: n={len(fs_all)}  (eval={len(fs_eval)}, test={len(fs_test)}, train={len(fs_train)})")
    print(f"final_six range=[{min(fs_all.values()):+.2f}, {max(fs_all.values()):+.2f}]  std={np.std(list(fs_all.values())):.2f}")

    overlap = sorted(set(fk) & set(fs_all))
    print(f"\nOverlap: {len(overlap)} tasks")
    if not overlap:
        print("No task ID overlap. Different task pools.")
        # Origin distributions
        def by_origin(d):
            from collections import Counter
            return dict(Counter(origin_from_id(t) for t in d))
        print(f"4k origin distribution:    {by_origin(fk)}")
        print(f"final_six origin dist:     {by_origin(fs_all)}")
        return

    a = np.array([fk[t] for t in overlap])
    b = np.array([fs_all[t] for t in overlap])
    print(f"\nOverall Pearson r = {pearsonr(a, b)[0]:+.3f}")
    print(f"Overall Spearman ρ = {spearmanr(a, b)[0]:+.3f}")
    print(f"  4k mean={a.mean():+.2f}  std={a.std():.2f}")
    print(f"  fs mean={b.mean():+.2f}  std={b.std():.2f}")

    print("\nPer-origin (overlapping tasks):")
    print(f"{'origin':<14} {'n':>5}  {'r':>7}  {'ρ':>7}  {'μ̄_4k':>8}  {'μ̄_fs':>8}  {'std_4k':>8}  {'std_fs':>8}")
    for origin in ["wildchat", "alpaca", "math", "bailbench", "stress_test"]:
        m = np.array([origin_from_id(t) == origin for t in overlap])
        if m.sum() < 5:
            continue
        r = pearsonr(a[m], b[m])[0]
        rho = spearmanr(a[m], b[m])[0]
        print(f"  {origin:<12} {m.sum():>5}  {r:+.3f}  {rho:+.3f}  {a[m].mean():+8.2f}  {b[m].mean():+8.2f}  {a[m].std():8.2f}  {b[m].std():8.2f}")

    # Top disagreements
    diffs = a - b
    abs_diffs = np.abs(diffs)
    top = np.argsort(-abs_diffs)[:10]
    print("\nTop 10 |Δμ| disagreements:")
    for idx in top:
        tid = overlap[idx]
        print(f"  {tid:<35}  4k={a[idx]:+6.2f}  fs={b[idx]:+6.2f}  Δ={diffs[idx]:+6.2f}")


if __name__ == "__main__":
    main()
