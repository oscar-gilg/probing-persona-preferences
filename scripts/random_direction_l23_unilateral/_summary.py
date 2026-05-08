"""Print across-seed mean curves + swing summary."""
import json
from pathlib import Path

import numpy as np

d = json.loads(Path("experiments/random_direction_l23_unilateral/agg.json").read_text())
pooled = d["pooled_across_seeds"]

print("Pooled across 5 seeds {0, 1, 2, 3, 42}:\n")
for cond in ("unilateral_first", "unilateral_second"):
    print(f"{cond}:")
    coefs = sorted(pooled[cond].values(), key=lambda v: v["applied_coef"])
    for v in coefs:
        m = v["mean_p_chose_steered"]
        s = v["sem_p_chose_steered"]
        r = v["mean_refusal_rate"]
        print(f"  applied={v['applied_coef']:+.2f}: P={m:.3f} ± {s:.3f} (refusal {r:.1%}, n_seeds={v['n_seeds']})")
    pos = pooled[cond]["+0.05"]["mean_p_chose_steered"]
    neg = pooled[cond]["-0.05"]["mean_p_chose_steered"]
    pos_sem = pooled[cond]["+0.05"]["sem_p_chose_steered"]
    neg_sem = pooled[cond]["-0.05"]["sem_p_chose_steered"]
    swing = pos - neg
    swing_sem = (pos_sem ** 2 + neg_sem ** 2) ** 0.5
    print(f"  swing P(+0.05) - P(-0.05) = {swing:+.3f} ± {swing_sem:.3f}\n")

# Combined null hypothesis: across all 5 seeds × both conditions, is the swing
# distinguishable from zero?
swings = []
for seed_str, agg in d["seeds"].items():
    for cond in ("unilateral_first", "unilateral_second"):
        pos = agg[cond]["+0.05"]["p_chose_steered"]
        neg = agg[cond]["-0.05"]["p_chose_steered"]
        swings.append(pos - neg)
swings = np.array(swings)
print(f"Per-seed × per-condition swings (N={len(swings)}):")
print(f"  mean = {swings.mean():+.3f}, std = {swings.std(ddof=1):.3f}, "
      f"sem = {swings.std(ddof=1) / np.sqrt(len(swings)):.3f}")
print(f"  range: {swings.min():+.3f} to {swings.max():+.3f}")
