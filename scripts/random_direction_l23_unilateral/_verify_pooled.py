"""Sanity-check the pooled load_random_single_task curve used in Fig 3 panel (b)."""
import sys
sys.path.insert(0, "paper/figures/panels")
from build_steering_integrated import load_random_single_task

xs, ys, elo, ehi = load_random_single_task()
for x, y, lo, hi in zip(xs, ys, elo, ehi):
    print(f"c={x:+.3f}  P(chose steered)={y:.3f}  [{y-lo:.3f}, {y+hi:.3f}]")
