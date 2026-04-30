"""Clean trajectory plot for App. G.1 (probe uniqueness)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "experiments/probe_direction_uniqueness/output/L32/trajectory.json"
OUT = ROOT / "paper/figures/appendix/plot_043026_uniqueness_trajectory_clean.png"

with TRAJ.open() as f:
    data = json.load(f)

iters = [t["iter"] for t in data["trajectory"]]
in_dist = [t["final_r"] for t in data["trajectory"]]
loo = [t["hoo_mean_r"] for t in data["trajectory"]]

fig, ax = plt.subplots(figsize=(6.0, 3.6))

ax.plot(iters, in_dist, "-o", color="#d97706", lw=2.2, ms=8,
        label="In-distribution (held-out r)")
ax.plot(iters, loo, "-D", color="#b91c1c", lw=2.2, ms=8,
        label="Cross-topic (leave-one-topic-out r)")

for x, y in zip(iters, in_dist):
    ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=9, color="#d97706")
for x, y in zip(iters, loo):
    ax.annotate(f"{y:.2f}", (x, y), xytext=(0, -16), textcoords="offset points",
                ha="center", fontsize=9, color="#b91c1c")

ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.6)

ax.set_xticks(iters)
ax.set_xticklabels([f"{i}\n({i} direction{'s' if i != 1 else ''} removed)" for i in iters])
ax.set_xlabel("Projection iteration")
ax.set_ylabel("Pearson r with utilities")
ax.set_ylim(-0.05, 1.0)
ax.set_xlim(-0.25, 2.25)
ax.legend(loc="lower left", frameon=False, fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
