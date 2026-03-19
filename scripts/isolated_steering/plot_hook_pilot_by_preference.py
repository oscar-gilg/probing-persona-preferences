"""Plot P(chose steered task) by pre-existing preference gap."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.steering.analysis import load_checkpoint, filter_valid, chose_steered_task

CHECKPOINT = "experiments/steering/isolated_steering/checkpoint_hook_pilot.jsonl"
ASSETS = Path("experiments/steering/isolated_steering/hook_patching_pilot/assets")

rows = filter_valid(load_checkpoint(CHECKPOINT))
rows = [r for r in rows if r["condition"] == "hook_patching" and r["layer"] == 25]

multipliers = sorted(set(abs(r["signed_multiplier"]) for r in rows))
bins = [(0, 0.5, "|Δμ| < 0.5"), (0.5, 1.0, "0.5–1.0"), (1.0, 3.0, "|Δμ| > 1.0")]

fig, ax = plt.subplots(figsize=(7, 5))

x = np.arange(len(bins))
width = 0.35
colors = {0.02: "#93c5fd", 0.05: "#2563eb"}

for i, mult in enumerate(multipliers):
    subset = [r for r in rows if abs(r["signed_multiplier"]) == mult]
    p_steered_per_bin = []
    ns = []
    for lo, hi, _ in bins:
        bin_rows = [r for r in subset if lo <= abs(r["delta_mu"]) < hi]
        n_success = sum(1 for r in bin_rows if chose_steered_task(r))
        p_steered_per_bin.append(n_success / len(bin_rows) if bin_rows else float("nan"))
        ns.append(len(bin_rows))

    offset = (i - 0.5) * width
    bars = ax.bar(x + offset, p_steered_per_bin, width, color=colors[mult],
                  label=f"strength={mult}")
    for j, (bar, n) in enumerate(zip(bars, ns)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={n}", ha="center", va="bottom", fontsize=8)

ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
ax.set_xticks(x)
ax.set_xticklabels([label for _, _, label in bins])
ax.set_xlabel("Pre-existing preference gap between tasks")
ax.set_ylabel("P(model chose steered task)")
ax.set_ylim(0, 1)
ax.set_title("Layer 25, splice only: steerability by preference gap", fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(ASSETS / "plot_031826_steerability_by_preference.png", dpi=150)
print("Saved steerability by preference plot")
