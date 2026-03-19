"""Plot KV cache steering results: dose-response and steerability by preference gap."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.steering.analysis import (
    aggregate_steered,
    chose_steered_task,
    filter_valid,
    load_checkpoint,
)

CHECKPOINT = "experiments/steering/isolated_steering/checkpoint_kv_full.jsonl"
ASSETS = Path("experiments/steering/isolated_steering/full_run/assets")
ASSETS.mkdir(parents=True, exist_ok=True)

rows = load_checkpoint(CHECKPOINT)
valid = filter_valid(rows)

# ── Plot 1: Dose-response ──
# Aggregate P(chose steered task) by |multiplier| using the analysis module
agg = aggregate_steered(valid, group_by=["condition"])

# Build symmetric x-axis: for each strength, plot both +strength and -strength (mirrored)
strengths = sorted(r["steering_strength"] for r in agg)
p_steered = {r["steering_strength"]: r["p_steered"] for r in agg}
ns = {r["steering_strength"]: r["n"] for r in agg}

# x values: negative (mirrored: 1 - p_steered), zero (0.5 baseline), positive (p_steered)
x_vals = []
y_vals = []
n_labels = []
for s in strengths:
    x_vals.append(-s)
    y_vals.append(1 - p_steered[s])
    n_labels.append(ns[s])
x_vals.append(0)
y_vals.append(0.5)
n_labels.append(None)
for s in strengths:
    x_vals.append(s)
    y_vals.append(p_steered[s])
    n_labels.append(ns[s])

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_vals, y_vals, "o-", color="#2563eb", markersize=8, linewidth=2)

# Annotate n per point (skip zero baseline)
for x, y, n in zip(x_vals, y_vals, n_labels):
    if n is not None:
        ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color="#555")

ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel(
    "Steering strength (fraction of per-layer KV norm)\n"
    r"($\leftarrow$ steer toward B | steer toward A $\rightarrow$)"
)
ax.set_ylabel("P(model chose steered task)")
ax.set_ylim(0, 1)
ax.set_title("KV steering: does modifying K+V cache shift task choice?", fontsize=12)
fig.tight_layout()
out1 = ASSETS / "plot_031826_kv_dose_response.png"
fig.savefig(out1, dpi=150)
plt.close(fig)
print(f"Saved {out1}")

# ── Plot 2: Steerability by preference gap ──
multipliers = sorted(set(abs(r["signed_multiplier"]) for r in valid))
bins = [(0, 0.5, "|Δμ| < 0.5"), (0.5, 1.0, "0.5–1.0"), (1.0, float("inf"), "|Δμ| > 1.0")]
colors = {0.003: "#93c5fd", 0.005: "#2563eb"}

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(bins))
width = 0.35

for i, mult in enumerate(multipliers):
    subset = [r for r in valid if abs(r["signed_multiplier"]) == mult]
    p_steered_per_bin = []
    ns_bin = []
    for lo, hi, _ in bins:
        bin_rows = [r for r in subset if lo <= abs(r["delta_mu"]) < hi]
        n_success = sum(1 for r in bin_rows if chose_steered_task(r))
        p_steered_per_bin.append(n_success / len(bin_rows) if bin_rows else float("nan"))
        ns_bin.append(len(bin_rows))

    offset = (i - 0.5) * width
    bars = ax.bar(x + offset, p_steered_per_bin, width, color=colors[mult],
                  label=f"strength={mult}")
    for bar, n in zip(bars, ns_bin):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={n}", ha="center", va="bottom", fontsize=8)

ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
ax.set_xticks(x)
ax.set_xticklabels([label for _, _, label in bins])
ax.set_xlabel("Pre-existing preference gap between tasks")
ax.set_ylabel("P(model chose steered task)")
ax.set_ylim(0, 1)
ax.set_title("KV steering: steerability by preference gap", fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()
out2 = ASSETS / "plot_031826_kv_steerability_by_preference.png"
fig.savefig(out2, dpi=150)
plt.close(fig)
print(f"Saved {out2}")
