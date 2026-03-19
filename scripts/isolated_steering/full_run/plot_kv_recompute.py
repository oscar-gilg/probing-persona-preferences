"""Plot KV steering vs KV + suffix recompute: dose-response comparison."""

from pathlib import Path

import matplotlib.pyplot as plt

from src.steering.analysis import aggregate_steered, filter_valid, load_checkpoint

CHECKPOINT = "experiments/steering/isolated_steering/checkpoint_kv_recompute.jsonl"
ASSETS = Path("experiments/steering/isolated_steering/full_run/assets")
ASSETS.mkdir(parents=True, exist_ok=True)

rows = load_checkpoint(CHECKPOINT)
valid = filter_valid(rows)

CONDITIONS = [
    ("kv_steering", "-", "KV only"),
    ("kv_steering_recompute", "--", "KV + suffix recompute"),
]
COLOR = "#2563eb"

agg = aggregate_steered(valid, group_by=["condition"])

fig, ax = plt.subplots(figsize=(8, 5))

for condition, linestyle, label in CONDITIONS:
    cond_rows = [r for r in agg if r["condition"] == condition]
    strengths = sorted(r["steering_strength"] for r in cond_rows)
    p_steered = {r["steering_strength"]: r["p_steered"] for r in cond_rows}
    ns = {r["steering_strength"]: r["n"] for r in cond_rows}

    x_vals = []
    y_vals = []
    n_labels = []
    for s in reversed(strengths):
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

    ax.plot(x_vals, y_vals, linestyle=linestyle, color=COLOR, marker="o",
            markersize=7, linewidth=2, label=label)

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
ax.set_title("Does suffix recomputation amplify KV steering?", fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()

out = ASSETS / "plot_031926_kv_recompute_comparison.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved {out}")
