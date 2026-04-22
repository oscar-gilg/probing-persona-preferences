import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path("experiments/steering/one_sided")
ASSETS = BASE / "assets"

with open(BASE / "analysis_harmful.json") as f:
    harmful = json.load(f)
with open(BASE / "analysis_benign.json") as f:
    benign = json.load(f)


def extract_condition(results: list[dict], condition: str) -> tuple[list[float], list[float]]:
    rows = [r for r in results if r["condition"] == condition]
    rows.sort(key=lambda r: r["signed_multiplier"])
    xs = [r["signed_multiplier"] for r in rows]
    ys = [r["p_steered"] for r in rows]
    return xs, ys


# ── Plot 1: Decomposition sigmoid ──────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

conditions = [
    ("steer_first_L25", "Steer first only", "tab:blue", 1.5),
    ("steer_second_L25", "Steer second only", "tab:orange", 1.5),
    ("differential_L25", "Differential", "black", 2.5),
]

for ax_idx, (data, panel_title) in enumerate([(harmful, "Harmful pairs"), (benign, "Benign pairs")]):
    ax = axes[ax_idx]
    results = data["results_judge"]

    for cond, label, color, lw in conditions:
        xs, ys = extract_condition(results, cond)
        ax.plot(xs, ys, marker="o", markersize=5, color=color, linewidth=lw, label=label)

    ax.set_xlabel("Signed multiplier")
    ax.set_ylabel("P(completed steered task)")
    ax.set_title(panel_title)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.11, 0.11)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)

    if ax_idx == 0:
        ax.legend(loc="upper left", framealpha=0.9)

fig.tight_layout()
fig.savefig(ASSETS / "plot_032526_decomposition_sigmoid.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {ASSETS / 'plot_032526_decomposition_sigmoid.png'}")


# ── Plot 2: Additivity deviation ───────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_idx, (data, panel_label) in enumerate([(harmful, "Harmful"), (benign, "Benign")]):
    ax = axes[ax_idx]
    results = data["results_judge"]
    mad = data["mad"]

    xs_diff, ys_diff = extract_condition(results, "differential_L25")
    xs_first, ys_first = extract_condition(results, "steer_first_L25")
    xs_second, ys_second = extract_condition(results, "steer_second_L25")

    # Build lookup dicts
    first_by_x = dict(zip(xs_first, ys_first))
    second_by_x = dict(zip(xs_second, ys_second))
    diff_by_x = dict(zip(xs_diff, ys_diff))

    # Non-zero multipliers only
    nonzero_xs = sorted([x for x in xs_diff if x != 0])

    deviations = []
    for x in nonzero_xs:
        additive_pred = first_by_x[x] + second_by_x[x] - 0.5
        dev = diff_by_x[x] - additive_pred
        deviations.append(dev)

    colors = ["tab:red" if abs(d) > 0.05 else "tab:green" for d in deviations]

    bar_width = 0.015
    ax.bar(nonzero_xs, deviations, width=bar_width, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(-0.05, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Signed multiplier")
    ax.set_ylabel("Deviation from additive prediction")
    ax.set_title(f"{panel_label} (MAD = {mad:.3f})")
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(-0.12, 0.12)
    ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(ASSETS / "plot_032526_additivity_deviation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {ASSETS / 'plot_032526_additivity_deviation.png'}")
