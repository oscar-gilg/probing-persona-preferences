"""Generate plots for the hook patching pilot report."""

from pathlib import Path

import matplotlib.pyplot as plt

from src.steering.analysis import load_checkpoint, aggregate_steered

CHECKPOINT = "experiments/steering/isolated_steering/checkpoint_hook_pilot.jsonl"
ASSETS = Path("experiments/steering/isolated_steering/hook_patching_pilot/assets")
ASSETS.mkdir(parents=True, exist_ok=True)

rows = load_checkpoint(CHECKPOINT)

agg = aggregate_steered(rows)
no_recompute = [r for r in agg if r["condition"] == "hook_patching"]
recompute = [r for r in agg if r["condition"] == "hook_patching_recompute"]

layers = sorted(set(r["layer"] for r in agg))
colors = {25: "#2563eb", 32: "#dc2626"}

# --- Plot 1: P(chose steered task) vs steering strength, faceted by mode ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

for ax, data, title in [
    (axes[0], no_recompute, "Splice only"),
    (axes[1], recompute, "Splice + suffix recompute"),
]:
    for layer in layers:
        subset = sorted([r for r in data if r["layer"] == layer], key=lambda r: r["steering_strength"])
        strengths = [r["steering_strength"] for r in subset]
        p_steered = [r["p_steered"] for r in subset]
        ax.plot(strengths, p_steered, "o-", color=colors[layer], label=f"Layer {layer}", markersize=7)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Steering strength (fraction of mean norm)")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, None)
    ax.legend(fontsize=8)

axes[0].set_ylabel("P(model chose steered task)")
fig.suptitle("Does activation patching causally shift task choice?", fontsize=13)
fig.tight_layout()
fig.savefig(ASSETS / "plot_031826_dose_response.png", dpi=150)
print("Saved dose-response plot")

# --- Plot 2: Recompute comparison ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

for ax, layer in zip(axes, layers):
    nr = sorted([r for r in no_recompute if r["layer"] == layer], key=lambda r: r["steering_strength"])
    rc = sorted([r for r in recompute if r["layer"] == layer], key=lambda r: r["steering_strength"])
    strengths = [r["steering_strength"] for r in nr]
    ax.plot(strengths, [r["p_steered"] for r in nr], "o-", color=colors[layer], label="Splice only", markersize=7)
    ax.plot(strengths, [r["p_steered"] for r in rc], "s--", color=colors[layer], alpha=0.6, label="Splice + recompute", markersize=7)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Steering strength (fraction of mean norm)")
    ax.set_title(f"Layer {layer}")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, None)
    ax.legend(fontsize=8)

axes[0].set_ylabel("P(model chose steered task)")
fig.suptitle("Does suffix recomputation amplify steering?", fontsize=13)
fig.tight_layout()
fig.savefig(ASSETS / "plot_031826_recompute_comparison.png", dpi=150)
print("Saved recompute comparison plot")
