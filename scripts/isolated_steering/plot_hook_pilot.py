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

# Consistent colors: layer = color, mode = linestyle
LAYER_COLORS = {25: "#2563eb", 32: "#dc2626"}
MODE_STYLES = {"splice": ("o", "-"), "recompute": ("s", "--")}


def add_zero_and_negatives(subset):
    """Mirror positive data points to negative strengths and add zero baseline."""
    result = [{"steering_strength": 0, "p_steered": 0.5, "n": 0}]
    for r in sorted(subset, key=lambda r: r["steering_strength"]):
        result.append(r)
        result.insert(0, {"steering_strength": -r["steering_strength"], "p_steered": 1 - r["p_steered"], "n": r["n"]})
    return sorted(result, key=lambda r: r["steering_strength"])


# --- Plot 1: Dose-response, all conditions on one plot ---
fig, ax = plt.subplots(figsize=(8, 5))

for layer in layers:
    color = LAYER_COLORS[layer]
    for data, mode_label, (marker, ls) in [
        (no_recompute, "splice", MODE_STYLES["splice"]),
        (recompute, "recompute", MODE_STYLES["recompute"]),
    ]:
        subset = add_zero_and_negatives([r for r in data if r["layer"] == layer])
        strengths = [r["steering_strength"] for r in subset]
        p_steered = [r["p_steered"] for r in subset]
        alpha = 1.0 if mode_label == "splice" else 0.6
        ax.plot(strengths, p_steered, marker=marker, linestyle=ls, color=color,
                label=f"L{layer} {mode_label}", markersize=7, alpha=alpha)

ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
ax.set_xlabel("Steering strength (fraction of mean norm)\n(← steer toward B | steer toward A →)")
ax.set_ylabel("P(model chose steered task)")
ax.set_ylim(0, 1)
ax.set_title("Does activation patching causally shift task choice?", fontsize=13)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(ASSETS / "plot_031826_dose_response.png", dpi=150)
print("Saved dose-response plot")

# --- Plot 2: Recompute comparison, one panel per layer, consistent colors ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

for ax, layer in zip(axes, layers):
    color = LAYER_COLORS[layer]
    for data, mode_label, (marker, ls) in [
        (no_recompute, "Splice only", MODE_STYLES["splice"]),
        (recompute, "Splice + recompute", MODE_STYLES["recompute"]),
    ]:
        subset = add_zero_and_negatives([r for r in data if r["layer"] == layer])
        strengths = [r["steering_strength"] for r in subset]
        p_steered = [r["p_steered"] for r in subset]
        alpha = 1.0 if "only" in mode_label else 0.6
        ax.plot(strengths, p_steered, marker=marker, linestyle=ls, color=color,
                label=mode_label, markersize=7, alpha=alpha)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Steering strength (fraction of mean norm)\n(← steer toward B | steer toward A →)")
    ax.set_title(f"Layer {layer}")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

axes[0].set_ylabel("P(model chose steered task)")
fig.suptitle("Does suffix recomputation amplify steering?", fontsize=13)
fig.tight_layout()
fig.savefig(ASSETS / "plot_031826_recompute_comparison.png", dpi=150)
print("Saved recompute comparison plot")
