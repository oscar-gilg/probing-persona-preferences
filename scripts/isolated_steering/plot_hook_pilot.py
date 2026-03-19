"""Generate plots for the hook patching pilot report."""

from pathlib import Path

import matplotlib.pyplot as plt

from src.steering.analysis import load_checkpoint, aggregate

CHECKPOINT = "experiments/steering/isolated_steering/checkpoint_hook_pilot.jsonl"
ASSETS = Path("experiments/steering/isolated_steering/hook_patching_pilot/assets")
ASSETS.mkdir(parents=True, exist_ok=True)

rows = load_checkpoint(CHECKPOINT)
agg = aggregate(rows)

# Separate recompute vs non-recompute
no_recompute = [r for r in agg if r["condition"] == "hook_patching"]
recompute = [r for r in agg if r["condition"] == "hook_patching_recompute"]

layers = sorted(set(r["layer"] for r in agg))
colors = {25: "#2563eb", 32: "#dc2626"}

# --- Plot 1: Dose-response, faceted by recompute mode ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

for ax, data, title in [
    (axes[0], no_recompute, "No suffix recompute"),
    (axes[1], recompute, "With suffix recompute"),
]:
    for layer in layers:
        subset = [r for r in data if r["layer"] == layer]
        subset.sort(key=lambda r: r["signed_multiplier"])
        mults = [r["signed_multiplier"] for r in subset]
        p_as = [r["p_a"] for r in subset]
        ax.plot(mults, p_as, "o-", color=colors[layer], label=f"L{layer}", markersize=7)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Signed multiplier")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()

axes[0].set_ylabel("P(chose task A)")
fig.suptitle("Hook patching pilot: dose-response", fontsize=13)
fig.tight_layout()
fig.savefig(ASSETS / "plot_031826_dose_response.png", dpi=150)
print(f"Saved dose-response plot")

# --- Plot 2: Recompute comparison ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

for ax, layer in zip(axes, layers):
    nr = sorted([r for r in no_recompute if r["layer"] == layer], key=lambda r: r["signed_multiplier"])
    rc = sorted([r for r in recompute if r["layer"] == layer], key=lambda r: r["signed_multiplier"])
    mults = [r["signed_multiplier"] for r in nr]
    ax.plot(mults, [r["p_a"] for r in nr], "o-", color=colors[layer], label="No recompute", markersize=7)
    ax.plot(mults, [r["p_a"] for r in rc], "s--", color=colors[layer], alpha=0.6, label="Recompute", markersize=7)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Signed multiplier")
    ax.set_title(f"Layer {layer}")
    ax.set_ylim(0, 1)
    ax.legend()

axes[0].set_ylabel("P(chose task A)")
fig.suptitle("Hook patching pilot: recompute comparison", fontsize=13)
fig.tight_layout()
fig.savefig(ASSETS / "plot_031826_recompute_comparison.png", dpi=150)
print(f"Saved recompute comparison plot")
