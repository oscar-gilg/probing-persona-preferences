"""Plot steering effect vs pre-existing preference strength."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.steering.analysis import load_checkpoint, filter_valid, chose_steered_task

CHECKPOINT = "experiments/steering/isolated_steering/checkpoint_hook_pilot.jsonl"
ASSETS = Path("experiments/steering/isolated_steering/hook_patching_pilot/assets")

rows = filter_valid(load_checkpoint(CHECKPOINT))

# Focus on L25 splice only (strongest clean signal)
rows = [r for r in rows if r["condition"] == "hook_patching" and r["layer"] == 25]

multipliers = sorted(set(abs(r["signed_multiplier"]) for r in rows))

# For each (pair, |mult|), compute:
#   toward = mean(chose_steered) for positive-signed trials
#   away = mean(chose_steered) for negative-signed trials
#   effect = toward - away
pair_ids = sorted(set(r["pair_id"] for r in rows))

for mult in multipliers:
    effects = []
    delta_mus = []

    for pid in pair_ids:
        toward = [r for r in rows if r["pair_id"] == pid and r["signed_multiplier"] == mult]
        away = [r for r in rows if r["pair_id"] == pid and r["signed_multiplier"] == -mult]
        if not toward or not away:
            continue

        p_toward = np.mean([1 if chose_steered_task(r) else 0 for r in toward])
        p_away = np.mean([1 if chose_steered_task(r) else 0 for r in away])
        effects.append(p_toward - p_away)
        delta_mus.append(abs(toward[0]["delta_mu"]))

    effects = np.array(effects)
    delta_mus = np.array(delta_mus)

    # Bin by |delta_mu|
    bins = [(0, 0.5, "small\n|delta_mu| < 0.5"), (0.5, 1.0, "medium\n0.5-1.0"), (1.0, 3.0, "large\n|delta_mu| > 1.0")]

    fig, ax = plt.subplots(figsize=(7, 5))
    positions = []
    labels = []
    for i, (lo, hi, label) in enumerate(bins):
        mask = (delta_mus >= lo) & (delta_mus < hi)
        bin_effects = effects[mask]
        if len(bin_effects) == 0:
            continue
        vp = ax.violinplot(bin_effects, positions=[i], showmedians=True, widths=0.7)
        for body in vp["bodies"]:
            body.set_facecolor("#2563eb")
            body.set_alpha(0.3)
        for key in ["cmins", "cmaxes", "cmedians", "cbars"]:
            vp[key].set_color("#2563eb")
        # Overlay individual points
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(bin_effects))
        ax.scatter(np.full_like(bin_effects, i) + jitter, bin_effects, color="#2563eb", alpha=0.6, s=30, zorder=3)
        positions.append(i)
        labels.append(f"{label}\n(n={len(bin_effects)})")

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Pre-existing preference gap between tasks")
    ax.set_ylabel("P(chose steered) - P(chose unsteered)")
    ax.set_ylim(-0.5, 1.2)
    ax.set_title(f"Layer 25, splice only, |mult|={mult}", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / f"plot_031826_effect_by_preference_{mult:.3f}.png", dpi=150)
    print(f"Saved effect_by_preference for |mult|={mult}")
