"""Regenerate heldout-r plot with clearer legend; add per-topic HOO plot."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

EXPERIMENT_DIR = Path(
    "/Users/oscargilg/Dev/MATS/Preferences/.claude/worktrees/"
    "probe_direction_uniqueness/experiments/probe_science/probe_direction_uniqueness"
)
OUTPUT_DIR = EXPERIMENT_DIR / "output" / "L32"
ASSETS_DIR = EXPERIMENT_DIR / "assets"
DATE = "041826"


def load_trajectory() -> dict:
    with open(OUTPUT_DIR / "trajectory.json") as f:
        return json.load(f)


def plot_heldout_r_vs_iter(traj: dict) -> Path:
    iters = [t["iter"] for t in traj["trajectory"]]
    final_r = [t["final_r"] for t in traj["trajectory"]]
    hoo_mean_r = [t["hoo_mean_r"] for t in traj["trajectory"]]

    shuffled_final_rs = [s["final_r"] for s in traj["shuffled_runs"]]
    r_chance_low = min(shuffled_final_rs)
    r_chance_high = max(shuffled_final_rs)
    r_chance = traj["r_chance"]
    threshold = traj["threshold"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhspan(
        r_chance_low,
        r_chance_high,
        alpha=0.15,
        color="gray",
        label=f"shuffled r range [{r_chance_low:.3f}, {r_chance_high:.3f}]",
    )
    ax.axhline(
        r_chance,
        color="gray",
        linestyle=":",
        linewidth=1,
        label=f"r_chance (p95 |r|) = {r_chance:.3f}",
    )
    ax.axhline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"stopping threshold = {threshold:.3f}",
    )

    ax.plot(
        iters,
        final_r,
        marker="s",
        color="tab:orange",
        label="Heldout r (final half)",
    )
    ax.plot(
        iters,
        hoo_mean_r,
        marker="D",
        color="tab:red",
        label="HOO r (mean across 13 topics)",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Pearson r (preference)")
    ax.set_title(
        "Preference signal vs iterations of probe projection (Gemma-3-27B L32)"
    )
    ax.set_xticks(iters)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    path = ASSETS_DIR / f"plot_{DATE}_heldout_r_vs_iter.png"
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def plot_hoo_per_topic(traj: dict) -> Path:
    iters = [t["iter"] for t in traj["trajectory"]]
    groups = [f["group"] for f in traj["trajectory"][0]["hoo_folds"]]
    hoo_by_group: dict[str, list[float]] = {g: [] for g in groups}
    for t in traj["trajectory"]:
        for f in t["hoo_folds"]:
            hoo_by_group[f["group"]].append(f["hoo_r"])

    iter0_values = {g: hoo_by_group[g][0] for g in groups}
    vmin = min(iter0_values.values())
    vmax = max(iter0_values.values())
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)

    for g in groups:
        ys = hoo_by_group[g]
        color = cmap(norm(iter0_values[g]))
        ax.plot(iters, ys, marker="o", color=color, linewidth=1.6, alpha=0.9)
        ax.annotate(
            g,
            xy=(iters[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color=color,
            va="center",
        )

    ax.set_xlabel(
        "Iteration (0 = no projection; 1 = after removing w_0; "
        "2 = after removing w_0 and w_1)"
    )
    ax.set_ylabel("HOO Pearson r")
    ax.set_title("Per-topic HOO r across iterations (14 topics)")
    ax.set_xticks(iters)
    ax.set_ylim(-0.6, 1.0)
    ax.set_xlim(iters[0] - 0.1, iters[-1] + 0.6)
    ax.grid(True, alpha=0.3)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.12)
    cbar.set_label("HOO r at iter 0 (dark = high)")

    path = ASSETS_DIR / f"plot_{DATE}_hoo_per_topic.png"
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    traj = load_trajectory()
    p1 = plot_heldout_r_vs_iter(traj)
    p2 = plot_hoo_per_topic(traj)
    print(f"Heldout-r plot: {p1}")
    print(f"Per-topic HOO plot: {p2}")


if __name__ == "__main__":
    main()
