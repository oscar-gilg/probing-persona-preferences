"""Plots for persona_prompt_tracking experiment.

Generates three plots from results.json:
1. Scatter of r_w1 vs r_w0 across conditions (headline).
2. Grouped per-condition bar chart: r_w0, r_w1, r_random.
3. Strip plot of baseline half-seed null distribution for w0 vs w1.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

RESULTS_PATH = Path(
    "/Users/oscargilg/Dev/MATS/Preferences/.claude/worktrees/probe_direction_uniqueness"
    "/experiments/probe_science/probe_direction_uniqueness/persona_prompt_tracking/results.json"
)
ASSETS_DIR = RESULTS_PATH.parent / "assets"

EXP_COLORS = {"1b": "#1f77b4", "2": "#ff7f0e", "3": "#2ca02c"}


def load_results() -> dict:
    with RESULTS_PATH.open() as f:
        return json.load(f)


def short_label(cond: dict) -> str:
    exp = cond["experiment"]
    if exp == "1b":
        persona = cond["persona"]
        if persona == "baseline":
            return "base"
        stem = persona.replace("_pos_persona", "").replace("_neg_persona", "")
        initials = "".join(part[0] for part in stem.split("_")).upper()
        sign = "+" if "pos_persona" in persona else "-"
        return f"{initials}{sign}"
    return f"{cond['persona']}/{cond['split']}"


def sort_conditions(conditions: list[dict]) -> list[dict]:
    """1b baseline first then alphabetical, then exp2 villain/midwest, then exp3 sadist."""
    def key(c: dict) -> tuple:
        exp = c["experiment"]
        if exp == "1b":
            # baseline sorts first
            return (0, 0 if c["is_baseline"] else 1, c["persona"])
        if exp == "2":
            persona_order = {"villain": 0, "midwest": 1}
            return (1, persona_order[c["persona"]], c["split"])
        return (2, 0, c["split"])
    return sorted(conditions, key=key)


def marker_for(cond: dict) -> str:
    if cond["experiment"] == "1b":
        if cond["is_baseline"]:
            return "*"
        return "^" if "pos_persona" in cond["persona"] else "v"
    return "o"


def plot_scatter(conditions: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))

    null_halfwidth = max(c["r_random_p95_abs"] for c in conditions)

    # Reference null band (centered at 0)
    ax.add_patch(
        Rectangle(
            (-null_halfwidth, -null_halfwidth),
            2 * null_halfwidth,
            2 * null_halfwidth,
            facecolor="0.9",
            edgecolor="0.7",
            linestyle="--",
            linewidth=1.0,
            zorder=0,
            label=f"max random-null p95 band (\u00b1{null_halfwidth:.2f})",
        )
    )

    # Identity line
    lo, hi = -0.3, 0.9
    ax.plot([lo, hi], [lo, hi], color="black", linestyle=":", linewidth=1.0, zorder=1, label="y = x")

    for cond in conditions:
        color = EXP_COLORS[cond["experiment"]]
        marker = marker_for(cond)
        ax.scatter(
            cond["r_w0"],
            cond["r_w1"],
            color=color,
            marker=marker,
            s=140,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            short_label(cond),
            (cond["r_w0"], cond["r_w1"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
            zorder=4,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.axhline(0, color="0.6", linewidth=0.5)
    ax.axvline(0, color="0.6", linewidth=0.5)
    ax.set_xlabel("r(\u0175\u2080 \u00b7 x, Thurstonian \u03bc) \u2014 canonical preference direction")
    ax.set_ylabel("r(\u0175\u2081 \u00b7 x, Thurstonian \u03bc) \u2014 orthogonal residual")
    ax.set_title("Does \u0175\u2081 track persona shifts like \u0175\u2080 does? (L32 tb-2, Gemma-3-27B)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    legend_elems = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray", markersize=14, label="baseline (1b)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=11, label="pos persona (1b)"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="gray", markersize=11, label="neg persona (1b)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label="split (exp 2/3)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=EXP_COLORS["1b"], markersize=10, label="exp 1b (persona prompts)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=EXP_COLORS["2"], markersize=10, label="exp 2 (villain / midwest)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=EXP_COLORS["3"], markersize=10, label="exp 3 (sadist)"),
        Line2D([0], [0], color="black", linestyle=":", label="y = x"),
        Line2D([0], [0], color="0.7", linestyle="--", label="null band"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_per_condition_bars(conditions: list[dict], out_path: Path) -> None:
    ordered = sort_conditions(conditions)
    labels = [short_label(c) for c in ordered]
    r_w0 = [c["r_w0"] for c in ordered]
    r_w1 = [c["r_w1"] for c in ordered]
    r_rand = [c["r_random_mean"] for c in ordered]
    r_rand_std = [c["r_random_std"] for c in ordered]

    x = np.arange(len(ordered))
    width = 0.27

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width, r_w0, width, label="\u0175\u2080", color="#1f77b4")
    ax.bar(x, r_w1, width, label="\u0175\u2081", color="#d62728")
    ax.bar(
        x + width,
        r_rand,
        width,
        yerr=r_rand_std,
        label="random (mean \u00b1 std over 5 seeds)",
        color="0.7",
        ecolor="black",
        capsize=3,
    )

    ax.axhline(0, color="black", linewidth=0.8)

    # Vertical separators between experiments
    prev_exp = ordered[0]["experiment"]
    for i, c in enumerate(ordered):
        if c["experiment"] != prev_exp:
            ax.axvline(i - 0.5, color="0.4", linewidth=1.0, linestyle="--")
            prev_exp = c["experiment"]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Pearson r with Thurstonian \u03bc")
    ax.set_title("Per-condition Pearson r: \u0175\u2080 vs \u0175\u2081 vs random direction")
    ax.set_ylim(-0.6, 1.0)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_baseline_null(baseline_halves: list[dict], out_path: Path) -> None:
    r_w0 = np.array([h["r_w0"] for h in baseline_halves])
    r_w1 = np.array([h["r_w1"] for h in baseline_halves])

    fig, ax = plt.subplots(figsize=(6, 6))

    data = [r_w0, r_w1]
    positions = [0, 1]

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.45,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], ["#1f77b4", "#d62728"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    rng = np.random.default_rng(0)
    for pos, values in zip(positions, data):
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            np.full_like(values, pos) + jitter,
            values,
            color="black",
            s=30,
            zorder=3,
            alpha=0.8,
        )

    ax.axhline(0, color="0.3", linestyle="--", linewidth=1.0)

    ax.set_xticks(positions)
    ax.set_xticklabels(["\u0175\u2080", "\u0175\u2081"])
    ax.set_ylabel("Pearson r (baseline half vs Thurstonian \u03bc)")
    ax.set_ylim(-0.35, 0.35)
    ax.set_title("Baseline halves null: \u0175\u2081 on random baseline-half splits (n=24 tasks each)")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    results = load_results()
    conditions = results["conditions"]
    baseline_halves = results["baseline_halves_null"]

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    scatter_path = ASSETS_DIR / "plot_041926_rw1_vs_rw0_scatter.png"
    bars_path = ASSETS_DIR / "plot_041926_per_condition_r_bar.png"
    null_path = ASSETS_DIR / "plot_041926_baseline_halves_null.png"

    plot_scatter(conditions, scatter_path)
    plot_per_condition_bars(conditions, bars_path)
    plot_baseline_null(baseline_halves, null_path)

    r_w0 = np.array([c["r_w0"] for c in conditions])
    r_w1 = np.array([c["r_w1"] for c in conditions])

    print(f"Saved: {scatter_path}")
    print(f"Saved: {bars_path}")
    print(f"Saved: {null_path}")
    print()
    print(f"n conditions: {len(conditions)}")
    print(f"r_w0  max={r_w0.max():+.4f}  median={np.median(r_w0):+.4f}  min={r_w0.min():+.4f}")
    print(f"r_w1  max={r_w1.max():+.4f}  median={np.median(r_w1):+.4f}  min={r_w1.min():+.4f}")
    n_w1_greater = int(np.sum(np.abs(r_w1) > np.abs(r_w0)))
    print(f"|r_w1| > |r_w0|: {n_w1_greater}/{len(conditions)}")


if __name__ == "__main__":
    main()
