"""Plots for persona_prompt_tracking experiment.

Generates plots from results.json:
1. Scatter of r_w1 vs r_w0 across conditions (parent comparison, y=x).
2. Scatter of r_w2 vs r_w0 across conditions (y=x).
3. Grouped per-condition bar chart: r_w0, r_w1, r_w2, r_random.
4. r_wk by exp-1b polarity (pos vs neg) with exp 2/3 grouping.
5. Strip/box plot of baseline half-seed null distribution for w0, w1, w2.
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

# Per-direction colors for bar charts and polarity plot.
W0_COLOR = "#1f77b4"
W1_COLOR = "#d62728"
W2_COLOR = "#9467bd"
RAND_COLOR = "0.7"


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


def plot_rwk_vs_rw0_scatter(
    conditions: list[dict],
    out_path: Path,
    y_key: str,
    y_label: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))

    null_halfwidth = max(c["r_random_p95_abs"] for c in conditions)

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

    lo, hi = -0.3, 0.9
    ax.plot([lo, hi], [lo, hi], color="black", linestyle=":", linewidth=1.0, zorder=1, label="y = x")

    for cond in conditions:
        color = EXP_COLORS[cond["experiment"]]
        marker = marker_for(cond)
        ax.scatter(
            cond["r_w0"],
            cond[y_key],
            color=color,
            marker=marker,
            s=140,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            short_label(cond),
            (cond["r_w0"], cond[y_key]),
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
    ax.set_ylabel(y_label)
    ax.set_title(title)
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
    r_w2 = [c["r_w2"] for c in ordered]
    r_rand = [c["r_random_mean"] for c in ordered]
    r_rand_std = [c["r_random_std"] for c in ordered]

    x = np.arange(len(ordered))
    width = 0.2

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(x - 1.5 * width, r_w0, width, label="\u0175\u2080", color=W0_COLOR)
    ax.bar(x - 0.5 * width, r_w1, width, label="\u0175\u2081", color=W1_COLOR)
    ax.bar(x + 0.5 * width, r_w2, width, label="\u0175\u2082", color=W2_COLOR)
    ax.bar(
        x + 1.5 * width,
        r_rand,
        width,
        yerr=r_rand_std,
        label="random (mean \u00b1 std over 5 seeds)",
        color=RAND_COLOR,
        ecolor="black",
        capsize=3,
    )

    ax.axhline(0, color="black", linewidth=0.8)

    prev_exp = ordered[0]["experiment"]
    for i, c in enumerate(ordered):
        if c["experiment"] != prev_exp:
            ax.axvline(i - 0.5, color="0.4", linewidth=1.0, linestyle="--")
            prev_exp = c["experiment"]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Pearson r with Thurstonian \u03bc")
    ax.set_title("Per-condition Pearson r: \u0175\u2080 vs \u0175\u2081 vs \u0175\u2082 vs random direction")
    ax.set_ylim(-0.6, 1.0)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_rwk_by_polarity(conditions: list[dict], out_path: Path) -> None:
    pos_1b = [
        c for c in conditions
        if c["experiment"] == "1b" and not c["is_baseline"] and "pos_persona" in c["persona"]
    ]
    neg_1b = [
        c for c in conditions
        if c["experiment"] == "1b" and not c["is_baseline"] and "neg_persona" in c["persona"]
    ]
    exp23 = [c for c in conditions if c["experiment"] in ("2", "3")]

    groups = [
        ("pos persona (1b, n=8)", pos_1b),
        ("neg persona (1b, n=8)", neg_1b),
        (f"exp 2/3 (n={len(exp23)})", exp23),
    ]

    def stats(conds: list[dict], key: str) -> tuple[float, float]:
        arr = np.array([c[key] for c in conds])
        return float(arr.mean()), float(arr.std())

    group_labels = [g[0] for g in groups]
    keys = [("r_w0", "\u0175\u2080", W0_COLOR), ("r_w1", "\u0175\u2081", W1_COLOR), ("r_w2", "\u0175\u2082", W2_COLOR)]

    x = np.arange(len(groups))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (key, label, color) in enumerate(keys):
        means = [stats(g[1], key)[0] for g in groups]
        stds = [stats(g[1], key)[1] for g in groups]
        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            label=label,
            color=color,
            ecolor="black",
            capsize=4,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Mean Pearson r (\u00b1 std across conditions)")
    ax.set_ylim(-0.2, 1.0)
    ax.set_title("r_wk by condition polarity \u2014 w\u2082 collapses on negative personas")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_baseline_null(baseline_halves: list[dict], out_path: Path) -> None:
    r_w0 = np.array([h["r_w0"] for h in baseline_halves])
    r_w1 = np.array([h["r_w1"] for h in baseline_halves])
    r_w2 = np.array([h["r_w2"] for h in baseline_halves])

    fig, ax = plt.subplots(figsize=(7, 6))

    data = [r_w0, r_w1, r_w2]
    positions = [0, 1, 2]
    colors = [W0_COLOR, W1_COLOR, W2_COLOR]

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.45,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], colors):
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
    ax.set_xticklabels(["\u0175\u2080", "\u0175\u2081", "\u0175\u2082"])
    ax.set_ylabel("Pearson r (baseline half vs Thurstonian \u03bc)")
    ax.set_ylim(-0.35, 0.35)
    ax.set_title("Baseline halves null: w\u2080/w\u2081/w\u2082 on random baseline-half splits (n=24 tasks each)")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    results = load_results()
    conditions = results["conditions"]
    baseline_halves = results["baseline_halves_null"]

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    rw1_scatter_path = ASSETS_DIR / "plot_041926_rw1_vs_rw0_scatter.png"
    rw2_scatter_path = ASSETS_DIR / "plot_041926_rw2_vs_rw0_scatter.png"
    bars_path = ASSETS_DIR / "plot_041926_per_condition_r_bar.png"
    polarity_path = ASSETS_DIR / "plot_041926_rw2_delta_by_polarity.png"
    null_path = ASSETS_DIR / "plot_041926_baseline_halves_null.png"

    plot_rwk_vs_rw0_scatter(
        conditions,
        rw1_scatter_path,
        y_key="r_w1",
        y_label="r(\u0175\u2081 \u00b7 x, Thurstonian \u03bc) \u2014 orthogonal residual (k=1)",
        title="Does \u0175\u2081 track persona shifts like \u0175\u2080 does? (L32 tb-2, Gemma-3-27B)",
    )
    plot_rwk_vs_rw0_scatter(
        conditions,
        rw2_scatter_path,
        y_key="r_w2",
        y_label="r(\u0175\u2082 \u00b7 x, Thurstonian \u03bc) \u2014 orthogonal residual (k=2)",
        title="Does \u0175\u2082 track persona shifts like \u0175\u2080 does? (L32 tb-2, Gemma-3-27B)",
    )
    plot_per_condition_bars(conditions, bars_path)
    plot_rwk_by_polarity(conditions, polarity_path)
    plot_baseline_null(baseline_halves, null_path)

    r_w0 = np.array([c["r_w0"] for c in conditions])
    r_w1 = np.array([c["r_w1"] for c in conditions])
    r_w2 = np.array([c["r_w2"] for c in conditions])

    print(f"Saved: {rw1_scatter_path}")
    print(f"Saved: {rw2_scatter_path}")
    print(f"Saved: {bars_path}")
    print(f"Saved: {polarity_path}")
    print(f"Saved: {null_path}")
    print()
    print(f"n conditions: {len(conditions)}")
    print(f"r_w0  max={r_w0.max():+.4f}  median={np.median(r_w0):+.4f}  min={r_w0.min():+.4f}")
    print(f"r_w1  max={r_w1.max():+.4f}  median={np.median(r_w1):+.4f}  min={r_w1.min():+.4f}")
    print(f"r_w2  max={r_w2.max():+.4f}  median={np.median(r_w2):+.4f}  min={r_w2.min():+.4f}")

    # Exp 1b polarity breakdown for headline observation.
    pos_1b = [
        c for c in conditions
        if c["experiment"] == "1b" and not c["is_baseline"] and "pos_persona" in c["persona"]
    ]
    neg_1b = [
        c for c in conditions
        if c["experiment"] == "1b" and not c["is_baseline"] and "neg_persona" in c["persona"]
    ]
    print()
    print(f"Exp 1b pos persona (n={len(pos_1b)}):")
    for k in ("r_w0", "r_w1", "r_w2"):
        arr = np.array([c[k] for c in pos_1b])
        print(f"  mean {k} = {arr.mean():+.4f}  (std {arr.std():.4f})")
    print(f"Exp 1b neg persona (n={len(neg_1b)}):")
    for k in ("r_w0", "r_w1", "r_w2"):
        arr = np.array([c[k] for c in neg_1b])
        print(f"  mean {k} = {arr.mean():+.4f}  (std {arr.std():.4f})")


if __name__ == "__main__":
    main()
