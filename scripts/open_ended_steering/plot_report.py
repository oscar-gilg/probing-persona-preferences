"""Generate plots for the open-ended steering experiment report."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = Path("experiments/steering/open_ended_steering/scored_results.jsonl")
ASSETS_DIR = Path("experiments/steering/open_ended_steering/assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

ITER0_CATEGORIES = {"introspective", "enjoyment", "creative", "neutral", "refusal"}
ITER1_CATEGORIES = {"choice", "willingness"}
ALL_CATEGORIES = ITER0_CATEGORIES | ITER1_CATEGORIES
MULTIPLIERS = [-0.05, -0.03, 0.0, 0.03, 0.05]
MODES = ["all_tokens", "generation_only", "prefill_only"]

MODE_COLORS = {
    "all_tokens": "#2196F3",
    "generation_only": "#FF9800",
    "prefill_only": "#4CAF50",
}

CATEGORY_COLORS = {
    "introspective": "#9C27B0",
    "enjoyment": "#E91E63",
    "creative": "#FF9800",
    "neutral": "#607D8B",
    "refusal": "#F44336",
    "choice": "#2196F3",
    "willingness": "#4CAF50",
}


def load_data() -> list[dict]:
    rows = []
    with open(DATA_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def setup_style():
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
    })


def mean_and_sem(values: list[float]) -> tuple[float, float]:
    arr = np.array(values)
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0


def plot_engagement_dose_response(data: list[dict]):
    """Plot 1: Engagement dose-response by steering mode (iter0 only)."""
    iter0 = [r for r in data if r["category"] in ITER0_CATEGORIES]

    # Group by (mode, multiplier)
    groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in iter0:
        groups[(r["steering_mode"], r["multiplier"])].append(r["engagement_score"])

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for mode in MODES:
        means, sems, mults = [], [], []
        for m in MULTIPLIERS:
            vals = groups[(mode, m)]
            if vals:
                mu, se = mean_and_sem(vals)
                means.append(mu)
                sems.append(se)
                mults.append(m)

        label = mode.replace("_", " ")
        ax.errorbar(
            mults, means, yerr=sems, marker="o", capsize=4, linewidth=2,
            label=label, color=MODE_COLORS[mode], markersize=6,
        )

    ax.set_xlabel("Steering multiplier")
    ax.set_ylabel("Mean engagement score")
    ax.set_ylim(-1, 1)
    ax.set_xlim(-0.06, 0.06)
    ax.set_xticks(MULTIPLIERS)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.legend(title="Steering mode")
    ax.set_title("Engagement score by steering strength and mode")
    fig.tight_layout()

    out = ASSETS_DIR / "plot_032326_engagement_dose_response.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_engagement_by_category(data: list[dict]):
    """Plot 2: Engagement by prompt category (all_tokens mode only)."""
    filtered = [r for r in data if r["steering_mode"] == "all_tokens"]

    # Group by (category, multiplier)
    groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in filtered:
        groups[(r["category"], r["multiplier"])].append(r["engagement_score"])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Sort categories: iter0 first, then iter1
    ordered_cats = sorted(ITER0_CATEGORIES) + sorted(ITER1_CATEGORIES)

    for cat in ordered_cats:
        means, sems, mults = [], [], []
        for m in MULTIPLIERS:
            vals = groups[(cat, m)]
            if vals:
                mu, se = mean_and_sem(vals)
                means.append(mu)
                sems.append(se)
                mults.append(m)

        if not means:
            continue

        ax.errorbar(
            mults, means, yerr=sems, marker="o", capsize=3, linewidth=2,
            label=cat, color=CATEGORY_COLORS[cat], markersize=5,
        )

    ax.set_xlabel("Steering multiplier")
    ax.set_ylabel("Mean engagement score")
    ax.set_ylim(-1, 1)
    ax.set_xlim(-0.06, 0.06)
    ax.set_xticks(MULTIPLIERS)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.legend(title="Category", loc="lower right", ncol=2)
    ax.set_title("Engagement by prompt category (all_tokens mode)")
    fig.tight_layout()

    out = ASSETS_DIR / "plot_032326_engagement_by_category.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_anomaly_rate(data: list[dict]):
    """Plot 3: Anomaly rate by multiplier and mode (iter0 only)."""
    iter0 = [r for r in data if r["category"] in ITER0_CATEGORIES]

    # Group by (mode, multiplier)
    totals: dict[tuple[str, float], int] = defaultdict(int)
    anomalous: dict[tuple[str, float], int] = defaultdict(int)
    for r in iter0:
        key = (r["steering_mode"], r["multiplier"])
        totals[key] += 1
        if r["is_anomalous"]:
            anomalous[key] += 1

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for mode in MODES:
        rates, mults = [], []
        for m in MULTIPLIERS:
            key = (mode, m)
            total = totals[key]
            if total > 0:
                rate = 100.0 * anomalous[key] / total
                rates.append(rate)
                mults.append(m)

        label = mode.replace("_", " ")
        ax.plot(
            mults, rates, marker="o", linewidth=2,
            label=label, color=MODE_COLORS[mode], markersize=6,
        )

    ax.set_xlabel("Steering multiplier")
    ax.set_ylabel("Anomaly rate (%)")
    ax.set_ylim(0, 50)
    ax.set_xlim(-0.06, 0.06)
    ax.set_xticks(MULTIPLIERS)
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.legend(title="Steering mode")
    ax.set_title("Anomaly rate by steering strength and mode")
    fig.tight_layout()

    out = ASSETS_DIR / "plot_032326_anomaly_rate.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    setup_style()
    data = load_data()
    print(f"Loaded {len(data)} rows")

    plot_engagement_dose_response(data)
    plot_engagement_by_category(data)
    plot_anomaly_rate(data)
    print("Done.")


if __name__ == "__main__":
    main()
