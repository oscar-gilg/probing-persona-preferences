"""Histogram of per-pair p_a (= #A choices / 3 across canonical-order seeds) for B0
and the four probe-ablated cells.

Usage: python -m scripts.preference_direction_ablation.plot_pa_distributions
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("experiments/preference_direction_ablation/results")
OUTPUT_PATH = Path(
    "experiments/preference_direction_ablation/assets/plot_042826_p_a_distributions.png"
)

CELLS = ["B0", "A_L25_probe", "A_L32_probe", "B_two_probe", "C_band_probe"]
CELL_COLORS = {
    "B0": "#888888",
    "A_L25_probe": "#1f77b4",
    "A_L32_probe": "#d62728",
    "B_two_probe": "#2ca02c",
    "C_band_probe": "#9467bd",
}

# Discrete p_a values when all 3 seeds responded with a/b
PA_LEVELS = [0.0, 1 / 3, 2 / 3, 1.0]
PA_LABELS = ["0", "1/3", "2/3", "1"]


def load_pa_per_pair(cell: str) -> tuple[dict[tuple[str, str], float], int]:
    """Return {(task_a, task_b): p_a} for pairs with all 3 canonical seeds responded
    plus the count of pairs with >=1 refusal/parse_error."""
    path = RESULTS_DIR / cell / "measurements.jsonl"
    by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    for line in path.open():
        row = json.loads(line)
        if row["order"] != "canonical":
            continue
        by_pair[(row["task_a"], row["task_b"])].append(row["choice_canonical"])

    pa_per_pair: dict[tuple[str, str], float] = {}
    refusal_pairs = 0
    for pair, choices in by_pair.items():
        if len(choices) != 3:
            raise ValueError(f"{cell} {pair}: expected 3 canonical seeds, got {len(choices)}")
        if any(c not in ("a", "b") for c in choices):
            refusal_pairs += 1
            continue
        pa_per_pair[pair] = sum(c == "a" for c in choices) / 3.0
    return pa_per_pair, refusal_pairs


def main() -> None:
    cell_data = {cell: load_pa_per_pair(cell) for cell in CELLS}
    n_pairs = {cell: len(d[0]) + d[1] for cell, d in cell_data.items()}
    print("Pairs per cell:")
    for cell in CELLS:
        responded, refusal = len(cell_data[cell][0]), cell_data[cell][1]
        print(f"  {cell}: {responded} responded, {refusal} with refusal/parse_error (total {n_pairs[cell]})")

    fig, axes = plt.subplots(len(CELLS), 1, figsize=(8.5, 10.5), sharex=True)

    # 5 bin positions on x: 0, 1/3, 2/3, 1, refusal
    x_positions = [0, 1, 2, 3, 4.2]  # gap before refusal
    x_labels = PA_LABELS + ["refusal"]

    for ax, cell in zip(axes, CELLS):
        pa_per_pair, refusal_count = cell_data[cell]
        counts = [
            sum(1 for v in pa_per_pair.values() if np.isclose(v, level))
            for level in PA_LEVELS
        ]
        bar_heights = counts + [refusal_count]
        ax.bar(
            x_positions,
            bar_heights,
            width=0.7,
            color=CELL_COLORS[cell],
            edgecolor="black",
            linewidth=0.5,
        )
        for x, h in zip(x_positions, bar_heights):
            ax.text(x, h + max(bar_heights) * 0.02, str(h), ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("count")
        ax.set_ylim(0, max(bar_heights) * 1.18)
        ax.set_title(
            f"{cell}  (n_responded={len(pa_per_pair)}, n_refusal={refusal_count})",
            loc="left",
            fontsize=11,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(3.6, color="lightgray", linestyle="--", linewidth=0.8)

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(x_labels)
    axes[-1].set_xlabel("p(chose A) across 3 canonical-order seeds")

    fig.suptitle(
        "Per-pair p(chose A) — B0 vs probe-ablated cells",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=160)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
