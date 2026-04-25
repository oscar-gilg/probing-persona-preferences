"""Headline grid: |Cohen's d| for every probe x (domain, turn) cell.

Rows: 6 (domain, turn) cells. Columns: 6 Qwen probes + 1 Gemma reference column.
Highlights the per-row max among Qwen probes with a thick border + bold text.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

EXP_DIR = Path("experiments/token_level_probes/qwen_canonical_probe_eval")
ASSETS_DIR = EXP_DIR / "assets"
QWEN_CSV = EXP_DIR / "headline_table.csv"
DATE = "042526"

QWEN_PROBES = [
    "qwen_tb-1_L33",
    "qwen_tb-1_L38",
    "qwen_tb-1_L43",
    "qwen_tb-4_L33",
    "qwen_tb-4_L38",
    "qwen_tb-4_L43",
]

# Rows: (domain_in_csv, turn_in_csv, display_label, gemma_reference_d).
ROWS = [
    ("truth", "user", "truth (user)", 3.35),
    ("truth", "assistant", "truth (asst)", 2.47),
    ("harm", "user", "harm (user)", 2.05),
    ("harm", "assistant", "harm (asst)", 2.12),
    ("politics_democrat", "assistant", "politics — democrat (asst)", 2.72),
    ("politics_republican", "assistant", "politics — republican (asst)", 1.11),
]

COLUMN_LABELS = [
    "tb-1 L33", "tb-1 L38", "tb-1 L43",
    "tb-4 L33", "tb-4 L38", "tb-4 L43",
    "Gemma ref",
]


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def lookup_abs_d(rows, domain, turn, probe):
    for r in rows:
        if r["domain"] == domain and r["turn"] == turn and r["probe"] == probe:
            return abs(float(r["d"]))
    raise KeyError(f"({domain}, {turn}, {probe}) not in CSV")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    qwen_rows = load_csv(QWEN_CSV)

    n_rows = len(ROWS)
    n_cols = len(QWEN_PROBES) + 1

    grid = np.zeros((n_rows, n_cols))
    for i, (domain, turn, _label, gemma_d) in enumerate(ROWS):
        for j, probe in enumerate(QWEN_PROBES):
            grid[i, j] = lookup_abs_d(qwen_rows, domain, turn, probe)
        grid[i, n_cols - 1] = gemma_d

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    vmax = max(grid.max(), 3.5)
    im = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)

    qwen_max_per_row = grid[:, : len(QWEN_PROBES)].argmax(axis=1)

    for i in range(n_rows):
        for j in range(n_cols):
            val = grid[i, j]
            is_qwen_max = j < len(QWEN_PROBES) and j == qwen_max_per_row[i]
            text_color = "white" if val < vmax * 0.55 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold" if is_qwen_max else "normal",
            )
            if is_qwen_max:
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="red", linewidth=2.2,
                ))

    # Vertical separator before Gemma reference column.
    ax.axvline(len(QWEN_PROBES) - 0.5, color="white", linewidth=2.5)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(COLUMN_LABELS, fontsize=10, rotation=20, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([r[2] for r in ROWS], fontsize=10)
    ax.set_xlabel("Probe")
    ax.set_title(
        "Qwen-3.5-122B: |Cohen's d| at the end-of-turn token, full grid\n"
        "(red border = best Qwen probe per row; right column = Gemma reference)",
        fontsize=11,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("|Cohen's d|")

    plt.tight_layout()
    out_path = ASSETS_DIR / f"plot_{DATE}_qwen_headline_grid.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")
    for i, (_d, _t, label, _g) in enumerate(ROWS):
        best_j = qwen_max_per_row[i]
        print(f"  {label}: best Qwen probe = {QWEN_PROBES[best_j]} "
              f"(|d| = {grid[i, best_j]:.2f})")


if __name__ == "__main__":
    main()
