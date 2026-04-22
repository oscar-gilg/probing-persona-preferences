"""Replot scatter and appendix grid with cleaner layouts.

Reads the same NPZ outputs as plot_transfer.py but produces less cluttered
versions for the report: scatter is zoomed to data range with selective
annotation, appendix grid is taller and uses shared colorbar.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "experiments/persona_sweep/probe_transfer/results"
ASSETS_DIR = REPO / "experiments/persona_sweep/probe_transfer/assets"

SELECTORS = [("tb-2", "turn_boundary:-2"), ("tb-5", "turn_boundary:-5")]
LAYERS = [25, 32, 39, 46, 53]
TODAY = "042226"  # preserve existing filenames


def _heatmap(ax, M, labels, title, vmin=-1.0, vmax=1.0, cmap="RdBu_r", fontsize=8):
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=fontsize - 1, color="black" if abs(M[i, j]) < 0.6 else "white")
    ax.set_title(title)
    return im


def main() -> None:
    util = np.load(RESULTS_DIR / "utility_similarity.npz", allow_pickle=True)
    personas = [str(p) for p in util["personas"]]
    utility_r = util["utility_r"]

    all_transfer = {}
    for tag, _ in SELECTORS:
        for layer in LAYERS:
            d = np.load(RESULTS_DIR / f"transfer_{tag}_L{layer}.npz", allow_pickle=True)
            all_transfer[(tag, layer)] = d["transfer_r"]

    headline_key = ("tb-5", 32)
    transfer = all_transfer[headline_key]
    n = len(personas)
    off_mask = ~np.eye(n, dtype=bool)

    # --- Scatter: zoomed to data, labels via adjustText ---
    fig, ax = plt.subplots(figsize=(7, 6.5))
    x = utility_r[off_mask]
    y = transfer[off_mask]
    # Reference y = x over the visible range
    ax.plot([-0.4, 0.7], [-0.4, 0.7], "k--", alpha=0.35, label="y = x (probe transfer equals utility similarity)")
    ax.scatter(x, y, s=42, alpha=0.8, color="#1f77b4", edgecolor="white", linewidth=0.4, zorder=3)

    labels = []
    short = {"sadist": "sad", "mathematician": "mat", "aura": "aur",
             "strategist": "str", "contrarian": "con", "slacker": "sla", "default": "def"}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            labels.append(ax.text(utility_r[i, j], transfer[i, j],
                                   f"{short[personas[i]]}→{short[personas[j]]}",
                                   fontsize=7.5, alpha=0.85))
    adjust_text(labels, ax=ax, expand=(1.2, 1.3),
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.4, lw=0.4))

    ax.set_xlabel("utility correlation between personas A and B (Pearson r on test)")
    ax.set_ylabel("probe transfer Pearson r (probe trained on A, predicting B)")
    ax.set_title(f"Probe transfer exceeds behavioural similarity for every ordered pair\n(tb-5, layer 32; 42/42 off-diagonal points above y = x)")
    ax.set_xlim(-0.4, 0.7)
    ax.set_ylim(-0.2, 1.0)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / f"plot_{TODAY}_transfer_vs_utility_scatter.png", dpi=160)
    plt.close(fig)

    # --- Appendix grid: 2x5, shared colorbar, larger fonts ---
    fig, axes = plt.subplots(2, 5, figsize=(24, 11), constrained_layout=True)
    ims = []
    for row, (tag, _) in enumerate(SELECTORS):
        for col, layer in enumerate(LAYERS):
            ax = axes[row, col]
            im = _heatmap(ax, all_transfer[(tag, layer)], personas,
                           f"{tag}, layer {layer}", fontsize=7)
            ims.append(im)
            if col == 0:
                ax.set_ylabel("train persona", fontsize=9)
            if row == 1:
                ax.set_xlabel("eval persona", fontsize=9)
    fig.suptitle("Probe-transfer Pearson r across every selector x layer combination", fontsize=14)
    fig.colorbar(ims[-1], ax=axes, shrink=0.7, label="Pearson r")
    plt.savefig(ASSETS_DIR / f"plot_{TODAY}_transfer_heatmap_grid.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("replotted scatter + appendix grid")


if __name__ == "__main__":
    main()
