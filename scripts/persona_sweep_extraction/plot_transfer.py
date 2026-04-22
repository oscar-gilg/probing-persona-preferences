"""Plot the four probe-transfer figures + the per-(selector, layer) appendix grid.

Reads the NPZ outputs of analyze_transfer.py and writes PNGs to
experiments/persona_sweep/probe_transfer/assets/.

Headline (selector, layer) is chosen as whichever minimises |diag - off_diag|
gap while keeping mean diagonal r above 0.7 (an indirect proxy for a reliable
probe); falls back to (tb:-2, layer 32) if none qualify.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "experiments/persona_sweep/probe_transfer/results"
ASSETS_DIR = REPO / "experiments/persona_sweep/probe_transfer/assets"

SELECTORS = [("tb-2", "turn_boundary:-2"), ("tb-5", "turn_boundary:-5")]
LAYERS = [25, 32, 39, 46, 53]
TODAY = datetime.now().strftime("%m%d%y")


def _heatmap(ax, M: np.ndarray, labels: list[str], title: str, vmin: float = -1.0, vmax: float = 1.0, cmap: str = "RdBu_r") -> None:
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(M[i, j]) < 0.6 else "white")
    ax.set_title(title)
    return im


def pick_headline(all_transfer: dict[tuple[str, int], np.ndarray]) -> tuple[str, int]:
    best_mean_off = -np.inf
    best_key = None
    for key, tr in all_transfer.items():
        n = tr.shape[0]
        diag_mean = np.diag(tr).mean()
        if diag_mean < 0.7:
            continue
        off_mean = tr[~np.eye(n, dtype=bool)].mean()
        if off_mean > best_mean_off:
            best_mean_off = off_mean
            best_key = key
    if best_key is None:
        return ("tb-2", 32)
    return best_key


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    util = np.load(RESULTS_DIR / "utility_similarity.npz", allow_pickle=True)
    personas = [str(p) for p in util["personas"]]
    utility_r = util["utility_r"]

    all_transfer: dict[tuple[str, int], np.ndarray] = {}
    all_cosine: dict[tuple[str, int], np.ndarray] = {}
    for tag, _ in SELECTORS:
        for layer in LAYERS:
            path = RESULTS_DIR / f"transfer_{tag}_L{layer}.npz"
            d = np.load(path, allow_pickle=True)
            all_transfer[(tag, layer)] = d["transfer_r"]
            all_cosine[(tag, layer)] = d["probe_cosine"]

    headline_key = pick_headline(all_transfer)
    print(f"headline: {headline_key}")

    # Headline transfer heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    _heatmap(ax, all_transfer[headline_key], personas,
             f"Probe-transfer Pearson r — {headline_key[0]}, layer {headline_key[1]}")
    ax.set_xlabel("eval persona (test utility)")
    ax.set_ylabel("train persona (probe)")
    plt.colorbar(ax.images[0], ax=ax, label="Pearson r", fraction=0.04)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / f"plot_{TODAY}_transfer_heatmap.png", dpi=150)
    plt.close(fig)

    # Utility similarity heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    _heatmap(ax, utility_r, personas, "Utility-utility Pearson r (test split)")
    ax.set_xlabel("persona B")
    ax.set_ylabel("persona A")
    plt.colorbar(ax.images[0], ax=ax, label="Pearson r", fraction=0.04)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / f"plot_{TODAY}_utility_similarity_heatmap.png", dpi=150)
    plt.close(fig)

    # Transfer vs utility scatter
    transfer = all_transfer[headline_key]
    n = len(personas)
    off_mask = ~np.eye(n, dtype=bool)
    x = utility_r[off_mask]
    y = transfer[off_mask]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([-1, 1], [-1, 1], "k--", alpha=0.3, label="y = x")
    ax.scatter(x, y, s=40, alpha=0.75)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ax.annotate(f"{personas[i][:3]}→{personas[j][:3]}", (utility_r[i, j], transfer[i, j]),
                        fontsize=6, alpha=0.6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("utility-utility Pearson r (A vs B on test)")
    ax.set_ylabel("probe-transfer Pearson r (A→B)")
    ax.set_title(f"Probe transfer vs utility similarity — {headline_key[0]}, L{headline_key[1]}")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / f"plot_{TODAY}_transfer_vs_utility_scatter.png", dpi=150)
    plt.close(fig)

    # Probe cosine heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    _heatmap(ax, all_cosine[headline_key], personas,
             f"Probe direction cosine — {headline_key[0]}, layer {headline_key[1]}")
    plt.colorbar(ax.images[0], ax=ax, label="cosine", fraction=0.04)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / f"plot_{TODAY}_probe_cosine_heatmap.png", dpi=150)
    plt.close(fig)

    # Appendix grid: 2 selectors × 5 layers = 10 panels
    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    for row, (tag, _) in enumerate(SELECTORS):
        for col, layer in enumerate(LAYERS):
            ax = axes[row, col]
            _heatmap(ax, all_transfer[(tag, layer)], personas, f"{tag}, L{layer}")
            if col == 0:
                ax.set_ylabel("train")
            if row == 1:
                ax.set_xlabel("eval")
    fig.suptitle("Probe-transfer Pearson r across selector × layer", fontsize=14)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / f"plot_{TODAY}_transfer_heatmap_grid.png", dpi=130)
    plt.close(fig)

    print(f"wrote 5 figures to {ASSETS_DIR.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
