"""Plot the raw transfer r and partial r(pred, u_eval | u_train) side by side."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments/qwen_replication/persona_transfer/probe_transfer/results"
ASSETS = REPO / "experiments/qwen_replication/persona_transfer/probe_transfer/assets"
TODAY = datetime.now().strftime("%m%d%y")
SELECTOR_TAG = "tb-4"
LAYER = 38


def _load_utility():
    d = np.load(RESULTS / "utility_similarity.npz", allow_pickle=True)
    return [str(p) for p in d["personas"]], d["U"]


def _qwen_order(personas, U):
    i_def = personas.index("default")
    others = [i for i in range(len(personas)) if i != i_def]
    others_sorted = sorted(others, key=lambda i: -U[i_def, i])
    return [i_def] + others_sorted


def _heatmap(ax, M, labels, title, vmin=-1, vmax=1):
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if abs(M[i, j]) < 0.6 else "white")
    ax.set_title(title, fontsize=11)
    return im


def main() -> None:
    personas, U = _load_utility()
    order = _qwen_order(personas, U)
    labels = [personas[i] for i in order]

    d = np.load(RESULTS / f"partial_eval_given_train_{SELECTOR_TAG}_L{LAYER}.npz", allow_pickle=True)
    raw = d["raw"][np.ix_(order, order)]
    partial = d["partial"][np.ix_(order, order)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    _heatmap(axes[0], raw, labels, "Raw transfer  r(pred, u_eval)")
    axes[0].set_xlabel("eval persona"); axes[0].set_ylabel("probe persona")
    im2 = _heatmap(axes[1], partial, labels, "Partial  r(pred, u_eval | u_train)")
    axes[1].set_xlabel("eval persona"); axes[1].set_ylabel("probe persona")
    fig.suptitle(f"Probe transfer with vs without train-bias control — tb-4, L{LAYER}", fontsize=13)
    fig.colorbar(im2, ax=axes, fraction=0.025, pad=0.015, label="Pearson r", shrink=0.85)

    out = ASSETS / f"plot_{TODAY}_partial_eval_given_train.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
