"""Figure: cross-persona probe transfer (7×7), single matrix coloured by Δ.

Each cell shows the Pearson r between the probe's predictions on the target
persona's activations and that target's own utilities (bold) plus the bare
Pearson r between train and target persona utilities (in parens, purple,
echoing the naive baseline of Fig.~ref{fig:default-probe}). Cell colour =
Δ = probe r − utility r. Diagonal is masked (utility r ≡ 1 there);
diagonal entries show the self-fit probe r for reference.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "experiments/persona_sweep/probe_transfer/results"
OUT_PNG = REPO / "paper/figures/main/plot_050626_persona_transfer_delta.png"

BLUE = "#1a73e8"
BLUE_LIGHT = "#bfdbfe"
NAIVE = "#7c3aed"
NAIVE_LIGHT = "#c4b5fd"

DELTA_CMAP = LinearSegmentedColormap.from_list(
    "delta_red", ["#ffffff", "#fee2e2", "#fca5a5", "#ef4444", "#991b1b"]
)

DISPLAY = {"sadist": "evil", "default": "Assistant"}


def load() -> dict:
    t = np.load(RESULTS / "transfer_tb-5_L32.npz", allow_pickle=True)
    u = np.load(RESULTS / "utility_similarity.npz", allow_pickle=True)
    personas = [str(p) for p in t["personas"]]
    transfer = t["transfer_r"]
    utility = u["utility_r"]
    assert [str(p) for p in u["personas"]] == personas

    i_def = personas.index("default")
    others = [i for i in range(len(personas)) if i != i_def]
    others_sorted = sorted(others, key=lambda i: -utility[i_def, i])
    order = [i_def] + others_sorted

    transfer = transfer[np.ix_(order, order)]
    utility = utility[np.ix_(order, order)]
    labels = [personas[i] for i in order]
    delta = transfer - utility
    delta_off = delta.copy()
    np.fill_diagonal(delta_off, np.nan)

    return {
        "labels": labels,
        "display": [DISPLAY.get(l, l) for l in labels],
        "transfer": transfer,
        "utility": utility,
        "delta_off": delta_off,
        "n": len(labels),
    }


def render(d: dict, path: Path) -> None:
    n = d["n"]
    delta = d["delta_off"]
    transfer = d["transfer"]
    utility = d["utility"]
    vmax = float(np.nanmax(delta))

    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    im = ax.imshow(delta, cmap=DELTA_CMAP, vmin=0.0, vmax=vmax,
                   aspect="equal", origin="upper")

    dot_x = -0.22
    text_x = -0.05
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor="#f3f4f6",
                                           edgecolor="white", lw=0.0,
                                           hatch="///", alpha=0.9))
                ax.text(j, i, f"{transfer[i, i]:.2f}", ha="center",
                        va="center", fontsize=9, color="#6b7280",
                        fontweight="bold")
                continue
            dark = delta[i, j] > 0.55 * vmax
            probe_color = BLUE_LIGHT if dark else BLUE
            util_color = NAIVE_LIGHT if dark else NAIVE
            ax.scatter([j + dot_x], [i - 0.13], color=BLUE, s=85, zorder=3,
                       edgecolors="none")
            ax.text(j + text_x, i - 0.13, f"{transfer[i, j]:.2f}",
                    ha="left", va="center", fontsize=11.5,
                    color=probe_color, fontweight="bold")
            ax.scatter([j + dot_x], [i + 0.18], color=NAIVE, s=55, zorder=3,
                       edgecolors="none")
            ax.text(j + text_x, i + 0.18, f"({utility[i, j]:.2f})",
                    ha="left", va="center", fontsize=7,
                    color=util_color, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(d["display"], rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(d["display"], fontsize=10)
    ax.set_xlabel("target persona  (probe applied to its activations)",
                  fontsize=10)
    ax.set_ylabel("probe trained on persona", fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, shrink=0.7)
    cbar.set_label("Δ  (probe r − utility r)", fontsize=9)
    cbar.outline.set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    d = load()
    print("Order:", d["display"])
    print("Δ off-diag range:",
          f"{np.nanmin(d['delta_off']):+.2f} to {np.nanmax(d['delta_off']):+.2f}")
    render(d, OUT_PNG)
    print(f"Saved {OUT_PNG.relative_to(REPO)}")


if __name__ == "__main__":
    main()
