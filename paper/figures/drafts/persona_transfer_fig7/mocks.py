"""Mock redesign for Fig 7 (cross-persona probe transfer 7x7).

Single matrix, colour = Δ (probe_r − utility_r). Cell text shows raw probe_r
(top, bold) and utility_r (bottom, faint) so the reader can verify the lift
without needing a second panel. Diagonal masked (utility_r ≡ 1 there).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "experiments/persona_sweep/probe_transfer/results"
OUT = Path(__file__).resolve().parent

BLUE = "#1a73e8"
NAIVE = "#7c3aed"
ARROW = "#1e3a8a"

DISPLAY = {"sadist": "evil", "default": "Assistant"}

# Single-sided blue ramp (white → blue) for Δ.
DELTA_CMAP = LinearSegmentedColormap.from_list(
    "delta_blue", ["#ffffff", "#dbeafe", "#93c5fd", "#3b82f6", "#1e3a8a"]
)


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
    display_labels = [DISPLAY.get(l, l) for l in labels]

    delta = transfer - utility
    n = len(labels)
    delta_off = delta.copy()
    np.fill_diagonal(delta_off, np.nan)

    return {
        "labels": labels,
        "display": display_labels,
        "transfer": transfer,
        "utility": utility,
        "delta": delta,
        "delta_off": delta_off,
        "n": n,
    }


def fig_delta_heatmap(d: dict, path: Path) -> None:
    n = d["n"]
    delta = d["delta_off"]
    transfer = d["transfer"]
    utility = d["utility"]

    vmax = float(np.nanmax(delta))
    vmin = 0.0

    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    im = ax.imshow(delta, cmap=DELTA_CMAP, vmin=vmin, vmax=vmax,
                   aspect="equal", origin="upper")

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
            probe_color = "white" if dark else "#0f172a"
            util_color = "#c4b5fd" if dark else NAIVE
            ax.text(j, i - 0.13, f"{transfer[i, j]:.2f}", ha="center",
                    va="center", fontsize=10, color=probe_color,
                    fontweight="bold")
            ax.text(j, i + 0.18, f"({utility[i, j]:.2f})", ha="center",
                    va="center", fontsize=8, color=util_color,
                    fontweight="bold")

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
    fig_delta_heatmap(d, OUT / "mock_a_delta_heatmap.png")
    print(f"Saved to {OUT}/")


if __name__ == "__main__":
    main()
