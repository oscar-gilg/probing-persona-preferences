"""Figure: Assistant probe vs naive utility-similarity baseline, with the
per-persona ceiling (probe trained on that persona itself).

For each non-Assistant persona, three quantities are plotted on a shared row:
  - purple  dot: Pearson r between Assistant utilities and that persona's
                 utilities (the naive baseline).
  - filled blue dot: Pearson r between the Assistant-trained probe's
                     predictions on that persona's activations and that
                     persona's own utilities.
  - hollow blue dot: Pearson r for a probe TRAINED on that persona — the
                     per-persona ceiling.
An arrow marks the uplift Δ from naive to Assistant probe. The top row anchors
the Assistant probe's in-domain r as a reference.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "experiments/persona_sweep/probe_transfer/results"
OUT_PNG = REPO / "paper/figures/main/plot_050626_persona_probe_uplift.png"

BLUE = "#1a73e8"
NAIVE = "#7c3aed"
ARROW = "#1e3a8a"
STRIPE = "#f5f5f4"

DISPLAY = {"sadist": "evil"}


def display(label: str) -> str:
    return DISPLAY.get(label, label)


def load() -> dict:
    t = np.load(RESULTS / "transfer_tb-5_L32.npz", allow_pickle=True)
    u = np.load(RESULTS / "utility_similarity.npz", allow_pickle=True)
    personas = [str(p) for p in t["personas"]]
    transfer = t["transfer_r"]
    utility = u["utility_r"]
    assert [str(p) for p in u["personas"]] == personas

    i_def = personas.index("default")
    others = [i for i in range(len(personas)) if i != i_def]
    others_sorted = sorted(others, key=lambda i: -transfer[i_def, i])

    labels = [personas[i] for i in others_sorted]
    return {
        "labels": labels,
        "display": [display(l) for l in labels],
        "probe": np.array([transfer[i_def, i] for i in others_sorted]),
        "utility": np.array([utility[i_def, i] for i in others_sorted]),
        "ceiling": np.array([transfer[i, i] for i in others_sorted]),
        "assistant_self_r": float(transfer[i_def, i_def]),
    }


def render(d: dict, path: Path) -> None:
    n = len(d["labels"])
    util, probe, ceiling = d["utility"], d["probe"], d["ceiling"]
    deltas = probe - util

    n_rows = n + 1
    y_assistant = n_rows - 1
    y_others = np.arange(n_rows - 2, -1, -1)
    asr = d["assistant_self_r"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    for k, yy in enumerate(np.concatenate([[y_assistant], y_others])):
        if k % 2 == 0:
            ax.axhspan(yy - 0.5, yy + 0.5, color=STRIPE, zorder=0)

    ax.axvline(0, color="black", lw=0.5, zorder=1)
    ax.axvline(asr, color="#94a3b8", lw=0.8, ls=(0, (3, 3)), zorder=1, alpha=0.7)

    ax.scatter([asr], [y_assistant], color=BLUE, s=140, zorder=4,
               edgecolor="white", linewidths=1.4)
    ax.text(asr + 0.025, y_assistant, f"{asr:.2f}", ha="left", va="center",
            fontsize=10, fontweight="bold", color=BLUE)
    ax.text(-0.32, y_assistant, "Assistant", ha="right", va="center",
            fontsize=10, fontweight="bold", color="#111827")
    ax.axhline(y_assistant - 0.5, color="#d4d4d8", lw=0.7, zorder=0.5)

    for i in range(n):
        yy = y_others[i]
        ax.scatter([ceiling[i]], [yy], facecolor="white", edgecolor=BLUE,
                   linewidths=1.8, s=90, zorder=3)
        ax.annotate(
            "",
            xy=(probe[i], yy), xytext=(util[i], yy),
            arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=2.2,
                            shrinkA=4, shrinkB=4, mutation_scale=15),
            zorder=2,
        )
        ax.scatter([util[i]], [yy], color=NAIVE, s=85, zorder=3,
                   edgecolor="white", linewidths=1.4)
        ax.scatter([probe[i]], [yy], color=BLUE, s=130, zorder=4,
                   edgecolor="white", linewidths=1.4)

        ax.text(probe[i] + 0.025, yy, f"{probe[i]:.2f}", ha="left",
                va="center", fontsize=10, fontweight="bold", color=BLUE)
        mid_x = (util[i] + probe[i]) / 2
        ax.text(mid_x, yy + 0.20, f"Δ +{deltas[i]:.2f}", ha="center",
                va="bottom", fontsize=9, color=ARROW, fontweight="bold")

    ax.set_yticks(np.concatenate([y_others, [y_assistant]]))
    ax.set_yticklabels(d["display"] + [""])
    ax.tick_params(axis="y", left=False, pad=4)
    ax.set_xlim(-0.32, 1.05)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_xlabel("Pearson correlation")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(10.5)

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=NAIVE,
                   markersize=10,
                   label="predict from Assistant's utilities"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE,
                   markersize=11,
                   label="Assistant probe on persona's activations"),
        plt.Line2D([0], [0], marker="o", color=BLUE, markerfacecolor="white",
                   markeredgewidth=1.8, markersize=10, lw=0,
                   label="probe trained on each persona  (ceiling)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False,
              fontsize=9, handletextpad=0.5, columnspacing=1.6)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    d = load()
    print("Order:", d["display"])
    render(d, OUT_PNG)
    print(f"Saved {OUT_PNG.relative_to(REPO)}")


if __name__ == "__main__":
    main()
