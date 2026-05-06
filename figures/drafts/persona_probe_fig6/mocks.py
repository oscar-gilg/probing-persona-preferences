"""Three throwaway mocks for redesigning Fig 6 (Assistant probe vs utility similarity).

Pain points to address:
  A. Hard to tell what the headline is (the lift / delta).
  B. Ugly / cluttered.
  C. No upper-bound reference — readers can't tell what "+0.63" means.

Mocks:
  1. Slopegraph: utility-similarity → Assistant probe, with a per-persona
     ceiling tick on the right (self-trained probe r, the diagonal of the
     transfer matrix).
  2. Three-bar grouped: utility-similarity | Assistant probe | self-trained
     ceiling, per persona.
  3. Lollipop / dumbbell: horizontal Δ marker per persona; absolute r values
     annotated at endpoints; companion right-hand panel highlighting evil
     (the only persona with a negative baseline).

Outputs go to scripts/persona_probe_fig6/assets/ so they don't pollute
paper/figures/ until one design is chosen.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "experiments/persona_sweep/probe_transfer/results"
OUT = Path(__file__).resolve().parent

BLUE = "#1a73e8"
GREY = "#9aa0a6"
CEIL = "#374151"
EVIL_RED = "#b91c1c"

DISPLAY = {"sadist": "evil", "default": "Assistant"}


def display(label: str) -> str:
    return DISPLAY.get(label, label)


def load() -> dict:
    t = np.load(RESULTS / "transfer_tb-5_L32.npz", allow_pickle=True)
    u = np.load(RESULTS / "utility_similarity.npz", allow_pickle=True)
    personas = [str(p) for p in t["personas"]]
    transfer = t["transfer_r"]
    utility = u["utility_r"]
    assert [str(p) for p in u["personas"]] == personas, "persona ordering mismatch"

    i_def = personas.index("default")
    others = [i for i in range(len(personas)) if i != i_def]
    others_sorted = sorted(others, key=lambda i: -utility[i_def, i])

    labels = [personas[i] for i in others_sorted]
    probe_r = np.array([transfer[i_def, i] for i in others_sorted])
    utility_r = np.array([utility[i_def, i] for i in others_sorted])
    ceiling_r = np.array([transfer[i, i] for i in others_sorted])
    return {
        "labels": labels,
        "display": [display(l) for l in labels],
        "probe": probe_r,
        "utility": utility_r,
        "ceiling": ceiling_r,
        "assistant_self_r": float(transfer[i_def, i_def]),
    }


def fig_slopegraph(d: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    n = len(d["labels"])
    x_left, x_right, x_ceil = 0.0, 1.0, 1.18

    cmap = plt.get_cmap("viridis")
    colors = []
    for i, lbl in enumerate(d["labels"]):
        if lbl == "sadist":
            colors.append(EVIL_RED)
        else:
            colors.append(cmap(0.15 + 0.65 * i / max(n - 1, 1)))

    for i, lbl in enumerate(d["labels"]):
        c = colors[i]
        ax.plot([x_left, x_right], [d["utility"][i], d["probe"][i]],
                color=c, lw=2.2, alpha=0.95, zorder=3)
        ax.scatter([x_left, x_right], [d["utility"][i], d["probe"][i]],
                   color=c, s=42, zorder=4)
        ax.scatter([x_ceil], [d["ceiling"][i]], color=c, s=64, marker="_",
                   linewidths=2.6, zorder=4)

    ax.axhline(0, color="#6B7C2A", lw=1.0, ls=(0, (4, 3)), zorder=1)

    label_offsets_left = {}
    label_offsets_right = {}
    for i, lbl in enumerate(d["labels"]):
        ax.text(x_left - 0.04, d["utility"][i], f"{d['utility'][i]:+.2f}",
                ha="right", va="center", fontsize=9, color=colors[i])
        ax.text(x_right + 0.04, d["probe"][i], f"{d['probe'][i]:+.2f}",
                ha="left", va="center", fontsize=9, fontweight="bold", color=colors[i])
        ax.text(x_ceil + 0.04, d["ceiling"][i], display(lbl),
                ha="left", va="center", fontsize=10, color=colors[i],
                fontweight="bold" if lbl == "sadist" else "normal")

    ax.set_xticks([x_left, x_right, x_ceil])
    ax.set_xticklabels([
        "Assistant–persona\nutility similarity\n(baseline)",
        "Assistant probe\non persona's\nactivations",
        "Self-trained\nceiling",
    ], fontsize=9)
    ax.set_xlim(-0.30, 1.62)
    ax.set_ylim(-0.25, 1.0)
    ax.set_ylabel("Pearson r vs persona's held-out utilities")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 6))
    ax.tick_params(axis="x", length=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_three_bar(d: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    n = len(d["labels"])
    x = np.arange(n)
    w = 0.26

    ax.bar(x - w, d["utility"], w, color=GREY,
           label="Utility similarity (baseline)")
    ax.bar(x, d["probe"], w, color=BLUE, label="Assistant probe")
    ax.bar(x + w, d["ceiling"], w, facecolor="none", edgecolor=CEIL,
           linewidth=1.6, hatch="///", label="Self-trained ceiling")

    for i in range(n):
        ax.text(x[i] - w, d["utility"][i] + (0.015 if d["utility"][i] >= 0 else -0.05),
                f"{d['utility'][i]:+.2f}", ha="center",
                va="bottom" if d["utility"][i] >= 0 else "top",
                fontsize=8, color="#555")
        ax.text(x[i], d["probe"][i] + 0.015, f"{d['probe'][i]:+.2f}",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=BLUE)
        ax.text(x[i] + w, d["ceiling"][i] + 0.015, f"{d['ceiling'][i]:.2f}",
                ha="center", va="bottom", fontsize=8, color=CEIL)

    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(d["display"])
    ax.set_ylabel("Pearson r vs persona's held-out utilities")
    ax.set_ylim(-0.25, 1.05)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_lollipop(d: dict, path: Path, *, assistant_self_r: float) -> None:
    """v4: probe-r sort, blue Assistant anchor, ceiling as faded headroom band
    + hollow dot, no '+' on absolute values, alternating row stripes."""
    n = len(d["labels"])

    AMBER = "#d97706"
    ARROW_BLUE = "#1e3a8a"
    STRIPE = "#f5f5f4"

    order_idx = np.argsort(-d["probe"])
    util = d["utility"][order_idx]
    probe = d["probe"][order_idx]
    ceiling = d["ceiling"][order_idx]
    labels = [d["display"][i] for i in order_idx]
    deltas = probe - util

    n_rows = n + 1
    y_assistant = n_rows - 1
    y_others = np.arange(n_rows - 2, -1, -1)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    for k, yy in enumerate(np.concatenate([[y_assistant], y_others])):
        if k % 2 == 0:
            ax.axhspan(yy - 0.5, yy + 0.5, color=STRIPE, zorder=0)

    ax.axvline(0, color="black", lw=0.5, zorder=1)
    ax.axvline(assistant_self_r, color="#94a3b8", lw=0.8, ls=(0, (3, 3)),
               zorder=1, alpha=0.7)

    ax.scatter([assistant_self_r], [y_assistant], color=BLUE, s=140,
               zorder=4, edgecolor="white", linewidths=1.4)
    ax.text(assistant_self_r + 0.025, y_assistant,
            f"{assistant_self_r:.2f}",
            ha="left", va="center", fontsize=10, fontweight="bold",
            color=BLUE)
    ax.text(-0.32, y_assistant, "Assistant", ha="right", va="center",
            fontsize=10, fontweight="bold", color="#111827")
    ax.axhline(y_assistant - 0.5, color="#d4d4d8", lw=0.7, zorder=0.5)

    for i in range(n):
        yy = y_others[i]
        ax.scatter([ceiling[i]], [yy], facecolor="white",
                   edgecolor=BLUE, linewidths=1.8, s=90, zorder=3)

        ax.annotate(
            "",
            xy=(probe[i], yy),
            xytext=(util[i], yy),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_BLUE, lw=2.2,
                            shrinkA=6, shrinkB=11,
                            mutation_scale=15),
            zorder=2,
        )
        ax.scatter([util[i]], [yy], color=AMBER, s=85, zorder=3,
                   edgecolor="white", linewidths=1.4)
        ax.scatter([probe[i]], [yy], color=BLUE, s=130, zorder=4,
                   edgecolor="white", linewidths=1.4)

        ax.text(probe[i] + 0.025, yy, f"{probe[i]:.2f}",
                ha="left", va="center", fontsize=10, fontweight="bold",
                color=BLUE)

        mid_x = (util[i] + probe[i]) / 2
        ax.text(mid_x, yy + 0.20, f"Δ +{deltas[i]:.2f}",
                ha="center", va="bottom", fontsize=9,
                color=ARROW_BLUE, fontweight="bold")

    ax.set_yticks(np.concatenate([y_others, [y_assistant]]))
    ax.set_yticklabels(labels + [""])
    ax.tick_params(axis="y", left=False, pad=4)
    ax.set_xlim(-0.32, 1.05)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_xlabel("Pearson correlation")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(10.5)

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=AMBER,
                   markersize=10,
                   label="predict from Assistant's utilities"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE,
                   markersize=11,
                   label="Assistant probe on persona's activations"),
        plt.Line2D([0], [0], marker="o", color=BLUE,
                   markerfacecolor="white", markeredgewidth=1.8,
                   markersize=10, lw=0,
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
    print(" probe r:  ", [f"{v:+.3f}" for v in d["probe"]])
    print(" utility r:", [f"{v:+.3f}" for v in d["utility"]])
    print(" ceiling r:", [f"{v:+.3f}" for v in d["ceiling"]])
    print(" gap (probe - utility):", [f"{v:+.3f}" for v in d["probe"] - d["utility"]])
    print(" headroom (ceiling - probe):", [f"{v:+.3f}" for v in d["ceiling"] - d["probe"]])

    fig_slopegraph(d, OUT / "mock_a_slopegraph.png")
    fig_three_bar(d, OUT / "mock_b_three_bar.png")
    fig_lollipop(d, OUT / "mock_c_lollipop.png",
                 assistant_self_r=d["assistant_self_r"])
    print(f"\nSaved to {OUT}/")


if __name__ == "__main__":
    main()
