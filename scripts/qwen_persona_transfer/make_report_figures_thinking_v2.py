"""Headline figures for the thinking-mode probe transfer report.

Produces the single-cell, paper-style figures that match the no-think parent's
headline plots (rather than the small-multiples grid which now becomes appendix).

Outputs:
  - plot_<date>_v2_default_probe_vs_utility_headline.png
      Single-panel bar chart for the headline cell, with red Δ arrows.
      Matches no-think `plot_042526_default_probe_vs_utility.png`.
  - plot_<date>_v2_transfer_utility_pair_headline.png
      Two-panel: probe transfer (left) + utility similarity (right) at headline.
      Matches no-think `plot_042526_transfer_utility_pair.png`.
  - plot_<date>_v2_headline_vs_nothink.png
      Two-panel: thinking headline transfer (left) vs no-think headline transfer (right),
      same cell ordering, shared colour scale. New comparison plot.
  - plot_<date>_v2_offdiag_vs_layer.png
      Bar chart: off-diagonal mean per cell (thinking vs no-think). Replaces the
      cell-by-cell summary table.
  - plot_<date>_v2_appendix_grid.png
      Cleaned-up 6-cell appendix grid with axis labels + shared colourbar.
      Replaces existing `plot_042626_transfer_grid.png`.
"""

from __future__ import annotations

from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
THINK_RESULTS = REPO / "experiments/qwen_replication/persona_transfer/probe_transfer/thinking/results"
NT_RESULTS = REPO / "experiments/qwen_replication/persona_transfer/probe_transfer/results"
ASSETS = REPO / "experiments/qwen_replication/persona_transfer/probe_transfer/thinking/assets"

SELECTORS = ["tb-1", "tb-4"]
LAYERS = [33, 38, 43]
HEADLINE = ("tb-4", 43)
NT_HEADLINE = ("tb-4", 38)

DATE = datetime.now().strftime("%m%d%y")


def load_cell(root: Path, sel: str, L: int) -> dict:
    return dict(np.load(root / f"transfer_{sel}_L{L}.npz", allow_pickle=True))


def load_utility(root: Path) -> dict:
    return dict(np.load(root / "utility_similarity.npz", allow_pickle=True))


def reorder(M: np.ndarray, order: list[int]) -> np.ndarray:
    return M[np.ix_(order, order)]


def heatmap(ax, M: np.ndarray, labels: list[str], vmin: float, vmax: float, title: str,
            cmap="RdBu_r", xlabel: str = "", ylabel: str = ""):
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=11)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(M[i, j]) > 0.5 else "black")
    return im


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    util_t = load_utility(THINK_RESULTS)
    personas_t = list(util_t["personas"])
    U_t = util_t["U"]

    util_nt = load_utility(NT_RESULTS)
    personas_nt = list(util_nt["personas"])
    U_nt = util_nt["U"]

    # ----- Headline cell load -----
    cell_t = load_cell(THINK_RESULTS, *HEADLINE)
    T_t = cell_t["T"]
    cell_nt = load_cell(NT_RESULTS, *NT_HEADLINE)
    T_nt = cell_nt["T"]

    # =========================================================
    # 1. Default-probe-vs-utility headline (single-panel)
    # =========================================================
    default_idx = personas_t.index("default")
    others = [k for k in range(len(personas_t)) if k != default_idx]
    # Sort by Δ descending for visual readability (matching parent)
    deltas = [T_t[default_idx, k] - U_t[default_idx, k] for k in others]
    order = sorted(range(len(others)), key=lambda i: -deltas[i])
    others_sorted = [others[i] for i in order]
    transfer_vals = [T_t[default_idx, k] for k in others_sorted]
    util_vals = [U_t[default_idx, k] for k in others_sorted]
    delta_vals = [t - u for t, u in zip(transfer_vals, util_vals)]
    labels = [personas_t[k] for k in others_sorted]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(labels))
    w = 0.36
    bars_p = ax.bar(x - w/2, transfer_vals, w,
                    label="probe transfer r  (default probe → persona)",
                    color="#1a73e8")
    bars_u = ax.bar(x + w/2, util_vals, w,
                    label="utility r  (default utilities vs persona)",
                    color="#9aa0a6")
    ax.axhline(0, color="black", lw=0.5)

    # Annotate bar values
    for k, (tp, up) in enumerate(zip(transfer_vals, util_vals)):
        ax.text(k - w/2, tp + (0.015 if tp >= 0 else -0.04),
                f"{tp:+.2f}", ha="center", fontsize=10, color="#1a73e8", fontweight="bold")
        ax.text(k + w/2, up + (0.015 if up >= 0 else -0.04),
                f"{up:+.2f}", ha="center", fontsize=10, color="#5f6368")

    # Δ arrows + labels
    for k, d in enumerate(delta_vals):
        x_arrow = k + w/2 + 0.12
        y_lo = util_vals[k]
        y_hi = transfer_vals[k]
        if d > 0:
            ax.annotate("", xy=(x_arrow, y_hi), xytext=(x_arrow, y_lo),
                        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4))
            ax.text(x_arrow + 0.04, (y_lo + y_hi) / 2, f"Δ {d:+.2f}",
                    color="#d62728", fontsize=9, va="center")
        else:
            ax.annotate("", xy=(x_arrow, y_hi), xytext=(x_arrow, y_lo),
                        arrowprops=dict(arrowstyle="->", color="#5f6368", lw=1.2, ls="--"))
            ax.text(x_arrow + 0.04, (y_lo + y_hi) / 2, f"Δ {d:+.2f}",
                    color="#5f6368", fontsize=9, va="center")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Pearson r (canonical test split)", fontsize=10)
    ax.set_title(f"Default probe vs utility similarity per persona  ({HEADLINE[0]}, L{HEADLINE[1]})\n"
                 "THINKING-mode utilities; red Δ = probe gain, grey Δ = probe loss",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(min(util_vals + transfer_vals) - 0.15,
                max(util_vals + transfer_vals) + 0.15)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{DATE}_v2_default_probe_vs_utility_headline.png", dpi=150)
    plt.close(fig)

    # =========================================================
    # 2. Transfer + utility pair headline (two-panel)
    # =========================================================
    # Reorder both by sim-with-default desc (Qwen-internal thinking order)
    sims = U_t[default_idx]
    order_t = sorted(range(len(personas_t)), key=lambda i: -sims[i])
    order_t.remove(default_idx)
    order_t = [default_idx] + order_t
    labels_t = [personas_t[i] for i in order_t]
    T_t_ord = reorder(T_t, order_t)
    U_t_ord = reorder(U_t, order_t)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im0 = heatmap(axes[0], T_t_ord, labels_t, -0.2, 0.9,
                  f"Probe transfer r — {HEADLINE[0]}, L{HEADLINE[1]} (THINKING)",
                  xlabel="eval persona", ylabel="probe persona")
    im1 = heatmap(axes[1], U_t_ord, labels_t, -0.5, 1,
                  "Utility similarity r (Thurstonian μ, test split)",
                  xlabel="persona B", ylabel="persona A")
    fig.colorbar(im0, ax=axes, fraction=0.025, pad=0.04, label="Pearson r")
    fig.suptitle("Headline cell: probe transfer (left) and utility similarity (right) — THINKING mode",
                 fontsize=12)
    plt.savefig(ASSETS / f"plot_{DATE}_v2_transfer_utility_pair_headline.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # =========================================================
    # 3. Thinking vs no-think headline comparison
    # =========================================================
    # Reorder no-think by ITS OWN sim-to-default ordering
    default_idx_nt = personas_nt.index("default")
    sims_nt = U_nt[default_idx_nt]
    order_nt = sorted(range(len(personas_nt)), key=lambda i: -sims_nt[i])
    order_nt.remove(default_idx_nt)
    order_nt = [default_idx_nt] + order_nt
    labels_nt = [personas_nt[i] for i in order_nt]
    T_nt_ord = reorder(T_nt, order_nt)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im0 = heatmap(axes[0], T_t_ord, labels_t, -0.2, 0.9,
                  f"THINKING (low effort) headline — {HEADLINE[0]}, L{HEADLINE[1]}",
                  xlabel="eval persona", ylabel="probe persona")
    im1 = heatmap(axes[1], T_nt_ord, labels_nt, -0.2, 0.9,
                  f"NO-THINK headline — {NT_HEADLINE[0]}, L{NT_HEADLINE[1]}",
                  xlabel="eval persona", ylabel="probe persona")
    fig.colorbar(im0, ax=axes, fraction=0.025, pad=0.04, label="Pearson r")
    fig.suptitle("Probe transfer matrices — THINKING vs NO-THINK headline cells\n"
                 "(persona ordering is sim-to-default within each regime; mathematician moves from #2 in no-think to #4 in thinking)",
                 fontsize=11)
    plt.savefig(ASSETS / f"plot_{DATE}_v2_headline_vs_nothink.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # =========================================================
    # 4. Off-diagonal mean per cell (thinking vs no-think)
    # =========================================================
    cells = [(s, L) for s in SELECTORS for L in LAYERS]
    off_t = []; off_nt = []; diag_t = []; diag_nt = []
    for s, L in cells:
        T = load_cell(THINK_RESULTS, s, L)["T"]
        n = T.shape[0]
        off_t.append(T[~np.eye(n, dtype=bool)].mean())
        diag_t.append(np.diag(T).mean())
        T = load_cell(NT_RESULTS, s, L)["T"]
        off_nt.append(T[~np.eye(n, dtype=bool)].mean())
        diag_nt.append(np.diag(T).mean())

    cell_labels = [f"{s}\nL{L}" for s, L in cells]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    x = np.arange(len(cells))
    w = 0.36

    axes[0].bar(x - w/2, diag_nt, w, color="#9aa0a6", label="no-think")
    axes[0].bar(x + w/2, diag_t, w, color="#1a73e8", label="thinking")
    axes[0].set_xticks(x); axes[0].set_xticklabels(cell_labels, fontsize=9)
    axes[0].set_ylabel("mean diagonal r (probe-on-self)")
    axes[0].set_title("Within-persona fit (diagonal mean)")
    axes[0].axhline(0, color="black", lw=0.4)
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].legend(frameon=False)
    for k, (a, b) in enumerate(zip(diag_nt, diag_t)):
        axes[0].text(k - w/2, a + 0.005, f"{a:.2f}", ha="center", fontsize=8)
        axes[0].text(k + w/2, b + 0.005, f"{b:.2f}", ha="center", fontsize=8)

    axes[1].bar(x - w/2, off_nt, w, color="#9aa0a6", label="no-think")
    axes[1].bar(x + w/2, off_t, w, color="#1a73e8", label="thinking")
    axes[1].set_xticks(x); axes[1].set_xticklabels(cell_labels, fontsize=9)
    axes[1].set_ylabel("mean off-diagonal r (cross-persona transfer)")
    axes[1].set_title("Cross-persona transfer (off-diagonal mean)")
    axes[1].axhline(0, color="black", lw=0.4)
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend(frameon=False)
    for k, (a, b) in enumerate(zip(off_nt, off_t)):
        axes[1].text(k - w/2, a + 0.003, f"{a:.2f}", ha="center", fontsize=8)
        axes[1].text(k + w/2, b + 0.003, f"{b:.2f}", ha="center", fontsize=8)

    fig.suptitle("Per-cell summary: thinking is uniformly weaker than no-think on both metrics", fontsize=12)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{DATE}_v2_offdiag_vs_layer.png", dpi=150)
    plt.close(fig)

    # =========================================================
    # 5. Cleaned-up appendix grid (6 cells, shared colourbar, axis labels)
    # =========================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5))
    for i, sel in enumerate(SELECTORS):
        for j, L in enumerate(LAYERS):
            cell = load_cell(THINK_RESULTS, sel, L)
            T = reorder(cell["T"], order_t)
            im = heatmap(axes[i, j], T, labels_t, -0.2, 0.9,
                         f"{sel}, L{L}",
                         xlabel="eval persona" if i == 1 else "",
                         ylabel="probe persona" if j == 0 else "")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.04, label="Pearson r")
    fig.suptitle("Probe transfer r — appendix grid (THINKING mode, all 6 cells)\n"
                 "rows = selector (tb-1, tb-4), cols = layer (33, 38, 43)",
                 fontsize=12)
    plt.savefig(ASSETS / f"plot_{DATE}_v2_appendix_grid.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(f"all v2 figures saved to {ASSETS.relative_to(REPO)}")


if __name__ == "__main__":
    main()
