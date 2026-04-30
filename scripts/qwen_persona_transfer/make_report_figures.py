"""Generate report figures for Qwen persona probe transfer.

Consumes only the per-cell NPZs from `analyze_transfer.py`. Produces:
  - transfer_grid: 2 selectors × 3 layers grid of 7×7 transfer heatmaps
  - utility_similarity: single 7×7 heatmap (utilities are layer/selector-independent)
  - probe_cosine_grid: 2×3 grid of 7×7 cosine heatmaps
  - layer_dependence: per-persona donor/target r vs layer (one panel per selector)
  - transfer_vs_utility_scatter: per-cell scatter of (utility-r, transfer-r) across pairs
  - default_probe_vs_utility: per non-default persona, default-probe transfer r vs default-vs-persona utility r
  - asymmetry_scatter: 21 unordered pairs, x=r(A→B), y=r(B→A), color=|gap|, per cell

All plots saved to experiments/qwen_replication/persona_transfer/probe_transfer/assets/
with date-stamped filenames. Persona ordering: Qwen-internal sort by similarity-with-default
(computed from utility_similarity.npz).
"""

from __future__ import annotations

from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments/qwen_replication/persona_transfer/probe_transfer/results"
ASSETS = REPO / "experiments/qwen_replication/persona_transfer/probe_transfer/assets"

SELECTORS = ["tb-1", "tb-4"]
LAYERS = [33, 38, 43]
DATE = datetime.now().strftime("%m%d%y")


def load_cell(selector_tag: str, layer: int) -> dict:
    return dict(np.load(RESULTS / f"transfer_{selector_tag}_L{layer}.npz", allow_pickle=True))


def load_utility() -> dict:
    return dict(np.load(RESULTS / "utility_similarity.npz", allow_pickle=True))


def qwen_internal_order(personas: list[str], U: np.ndarray) -> list[int]:
    """Sort by similarity-with-default descending, default first."""
    default_idx = personas.index("default")
    sims = U[default_idx]
    order = sorted(range(len(personas)), key=lambda i: -sims[i])
    # ensure default is first (it has self-similarity 1)
    order.remove(default_idx)
    return [default_idx] + order


def reorder(M: np.ndarray, order: list[int]) -> np.ndarray:
    return M[np.ix_(order, order)]


def heatmap(ax, M: np.ndarray, labels: list[str], vmin: float, vmax: float, title: str, cmap="RdBu_r"):
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(M[i, j]) > 0.5 else "black")
    return im


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    util = load_utility()
    personas = list(util["personas"])
    U = util["U"]
    order = qwen_internal_order(personas, U)
    labels = [personas[i] for i in order]
    print(f"persona ordering (sim-to-default desc): {labels}")

    # --- Utility similarity (single heatmap) ---
    fig, ax = plt.subplots(figsize=(7, 6))
    heatmap(ax, reorder(U, order), labels, -1, 1, "Utility similarity (Pearson r, test split)")
    fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ASSETS / f"plot_{DATE}_utility_similarity.png", dpi=150)
    plt.close(fig)

    # --- Transfer grid (2 × 3) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for i, sel in enumerate(SELECTORS):
        for j, layer in enumerate(LAYERS):
            cell = load_cell(sel, layer)
            T = reorder(cell["T"], order)
            heatmap(axes[i, j], T, labels, -0.2, 1, f"{sel}, L{layer}")
    fig.suptitle("Probe transfer r — rows = train, cols = eval", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / f"plot_{DATE}_transfer_grid.png", dpi=150)
    plt.close(fig)

    # --- Probe cosine grid (2 × 3) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for i, sel in enumerate(SELECTORS):
        for j, layer in enumerate(LAYERS):
            cell = load_cell(sel, layer)
            C = reorder(cell["C"], order)
            heatmap(axes[i, j], C, labels, -1, 1, f"{sel}, L{layer}")
    fig.suptitle("Probe weight cosine", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / f"plot_{DATE}_probe_cosine_grid.png", dpi=150)
    plt.close(fig)

    # --- Layer dependence (per selector) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, sel in zip(axes, SELECTORS):
        per_persona_outbound = {p: [] for p in labels}
        per_persona_inbound = {p: [] for p in labels}
        for layer in LAYERS:
            cell = load_cell(sel, layer)
            T = reorder(cell["T"], order)
            for k, p in enumerate(labels):
                row = T[k]; col = T[:, k]
                per_persona_outbound[p].append(row[np.arange(len(row)) != k].mean())
                per_persona_inbound[p].append(col[np.arange(len(col)) != k].mean())
        for p in labels:
            ax.plot(LAYERS, per_persona_outbound[p], marker="o", label=p)
        ax.set_title(f"{sel}: mean outbound r vs layer")
        ax.set_xlabel("layer"); ax.set_ylabel("mean off-diagonal r")
        ax.set_xticks(LAYERS); ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / f"plot_{DATE}_layer_dependence_outbound.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, sel in zip(axes, SELECTORS):
        per_persona_inbound = {p: [] for p in labels}
        for layer in LAYERS:
            cell = load_cell(sel, layer)
            T = reorder(cell["T"], order)
            for k, p in enumerate(labels):
                col = T[:, k]
                per_persona_inbound[p].append(col[np.arange(len(col)) != k].mean())
        for p in labels:
            ax.plot(LAYERS, per_persona_inbound[p], marker="o", label=p)
        ax.set_title(f"{sel}: mean inbound r vs layer")
        ax.set_xlabel("layer"); ax.set_ylabel("mean off-diagonal r")
        ax.set_xticks(LAYERS); ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / f"plot_{DATE}_layer_dependence_inbound.png", dpi=150)
    plt.close(fig)

    # --- Transfer vs utility scatter (per cell) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    n = len(labels)
    U_ord = reorder(U, order)
    off_ij = [(i, j) for i in range(n) for j in range(n) if i != j]
    for i, sel in enumerate(SELECTORS):
        for j, layer in enumerate(LAYERS):
            ax = axes[i, j]
            cell = load_cell(sel, layer)
            T = reorder(cell["T"], order)
            x = [U_ord[a, b] for a, b in off_ij]
            y = [T[a, b] for a, b in off_ij]
            ax.scatter(x, y, alpha=0.7, s=30)
            lo, hi = -0.3, 1.0
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1)
            ax.axhline(0, color="grey", linewidth=0.5); ax.axvline(0, color="grey", linewidth=0.5)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_title(f"{sel}, L{layer}", fontsize=10)
            ax.set_xlabel("utility-utility r"); ax.set_ylabel("transfer r")
            ax.grid(alpha=0.3)
            above = sum(1 for xx, yy in zip(x, y) if yy > xx)
            ax.text(0.05, 0.95, f"{above}/{len(x)} above y=x",
                    transform=ax.transAxes, va="top", fontsize=9)
    fig.suptitle("Transfer r vs utility-utility r (off-diagonal pairs)", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / f"plot_{DATE}_transfer_vs_utility_scatter.png", dpi=150)
    plt.close(fig)

    # --- Default-probe vs utility (per cell) ---
    default_idx_ord = labels.index("default")
    other = [k for k in range(n) if k != default_idx_ord]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    for i, sel in enumerate(SELECTORS):
        for j, layer in enumerate(LAYERS):
            ax = axes[i, j]
            cell = load_cell(sel, layer)
            T = reorder(cell["T"], order)
            transfer = [T[default_idx_ord, k] for k in other]
            sim = [U_ord[default_idx_ord, k] for k in other]
            xs = np.arange(len(other))
            w = 0.35
            ax.bar(xs - w/2, transfer, w, label="default probe r", color="steelblue")
            ax.bar(xs + w/2, sim, w, label="default-vs-persona utility r", color="grey")
            ax.set_xticks(xs)
            ax.set_xticklabels([labels[k] for k in other], rotation=45, ha="right", fontsize=8)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_title(f"{sel}, L{layer}", fontsize=10)
            ax.set_ylabel("r"); ax.grid(alpha=0.3, axis="y")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Default probe transfer vs default-vs-persona utility (per non-default persona)", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / f"plot_{DATE}_default_probe_vs_utility.png", dpi=150)
    plt.close(fig)

    # --- Asymmetry scatter (per cell) ---
    pairs = list(combinations(range(n), 2))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for i, sel in enumerate(SELECTORS):
        for j, layer in enumerate(LAYERS):
            ax = axes[i, j]
            cell = load_cell(sel, layer)
            T = reorder(cell["T"], order)
            x = [T[a, b] for a, b in pairs]
            y = [T[b, a] for a, b in pairs]
            gaps = [abs(xx - yy) for xx, yy in zip(x, y)]
            sc = ax.scatter(x, y, c=gaps, cmap="viridis", alpha=0.85, s=40)
            ax.plot([-0.2, 1], [-0.2, 1], "k--", alpha=0.5, linewidth=1)
            ax.set_xlim(-0.2, 1); ax.set_ylim(-0.2, 1)
            ax.set_xlabel("r(A→B)"); ax.set_ylabel("r(B→A)")
            ax.set_title(f"{sel}, L{layer}", fontsize=10)
            ax.grid(alpha=0.3)
            for k, (a, b) in enumerate(pairs):
                if gaps[k] > 0.25:
                    ax.annotate(f"{labels[a][:3]}↔{labels[b][:3]}", (x[k], y[k]),
                                fontsize=6, alpha=0.8)
            if j == 2:
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="|gap|")
    fig.suptitle("Transfer asymmetry (21 unordered pairs)", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / f"plot_{DATE}_asymmetry_scatter.png", dpi=150)
    plt.close(fig)

    print(f"all figures saved to {ASSETS.relative_to(REPO)}")


if __name__ == "__main__":
    main()
