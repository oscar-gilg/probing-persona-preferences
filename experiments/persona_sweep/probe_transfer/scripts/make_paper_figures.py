"""Generate the paper figures for the persona probe-transfer experiment.

Reads the precomputed transfer and utility-similarity NPZs under
experiments/persona_sweep/probe_transfer/results/ and writes PNGs to
experiments/persona_sweep/probe_transfer/assets/.

Figure set (tb-5, renamed 'eot' in text):
- plot_<mmddyy>_default_probe_vs_utility.png     — Fig A: bars, 6 personas × {probe r, utility r, gap}
- plot_<mmddyy>_transfer_utility_pair.png         — Fig B: 7×7 transfer and 7×7 utility side-by-side, cluster-ordered
- plot_<mmddyy>_layer_dependence.png              — Fig C: 2-panel line plot, outbound and inbound mean r vs layer
- plot_<mmddyy>_asymmetry_scatter.png             — Fig D: 21 unordered pairs, r(A→B) vs r(B→A) with y=x
- plot_<mmddyy>_self_fit_vs_donor.png             — appendix: diagonal r vs mean outbound r, 35 points
- plot_<mmddyy>_cosine_by_layer.png               — appendix: mean off-diagonal probe-cosine vs layer
- plot_<mmddyy>_transfer_heatmap_grid.png         — appendix: 2×5 heatmap grid across selector×layer
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "experiments/persona_sweep/probe_transfer/results"
ASSETS = REPO / "experiments/persona_sweep/probe_transfer/assets"
TODAY = datetime.now().strftime("%m%d%y")
LAYERS = [25, 32, 39, 46, 53]
SELECTORS = [("tb-2", "turn_boundary:-2"), ("tb-5", "turn_boundary:-5")]
HEADLINE = ("tb-5", 32)  # selector, layer


def _load_matrix(tag: str, layer: int):
    d = np.load(RESULTS / f"transfer_{tag}_L{layer}.npz", allow_pickle=True)
    personas = [str(p) for p in d["personas"]]
    return personas, d["transfer_r"], d["probe_cosine"]


def _cluster_order(M: np.ndarray) -> np.ndarray:
    """Hierarchical cluster order from a correlation matrix."""
    D = np.clip(1.0 - M, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    Z = linkage(squareform(D, checks=False), method="average")
    return leaves_list(Z)


def _heatmap(ax, M: np.ndarray, labels, title: str, vmin=-1.0, vmax=1.0, cmap="RdBu_r") -> None:
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


def fig_default_probe(personas, transfer, utility_r):
    idx_default = personas.index("default")
    others = [p for p in personas if p != "default"]
    probe_vs_default = np.array([transfer[idx_default, personas.index(p)] for p in others])
    util_vs_default = np.array([utility_r[idx_default, personas.index(p)] for p in others])
    gap = probe_vs_default - util_vs_default

    order = np.argsort(-gap)
    labels = [others[i] for i in order]
    probe = probe_vs_default[order]
    util = util_vs_default[order]
    gap_sorted = gap[order]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(labels))
    w = 0.38
    bars_util = ax.bar(x - w/2, util, w, label="utility r (default vs persona)", color="#9aa0a6")
    bars_probe = ax.bar(x + w/2, probe, w, label="probe transfer r (default → persona)", color="#1a73e8")
    for i, (u, p, g) in enumerate(zip(util, probe, gap_sorted)):
        top = max(u, p)
        ax.annotate(f"Δ = +{g:.2f}", xy=(x[i], top + 0.03), ha="center", fontsize=9,
                    color="#1a73e8", fontweight="bold")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pearson r (canonical test split)")
    ax.set_ylim(min(0.0, util.min() - 0.05), max(probe.max() + 0.15, 1.0))
    ax.set_title("Default probe beats utility correlation at every persona (eot, layer 32)")
    ax.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_default_probe_vs_utility.png", dpi=150)
    plt.close(fig)


def fig_transfer_utility_pair(personas, transfer, utility_r):
    order = _cluster_order(utility_r)
    ordered_labels = [personas[i] for i in order]
    transfer_o = transfer[np.ix_(order, order)]
    utility_o = utility_r[np.ix_(order, order)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    im1 = _heatmap(axes[0], transfer_o, ordered_labels,
                   "Probe transfer r (rows: probe trained on, cols: evaluated on)")
    axes[0].set_xlabel("eval persona")
    axes[0].set_ylabel("train persona (probe)")
    im2 = _heatmap(axes[1], utility_o, ordered_labels, "Utility similarity r (Thurstonian μ)")
    axes[1].set_xlabel("persona B")
    axes[1].set_ylabel("persona A")
    fig.suptitle("Transfer vs utility similarity — eot, layer 32 (cluster-ordered)", fontsize=13)
    fig.colorbar(im2, ax=axes, fraction=0.03, pad=0.02, label="Pearson r")
    plt.savefig(ASSETS / f"plot_{TODAY}_transfer_utility_pair.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_layer_dependence(personas):
    outbound = np.full((len(personas), len(LAYERS)), np.nan)
    inbound = np.full((len(personas), len(LAYERS)), np.nan)
    for li, layer in enumerate(LAYERS):
        _, tr, _ = _load_matrix("tb-5", layer)
        n = tr.shape[0]
        mask = ~np.eye(n, dtype=bool)
        for pi in range(n):
            row_mask = mask[pi]
            col_mask = mask[:, pi]
            outbound[pi, li] = tr[pi][row_mask].mean()
            inbound[pi, li] = tr[:, pi][col_mask].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(personas)))
    for pi, persona in enumerate(personas):
        axes[0].plot(LAYERS, outbound[pi], "-o", color=colors[pi], label=persona,
                     lw=2 if persona == "contrarian" else 1.4,
                     alpha=1.0 if persona == "contrarian" else 0.85)
        axes[1].plot(LAYERS, inbound[pi], "-o", color=colors[pi], label=persona,
                     lw=2 if persona == "contrarian" else 1.4,
                     alpha=1.0 if persona == "contrarian" else 0.85)
    for ax in axes:
        ax.set_xlabel("layer")
        ax.set_xticks(LAYERS)
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean Pearson r (6 off-diagonal pairs)")
    axes[0].set_title("Outbound — how well this probe predicts others")
    axes[1].set_title("Inbound — how well others predict this persona's utilities")
    axes[0].legend(loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Donor and target quality across layers (eot)", fontsize=13)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_layer_dependence.png", dpi=150)
    plt.close(fig)


def fig_asymmetry(personas, transfer):
    n = len(personas)
    pairs = []
    ab = []
    ba = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(f"{personas[i]}↔{personas[j]}")
            ab.append(transfer[i, j])  # A(i) → B(j)
            ba.append(transfer[j, i])  # B(j) → A(i)
    ab = np.array(ab)
    ba = np.array(ba)
    asym = np.abs(ab - ba)

    fig, ax = plt.subplots(figsize=(7, 7))
    lims = (-0.2, 1.0)
    ax.plot(lims, lims, "k--", alpha=0.3, label="y = x (symmetric)")
    ax.scatter(ab, ba, s=80, c=asym, cmap="viridis", vmin=0, vmax=max(0.5, asym.max()), edgecolor="black", lw=0.5)
    cbar = plt.colorbar(ax.collections[0], ax=ax, label="|r(A→B) − r(B→A)|", fraction=0.04)
    for label, x, y in zip(pairs, ab, ba):
        ax.annotate(label, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points", alpha=0.85)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("r(persona A probe → persona B utilities)")
    ax.set_ylabel("r(persona B probe → persona A utilities)")
    ax.set_title(f"Transfer asymmetry — 21 pairs, eot, layer 32\n"
                 f"max gap = {asym.max():.2f}, mean = {asym.mean():.2f}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_asymmetry_scatter.png", dpi=150)
    plt.close(fig)


def fig_self_fit_vs_donor(personas):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(personas)))
    markers = ["o", "s", "^", "D", "P"]
    for li, layer in enumerate(LAYERS):
        _, tr, _ = _load_matrix("tb-5", layer)
        n = tr.shape[0]
        mask = ~np.eye(n, dtype=bool)
        for pi, persona in enumerate(personas):
            self_fit = tr[pi, pi]
            donor = tr[pi][mask[pi]].mean()
            ax.scatter(self_fit, donor, s=80, c=[colors[pi]], marker=markers[li],
                       edgecolor="black", lw=0.4,
                       label=f"{persona}" if li == 0 else None)
            if persona == "contrarian":
                ax.annotate(f"L{layer}", (self_fit, donor), fontsize=7, xytext=(5, 0),
                            textcoords="offset points")
    ax.set_xlabel("within-persona r (diagonal)")
    ax.set_ylabel("mean outbound off-diagonal r")
    ax.set_title("Self-fit vs donor quality — 7 personas × 5 layers (eot)\n"
                 "markers: ○=L25  □=L32  △=L39  ◇=L46  ✚=L53")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_self_fit_vs_donor.png", dpi=150)
    plt.close(fig)


def fig_cosine_by_layer(personas):
    n = len(personas)
    mask = ~np.eye(n, dtype=bool)
    mean_cos_tb2 = []
    mean_cos_tb5 = []
    for layer in LAYERS:
        _, _, cos_tb2 = _load_matrix("tb-2", layer)
        _, _, cos_tb5 = _load_matrix("tb-5", layer)
        mean_cos_tb2.append(cos_tb2[mask].mean())
        mean_cos_tb5.append(cos_tb5[mask].mean())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(LAYERS, mean_cos_tb2, "-o", label="tb-2", color="#9aa0a6", lw=2)
    ax.plot(LAYERS, mean_cos_tb5, "-o", label="eot (tb-5)", color="#1a73e8", lw=2)
    ax.set_xlabel("layer")
    ax.set_xticks(LAYERS)
    ax.set_ylabel("mean probe-direction cosine\n(21 off-diagonal pairs)")
    ax.set_title("Probe direction alignment dips mid-network, rises in late layers")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_cosine_by_layer.png", dpi=150)
    plt.close(fig)


def fig_heatmap_grid(personas):
    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    for row, (tag, _) in enumerate(SELECTORS):
        label = "eot" if tag == "tb-5" else tag
        for col, layer in enumerate(LAYERS):
            _, tr, _ = _load_matrix(tag, layer)
            ax = axes[row, col]
            _heatmap(ax, tr, personas, f"{label}, L{layer}")
            if col == 0:
                ax.set_ylabel("train")
            if row == 1:
                ax.set_xlabel("eval")
    fig.suptitle("Transfer r across selector × layer", fontsize=14)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_transfer_heatmap_grid.png", dpi=130)
    plt.close(fig)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    personas, transfer, _ = _load_matrix(*HEADLINE)
    util = np.load(RESULTS / "utility_similarity.npz", allow_pickle=True)
    utility_r = util["utility_r"]

    fig_default_probe(personas, transfer, utility_r)
    fig_transfer_utility_pair(personas, transfer, utility_r)
    fig_layer_dependence(personas)
    fig_asymmetry(personas, transfer)
    fig_self_fit_vs_donor(personas)
    fig_cosine_by_layer(personas)
    fig_heatmap_grid(personas)

    print(f"7 figures written to {ASSETS.relative_to(REPO)}/ with date stamp {TODAY}")


if __name__ == "__main__":
    main()
