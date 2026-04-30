"""Paper-format figures for the Qwen-3.5-122B persona probe-transfer experiment.

Direct port of `experiments/persona_sweep/probe_transfer/scripts/make_paper_figures.py`
(Gemma paper figures), adapted for Qwen:
  - Headline cell: (tb-4, L38) — picked by off-diagonal mean r
  - Layers: [33, 38, 43] (3 layers vs Gemma's 5)
  - Persona ordering: Qwen-internal sort by sim-with-default

Reads from experiments/qwen_replication/persona_transfer/probe_transfer/results/*.npz
Writes to experiments/qwen_replication/persona_transfer/probe_transfer/assets/.

Skips the Gemma `corr_bias`/`full_bias`/`profile_bias` figures (need precomputed
NPZs not yet generated for Qwen — separate follow-up).
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
TODAY = datetime.now().strftime("%m%d%y")

LAYERS = [33, 38, 43]
SELECTORS = [("tb-1", "turn_boundary:-1"), ("tb-4", "turn_boundary:-4")]
HEADLINE_TAG, HEADLINE_LAYER = ("tb-4", 38)
HEADLINE_LABEL = f"{HEADLINE_TAG}, L{HEADLINE_LAYER}"

BLUE = "#1a73e8"
GREY = "#9aa0a6"


def _load_matrix(tag: str, layer: int):
    d = np.load(RESULTS / f"transfer_{tag}_L{layer}.npz", allow_pickle=True)
    personas = [str(p) for p in d["personas"]]
    return personas, d["T"], d["C"]


def _load_utility():
    d = np.load(RESULTS / "utility_similarity.npz", allow_pickle=True)
    return [str(p) for p in d["personas"]], d["U"]


def _qwen_order(personas: list[str], utility_r: np.ndarray) -> list[int]:
    """Default first, then by descending utility_r with default."""
    i_def = personas.index("default")
    others = [i for i in range(len(personas)) if i != i_def]
    others_sorted = sorted(others, key=lambda i: -utility_r[i_def, i])
    return [i_def] + others_sorted


def _reorder(M: np.ndarray, order: list[int]) -> np.ndarray:
    return M[np.ix_(order, order)]


def _heatmap(ax, M: np.ndarray, labels, title: str, vmin=-1.0, vmax=1.0, cmap="RdBu_r"):
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap)
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


# --- Figure A: default probe vs utility (bar + delta arrows) ---

def fig_default_probe(personas, transfer, utility_r, order):
    ordered = [personas[i] for i in order]
    labels = [p for p in ordered if p != "default"]
    i_def = personas.index("default")
    probe = np.array([transfer[i_def, personas.index(p)] for p in labels])
    util = np.array([utility_r[i_def, personas.index(p)] for p in labels])
    gap = probe - util

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w/2, probe, w, label="probe transfer r  (default probe → persona)", color=BLUE)
    ax.bar(x + w/2, util,  w, label="utility r  (default utilities vs persona)",  color=GREY)

    ymin_data = min(util.min(), 0.0)
    ymax_data = probe.max()
    span = ymax_data - ymin_data
    label_pad = 0.012 * span
    ymin = ymin_data - 0.15 * span
    ymax = ymax_data + 0.18 * span
    ax.set_ylim(ymin, ymax)

    for i in range(len(labels)):
        ax.text(x[i] - w/2, probe[i] + label_pad, f"{probe[i]:+.2f}",
                ha="center", va="bottom", fontsize=10, color=BLUE, fontweight="bold")
        x_arrow = x[i] + 0.02
        ax.annotate(
            "", xy=(x_arrow, probe[i]), xytext=(x_arrow, util[i]),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4, shrinkA=0, shrinkB=0),
        )
        mid_y = (probe[i] + util[i]) / 2
        ax.text(x_arrow + 0.03, mid_y, f"Δ {gap[i]:+.2f}",
                ha="left", va="center", fontsize=9, color="#c0392b", fontweight="bold")

    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pearson r (canonical test split)")
    ax.set_title(f"Default probe vs utility similarity per persona  ({HEADLINE_LABEL})", fontsize=12)
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_default_probe_vs_utility.png", dpi=150)
    plt.close(fig)


# --- Figure B: transfer + utility heatmap pair ---

def fig_transfer_utility_pair(personas, transfer, utility_r, order):
    labels = [personas[i] for i in order]
    transfer_o = _reorder(transfer, order)
    utility_o = _reorder(utility_r, order)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    _heatmap(axes[0], transfer_o, labels,
             "Probe transfer r  (rows: probe trained on; cols: evaluated on)")
    axes[0].set_xlabel("eval persona")
    axes[0].set_ylabel("probe persona")
    im2 = _heatmap(axes[1], utility_o, labels, "Utility similarity r  (Thurstonian μ)")
    axes[1].set_xlabel("persona B")
    axes[1].set_ylabel("persona A")
    fig.suptitle(f"Transfer vs utility similarity — {HEADLINE_LABEL}", fontsize=13)
    fig.colorbar(im2, ax=axes, fraction=0.025, pad=0.015, label="Pearson r", shrink=0.85)
    plt.savefig(ASSETS / f"plot_{TODAY}_transfer_utility_pair.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- Figure C: layer dependence (outbound + inbound, headline selector) ---

def fig_layer_dependence(personas, order):
    ordered = [personas[i] for i in order]
    n = len(personas)
    outbound = np.full((n, len(LAYERS)), np.nan)
    inbound = np.full((n, len(LAYERS)), np.nan)
    for li, layer in enumerate(LAYERS):
        _, tr, _ = _load_matrix(HEADLINE_TAG, layer)
        mask = ~np.eye(n, dtype=bool)
        for pi in range(n):
            outbound[pi, li] = tr[pi][mask[pi]].mean()
            inbound[pi, li] = tr[:, pi][mask[:, pi]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    color_by_persona = {p: colors[i] for i, p in enumerate(ordered)}
    for persona in ordered:
        pi = personas.index(persona)
        # Highlight strongest donor (computed from data)
        outbound_mean = np.nanmean(outbound[pi])
        is_top = outbound_mean == np.nanmax([np.nanmean(outbound[personas.index(p)]) for p in ordered])
        lw = 2.4 if is_top else 1.4
        alpha = 1.0 if is_top else 0.85
        axes[0].plot(LAYERS, outbound[pi], "-o", color=color_by_persona[persona],
                     label=persona, lw=lw, alpha=alpha)
        axes[1].plot(LAYERS, inbound[pi], "-o", color=color_by_persona[persona],
                     label=persona, lw=lw, alpha=alpha)
    for ax in axes:
        ax.set_xlabel("layer")
        ax.set_xticks(LAYERS)
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean Pearson r (6 off-diagonal pairs)")
    axes[0].set_title("Outbound — how well this probe predicts others")
    axes[1].set_title("Inbound — how well others predict this persona's utilities")
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(f"Per-persona donor & target quality vs layer  ({HEADLINE_TAG} selector)", fontsize=12)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_layer_dependence.png", dpi=150)
    plt.close(fig)


# --- Figure D: asymmetry scatter ---

def fig_asymmetry(personas, transfer, order):
    n = len(personas)
    pairs = []; ab = []; ba = []
    for oi in range(n):
        for oj in range(oi + 1, n):
            i, j = order[oi], order[oj]
            pairs.append(f"{personas[i]}↔{personas[j]}")
            ab.append(transfer[i, j])
            ba.append(transfer[j, i])
    ab = np.array(ab); ba = np.array(ba)
    asym = np.abs(ab - ba)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    lims = (-0.2, 1.0)
    ax.plot(lims, lims, "k--", alpha=0.3, label="y = x (symmetric)")
    sc = ax.scatter(ab, ba, s=80, c=asym, cmap="viridis", vmin=0, vmax=max(0.5, asym.max()),
                    edgecolor="black", lw=0.5)
    plt.colorbar(sc, ax=ax, label="|r(A→B) − r(B→A)|", fraction=0.04)
    for label, x, y in zip(pairs, ab, ba):
        ax.annotate(label, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points", alpha=0.85)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("r(persona A probe → persona B utilities)")
    ax.set_ylabel("r(persona B probe → persona A utilities)")
    ax.set_title(f"Transfer asymmetry — 21 pairs, {HEADLINE_LABEL}\n"
                 f"max gap = {asym.max():.2f}, mean = {asym.mean():.2f}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_asymmetry_scatter.png", dpi=150)
    plt.close(fig)


# --- Figure E: receiver vs default similarity ---

def fig_receiver_vs_default_similarity(personas, transfer, utility_r):
    from scipy.stats import pearsonr
    i_def = personas.index("default")
    n = len(personas)
    mask = ~np.eye(n, dtype=bool)
    rows = []
    for p in personas:
        i = personas.index(p)
        sim = 1.0 if p == "default" else utility_r[i_def, i]
        inbound = transfer[:, i][mask[:, i]].mean()
        rows.append((p, sim, inbound))
    labels = [r[0] for r in rows]
    xs = np.array([r[1] for r in rows])
    ys = np.array([r[2] for r in rows])

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    colors = ["#c0392b" if p == "default" else BLUE for p in labels]
    ax.scatter(xs, ys, s=130, c=colors, edgecolor="black", lw=0.5, zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), fontsize=10, xytext=(7, 2), textcoords="offset points")

    non_default = np.array([p != "default" for p in labels])
    r_nd, _ = pearsonr(xs[non_default], ys[non_default])
    xr = np.linspace(xs.min() - 0.05, 1.05, 50)
    slope, intercept = np.polyfit(xs[non_default], ys[non_default], 1)
    ax.plot(xr, slope * xr + intercept, color="#c0392b", lw=1.5, alpha=0.7,
            label=f"OLS fit on non-default points  (r = {r_nd:+.2f})")

    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("utility similarity with default  (Pearson r)")
    ax.set_ylabel("mean inbound transfer r")
    ax.set_title(f"Are personas more different from default harder to predict?\n"
                 f"({HEADLINE_LABEL} — all 7 personas; red = default)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_receiver_vs_default_similarity.png", dpi=150)
    plt.close(fig)


# --- Figure G: cosine by layer (one line per selector) ---

def fig_cosine_by_layer(personas):
    n = len(personas)
    mask = ~np.eye(n, dtype=bool)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"tb-1": GREY, "tb-4": BLUE}
    for tag, _ in SELECTORS:
        means = []
        for layer in LAYERS:
            _, _, cos = _load_matrix(tag, layer)
            means.append(cos[mask].mean())
        ax.plot(LAYERS, means, "-o", color=colors[tag], lw=2.2, markersize=7, label=tag)
        for x, y in zip(LAYERS, means):
            ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color=colors[tag])
    ax.set_xlabel("layer"); ax.set_xticks(LAYERS)
    ax.set_ylabel("mean probe-direction cosine\n(21 off-diagonal pairs)")
    ax.set_title("Probe-direction alignment by layer")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_cosine_by_layer.png", dpi=150)
    plt.close(fig)


# --- Appendix: per-cell heatmap grid (already in old script, kept for completeness) ---

def fig_heatmap_grid(personas, order):
    labels = [personas[i] for i in order]
    fig, axes = plt.subplots(2, 3, figsize=(20, 13), constrained_layout=True)
    last_im = None
    for i, (tag, _) in enumerate(SELECTORS):
        for j, layer in enumerate(LAYERS):
            _, transfer_o, _ = _load_matrix(tag, layer)
            T = _reorder(transfer_o, order)
            last_im = _heatmap(axes[i, j], T, labels, f"{tag}, L{layer}")
            axes[i, j].set_xlabel("eval persona")
            axes[i, j].set_ylabel("probe persona")
    fig.suptitle("Probe transfer r — appendix grid (2 selectors × 3 layers)", fontsize=13)
    fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.015, label="Pearson r", shrink=0.7)
    plt.savefig(ASSETS / f"plot_{TODAY}_transfer_heatmap_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    personas, utility_r = _load_utility()
    order = _qwen_order(personas, utility_r)
    print(f"persona ordering: {[personas[i] for i in order]}")

    _, headline_T, _ = _load_matrix(HEADLINE_TAG, HEADLINE_LAYER)

    fig_default_probe(personas, headline_T, utility_r, order)
    fig_transfer_utility_pair(personas, headline_T, utility_r, order)
    fig_layer_dependence(personas, order)
    fig_asymmetry(personas, headline_T, order)
    fig_receiver_vs_default_similarity(personas, headline_T, utility_r)
    fig_cosine_by_layer(personas)
    fig_heatmap_grid(personas, order)

    print(f"figures saved to {ASSETS.relative_to(REPO)}")


if __name__ == "__main__":
    main()
