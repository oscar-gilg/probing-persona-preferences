"""Generate the paper figures for the persona probe-transfer experiment.

Reads the precomputed transfer and utility-similarity NPZs under
experiments/persona_sweep/probe_transfer/results/ and writes PNGs to
experiments/persona_sweep/probe_transfer/assets/.

All figures use a fixed persona ordering — sorted by utility similarity to
`default` — so cross-plot comparisons read left-to-right as "most default-like
→ most default-opposed":

    default, aura, mathematician, strategist, contrarian, slacker, sadist
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "experiments/persona_sweep/probe_transfer/results"
ASSETS = REPO / "experiments/persona_sweep/probe_transfer/assets"
TODAY = datetime.now().strftime("%m%d%y")
LAYERS = [25, 32, 39, 46, 53]
SELECTORS = [("tb-2", "turn_boundary:-2"), ("tb-5", "turn_boundary:-5")]
HEADLINE = ("tb-5", 32)

BLUE = "#1a73e8"
GREY = "#9aa0a6"


def _load_matrix(tag: str, layer: int):
    d = np.load(RESULTS / f"transfer_{tag}_L{layer}.npz", allow_pickle=True)
    personas = [str(p) for p in d["personas"]]
    return personas, d["transfer_r"], d["probe_cosine"]


def _fixed_order(personas: list[str], utility_r: np.ndarray) -> list[int]:
    """Persona order: default first, then by descending utility_r with default."""
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


def fig_default_probe(personas, transfer, utility_r, order):
    ordered = [personas[i] for i in order]
    labels = [p for p in ordered if p != "default"]  # already in similarity order
    i_def = personas.index("default")
    probe = np.array([transfer[i_def, personas.index(p)] for p in labels])
    util = np.array([utility_r[i_def, personas.index(p)] for p in labels])
    gap = probe - util

    fig, ax = plt.subplots(figsize=(10, 6.5))
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w/2, probe, w, label="probe transfer r  (default probe → persona)", color=BLUE)
    ax.bar(x + w/2, util,  w, label="utility r  (default utilities vs persona)",  color=GREY)

    ymin_data = min(util.min(), 0.0)
    ymax_data = probe.max()
    span = ymax_data - ymin_data
    pad = 0.05 * span
    ymin = ymin_data - 0.15 * span
    ymax = ymax_data + 0.25 * span
    ax.set_ylim(ymin, ymax)

    for i in range(len(labels)):
        ax.text(x[i] - w/2, probe[i] + pad, f"{probe[i]:+.2f}",
                ha="center", va="bottom", fontsize=10, color=BLUE, fontweight="bold")
        va_util = "bottom" if util[i] >= 0 else "top"
        offset = pad if util[i] >= 0 else -pad
        ax.text(x[i] + w/2, util[i] + offset, f"{util[i]:+.2f}",
                ha="center", va=va_util, fontsize=10, color="#555")

        # Vertical arrow spanning the gap between the two bar tops.
        x_arrow = x[i] + w/2 + 0.12
        ax.annotate(
            "",
            xy=(x_arrow, probe[i]),
            xytext=(x_arrow, util[i]),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4, shrinkA=0, shrinkB=0),
        )
        mid_y = (probe[i] + util[i]) / 2
        ax.text(x_arrow + 0.04, mid_y, f"Δ {gap[i]:+.2f}",
                ha="left", va="center", fontsize=10, color="#c0392b", fontweight="bold")

    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pearson r (canonical test split)")
    ax.set_title("Default probe beats utility similarity at every persona  (eot, layer 32)",
                 fontsize=12)
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_default_probe_vs_utility.png", dpi=150)
    plt.close(fig)


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
    fig.suptitle("Transfer vs utility similarity — eot, layer 32", fontsize=13)
    fig.colorbar(im2, ax=axes, fraction=0.025, pad=0.015, label="Pearson r", shrink=0.85)
    plt.savefig(ASSETS / f"plot_{TODAY}_transfer_utility_pair.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_layer_dependence(personas, order):
    ordered = [personas[i] for i in order]
    outbound = np.full((len(personas), len(LAYERS)), np.nan)
    inbound = np.full((len(personas), len(LAYERS)), np.nan)
    for li, layer in enumerate(LAYERS):
        _, tr, _ = _load_matrix("tb-5", layer)
        n = tr.shape[0]
        mask = ~np.eye(n, dtype=bool)
        for pi in range(n):
            outbound[pi, li] = tr[pi][mask[pi]].mean()
            inbound[pi, li] = tr[:, pi][mask[:, pi]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(personas)))
    color_by_persona = {p: colors[i] for i, p in enumerate(ordered)}
    for persona in ordered:
        pi = personas.index(persona)
        lw = 2.4 if persona == "contrarian" else 1.4
        alpha = 1.0 if persona == "contrarian" else 0.85
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
    axes[0].legend(loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Donor and target quality across layers (eot)", fontsize=13)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_layer_dependence.png", dpi=150)
    plt.close(fig)


def fig_asymmetry(personas, transfer, order):
    n = len(personas)
    pairs = []
    ab = []
    ba = []
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
    ax.scatter(ab, ba, s=80, c=asym, cmap="viridis", vmin=0, vmax=max(0.5, asym.max()),
               edgecolor="black", lw=0.5)
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


def fig_receiver_vs_default_similarity(personas, transfer, utility_r):
    """Does distance from default predict how bad a target a persona is?

    x = utility_r with default (how behaviourally similar to default)
    y = mean inbound r (how well other probes predict this persona's utilities)
    One point per persona (excluding default itself).
    """
    i_def = personas.index("default")
    n = len(personas)
    mask = ~np.eye(n, dtype=bool)
    xs, ys, labels = [], [], []
    for p in personas:
        if p == "default":
            continue
        i = personas.index(p)
        xs.append(utility_r[i_def, i])
        ys.append(transfer[:, i][mask[:, i]].mean())
        labels.append(p)
    xs = np.array(xs); ys = np.array(ys)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(xs, ys, s=110, c=BLUE, edgecolor="black", lw=0.5, zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), fontsize=10, xytext=(7, 2), textcoords="offset points")

    # Linear fit (exclude sadist for robust fit if desired? no — show raw).
    from scipy.stats import pearsonr
    r_fit, p_fit = pearsonr(xs, ys)
    xr = np.linspace(xs.min() - 0.05, xs.max() + 0.05, 50)
    slope, intercept = np.polyfit(xs, ys, 1)
    ax.plot(xr, slope * xr + intercept, color="#c0392b", lw=1.5, alpha=0.7,
            label=f"OLS  (r = {r_fit:+.2f}, p = {p_fit:.2f})")

    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("utility similarity with default  (Pearson r)")
    ax.set_ylabel("mean inbound transfer r  (6 probes → this persona)")
    ax.set_title("Are personas more different from default harder to predict?\n"
                 "(eot, layer 32 — one point per non-default persona)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_receiver_vs_default_similarity.png", dpi=150)
    plt.close(fig)


def fig_cosine_by_layer(personas):
    n = len(personas)
    mask = ~np.eye(n, dtype=bool)
    mean_cos_tb5 = []
    for layer in LAYERS:
        _, _, cos = _load_matrix("tb-5", layer)
        mean_cos_tb5.append(cos[mask].mean())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(LAYERS, mean_cos_tb5, "-o", color=BLUE, lw=2.2, markersize=7)
    for x, y in zip(LAYERS, mean_cos_tb5):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.set_xlabel("layer")
    ax.set_xticks(LAYERS)
    ax.set_ylabel("mean probe-direction cosine\n(21 off-diagonal pairs)")
    ax.set_title("Probe directions dip mid-network, converge in late layers (eot)")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_cosine_by_layer.png", dpi=150)
    plt.close(fig)


def fig_self_fit_vs_donor(personas, order):
    ordered = [personas[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(personas)))
    color_by_persona = {p: colors[i] for i, p in enumerate(ordered)}
    markers = ["o", "s", "^", "D", "P"]
    legend_shown = set()
    for li, layer in enumerate(LAYERS):
        _, tr, _ = _load_matrix("tb-5", layer)
        n = tr.shape[0]
        mask = ~np.eye(n, dtype=bool)
        for persona in ordered:
            pi = personas.index(persona)
            self_fit = tr[pi, pi]
            donor = tr[pi][mask[pi]].mean()
            label = persona if persona not in legend_shown else None
            legend_shown.add(persona)
            ax.scatter(self_fit, donor, s=80, c=[color_by_persona[persona]],
                       marker=markers[li], edgecolor="black", lw=0.4, label=label)
    ax.set_xlabel("within-persona r (diagonal)")
    ax.set_ylabel("mean outbound off-diagonal r")
    ax.set_title("Self-fit vs donor quality — 7 personas × 5 layers (eot)\n"
                 "markers: ○=L25  □=L32  △=L39  ◇=L46  ✚=L53")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_self_fit_vs_donor.png", dpi=150)
    plt.close(fig)


def fig_heatmap_grid(personas, order):
    labels = [personas[i] for i in order]
    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    for row, (tag, _) in enumerate(SELECTORS):
        label_sel = "eot" if tag == "tb-5" else tag
        for col, layer in enumerate(LAYERS):
            _, tr, _ = _load_matrix(tag, layer)
            ax = axes[row, col]
            _heatmap(ax, _reorder(tr, order), labels, f"{label_sel}, L{layer}")
            if col == 0:
                ax.set_ylabel("probe")
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
    order = _fixed_order(personas, utility_r)
    print("fixed order:", [personas[i] for i in order])

    fig_default_probe(personas, transfer, utility_r, order)
    fig_transfer_utility_pair(personas, transfer, utility_r, order)
    fig_layer_dependence(personas, order)
    fig_asymmetry(personas, transfer, order)
    fig_receiver_vs_default_similarity(personas, transfer, utility_r)
    fig_cosine_by_layer(personas)
    fig_self_fit_vs_donor(personas, order)
    fig_heatmap_grid(personas, order)

    print(f"8 figures written to {ASSETS.relative_to(REPO)}/ with date stamp {TODAY}")


if __name__ == "__main__":
    main()
