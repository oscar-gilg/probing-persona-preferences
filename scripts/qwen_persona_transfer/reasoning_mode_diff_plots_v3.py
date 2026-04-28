"""Refined plots for the reasoning_mode_diff_v2 report.

Reads the same diff_v2.npz / persona_diff.npz / summary_v2.json data and produces
better-laid-out plots for the report:

- v3_scatter_by_origin.png    larger, readable markers + better-placed legend
- v3_per_origin_correlation.png  origins sorted by Pearson r ascending
- v3_mean_mu_per_origin.png    same ordering, y-axis labelled, no overlap
- v3_mu_distribution.png       same ordering, y-axis labelled
- v3_shift_magnitude.png       same ordering
- v3_persona_overall_agreement.png  bar chart of per-persona overall r (replaces table)
- v3_per_persona_per_origin_pearson.png  personas sorted by overall r descending
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments/qwen_replication/persona_transfer/reasoning_mode_diff/results"
ASSETS = REPO / "experiments/qwen_replication/persona_transfer/reasoning_mode_diff/assets"
DATE = datetime.now().strftime("%m%d%y")

ORIGIN_COLORS = {
    "wildchat": "#1f77b4",
    "alpaca": "#ff7f0e",
    "math": "#2ca02c",
    "bailbench": "#d62728",
    "stress_test": "#9467bd",
}


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    diff = np.load(RESULTS / "diff_v2.npz", allow_pickle=True)
    nt = diff["no_think_mu"]; th = diff["think_mu"]; origins = diff["origins"]
    n_total = len(nt)

    with open(RESULTS / "summary_v2.json") as f:
        summary = json.load(f)

    # Sort origins by within-origin Pearson r (ascending — least agreement first)
    rows = sorted(summary["per_origin"], key=lambda r: r["pearson"])
    origin_order = [r["origin"] for r in rows]
    print(f"origin order (asc r): {origin_order}")

    # Sort personas by overall r (descending — most regime-stable first)
    pers_rows = sorted(summary["per_persona"], key=lambda r: -r["pearson"])

    # =========================================================
    # 1. Scatter by origin (large, readable)
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 9))
    for origin in origin_order:
        m = origins == origin
        if m.sum() == 0: continue
        ax.scatter(nt[m], th[m], s=14, alpha=0.45, color=ORIGIN_COLORS[origin],
                   edgecolors="none", label=f"{origin} (n={m.sum()})")
    lims = (-12, 12)
    ax.plot(lims, lims, "k--", alpha=0.4, lw=1, label="y = x (perfect agreement)")
    ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("μ — reasoning OFF (no-think)", fontsize=12)
    ax.set_ylabel("μ — reasoning ON, low effort", fontsize=12)
    overall = summary["overall_pearson"]
    overall_sp = summary["overall_spearman"]
    ax.set_title(f"Default-Qwen preference shift by reasoning mode\n"
                 f"{n_total} shared canonical tasks · "
                 f"overall Pearson r = {overall:+.2f} · Spearman ρ = {overall_sp:+.2f}",
                 fontsize=12)
    # Move legend outside plot area
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10, frameon=False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{DATE}_v3_scatter_by_origin.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # =========================================================
    # 2. Per-origin correlation (ordered by r ascending)
    # =========================================================
    fig, ax = plt.subplots(figsize=(9, 5))
    rs = [r["pearson"] for r in rows]
    rhos = [r["spearman"] for r in rows]
    x = np.arange(len(origin_order))
    w = 0.36
    colors = [ORIGIN_COLORS[o] for o in origin_order]
    ax.bar(x - w/2, rs, w, label="Pearson r", color=colors, alpha=0.85)
    ax.bar(x + w/2, rhos, w, label="Spearman ρ",
           color=colors, alpha=0.5, hatch="//", edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(0.9, color="green", lw=0.5, ls=":", alpha=0.7)
    ax.text(len(origin_order) - 0.5, 0.91, "same-regime baseline (r ≈ 0.9)",
            color="green", fontsize=9, ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(origin_order)
    ax.set_ylabel("correlation between regimes")
    ax.set_ylim(-0.1, 1.0)
    ax.set_title("Within-origin agreement: reasoning ON (low effort) vs OFF\n"
                 "ordered by Pearson r — math, alpaca, wildchat are at noise level")
    for i, (rr, rho) in enumerate(zip(rs, rhos)):
        ax.text(i - w/2, rr + 0.02, f"{rr:+.2f}", ha="center", fontsize=10)
        ax.text(i + w/2, rho + 0.02, f"{rho:+.2f}", ha="center", fontsize=10)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{DATE}_v3_per_origin_correlation.png", dpi=150)
    plt.close(fig)

    # =========================================================
    # 3. Mean μ per origin per regime
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    nt_means = [r["mean_no_think"] for r in rows]
    th_means = [r["mean_think"] for r in rows]
    nt_stds = [r["std_no_think"] for r in rows]
    th_stds = [r["std_think"] for r in rows]
    x = np.arange(len(origin_order))
    w = 0.36
    ax.bar(x - w/2, nt_means, w, yerr=nt_stds, label="reasoning OFF",
           color="#9aa0a6", capsize=4)
    ax.bar(x + w/2, th_means, w, yerr=th_stds, label="reasoning ON (low effort)",
           color="#1a73e8", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(origin_order)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("mean μ ± std per origin")
    ax.set_ylim(-12, 13)
    ax.set_title("Mean preference per task category — does default-Qwen like or dislike each?\n"
                 "(error bars = std; same origin ordering as Figure 2)")
    # Place text labels above/below bars without overlap, accounting for error bars
    for i, (nm, tm, nstd, tstd) in enumerate(zip(nt_means, th_means, nt_stds, th_stds)):
        nt_top = nm + nstd if nm >= 0 else nm - nstd
        th_top = tm + tstd if tm >= 0 else tm - tstd
        nt_offset = 0.7 if nm >= 0 else -1.0
        th_offset = 0.7 if tm >= 0 else -1.0
        ax.text(i - w/2, nt_top + nt_offset,
                f"{nm:+.1f}", ha="center", fontsize=10, color="#5f6368")
        ax.text(i + w/2, th_top + th_offset,
                f"{tm:+.1f}", ha="center", fontsize=10, color="#1a73e8", fontweight="bold")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{DATE}_v3_mean_mu_per_origin.png", dpi=150)
    plt.close(fig)

    # =========================================================
    # 4. μ distribution per origin (with y-axis label)
    # =========================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
    bins = np.linspace(-12, 12, 30)
    for ax, origin in zip(axes.flat, origin_order + [None]):
        if origin is None:
            ax.axis("off"); continue
        m = origins == origin
        if m.sum() == 0: continue
        ax.hist(nt[m], bins=bins, alpha=0.55, color="#9aa0a6", label="reasoning OFF")
        ax.hist(th[m], bins=bins, alpha=0.55, color=ORIGIN_COLORS[origin],
                label="reasoning ON")
        ax.axvline(0, color="black", lw=0.5)
        ax.set_title(f"{origin} (n={m.sum()})")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.3)
        ax.set_xlabel("μ (Thurstonian preference)")
        ax.set_ylabel("count")
    fig.suptitle("μ distribution per origin — full 6000-task overlap", fontsize=12)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{DATE}_v3_mu_distribution.png", dpi=150)
    plt.close(fig)

    # =========================================================
    # 5. Per-persona overall agreement (NEW: replaces 7-row table)
    # =========================================================
    fig, ax = plt.subplots(figsize=(9.5, 5))
    pers_names = [r["persona"] for r in pers_rows]
    pers_pearson = [r["pearson"] for r in pers_rows]
    pers_spearman = [r["spearman"] for r in pers_rows]
    x = np.arange(len(pers_names))
    w = 0.36
    ax.bar(x - w/2, pers_pearson, w, label="Pearson r", color="#4c8cd6")
    ax.bar(x + w/2, pers_spearman, w, label="Spearman ρ", color="#d68c4c")
    ax.set_xticks(x); ax.set_xticklabels(pers_names, rotation=20)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(0.9, color="green", lw=0.5, ls=":", alpha=0.7)
    ax.text(0.02, 0.91, "same-regime baseline (r ≈ 0.9)",
            color="green", fontsize=9, ha="left", va="bottom")
    for i, (p, s) in enumerate(zip(pers_pearson, pers_spearman)):
        ax.text(i - w/2, p + 0.015, f"{p:+.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, s + 0.015, f"{s:+.2f}", ha="center", fontsize=9)
    ax.set_ylim(-0.1, 1.0)
    ax.set_ylabel("agreement between regimes")
    ax.set_title("Per-persona regime invariance (overall, all 6000 tasks)\n"
                 "ordered by Pearson r — sadist + strategist nearly orthogonal across regimes")
    ax.legend(frameon=False, loc="center right")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{DATE}_v3_persona_overall_agreement.png", dpi=150)
    plt.close(fig)

    # =========================================================
    # 6. Per-persona per-origin heatmap (personas sorted by overall r desc)
    # =========================================================
    M = np.array([[r["per_origin_pearson"][o] for o in origin_order] for r in pers_rows])
    fig, ax = plt.subplots(figsize=(8.5, 5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.4, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(origin_order))); ax.set_xticklabels(origin_order)
    ax.set_yticks(range(len(pers_names))); ax.set_yticklabels(pers_names)
    for i in range(len(pers_names)):
        for j in range(len(origin_order)):
            v = M[i, j]
            txt = f"{v:+.2f}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color="white" if abs(v) > 0.3 else "black")
    fig.colorbar(im, ax=ax, label="within-origin Pearson r between regimes")
    ax.set_title("Per-persona × per-origin agreement (reasoning ON vs OFF)\n"
                 "rows: personas sorted by overall regime agreement (high → low)\n"
                 "cols: origins sorted by Pearson r (low → high)")
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{DATE}_v3_per_persona_per_origin_pearson.png", dpi=150)
    plt.close(fig)

    # =========================================================
    # 7. Shift magnitude boxplot (origins sorted by agreement asc)
    # =========================================================
    fig, ax = plt.subplots(figsize=(9, 4.5))
    delta_data = []
    delta_labels = []
    for origin in origin_order:
        m = origins == origin
        if m.sum() == 0: continue
        delta_data.append(np.abs(th[m] - nt[m]))
        delta_labels.append(f"{origin}\n(n={m.sum()})")
    bp = ax.boxplot(delta_data, tick_labels=delta_labels, patch_artist=True, showfliers=False)
    for patch, origin in zip(bp["boxes"], origin_order):
        patch.set_facecolor(ORIGIN_COLORS[origin])
        patch.set_alpha(0.6)
    ax.set_ylabel("|μ_ON − μ_OFF| per task")
    ax.set_title("Magnitude of preference shift per task, by origin\n"
                 "(same origin ordering as Figure 2; lower = less affected by regime swap)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{DATE}_v3_shift_magnitude.png", dpi=150)
    plt.close(fig)

    print(f"v3 plots saved to {ASSETS.relative_to(REPO)}")


if __name__ == "__main__":
    main()
