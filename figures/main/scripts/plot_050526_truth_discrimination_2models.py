"""Truth-only base discrimination figure (Gemma + Qwen, user-turn).

1 row x 2 cols of violin panels: Gemma-3-27B and Qwen-3.5-122B end-of-turn
probe scores on CREAK true/false statements. No encoder baseline, no
persona/system-prompt qualifier in the figure framing — this is the
"probe discriminates truth/false" panel for §2.4.

Usage:
    python paper/figures/main/scripts/plot_050526_truth_discrimination_2models.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3].parent
GEMMA = REPO / "experiments/eot_discrimination_v2/scoring/gemma3_27b/user_turn_scoring_results.json"
QWEN  = REPO / "experiments/eot_discrimination_v2/scoring/qwen35_122b/user_turn_scoring_results.json"
OUT   = REPO / "paper/figures/main/plot_050526_truth_discrimination_2models.png"

COLOR_TRUE  = "#2196F3"
COLOR_FALSE = "#D32F2F"


def cohen_d_with_ci(pos, neg, z=1.96):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    n1, n2 = len(pos), len(neg)
    pooled = np.sqrt(((n1 - 1) * pos.var(ddof=1) + (n2 - 1) * neg.var(ddof=1)) / (n1 + n2 - 2))
    d = (pos.mean() - neg.mean()) / pooled
    se = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2 - 2)))
    return float(d), float(d - z * se), float(d + z * se)


def truth_scores(path: Path, probe: str):
    items = json.load(open(path))["items"]
    sub = [it for it in items
           if it["domain"] == "truth"
           and it.get("turn", "user") == "user"
           and it.get("system_prompt") == "neutral"]
    pos = np.array([it["probe_scores"][probe] for it in sub if it["condition"] == "true"])
    neg = np.array([it["probe_scores"][probe] for it in sub if it["condition"] == "false"])
    return pos, neg


def violin_panel(ax, pos, neg, title, ylabel=None):
    parts = ax.violinplot([pos, neg], positions=[0, 1], widths=0.7,
                          showmeans=True, showextrema=False)
    for body, color in zip(parts["bodies"], [COLOR_TRUE, COLOR_FALSE]):
        body.set_facecolor(color)
        body.set_alpha(0.7)
        body.set_edgecolor("black")
    parts["cmeans"].set_color("black")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["true", "false"], fontsize=10)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", alpha=0.3)

    d, lo, hi = cohen_d_with_ci(pos, neg)
    half_ci = (hi - lo) / 2.0
    ax.set_title(f"{title}\n$d = {d:+.2f} \\pm {half_ci:.2f}$", fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    return d, lo, hi


def main():
    g_pos, g_neg = truth_scores(GEMMA, "tb-5_L32")
    q_pos, q_neg = truth_scores(QWEN,  "qwen_tb-4_L38")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    d_g, lo_g, hi_g = violin_panel(axes[0], g_pos, g_neg,
                                    "Gemma-3-27B — Truth (CREAK)",
                                    ylabel="End-of-turn probe score")
    d_q, lo_q, hi_q = violin_panel(axes[1], q_pos, q_neg,
                                    "Qwen-3.5-122B — Truth (CREAK)")

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"wrote {OUT}")
    print(f"  Gemma truth d = {d_g:+.2f} [{lo_g:+.2f}, {hi_g:+.2f}] (n={len(g_pos)}/{len(g_neg)})")
    print(f"  Qwen  truth d = {d_q:+.2f} [{lo_q:+.2f}, {hi_q:+.2f}] (n={len(q_pos)}/{len(q_neg)})")


if __name__ == "__main__":
    main()
