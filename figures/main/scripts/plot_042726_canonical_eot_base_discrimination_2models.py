"""Two-model end-of-turn base discrimination figure (Gemma + Qwen, user-turn, neutral).

2 rows x 2 cols of violin panels:
    Top row    Gemma-3-27B  : Truth (tb-5_L32),   Harm (tb-5_L39)
    Bottom row Qwen-3.5-122B: Truth (qwen_tb-4_L38), Harm (qwen_tb-4_L38)

Gemma panels include the nonsense control violin (3 violins). Qwen panels show
true/false and harmful/benign only (2 violins) per the spec, even though the
qwen user-turn data also contains a nonsense bucket.

Usage:
    python paper/figures/main/scripts/plot_042726_canonical_eot_base_discrimination_2models.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3].parent  # repo root
GEMMA_PARENT = REPO / "experiments/token_level_probes/system_prompt_modulation_v2/parent_eot_scores.json"
QWEN_USER_TURN = REPO / "experiments/token_level_probes/qwen_canonical_probe_eval/user_turn_scoring_results.json"
OUT_PATH = REPO / "paper/figures/main/plot_042726_canonical_eot_base_discrimination_2models.png"

# Match colors used in the reference Gemma plot
COLORS = {
    "true": "#2196F3",     # blue
    "false": "#D32F2F",    # red
    "benign": "#2E7D32",   # green
    "harmful": "#D32F2F",  # red
    "nonsense": "#9E9E9E", # grey
}


def cohen_d_with_ci(pos, neg, z=1.96):
    """Pooled Cohen's d with Hedges/Olkin analytical 95% CI.

    SE(d) = sqrt((n1+n2)/(n1*n2) + d^2 / (2*(n1+n2-2)))
    """
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    n1, n2 = len(pos), len(neg)
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan"), float("nan")
    pooled = np.sqrt(
        ((n1 - 1) * pos.var(ddof=1) + (n2 - 1) * neg.var(ddof=1)) / (n1 + n2 - 2)
    )
    if pooled == 0:
        return 0.0, 0.0, 0.0
    d = (pos.mean() - neg.mean()) / pooled
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2 - 2)))
    return float(d), float(d - z * se), float(d + z * se)


def gemma_neutral_user_turn(domain, probe, c_pos, c_neg, c_non="nonsense"):
    items = json.load(open(GEMMA_PARENT))["items"]
    sub = [it for it in items if it["domain"] == domain and it.get("turn") == "user"]
    pos = np.array([it["eot_scores"][probe] for it in sub if it["condition"] == c_pos])
    neg = np.array([it["eot_scores"][probe] for it in sub if it["condition"] == c_neg])
    non = np.array([it["eot_scores"][probe] for it in sub if it["condition"] == c_non])
    return pos, neg, non


def qwen_neutral_user_turn(domain, probe, c_pos, c_neg):
    items = json.load(open(QWEN_USER_TURN))["items"]
    sub = [it for it in items
           if it["domain"] == domain and it.get("system_prompt") == "neutral"]
    pos = np.array([it["probe_scores"][probe] for it in sub if it["condition"] == c_pos])
    neg = np.array([it["probe_scores"][probe] for it in sub if it["condition"] == c_neg])
    return pos, neg


def violin_panel(ax, series, colors_used, tick_labels, title_top, d, lo, hi,
                 n_pos, n_neg, ylabel=None):
    positions = list(range(len(series)))
    parts = ax.violinplot(series, positions=positions, widths=0.7,
                          showmeans=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors_used):
        body.set_facecolor(color)
        body.set_alpha(0.7)
        body.set_edgecolor("black")
    parts["cmeans"].set_color("black")

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        f"{title_top}\n(d = {d:+.2f} [{lo:+.2f}, {hi:+.2f}], n = {n_pos}/{n_neg})",
        fontsize=10,
    )
    if ylabel:
        ax.set_ylabel(ylabel)


def main():
    # --- Gemma panels ---
    g_truth_pos, g_truth_neg, g_truth_non = gemma_neutral_user_turn(
        "truth", "tb-5_L32", "true", "false"
    )
    g_harm_pos, g_harm_neg, g_harm_non = gemma_neutral_user_turn(
        "harm", "tb-5_L39", "harmful", "benign"
    )
    d_g_truth, lo_g_truth, hi_g_truth = cohen_d_with_ci(g_truth_pos, g_truth_neg)
    d_g_harm, lo_g_harm, hi_g_harm = cohen_d_with_ci(g_harm_pos, g_harm_neg)

    # --- Qwen panels (no nonsense, per spec) ---
    q_truth_pos, q_truth_neg = qwen_neutral_user_turn(
        "truth", "qwen_tb-4_L38", "true", "false"
    )
    q_harm_pos, q_harm_neg = qwen_neutral_user_turn(
        "harm", "qwen_tb-4_L38", "harmful", "benign"
    )
    d_q_truth, lo_q_truth, hi_q_truth = cohen_d_with_ci(q_truth_pos, q_truth_neg)
    d_q_harm, lo_q_harm, hi_q_harm = cohen_d_with_ci(q_harm_pos, q_harm_neg)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))

    # axes[0]: Gemma truth
    violin_panel(
        axes[0],
        series=[g_truth_pos, g_truth_neg],
        colors_used=[COLORS["true"], COLORS["false"]],
        tick_labels=["true", "false"],
        title_top="Gemma-3-27B — Truth (CREAK)",
        d=d_g_truth, lo=lo_g_truth, hi=hi_g_truth,
        n_pos=len(g_truth_pos), n_neg=len(g_truth_neg),
        ylabel="End-of-turn probe score",
    )

    # axes[1]: Gemma harm
    violin_panel(
        axes[1],
        series=[g_harm_pos, g_harm_neg],
        colors_used=[COLORS["harmful"], COLORS["benign"]],
        tick_labels=["harmful", "benign"],
        title_top="Gemma-3-27B — Harm (BailBench+STRESS-TEST)",
        d=d_g_harm, lo=lo_g_harm, hi=hi_g_harm,
        n_pos=len(g_harm_pos), n_neg=len(g_harm_neg),
    )

    # axes[2]: Qwen truth
    violin_panel(
        axes[2],
        series=[q_truth_pos, q_truth_neg],
        colors_used=[COLORS["true"], COLORS["false"]],
        tick_labels=["true", "false"],
        title_top="Qwen3.5-122B-A10B — Truth (CREAK)",
        d=d_q_truth, lo=lo_q_truth, hi=hi_q_truth,
        n_pos=len(q_truth_pos), n_neg=len(q_truth_neg),
    )

    # axes[3]: Qwen harm
    violin_panel(
        axes[3],
        series=[q_harm_pos, q_harm_neg],
        colors_used=[COLORS["harmful"], COLORS["benign"]],
        tick_labels=["harmful", "benign"],
        title_top="Qwen3.5-122B-A10B — Harm (BailBench+STRESS-TEST)",
        d=d_q_harm, lo=lo_q_harm, hi=hi_q_harm,
        n_pos=len(q_harm_pos), n_neg=len(q_harm_neg),
    )

    fig.suptitle(
        "Base discrimination at the end-of-turn token (user-turn, neutral persona)",
        fontsize=11, y=1.00,
    )
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"wrote {OUT_PATH}")
    print(f"  Gemma truth d = {d_g_truth:+.2f} [{lo_g_truth:+.2f}, {hi_g_truth:+.2f}] (n={len(g_truth_pos)}/{len(g_truth_neg)})")
    print(f"  Gemma harm  d = {d_g_harm:+.2f} [{lo_g_harm:+.2f}, {hi_g_harm:+.2f}] (n={len(g_harm_pos)}/{len(g_harm_neg)})")
    print(f"  Qwen  truth d = {d_q_truth:+.2f} [{lo_q_truth:+.2f}, {hi_q_truth:+.2f}] (n={len(q_truth_pos)}/{len(q_truth_neg)})")
    print(f"  Qwen  harm  d = {d_q_harm:+.2f} [{lo_q_harm:+.2f}, {hi_q_harm:+.2f}] (n={len(q_harm_pos)}/{len(q_harm_neg)})")


if __name__ == "__main__":
    main()
