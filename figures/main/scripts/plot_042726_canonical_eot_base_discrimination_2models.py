"""Two-model end-of-turn base discrimination figure (Gemma + Qwen, user-turn, neutral).

2 rows x 2 cols of violin panels:
    Top row    Gemma-3-27B  : Truth (tb-5_L32),   Harm (tb-5_L32)
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
GEMMA_PARENT = REPO / "experiments/eot_discrimination_v2/scoring/gemma3_27b/user_turn_scoring_results.json"
QWEN_USER_TURN = REPO / "experiments/eot_discrimination_v2/scoring/qwen35_122b/user_turn_scoring_results.json"
ENCODER_GEMMA = REPO / "experiments/descriptive_baseline_extensions/eot_baseline_user_gemma-3-27b.json"
ENCODER_QWEN = REPO / "experiments/descriptive_baseline_extensions/eot_baseline_user_qwen-3.5-122b.json"
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
    sub = [it for it in items
           if it["domain"] == domain and it.get("turn") == "user"
           and it.get("system_prompt") == "neutral"]
    pos = np.array([it["probe_scores"][probe] for it in sub if it["condition"] == c_pos])
    neg = np.array([it["probe_scores"][probe] for it in sub if it["condition"] == c_neg])
    non = np.array([it["probe_scores"][probe] for it in sub if it["condition"] == c_non])
    return pos, neg, non


def qwen_neutral_user_turn(domain, probe, c_pos, c_neg):
    items = json.load(open(QWEN_USER_TURN))["items"]
    sub = [it for it in items
           if it["domain"] == domain and it.get("system_prompt") == "neutral"]
    pos = np.array([it["probe_scores"][probe] for it in sub if it["condition"] == c_pos])
    neg = np.array([it["probe_scores"][probe] for it in sub if it["condition"] == c_neg])
    return pos, neg


def load_encoder_neutral_d(path: Path, domain: str) -> float | None:
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    for r in d["rows"]:
        if r["domain"] == domain and r["system_prompt"] == "neutral":
            return r["cohen_d"]
    return None


def lm_pooled_sd(pos, neg):
    """LM-probe pooled SD (matches denominator of cohen_d_with_ci)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    n1, n2 = len(pos), len(neg)
    return float(np.sqrt(
        ((n1 - 1) * pos.var(ddof=1) + (n2 - 1) * neg.var(ddof=1)) / (n1 + n2 - 2)
    ))


def violin_panel(ax, series, colors_used, tick_labels, title_top, d, lo, hi,
                 ylabel=None, encoder_d=None, lm_pos=None, lm_neg=None,
                 baseline_handle=None):
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

    # Encoder dotted segments: anchored at the LM-probe class-mean midpoint,
    # spread by (enc_d / 2) * LM-probe pooled SD so the gap between the two
    # dotted lines visually matches the encoder's Cohen's d expressed in
    # LM-probe score units.
    if encoder_d is not None and lm_pos is not None and lm_neg is not None:
        sigma = lm_pooled_sd(lm_pos, lm_neg)
        mid = (np.mean(lm_pos) + np.mean(lm_neg)) / 2.0
        offset = (encoder_d / 2.0) * sigma
        # series[0] is the positive class, series[1] is the negative class
        for pos_idx, sign in zip(positions, (+1.0, -1.0)):
            line = ax.hlines(mid + sign * offset,
                             pos_idx - 0.40, pos_idx + 0.40,
                             colors="#ff7f0e", linestyles="--", linewidth=2.6,
                             zorder=5)
            if baseline_handle is not None and baseline_handle[0] is None:
                baseline_handle[0] = line

    half_ci = (hi - lo) / 2.0
    ax.set_title(f"{title_top}\n$d = {d:+.2f} \\pm {half_ci:.2f}$", fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel)


def main():
    g_truth_pos, g_truth_neg, _ = gemma_neutral_user_turn(
        "truth", "tb-5_L32", "true", "false"
    )
    d_g_truth, lo_g_truth, hi_g_truth = cohen_d_with_ci(g_truth_pos, g_truth_neg)

    q_truth_pos, q_truth_neg = qwen_neutral_user_turn(
        "truth", "qwen_tb-4_L38", "true", "false"
    )
    d_q_truth, lo_q_truth, hi_q_truth = cohen_d_with_ci(q_truth_pos, q_truth_neg)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    enc_g_truth = load_encoder_neutral_d(ENCODER_GEMMA, "truth")
    enc_q_truth = load_encoder_neutral_d(ENCODER_QWEN, "truth")

    baseline_handle = [None]

    violin_panel(
        axes[0],
        series=[g_truth_pos, g_truth_neg],
        colors_used=[COLORS["true"], COLORS["false"]],
        tick_labels=["true", "false"],
        title_top="Gemma-3-27B — Truth (CREAK)",
        d=d_g_truth, lo=lo_g_truth, hi=hi_g_truth,
        ylabel="End-of-turn probe score",
        encoder_d=enc_g_truth,
        lm_pos=g_truth_pos, lm_neg=g_truth_neg,
        baseline_handle=baseline_handle,
    )

    violin_panel(
        axes[1],
        series=[q_truth_pos, q_truth_neg],
        colors_used=[COLORS["true"], COLORS["false"]],
        tick_labels=["true", "false"],
        title_top="Qwen3.5-122B-A10B — Truth (CREAK)",
        d=d_q_truth, lo=lo_q_truth, hi=hi_q_truth,
        encoder_d=enc_q_truth,
        lm_pos=q_truth_pos, lm_neg=q_truth_neg,
        baseline_handle=baseline_handle,
    )

    fig.suptitle(
        "Base discrimination on truth at the user end-of-turn (neutral persona)",
        fontsize=11, y=1.02,
    )
    if baseline_handle[0] is not None:
        fig.legend(
            handles=[baseline_handle[0]],
            labels=["Qwen3-Embedding-8B (text-encoder baseline)"],
            loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=1,
            frameon=False, fontsize=11,
        )
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"wrote {OUT_PATH}")
    print(f"  Gemma truth d = {d_g_truth:+.2f} [{lo_g_truth:+.2f}, {hi_g_truth:+.2f}] (n={len(g_truth_pos)}/{len(g_truth_neg)})")
    print(f"  Qwen  truth d = {d_q_truth:+.2f} [{lo_q_truth:+.2f}, {hi_q_truth:+.2f}] (n={len(q_truth_pos)}/{len(q_truth_neg)})")


if __name__ == "__main__":
    main()
