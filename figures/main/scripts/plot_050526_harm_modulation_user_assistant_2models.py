"""Harm-only persona modulation, both turn positions, both models.

Main-text §3.1 figure. 2x2 grid:
    Row: Gemma-3-27B | Qwen-3.5-122B-A10B
    Col: User end-of-turn | Assistant end-of-turn
Each panel: 3 personas (Assistant, aura, evil) x harmful/benign violins,
with the Qwen3-Embedding-8B text-encoder baseline overlaid as orange
dashed segments anchored at the LM-probe class-mean midpoint, spread by
(enc_d / 2) * LM-probe pooled SD.

Usage:
    python paper/figures/main/scripts/plot_050526_harm_modulation_user_assistant_2models.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3].parent
GEMMA_USER = REPO / "experiments/eot_discrimination_v2/scoring/gemma3_27b/user_turn_scoring_results.json"
GEMMA_ASSIST = REPO / "experiments/eot_discrimination_v2/scoring/gemma3_27b/scoring_results.json"
QWEN_USER = REPO / "experiments/eot_discrimination_v2/scoring/qwen35_122b/user_turn_scoring_results.json"
QWEN_ASSIST = REPO / "experiments/eot_discrimination_v2/scoring/qwen35_122b/scoring_results.json"
ENC_GEMMA_USER = REPO / "experiments/descriptive_baseline_extensions/eot_baseline_user_gemma-3-27b.json"
ENC_GEMMA_ASSIST = REPO / "experiments/descriptive_baseline_extensions/eot_baseline_assistant_gemma-3-27b.json"
ENC_QWEN_USER = REPO / "experiments/descriptive_baseline_extensions/eot_baseline_user_qwen-3.5-122b.json"
ENC_QWEN_ASSIST = REPO / "experiments/descriptive_baseline_extensions/eot_baseline_assistant_qwen-3.5-122b.json"
OUT_PATH = REPO / "paper/figures/main/plot_050526_harm_modulation_user_assistant_2models.png"

COLORS = {"benign": "#2E7D32", "harmful": "#D32F2F"}
DISPLAY_LABELS = {"neutral": "Assistant", "sadist": "evil"}


def display(sp):
    return DISPLAY_LABELS.get(sp, sp)


def cohen_d_with_ci(pos, neg, z=1.96):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    n1, n2 = len(pos), len(neg)
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan"), float("nan")
    pooled = np.sqrt(((n1 - 1) * pos.var(ddof=1) + (n2 - 1) * neg.var(ddof=1)) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0, 0.0, 0.0
    d = (pos.mean() - neg.mean()) / pooled
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2 - 2)))
    return float(d), float(d - z * se), float(d + z * se)


def lm_pooled_sd(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    n1, n2 = len(pos), len(neg)
    return float(np.sqrt(((n1 - 1) * pos.var(ddof=1) + (n2 - 1) * neg.var(ddof=1)) / (n1 + n2 - 2)))


def load_encoder_d(path):
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    return {(r["domain"], r["system_prompt"]): r["cohen_d"] for r in d["rows"]}


def panel(ax, items, prompts, probe, c_pos, c_neg, title,
          ylabel=None, encoder_d=None, baseline_handle=None):
    by_sp = defaultdict(list)
    for it in items:
        by_sp[it["system_prompt"]].append(it)

    positions, all_series, all_colors, d_values, enc_segments = [], [], [], [], []
    valid_prompts = []
    for sp in prompts:
        pos_vals = [it["probe_scores"][probe] for it in by_sp.get(sp, []) if it["condition"] == c_pos]
        neg_vals = [it["probe_scores"][probe] for it in by_sp.get(sp, []) if it["condition"] == c_neg]
        if not pos_vals or not neg_vals:
            continue
        valid_prompts.append(sp)
        pi = len(valid_prompts) - 1
        d, lo, hi = cohen_d_with_ci(pos_vals, neg_vals)
        d_values.append((sp, d, lo, hi))
        positions.extend([pi * 3, pi * 3 + 1])
        all_series.extend([pos_vals, neg_vals])
        all_colors.extend([COLORS[c_pos], COLORS[c_neg]])
        if encoder_d is not None:
            ed = encoder_d.get(("harm", sp))
            if ed is not None and not np.isnan(ed):
                sigma = lm_pooled_sd(pos_vals, neg_vals)
                mid = (np.mean(pos_vals) + np.mean(neg_vals)) / 2.0
                offset = (ed / 2.0) * sigma
                enc_segments.append((pi * 3, pi * 3 + 1, mid, offset))

    parts = ax.violinplot(all_series, positions=positions, widths=0.9,
                          showmeans=True, showextrema=False)
    for body, color in zip(parts["bodies"], all_colors):
        body.set_facecolor(color)
        body.set_alpha(0.75)
        body.set_edgecolor("black")
    parts["cmeans"].set_color("black")

    for pos_x, neg_x, mid, offset in enc_segments:
        for x_idx, sign in ((pos_x, +1.0), (neg_x, -1.0)):
            line = ax.hlines(mid + sign * offset,
                             x_idx - 0.45, x_idx + 0.45,
                             colors="#ff7f0e", linestyles="--", linewidth=2.4,
                             zorder=5)
            if baseline_handle is not None and baseline_handle[0] is None:
                baseline_handle[0] = line

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xticks([pi * 3 + 0.5 for pi in range(len(valid_prompts))])
    labels = [f"{display(sp)}\n$d = {d:+.2f} \\pm {(hi - lo) / 2.0:.2f}$"
              for sp, d, lo, hi in d_values]
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[c_pos], alpha=0.75),
               plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[c_neg], alpha=0.75)]
    ax.legend(handles, [c_pos, c_neg], loc="best", fontsize=9)

    return {sp: (d, lo, hi) for sp, d, lo, hi in d_values}


def main():
    g_user = json.load(open(GEMMA_USER))["items"]
    g_assist = json.load(open(GEMMA_ASSIST))["items"]
    q_user = json.load(open(QWEN_USER))["items"]
    q_assist = json.load(open(QWEN_ASSIST))["items"]

    g_user_harm = [it for it in g_user if it["domain"] == "harm"]
    g_assist_harm = [it for it in g_assist if it["domain"] == "harm"]
    q_user_harm = [it for it in q_user if it["domain"] == "harm"]
    q_assist_harm = [it for it in q_assist if it["domain"] == "harm"]

    enc_g_user = load_encoder_d(ENC_GEMMA_USER)
    enc_g_assist = load_encoder_d(ENC_GEMMA_ASSIST)
    enc_q_user = load_encoder_d(ENC_QWEN_USER)
    enc_q_assist = load_encoder_d(ENC_QWEN_ASSIST)

    harm_prompts = ["neutral", "aura", "sadist"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    baseline_handle = [None]

    panel(axes[0, 0], g_user_harm, harm_prompts, "tb-5_L32",
          "harmful", "benign",
          "Gemma-3-27B  |  user end-of-turn",
          ylabel="Probe score",
          encoder_d=enc_g_user, baseline_handle=baseline_handle)
    panel(axes[0, 1], g_assist_harm, harm_prompts, "tb-5_L32",
          "harmful", "benign",
          "Gemma-3-27B  |  assistant end-of-turn",
          encoder_d=enc_g_assist, baseline_handle=baseline_handle)
    panel(axes[1, 0], q_user_harm, harm_prompts, "qwen_tb-4_L38",
          "harmful", "benign",
          "Qwen3.5-122B-A10B  |  user end-of-turn",
          ylabel="Probe score",
          encoder_d=enc_q_user, baseline_handle=baseline_handle)
    panel(axes[1, 1], q_assist_harm, harm_prompts, "qwen_tb-4_L38",
          "harmful", "benign",
          "Qwen3.5-122B-A10B  |  assistant end-of-turn",
          encoder_d=enc_q_assist, baseline_handle=baseline_handle)

    fig.suptitle(
        "Persona modulation on harm: collapse at user turn, sign flip at assistant turn (Gemma)",
        fontsize=11, y=1.00,
    )
    if baseline_handle[0] is not None:
        fig.legend(
            handles=[baseline_handle[0]],
            labels=["Qwen3-Embedding-8B (text-encoder baseline)"],
            loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=1,
            frameon=False, fontsize=11,
        )
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
