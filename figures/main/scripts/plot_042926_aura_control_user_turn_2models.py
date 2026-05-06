"""Two-model persona-modulation figure with aura positive-persona control (USER turn).

Truth-only (harm panels live in plot_050526_harm_modulation_user_assistant_2models.py).
Aura is the "positive persona" control: under aura, the readout should preserve
(or strengthen) true/false separation.

The Qwen3-Embedding-8B text-encoder baseline is overlaid as orange dashed
segments anchored at the LM-probe class-mean midpoint, spread by
(enc_d / 2) * LM-probe pooled SD.

Usage:
    python paper/figures/main/scripts/plot_042926_aura_control_user_turn_2models.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3].parent
GEMMA_PATH = REPO / "experiments/eot_discrimination_v2/scoring/gemma3_27b/user_turn_scoring_results.json"
GEMMA_AURA_PATH = REPO / "experiments/eot_discrimination_v2/scoring/gemma3_27b/_AURA_IN_BASE_FILE.json"
QWEN_PATH = REPO / "experiments/eot_discrimination_v2/scoring/qwen35_122b/user_turn_scoring_results.json"
QWEN_AURA_PATH = REPO / "experiments/eot_discrimination_v2/scoring/qwen35_122b/_AURA_IN_BASE_FILE.json"
ENCODER_GEMMA = REPO / "experiments/descriptive_baseline_extensions/eot_baseline_user_gemma-3-27b.json"
ENCODER_QWEN = REPO / "experiments/descriptive_baseline_extensions/eot_baseline_user_qwen-3.5-122b.json"
OUT_PATH = REPO / "paper/figures/main/plot_042926_aura_control_user_turn_2models.png"

COLORS = {
    "true": "#2196F3",
    "false": "#D32F2F",
}

DISPLAY_LABELS = {"neutral": "Assistant"}


def display(sp: str) -> str:
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


def load_encoder_d(path: Path) -> dict[tuple[str, str], float]:
    with open(path) as f:
        d = json.load(f)
    return {(r["domain"], r["system_prompt"]): r["cohen_d"] for r in d["rows"]}


def panel(ax, items, prompts, probe, c_pos, c_neg, score_key, domain_label,
          ylabel=None, show_legend=True,
          encoder_d=None, domain=None, baseline_handle=None):
    by_sp = defaultdict(list)
    for it in items:
        by_sp[it["system_prompt"]].append(it)

    positions, all_series, all_colors, d_values, enc_segments = [], [], [], [], []
    valid_prompts = []
    for sp in prompts:
        pos_vals = [it[score_key][probe] for it in by_sp.get(sp, []) if it["condition"] == c_pos]
        neg_vals = [it[score_key][probe] for it in by_sp.get(sp, []) if it["condition"] == c_neg]
        if not pos_vals or not neg_vals:
            continue
        valid_prompts.append(sp)
        pi = len(valid_prompts) - 1
        d, lo, hi = cohen_d_with_ci(pos_vals, neg_vals)
        d_values.append((sp, d, lo, hi))
        positions.extend([pi * 3, pi * 3 + 1])
        all_series.extend([pos_vals, neg_vals])
        all_colors.extend([COLORS[c_pos], COLORS[c_neg]])
        if encoder_d is not None and domain is not None:
            ed = encoder_d.get((domain, sp))
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
    ax.set_title(domain_label, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel)

    if show_legend:
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[c_pos], alpha=0.75),
                   plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[c_neg], alpha=0.75)]
        ax.legend(handles, [c_pos, c_neg], loc="best", fontsize=9)

    return {sp: (d, lo, hi) for sp, d, lo, hi in d_values}


def load_with_aura(base_path: Path, aura_path: Path) -> list[dict]:
    items = json.load(open(base_path))["items"]
    if aura_path.exists():
        items = items + json.load(open(aura_path))["items"]
    return items


def main():
    g = load_with_aura(GEMMA_PATH, GEMMA_AURA_PATH)
    q = load_with_aura(QWEN_PATH, QWEN_AURA_PATH)

    g_truth = [it for it in g if it["domain"] == "truth"]
    q_truth = [it for it in q if it["domain"] == "truth"]

    truth_prompts = ["neutral", "aura", "lie_directive", "pathological_liar"]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8.4))
    baseline_handle = [None]

    enc_g = load_encoder_d(ENCODER_GEMMA) if ENCODER_GEMMA.exists() else None
    enc_q = load_encoder_d(ENCODER_QWEN) if ENCODER_QWEN.exists() else None

    g_truth_d = panel(axes[0], g_truth, truth_prompts, "tb-5_L32",
                      "true", "false", "probe_scores",
                      "Gemma-3-27B — Truth (true vs false)",
                      ylabel="End-of-turn probe score",
                      encoder_d=enc_g, domain="truth",
                      baseline_handle=baseline_handle)
    q_truth_d = panel(axes[1], q_truth, truth_prompts, "qwen_tb-4_L38",
                      "true", "false", "probe_scores",
                      "Qwen3.5-122B-A10B — Truth (true vs false)",
                      ylabel="End-of-turn probe score",
                      encoder_d=enc_q, domain="truth",
                      baseline_handle=baseline_handle)

    fig.suptitle(
        "Persona-relative readout on truth at the user end-of-turn: "
        "aura preserves separation, lying personas collapse or flip it",
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

    def fmt(d): return {k: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]" for k, v in d.items()}
    print(f"wrote {OUT_PATH}")
    print("  Gemma truth d:", fmt(g_truth_d))
    print("  Qwen  truth d:", fmt(q_truth_d))


if __name__ == "__main__":
    main()
