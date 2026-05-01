"""Two-model persona-modulation figure with aura positive-persona control (assistant turn).

Extends plot_042726_canonical_eot_induced_shifts_2models.py by adding the aura
sysprompt to truth and harm panels. Politics is unchanged (aura is not a
political stance, so the partisan-vs-neutral structure stays as-is).

Aura is the "positive persona" control: under aura, the readout should preserve
(or strengthen) harm/benign and true/false separation, distinguishing "tracks
active persona" from "noise under non-default persona".

Usage:
    python paper/figures/main/scripts/plot_042926_aura_control_2models.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3].parent
GEMMA_TH = REPO / "experiments/eot_discrimination_v2/scoring/gemma3_27b/scoring_results.json"
GEMMA_TH_AURA = REPO / "experiments/eot_discrimination_v2/scoring/gemma3_27b/_AURA_IN_BASE_FILE.json"  # sentinel: aura is in base file
GEMMA_POL = REPO / "experiments/eot_discrimination_v2/scoring/gemma3_27b/scoring_results.json"
QWEN_TH = REPO / "experiments/eot_discrimination_v2/scoring/qwen35_122b/scoring_results.json"
QWEN_TH_AURA = REPO / "experiments/eot_discrimination_v2/scoring/qwen35_122b/_AURA_IN_BASE_FILE.json"
QWEN_POL = REPO / "experiments/eot_discrimination_v2/scoring/qwen35_122b/scoring_results.json"
OUT_PATH = REPO / "paper/figures/main/plot_042926_aura_control_2models.png"

COLORS = {
    "true": "#2196F3",
    "false": "#D32F2F",
    "benign": "#2E7D32",
    "harmful": "#D32F2F",
    "left": "#2196F3",
    "right": "#D32F2F",
}


def cohen_d_with_ci(pos, neg, z=1.96):
    """Pooled Cohen's d with Hedges/Olkin analytical 95% CI."""
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


DISPLAY_LABELS = {"neutral": "Assistant"}


def display(sp: str) -> str:
    return DISPLAY_LABELS.get(sp, sp)


def panel(ax, items, prompts, probe, c_pos, c_neg, score_key, domain_label,
          ylabel=None, show_legend=True, highlight_sp=None):
    by_sp = defaultdict(list)
    for it in items:
        by_sp[it["system_prompt"]].append(it)

    positions = []
    all_series = []
    all_colors = []
    d_values = []

    valid_prompts = []
    for sp in prompts:
        pos_vals = [it[score_key][probe] for it in by_sp.get(sp, [])
                    if it["condition"] == c_pos]
        neg_vals = [it[score_key][probe] for it in by_sp.get(sp, [])
                    if it["condition"] == c_neg]
        if not pos_vals or not neg_vals:
            continue
        valid_prompts.append(sp)
        pi = len(valid_prompts) - 1
        d, lo, hi = cohen_d_with_ci(pos_vals, neg_vals)
        d_values.append((sp, d, lo, hi))
        positions.extend([pi * 3, pi * 3 + 1])
        all_series.extend([pos_vals, neg_vals])
        all_colors.extend([COLORS[c_pos], COLORS[c_neg]])
    prompts = valid_prompts

    parts = ax.violinplot(all_series, positions=positions, widths=0.9,
                          showmeans=True, showextrema=False)
    for body, color in zip(parts["bodies"], all_colors):
        body.set_facecolor(color)
        body.set_alpha(0.75)
        body.set_edgecolor("black")
    parts["cmeans"].set_color("black")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xticks([pi * 3 + 0.5 for pi in range(len(prompts))])

    labels = [f"{display(sp)}\nd = {d:+.2f}\n[{lo:+.2f}, {hi:+.2f}]" for sp, d, lo, hi in d_values]
    ax.set_xticklabels(labels, fontsize=8)

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
    # else: aura is already in the base file (v2 unified output), no concat needed.
    return items


def main():
    g_th = load_with_aura(GEMMA_TH, GEMMA_TH_AURA)
    g_pol = json.load(open(GEMMA_POL))["items"]
    q_th = load_with_aura(QWEN_TH, QWEN_TH_AURA)
    q_pol = json.load(open(QWEN_POL))["items"]

    g_truth = [it for it in g_th if it["domain"] == "truth"]
    g_harm = [it for it in g_th if it["domain"] == "harm"]
    q_truth = [it for it in q_th if it["domain"] == "truth"]
    q_harm = [it for it in q_th if it["domain"] == "harm"]

    truth_prompts = ["neutral", "aura", "lie_directive", "pathological_liar"]
    harm_prompts = ["neutral", "aura", "sadist"]
    politics_prompts = ["democrat", "republican"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4),
                             gridspec_kw={"width_ratios": [4, 3, 2]})

    g_truth_d = panel(axes[0, 0], g_truth, truth_prompts, "tb-5_L32",
                      "true", "false", "probe_scores",
                      "Gemma-3-27B — Truth (true vs false)",
                      ylabel="End-of-turn probe score",
                      highlight_sp="aura")
    g_harm_d = panel(axes[0, 1], g_harm, harm_prompts, "tb-5_L39",
                     "harmful", "benign", "probe_scores",
                     "Gemma-3-27B — Harm (harmful vs benign)",
                     highlight_sp="aura")
    g_pol_d = panel(axes[0, 2], g_pol, politics_prompts, "tb-5_L39",
                    "left", "right", "probe_scores",
                    "Gemma-3-27B — Politics (left vs right)")

    q_truth_d = panel(axes[1, 0], q_truth, truth_prompts, "qwen_tb-4_L38",
                      "true", "false", "probe_scores",
                      "Qwen3.5-122B-A10B — Truth (true vs false)",
                      ylabel="End-of-turn probe score",
                      highlight_sp="aura")
    q_harm_d = panel(axes[1, 1], q_harm, harm_prompts, "qwen_tb-4_L38",
                     "harmful", "benign", "probe_scores",
                     "Qwen3.5-122B-A10B — Harm (harmful vs benign)",
                     highlight_sp="aura")
    q_pol_d = panel(axes[1, 2], q_pol, politics_prompts, "qwen_tb-4_L38",
                    "left", "right", "probe_scores",
                    "Qwen3.5-122B-A10B — Politics (left vs right)")

    fig.suptitle(
        "Persona-relative readout: aura preserves separation; "
        "lying/sadist personas collapse or flip it (assistant-turn)",
        fontsize=11, y=1.00,
    )
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    def fmt(d): return {k: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]" for k, v in d.items()}
    print(f"wrote {OUT_PATH}")
    print("  Gemma truth d:", fmt(g_truth_d))
    print("  Gemma harm  d:", fmt(g_harm_d))
    print("  Gemma pol   d:", fmt(g_pol_d))
    print("  Qwen  truth d:", fmt(q_truth_d))
    print("  Qwen  harm  d:", fmt(q_harm_d))
    print("  Qwen  pol   d:", fmt(q_pol_d))


if __name__ == "__main__":
    main()
