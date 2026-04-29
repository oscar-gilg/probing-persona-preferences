"""Two-model persona-modulation figure with aura positive-persona control (USER turn).

Extends plot_042726_canonical_eot_induced_shifts_user_turn_2models.py by adding
the aura sysprompt to truth and harm panels. Aura is the "positive persona"
control: under aura, the readout should preserve (or strengthen) harm/benign
and true/false separation, distinguishing "tracks active persona" from "noise
under non-default persona".

Usage:
    python paper/figures/main/scripts/plot_042926_aura_control_user_turn_2models.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3].parent
GEMMA_PATH = REPO / "experiments/token_level_probes/system_prompt_modulation_v2/scoring_results_user_turn.json"
GEMMA_AURA_PATH = REPO / "experiments/token_level_probes/system_prompt_modulation_v2/scoring_results_user_turn_aura.json"
QWEN_PATH = REPO / "experiments/token_level_probes/qwen_canonical_probe_eval/user_turn_scoring_results.json"
QWEN_AURA_PATH = REPO / "experiments/token_level_probes/qwen_canonical_probe_eval/user_turn_scoring_results_aura.json"
OUT_PATH = REPO / "paper/figures/main/plot_042926_aura_control_user_turn_2models.png"

COLORS = {
    "true": "#2196F3",
    "false": "#D32F2F",
    "benign": "#2E7D32",
    "harmful": "#D32F2F",
}


def cohen_d_pooled(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    pooled = np.sqrt(
        ((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1))
        / (len(pos) + len(neg) - 2)
    )
    return (pos.mean() - neg.mean()) / pooled if pooled > 0 else 0.0


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
        d = cohen_d_pooled(pos_vals, neg_vals)
        d_values.append((sp, d))
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

    labels = []
    for sp, d in d_values:
        if sp == highlight_sp:
            labels.append(f"$\\bf{{{sp}}}$\n(d = {d:+.2f})")
        else:
            labels.append(f"{sp}\n(d = {d:+.2f})")
    ax.set_xticklabels(labels, fontsize=9)

    ax.grid(axis="y", alpha=0.3)
    ax.set_title(domain_label, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel)

    if show_legend:
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[c_pos], alpha=0.75),
                   plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[c_neg], alpha=0.75)]
        ax.legend(handles, [c_pos, c_neg], loc="best", fontsize=9)

    return dict(d_values)


def load_with_aura(base_path: Path, aura_path: Path) -> list[dict]:
    items = json.load(open(base_path))["items"]
    if aura_path.exists():
        items = items + json.load(open(aura_path))["items"]
    else:
        print(f"WARNING: aura data missing at {aura_path}; rendering without aura column.")
    return items


def main():
    g = load_with_aura(GEMMA_PATH, GEMMA_AURA_PATH)
    q = load_with_aura(QWEN_PATH, QWEN_AURA_PATH)

    g_truth = [it for it in g if it["domain"] == "truth"]
    g_harm = [it for it in g if it["domain"] == "harm"]
    q_truth = [it for it in q if it["domain"] == "truth"]
    q_harm = [it for it in q if it["domain"] == "harm"]

    truth_prompts = ["neutral", "aura", "lie_directive", "pathological_liar"]
    harm_prompts = ["neutral", "aura", "sadist"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4),
                             gridspec_kw={"width_ratios": [4, 3]})

    g_truth_d = panel(axes[0, 0], g_truth, truth_prompts, "tb-5_L32",
                      "true", "false", "eot_scores",
                      "Gemma-3-27B — Truth (true vs false)",
                      ylabel="End-of-turn probe score",
                      highlight_sp="aura")
    g_harm_d = panel(axes[0, 1], g_harm, harm_prompts, "tb-5_L39",
                     "harmful", "benign", "eot_scores",
                     "Gemma-3-27B — Harm (harmful vs benign)",
                     highlight_sp="aura")

    q_truth_d = panel(axes[1, 0], q_truth, truth_prompts, "qwen_tb-4_L38",
                      "true", "false", "probe_scores",
                      "Qwen3.5-122B-A10B — Truth (true vs false)",
                      ylabel="End-of-turn probe score",
                      highlight_sp="aura")
    q_harm_d = panel(axes[1, 1], q_harm, harm_prompts, "qwen_tb-4_L38",
                     "harmful", "benign", "probe_scores",
                     "Qwen3.5-122B-A10B — Harm (harmful vs benign)",
                     highlight_sp="aura")

    fig.suptitle(
        "Persona-relative readout: aura (positive persona) preserves separation; "
        "lying/sadist personas collapse or flip it (user-turn)",
        fontsize=11, y=1.00,
    )
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"wrote {OUT_PATH}")
    print("  Gemma truth d:", {k: f"{v:+.2f}" for k, v in g_truth_d.items()})
    print("  Gemma harm  d:", {k: f"{v:+.2f}" for k, v in g_harm_d.items()})
    print("  Qwen  truth d:", {k: f"{v:+.2f}" for k, v in q_truth_d.items()})
    print("  Qwen  harm  d:", {k: f"{v:+.2f}" for k, v in q_harm_d.items()})


if __name__ == "__main__":
    main()
