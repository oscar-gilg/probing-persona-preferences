"""Paper-ready Fig 6 analogues for Qwen-3.5-122B canonical probe replication.

Produces:
- plot_042526_qwen_eot_persona_modulation_minimal.png — selected sysprompts that
  flip/collapse the evaluative readout per domain (truth, harm, politics).
- plot_042526_qwen_eot_persona_modulation_full.png — all sysprompts per domain.

Headline probe: `qwen_tb-1_L38`.

Usage:
    python experiments/token_level_probes/qwen_canonical_probe_eval/scripts/plot_qwen_eot_persona_modulation.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path("experiments/token_level_probes/qwen_canonical_probe_eval")
ASSETS_DIR = EXP_DIR / "assets"
TRUTHHARM_PATH = EXP_DIR / "scoring_results.json"
POLITICS_PATH = EXP_DIR / "politics_scoring_results.json"

PROBE = "qwen_tb-1_L38"
DATE = "042526"

COLORS = {
    "true": "#2196F3", "false": "#D32F2F",
    "benign": "#2E7D32", "harmful": "#D32F2F",
    "left": "#2196F3", "right": "#D32F2F",
}

TRUTH_ORDER_FULL = [
    "truthful", "neutral", "unreliable_narrator", "contrarian",
    "opposite_day", "lie_directive", "pathological_liar", "con_artist", "gaslighter",
]
HARM_ORDER_FULL = ["safe", "neutral", "unrestricted", "sinister_ai", "sadist"]
POLITICS_ORDER_FULL = [
    "socialist", "democrat", "centrist", "apolitical", "neutral",
    "libertarian", "republican", "nationalist", "contrarian",
]


def load(path):
    return json.load(open(path))["items"]


def cohen_d_pooled(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    pooled = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1))
                     / (len(pos) + len(neg) - 2))
    return (pos.mean() - neg.mean()) / pooled if pooled > 0 else 0.0


def panel_violins(ax, items, prompts, c_pos, c_neg, title):
    by_sp = defaultdict(list)
    for it in items:
        by_sp[it["system_prompt"]].append(it)

    positions, all_series, all_colors = [], [], []
    d_values = []

    for pi, sp in enumerate(prompts):
        pos_vals = [it["probe_scores"][PROBE] for it in by_sp[sp] if it["condition"] == c_pos]
        neg_vals = [it["probe_scores"][PROBE] for it in by_sp[sp] if it["condition"] == c_neg]
        d_raw = cohen_d_pooled(np.array(pos_vals), np.array(neg_vals))
        d = round(float(d_raw), 2) if not np.isnan(d_raw) else float("nan")
        d_values.append((sp, d))
        positions.extend([pi * 3, pi * 3 + 1])
        all_series.extend([pos_vals, neg_vals])
        all_colors.extend([COLORS[c_pos], COLORS[c_neg]])

    parts = ax.violinplot(all_series, positions=positions, widths=0.9,
                          showmeans=True, showextrema=False)
    for body, color in zip(parts["bodies"], all_colors):
        body.set_facecolor(color); body.set_alpha(0.75); body.set_edgecolor("black")
    parts["cmeans"].set_color("black")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xticks([pi * 3 + 0.5 for pi in range(len(prompts))])
    ax.set_xticklabels(
        [f"{sp}\n(d = {d:+.2f})" if not np.isnan(d) else f"{sp}\n(d = n/a)" for sp, d in d_values],
        fontsize=8.5,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(f"Probe score ({PROBE})")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[c_pos], alpha=0.75),
               plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[c_neg], alpha=0.75)]
    ax.legend(handles, [c_pos, c_neg], loc="best", fontsize=9)
    return d_values


def make_minimal():
    truth_items = [it for it in load(TRUTHHARM_PATH) if it["domain"] == "truth"]
    harm_items = [it for it in load(TRUTHHARM_PATH) if it["domain"] == "harm"]
    politics_items = load(POLITICS_PATH)

    truth_prompts = ["neutral", "lie_directive", "pathological_liar"]
    harm_prompts = ["neutral", "sadist"]
    politics_prompts = ["democrat", "republican"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2),
                             gridspec_kw={"width_ratios": [3, 2, 2]})

    d_truth = panel_violins(axes[0], truth_items, truth_prompts, "true", "false",
                            "Truth: lying personas collapse or flip the signal")
    d_harm = panel_violins(axes[1], harm_items, harm_prompts, "harmful", "benign",
                           "Harm: sadist persona collapses the signal")
    d_pol = panel_violins(axes[2], politics_items, politics_prompts, "left", "right",
                          "Politics: partisan prompt flips the sign")

    fig.suptitle(f"Qwen-3.5-122B at probe {PROBE}: persona prompts modulate the probe's evaluative readout",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    path = ASSETS_DIR / f"plot_{DATE}_qwen_eot_persona_modulation_minimal.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")
    print("  truth:", d_truth)
    print("  harm:", d_harm)
    print("  politics:", d_pol)


def make_full():
    truth_items = [it for it in load(TRUTHHARM_PATH) if it["domain"] == "truth"]
    harm_items = [it for it in load(TRUTHHARM_PATH) if it["domain"] == "harm"]
    politics_items = load(POLITICS_PATH)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(22, 4.6),
        gridspec_kw={"width_ratios": [len(TRUTH_ORDER_FULL),
                                      len(HARM_ORDER_FULL),
                                      len(POLITICS_ORDER_FULL)]},
    )

    d_truth = panel_violins(axes[0], truth_items, TRUTH_ORDER_FULL, "true", "false",
                            "Truth (true vs false)")
    d_harm = panel_violins(axes[1], harm_items, HARM_ORDER_FULL, "harmful", "benign",
                           "Harm (harmful vs benign)")
    d_pol = panel_violins(axes[2], politics_items, POLITICS_ORDER_FULL, "left", "right",
                          "Politics (left vs right)")

    fig.suptitle(f"Qwen-3.5-122B at probe {PROBE}: persona modulation across all sysprompts",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    path = ASSETS_DIR / f"plot_{DATE}_qwen_eot_persona_modulation_full.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")
    print("  truth:", d_truth)
    print("  harm:", d_harm)
    print("  politics:", d_pol)


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    make_minimal()
    make_full()


if __name__ == "__main__":
    main()
