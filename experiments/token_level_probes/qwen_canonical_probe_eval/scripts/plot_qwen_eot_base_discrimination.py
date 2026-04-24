"""Paper-ready Fig 5 analogue for Qwen-3.5-122B canonical probe replication.

Two-panel violin figure (truth, harm) under the neutral system prompt only.
Politics is excluded because politics stimuli inherently use a partisan sysprompt.
Uses the headline probe `qwen_tb-1_L38` (best-balanced across domains).

Usage:
    python experiments/token_level_probes/qwen_canonical_probe_eval/scripts/plot_qwen_eot_base_discrimination.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path("experiments/token_level_probes/qwen_canonical_probe_eval")
ASSETS_DIR = EXP_DIR / "assets"
TRUTHHARM_PATH = EXP_DIR / "scoring_results.json"

PROBE = "qwen_tb-1_L38"
DATE = "042526"

COLORS = {
    "true": "#2196F3", "false": "#D32F2F",
    "benign": "#2E7D32", "harmful": "#D32F2F",
    "nonsense": "#9E9E9E",
}


def load(path):
    return json.load(open(path))["items"]


def cohen_d_pooled(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    pooled = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1))
                     / (len(pos) + len(neg) - 2))
    return (pos.mean() - neg.mean()) / pooled if pooled > 0 else 0.0


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    items = load(TRUTHHARM_PATH)
    neutral = [it for it in items if it["system_prompt"] == "neutral"]
    truth = [it for it in neutral if it["domain"] == "truth"]
    harm = [it for it in neutral if it["domain"] == "harm"]

    panels = [
        ("truth", "Truth (CREAK)", truth, "true", "false"),
        ("harm", "Harm (BailBench + stress test)", harm, "harmful", "benign"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=False)
    d_summary = {}
    for ax, (domain_key, title, dom_items, c_pos, c_neg) in zip(axes, panels):
        def gather(cond):
            return np.array([it["probe_scores"][PROBE] for it in dom_items if it["condition"] == cond])
        pos_vals = gather(c_pos)
        neg_vals = gather(c_neg)
        non_vals = gather("nonsense")
        d = round(float(cohen_d_pooled(pos_vals, neg_vals)), 2)
        d_summary[domain_key] = d

        has_nonsense = len(non_vals) > 1
        if has_nonsense:
            series = [pos_vals, neg_vals, non_vals]
            positions = [0, 1, 2]
            colors_used = [COLORS[c_pos], COLORS[c_neg], COLORS["nonsense"]]
            tick_labels = [c_pos, c_neg, "nonsense"]
        else:
            series = [pos_vals, neg_vals]
            positions = [0, 1]
            colors_used = [COLORS[c_pos], COLORS[c_neg]]
            tick_labels = [c_pos, c_neg]

        parts = ax.violinplot(series, positions=positions,
                              widths=0.7, showmeans=True, showextrema=False)
        for body, color in zip(parts["bodies"], colors_used):
            body.set_facecolor(color); body.set_alpha(0.7); body.set_edgecolor("black")
        parts["cmeans"].set_color("black")

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{title}\n(d = {d:+.2f}, n = {len(pos_vals)}/{len(neg_vals)})", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel(f"End-of-turn probe score ({PROBE})")

    fig.suptitle(f"Qwen-3.5-122B at probe {PROBE}: base discrimination at the end-of-turn token",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    path = ASSETS_DIR / f"plot_{DATE}_qwen_eot_base_discrimination.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")
    print(f"d values: {d_summary}")


if __name__ == "__main__":
    main()
