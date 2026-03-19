"""Fullstop score violin plots split by prompt type."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ASSETS = Path("experiments/token_level_probes/assets")
DATA = json.load(open("experiments/token_level_probes/scoring_results.json"))
ITEMS = DATA["items"]

COLORS = {
    "true": "#4CAF50", "false": "#F44336", "nonsense": "#9E9E9E",
    "benign": "#4CAF50", "harmful": "#F44336",
}

DOMAIN_CFG = {
    "truth": {
        "probe": "task_mean_L32",
        "conds": ["true", "false", "nonsense"],
        "d_conds": ("true", "false"),
        "turns": {
            "user": '"Repeat the following statement: [claim]"',
            "assistant": 'Q→A prefill with [answer]',
        },
    },
    "harm": {
        "probe": "task_mean_L39",
        "conds": ["benign", "harmful", "nonsense"],
        "d_conds": ("benign", "harmful"),
        "turns": {
            "user": '"I want to [harmful/benign verb] someone"',
            "assistant": '"Sure! Start by [harmful/benign action]"',
        },
    },
}


def cohens_d(a, b):
    return (np.mean(a) - np.mean(b)) / np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)


def make_violin(ax, groups, conds, title, subtitle):
    data = []
    positions = []
    colors = []
    for i, c in enumerate(conds):
        if c in groups and groups[c]:
            data.append(groups[c])
            positions.append(i)
            colors.append(COLORS[c])

    if not data:
        return

    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    for key in ["cmeans", "cmins", "cmaxes", "cbars"]:
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(0.8)

    for i, (pos, vals) in enumerate(zip(positions, data)):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
        ax.scatter(pos + jitter, vals, c=colors[i], alpha=0.3, s=8, zorder=3)

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds)
    ax.set_ylabel("Mean fullstop score")
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)

    hi, lo = conds[0], conds[1]
    if hi in groups and lo in groups:
        d = cohens_d(np.array(groups[hi]), np.array(groups[lo]))
        _, p = stats.mannwhitneyu(groups[hi], groups[lo], alternative="two-sided")
        ax.text(0.02, 0.98, f"d={d:.2f}, p={p:.1e}", transform=ax.transAxes,
                va="top", ha="left", fontsize=9, color="dimgray")


for domain, cfg in DOMAIN_CFG.items():
    probe = cfg["probe"]
    turns = cfg["turns"]
    fig, axes = plt.subplots(1, len(turns), figsize=(10, 5), sharey=True)

    for ax, (turn, turn_label) in zip(axes, turns.items()):
        groups = defaultdict(list)
        for item in ITEMS:
            if item["domain"] != domain or item["turn"] != turn:
                continue
            fs = item["fullstop_scores"][probe]
            if fs:
                groups[item["condition"]].append(np.mean(fs))

        n = len(groups.get(cfg["conds"][0], []))
        make_violin(ax, groups, cfg["conds"],
                    f"{turn.capitalize()} turn: {turn_label}",
                    f"probe: {probe}, n={n}/condition")

    fig.suptitle(f"{domain.capitalize()} domain: mean fullstop scores by prompt type",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = ASSETS / f"plot_031426_{domain}_fullstop_by_prompt_type.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()
