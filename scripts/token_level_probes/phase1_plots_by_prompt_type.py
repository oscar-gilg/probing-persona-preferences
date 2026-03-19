"""Generate Phase 1 violin plots split by prompt type with descriptive titles."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ASSETS = Path("experiments/token_level_probes/assets")
DATA = json.load(open("experiments/token_level_probes/scoring_results.json"))
ITEMS = DATA["items"]

BEST_PROBE = {"truth": "task_mean_L32", "harm": "task_mean_L39", "politics": "task_mean_L39"}

COLORS = {
    "true": "#4CAF50", "false": "#F44336", "nonsense": "#9E9E9E",
    "benign": "#4CAF50", "harmful": "#F44336",
    "left": "#2196F3", "right": "#F44336",
}

COND_ORDER = {
    "truth": ["true", "false", "nonsense"],
    "harm": ["benign", "harmful", "nonsense"],
    "politics": ["left", "right", "nonsense"],
}


def get_system_prompt(item_id):
    if "democrat" in item_id:
        return "democrat"
    elif "republican" in item_id:
        return "republican"
    return "neutral"


def cohens_d(a, b):
    return (np.mean(a) - np.mean(b)) / np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)


def make_violin(ax, groups, cond_order, title, subtitle=None):
    data = []
    positions = []
    colors = []
    labels = []
    for i, cond in enumerate(cond_order):
        if cond in groups:
            data.append(groups[cond])
            positions.append(i)
            colors.append(COLORS[cond])
            labels.append(cond)

    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    for key in ["cmeans", "cmins", "cmaxes", "cbars"]:
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(0.8)

    # scatter points
    for i, (pos, vals) in enumerate(zip(positions, data)):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
        ax.scatter(pos + jitter, vals, c=colors[i], alpha=0.3, s=8, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Probe score")
    ax.set_title(title, fontsize=11, fontweight="bold")
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)

    # stats annotation
    hi, lo = cond_order[0], cond_order[1]
    if hi in groups and lo in groups:
        d = cohens_d(np.array(groups[hi]), np.array(groups[lo]))
        _, p = stats.mannwhitneyu(groups[hi], groups[lo], alternative="two-sided")
        ax.text(0.02, 0.98, f"d={d:.2f}, p={p:.1e}", transform=ax.transAxes,
                va="top", ha="left", fontsize=9, color="dimgray")


def group_scores(items, domain, probe, filters=None):
    groups = defaultdict(list)
    for item in items:
        if item["domain"] != domain:
            continue
        if filters and not all(filters[k](item) for k in filters):
            continue
        groups[item["condition"]].append(item["critical_span_mean_scores"][probe])
    return groups


# === TRUTH: user vs assistant ===
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
probe = BEST_PROBE["truth"]

user_groups = group_scores(ITEMS, "truth", probe, {"turn": lambda i: i["turn"] == "user"})
make_violin(axes[0], user_groups, COND_ORDER["truth"],
            'User turn: "Repeat the following statement: [claim]"',
            f"probe: {probe}, n=88/condition")

asst_groups = group_scores(ITEMS, "truth", probe, {"turn": lambda i: i["turn"] == "assistant"})
make_violin(axes[1], asst_groups, COND_ORDER["truth"],
            'Assistant turn: Q\u2192A prefill with [answer]',
            f"probe: {probe}, n=88/condition")

fig.suptitle("Truth domain: critical span scores by prompt type", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(ASSETS / "plot_031426_truth_by_prompt_type.png", dpi=150, bbox_inches="tight")
print(f"Saved truth plot")
plt.close()


# === HARM: user vs assistant ===
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
probe = BEST_PROBE["harm"]

user_groups = group_scores(ITEMS, "harm", probe, {"turn": lambda i: i["turn"] == "user"})
make_violin(axes[0], user_groups, COND_ORDER["harm"],
            'User turn: "I want to [harmful/benign verb] someone"',
            f"probe: {probe}, n=77/condition")

asst_groups = group_scores(ITEMS, "harm", probe, {"turn": lambda i: i["turn"] == "assistant"})
make_violin(axes[1], asst_groups, COND_ORDER["harm"],
            'Assistant turn: "Sure! Start by [harmful/benign action]"',
            f"probe: {probe}, n=77/condition")

fig.suptitle("Harm domain: critical span scores by prompt type", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(ASSETS / "plot_031426_harm_by_prompt_type.png", dpi=150, bbox_inches="tight")
print(f"Saved harm plot")
plt.close()


# === POLITICS: by system prompt ===
fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
probe = BEST_PROBE["politics"]

for ax, sp, sp_label in zip(axes, ["democrat", "republican", "neutral"], [
    'System: "You are a progressive Democrat..."',
    'System: "You are a conservative Republican..."',
    'No system prompt',
]):
    groups = defaultdict(list)
    for item in ITEMS:
        if item["domain"] != "politics":
            continue
        item_sp = get_system_prompt(item["id"])
        if item_sp != sp:
            continue
        groups[item["condition"]].append(item["critical_span_mean_scores"][probe])

    order = ["left", "right", "nonsense"] if "nonsense" in groups else ["left", "right"]
    n = len(groups.get("left", []))
    make_violin(ax, groups, order, sp_label, f"probe: {probe}, n={n}/condition")

fig.suptitle("Politics domain: critical span scores by system prompt", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(ASSETS / "plot_031426_politics_by_system_prompt.png", dpi=150, bbox_inches="tight")
print(f"Saved politics plot")
plt.close()

print("Done")
