"""Plot induced-shift EOT scores by system prompt for tb-5 and tb-2 probes.

Usage:
    python -m experiments.token_level_probes.canonical_probe_eval.scripts.plot_induced_shifts
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path("experiments/token_level_probes")
V2_DIR = EXP_DIR / "system_prompt_modulation_v2"
OUT_DIR = EXP_DIR / "canonical_probe_eval"
ASSETS_DIR = OUT_DIR / "assets"

V2_TRUTHHARM_PATH = V2_DIR / "scoring_results.json"
V2_POLITICS_PATH = V2_DIR / "politics_scoring_results.json"

DOMAIN_CFG = {
    "truth": {
        "path": V2_TRUTHHARM_PATH,
        "conditions": ("true", "false"),
        "probes": ("tb-5_L32", "tb-2_L32"),
        "task_mean_probe": "task_mean_L32",
        "prompt_order": [
            "truthful", "neutral", "unreliable_narrator", "contrarian",
            "opposite_day", "lie_directive", "pathological_liar", "con_artist", "gaslighter",
        ],
    },
    "harm": {
        "path": V2_TRUTHHARM_PATH,
        "conditions": ("harmful", "benign"),
        "probes": ("tb-5_L39", "tb-2_L39"),
        "task_mean_probe": "task_mean_L39",
        "prompt_order": ["safe", "neutral", "unrestricted", "sinister_ai", "sadist"],
    },
    "politics": {
        "path": V2_POLITICS_PATH,
        "conditions": ("right", "left"),
        "probes": ("tb-5_L39", "tb-2_L39"),
        "task_mean_probe": "task_mean_L39",
        "prompt_order": [
            "socialist", "democrat", "centrist", "apolitical", "neutral",
            "libertarian", "republican", "nationalist", "contrarian",
        ],
    },
}

COLORS = {
    "true": "#2196F3", "false": "#F44336",
    "benign": "#4CAF50", "harmful": "#F44336",
    "left": "#2196F3", "right": "#F44336",
}


def load(path):
    return json.load(open(path))["items"]


def plot_domain(domain, date_tag="042126"):
    cfg = DOMAIN_CFG[domain]
    items = [it for it in load(cfg["path"]) if domain in (it["domain"], it.get("domain", ""))]
    if domain != "politics":
        items = [it for it in items if it["domain"] == domain]
    prompts = cfg["prompt_order"]
    c_pos, c_neg = cfg["conditions"]
    probes = list(cfg["probes"]) + [cfg["task_mean_probe"]]

    by_sp = defaultdict(list)
    for it in items:
        by_sp[it["system_prompt"]].append(it)

    fig, axes = plt.subplots(1, 3, figsize=(max(12, len(prompts) * 1.1) * 1.6, 5), sharey=True)
    for ax, probe in zip(axes, probes):
        means_pos, means_neg, sems_pos, sems_neg = [], [], [], []
        for sp in prompts:
            sp_items = by_sp.get(sp, [])
            pos_vals = [it["eot_scores"][probe] for it in sp_items if it["condition"] == c_pos]
            neg_vals = [it["eot_scores"][probe] for it in sp_items if it["condition"] == c_neg]
            means_pos.append(np.mean(pos_vals) if pos_vals else np.nan)
            means_neg.append(np.mean(neg_vals) if neg_vals else np.nan)
            sems_pos.append(np.std(pos_vals, ddof=1) / np.sqrt(max(len(pos_vals), 1)) if len(pos_vals) > 1 else 0)
            sems_neg.append(np.std(neg_vals, ddof=1) / np.sqrt(max(len(neg_vals), 1)) if len(neg_vals) > 1 else 0)

        x = np.arange(len(prompts))
        w = 0.4
        ax.bar(x - w/2, means_pos, w, yerr=sems_pos, color=COLORS[c_pos], alpha=0.75,
               label=c_pos, capsize=3, edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, means_neg, w, yerr=sems_neg, color=COLORS[c_neg], alpha=0.75,
               label=c_neg, capsize=3, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(prompts, rotation=40, ha="right", fontsize=8)
        ax.set_title(probe + (" (parent overlay)" if probe == cfg["task_mean_probe"] else ""))
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Mean EOT probe score")
            ax.legend(loc="best")
    fig.suptitle(f"{domain.capitalize()}: EOT score by system prompt × probe family", y=1.02)
    plt.tight_layout()
    path = ASSETS_DIR / f"plot_{date_tag}_tb_eot_by_system_prompt_{domain}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")


def plot_headline():
    """Headline: |d| per domain, grouped by probe family, with parent overlay."""
    import csv
    path = OUT_DIR / "headline_table.csv"
    rows = list(csv.DictReader(open(path)))

    def family_of(probe_str):
        if "task_mean" in probe_str:
            return "task_mean (parent)"
        if probe_str.startswith("tb-5"):
            return "tb-5 (EOT probe)"
        if probe_str.startswith("tb-2"):
            return "tb-2 (model marker)"
        return "other"

    def layer_of(probe_str):
        for L in ("L32", "L39", "L53"):
            if L in probe_str:
                return L
        return "?"

    domain_order = ["truth", "harm", "politics_democrat", "politics_republican"]
    domain_labels = {
        "truth": "Truth\n(true vs false)",
        "harm": "Harm\n(harmful vs benign)",
        "politics_democrat": "Politics\n(democrat prompt)",
        "politics_republican": "Politics\n(republican prompt)",
    }
    family_color = {
        "tb-5 (EOT probe)": "#1976D2",
        "tb-2 (model marker)": "#26A69A",
        "task_mean (parent)": "#E65100",
    }

    data = defaultdict(dict)
    for r in rows:
        fam = family_of(r["probe"])
        L = layer_of(r["probe"])
        d = abs(float(r["d"])) if r["d"] else None
        data[r["domain"]][(fam, L)] = d

    fig, ax = plt.subplots(figsize=(11, 5))
    families = ["tb-5 (EOT probe)", "tb-2 (model marker)", "task_mean (parent)"]
    layers_per_family = {
        "tb-5 (EOT probe)": ["L32", "L39", "L53"],
        "tb-2 (model marker)": ["L32", "L39", "L53"],
        "task_mean (parent)": ["L32", "L39"],
    }

    all_bars = []
    for fam in families:
        for L in layers_per_family[fam]:
            heights = []
            for d in domain_order:
                heights.append(data.get(d, {}).get((fam, L)))
            all_bars.append((fam, L, heights))

    n_bars = len(all_bars)
    total_w = 0.85
    w = total_w / n_bars
    x = np.arange(len(domain_order))

    for bi, (fam, L, heights) in enumerate(all_bars):
        offset = (bi - (n_bars - 1) / 2) * w
        vals = [h if h is not None else 0 for h in heights]
        ax.bar(x + offset, vals, w, color=family_color[fam],
               alpha=0.9 if L == "L32" else (0.7 if L == "L39" else 0.5),
               label=f"{fam} {L}", edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([domain_labels[d] for d in domain_order])
    ax.set_ylabel("|Cohen's d| at EOT token")
    ax.set_title("Do canonical EOT-trained probes match the parent's task_mean numbers?")
    ax.axhline(2, color="gray", linestyle=":", linewidth=0.7, label="large effect (d=2)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    path = ASSETS_DIR / "plot_042126_tb_eot_headline_d.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")


def plot_per_turn():
    """Per-turn d for truth and harm: tb-5 and tb-2 with parent reference lines."""
    import csv
    rows = list(csv.DictReader(open(OUT_DIR / "per_turn_table.csv")))

    parent = {
        ("truth", "user"): 3.19, ("truth", "assistant"): 3.00,
        ("harm", "user"): 1.97, ("harm", "assistant"): 1.91,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, domain in zip(axes, ["truth", "harm"]):
        probes = ["tb-5_L32", "tb-2_L32"] if domain == "truth" else ["tb-5_L39", "tb-2_L39"]
        turns = ["user", "assistant"]
        x = np.arange(len(turns))
        w = 0.35
        for pi, probe in enumerate(probes):
            vals = []
            for t in turns:
                matches = [r for r in rows
                           if r["domain"] == f"{domain} ({t})" and r["probe"] == probe]
                vals.append(abs(float(matches[0]["d"])) if matches else 0)
            color = "#1976D2" if probe.startswith("tb-5") else "#26A69A"
            ax.bar(x + (pi - 0.5) * w, vals, w, color=color, edgecolor="black",
                   linewidth=0.4, label=probe)
        parent_user = parent[(domain, "user")]
        parent_assist = parent[(domain, "assistant")]
        ax.hlines([parent_user, parent_assist], [-0.5, 0.5], [0.5, 1.5],
                  colors="#E65100", linestyles="--", linewidth=2,
                  label="task_mean (parent)")
        ax.set_xticks(x)
        ax.set_xticklabels(["user turn", "assistant turn"])
        ax.set_ylabel("|Cohen's d| at EOT token")
        ax.set_title(f"{domain.capitalize()}: does per-turn d recover the parent's signal?")
        ax.axhline(2, color="gray", linestyle=":", linewidth=0.7)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    path = ASSETS_DIR / "plot_042126_tb_eot_per_turn_d.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for domain in ("truth", "harm", "politics"):
        plot_domain(domain)
    plot_headline()
    plot_per_turn()


if __name__ == "__main__":
    main()
