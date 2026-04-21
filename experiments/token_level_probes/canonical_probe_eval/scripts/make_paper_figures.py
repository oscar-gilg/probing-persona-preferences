"""Generate paper-ready figures for §5.1.2 (main text) and cross-token appendix.

Single canonical probe (trained at end-of-turn token, layer matched to parent:
L32 for truth, L39 for harm and politics). Uses descriptive labels rather than
code-level selector names (no tb-5 / tb-2 / turn_boundary:-N).

Usage:
    python -m experiments.token_level_probes.canonical_probe_eval.scripts.make_paper_figures
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
PAPER_FIG_DIR = Path("paper/figures")

PARENT_PATH = V2_DIR / "parent_eot_scores.json"
V2_TRUTHHARM_PATH = V2_DIR / "scoring_results.json"
V2_POLITICS_PATH = V2_DIR / "politics_scoring_results.json"

# Canonical probe: end-of-turn-trained, layer matched to parent per domain
CANONICAL = {
    "truth": "tb-5_L32",  # internal name; paper calls it "the canonical preference probe"
    "harm": "tb-5_L39",
    "politics": "tb-5_L39",
}
# Display names for probe families (appendix only uses these)
PROBE_DISPLAY = {
    "tb-5_L32": "end-of-turn probe",
    "tb-5_L39": "end-of-turn probe",
    "tb-5_L53": "end-of-turn probe (L53)",
    "tb-2_L32": "model-marker probe",
    "tb-2_L39": "model-marker probe",
    "tb-2_L53": "model-marker probe (L53)",
    "task_mean_L32": "task-averaged probe",
    "task_mean_L39": "task-averaged probe",
    "task_mean_L53": "task-averaged probe (L53)",
}

COLORS = {
    "true": "#2196F3", "false": "#D32F2F",
    "benign": "#2E7D32", "harmful": "#D32F2F",
    "left": "#2196F3", "right": "#D32F2F",
    "nonsense": "#9E9E9E",
}

DATE = "042126"


def load(path):
    return json.load(open(path))["items"]


def cohen_d_pooled(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    pooled = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1))
                     / (len(pos) + len(neg) - 2))
    return (pos.mean() - neg.mean()) / pooled if pooled > 0 else 0.0


def fig_main_base_discrimination():
    """Figure for §5.1.2: probe score distributions by condition, three domains."""
    parent = load(PARENT_PATH)
    truth = [it for it in parent if it["domain"] == "truth"]
    harm = [it for it in parent if it["domain"] == "harm"]
    pol = [it for it in parent if it["domain"] == "politics"]

    pol_dem = [it for it in pol if it.get("system_prompt") == "democrat"]
    pol_rep = [it for it in pol if it.get("system_prompt") == "republican"]

    panels = [
        ("Truth (CREAK, user turn)", truth, "tb-5_L32", "true", "false",
         lambda it: it.get("turn") == "user"),
        ("Harm (BailBench+stress test)", harm, "tb-5_L39", "harmful", "benign", None),
        ("Politics, democrat prompt", pol_dem, "tb-5_L39", "left", "right", None),
        ("Politics, republican prompt", pol_rep, "tb-5_L39", "left", "right", None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=False)
    for ax, (title, items, probe, c_pos, c_neg, tfilter) in zip(axes, panels):
        def gather(cond):
            vals = [it["eot_scores"][probe] for it in items
                    if it["condition"] == cond and (tfilter is None or tfilter(it))]
            return np.array(vals)
        pos_vals = gather(c_pos)
        neg_vals = gather(c_neg)
        non_vals = gather("nonsense")
        d = cohen_d_pooled(pos_vals, neg_vals)

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
            ax.set_ylabel("End-of-turn probe score")

    fig.suptitle("Does the preference probe discriminate truth / harm / politics at the end-of-turn token?",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    path = ASSETS_DIR / f"plot_{DATE}_canonical_eot_base_discrimination.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")
    paper_path = PAPER_FIG_DIR / path.name
    import shutil
    shutil.copy(path, paper_path)
    print(f"copied to {paper_path}")


def fig_main_persona_modulation():
    """Figure for §5.1.2: d by system prompt, three domains, single probe."""
    def load_items(path, domain):
        return [it for it in load(path) if it["domain"] == domain]

    domains = [
        {
            "name": "truth",
            "title": "Truth (true vs false)",
            "items": load_items(V2_TRUTHHARM_PATH, "truth"),
            "probe": "tb-5_L32", "pos": "true", "neg": "false",
            "order": ["truthful", "neutral", "unreliable_narrator", "contrarian",
                      "opposite_day", "lie_directive", "pathological_liar", "con_artist", "gaslighter"],
        },
        {
            "name": "harm",
            "title": "Harm (harmful vs benign)",
            "items": load_items(V2_TRUTHHARM_PATH, "harm"),
            "probe": "tb-5_L39", "pos": "harmful", "neg": "benign",
            "order": ["safe", "neutral", "unrestricted", "sinister_ai", "sadist"],
        },
        {
            "name": "politics",
            "title": "Politics (left vs right)",
            "items": load_items(V2_POLITICS_PATH, "politics"),
            "probe": "tb-5_L39", "pos": "left", "neg": "right",
            "order": ["socialist", "democrat", "centrist", "apolitical", "neutral",
                      "libertarian", "republican", "nationalist", "contrarian"],
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, dom in zip(axes, domains):
        by_sp = defaultdict(list)
        for it in dom["items"]:
            by_sp[it["system_prompt"]].append(it)

        ds = []
        for sp in dom["order"]:
            sp_items = by_sp.get(sp, [])
            pos = [it["eot_scores"][dom["probe"]] for it in sp_items if it["condition"] == dom["pos"]]
            neg = [it["eot_scores"][dom["probe"]] for it in sp_items if it["condition"] == dom["neg"]]
            ds.append(cohen_d_pooled(pos, neg) if pos and neg else float("nan"))

        x = np.arange(len(dom["order"]))
        colors = ["#1976D2" if d >= 0 else "#D32F2F" for d in ds]
        ax.bar(x, ds, color=colors, edgecolor="black", linewidth=0.4)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.axhline(2, color="gray", linestyle=":", linewidth=0.6)
        ax.axhline(-2, color="gray", linestyle=":", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(dom["order"], rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Cohen's d" if ax is axes[0] else "")
        ax.set_title(dom["title"], fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Does the sign of the probe flip under persona prompts that invert the evaluative stance?",
                 fontsize=11, y=1.03)
    plt.tight_layout()
    path = ASSETS_DIR / f"plot_{DATE}_canonical_eot_persona_modulation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")
    import shutil
    paper_path = PAPER_FIG_DIR / path.name
    shutil.copy(path, paper_path)
    print(f"copied to {paper_path}")


def fig_appendix_cross_training_selector():
    """Appendix figure: different training-token probes agree on d at EOT."""
    import csv
    rows = list(csv.DictReader(open(OUT_DIR / "headline_table.csv")))

    domain_order = ["truth", "harm", "politics_democrat", "politics_republican"]
    domain_labels = {
        "truth": "Truth",
        "harm": "Harm",
        "politics_democrat": "Politics\n(democrat)",
        "politics_republican": "Politics\n(republican)",
    }
    probe_order = [
        ("tb-5_L32", "end-of-turn probe"),
        ("tb-2_L32", "model-marker probe"),
        ("task_mean_L32", "task-averaged probe"),
        ("tb-5_L39", "end-of-turn probe"),
        ("tb-2_L39", "model-marker probe"),
        ("task_mean_L39", "task-averaged probe"),
    ]
    probe_color = {
        "end-of-turn probe": "#1976D2",
        "model-marker probe": "#00897B",
        "task-averaged probe": "#E65100",
    }

    best_probe_for_domain = {
        "truth": ["tb-5_L32", "tb-2_L32", "task_mean_L32"],
        "harm": ["tb-5_L39", "tb-2_L39", "task_mean_L39"],
        "politics_democrat": ["tb-5_L39", "tb-2_L39", "task_mean_L39"],
        "politics_republican": ["tb-5_L39", "tb-2_L39", "task_mean_L39"],
    }

    data = {}
    for r in rows:
        data[(r["domain"], r["probe"])] = r

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(domain_order))
    n_fam = 3
    w = 0.25
    family_to_offset = {"end-of-turn probe": -w, "model-marker probe": 0, "task-averaged probe": w}

    for probe, label in [("tb-5", "end-of-turn probe"),
                         ("tb-2", "model-marker probe"),
                         ("task_mean", "task-averaged probe")]:
        heights = []
        for d in domain_order:
            probes_here = best_probe_for_domain[d]
            match = next((p for p in probes_here if p.startswith(probe + "_")), None)
            if match is None:
                heights.append(0)
                continue
            row = data.get((d, match))
            heights.append(abs(float(row["d"])) if row and row["d"] else 0)
        ax.bar(x + family_to_offset[label], heights, w,
               color=probe_color[label], label=label,
               edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([domain_labels[d] for d in domain_order])
    ax.set_ylabel("|Cohen's d| at end-of-turn token")
    ax.axhline(2, color="gray", linestyle=":", linewidth=0.7, label="large effect (d=2)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("The same evaluative discrimination emerges across training-token choices")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    path = ASSETS_DIR / f"plot_{DATE}_cross_training_selector.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")
    import shutil
    paper_path = PAPER_FIG_DIR / path.name
    shutil.copy(path, paper_path)
    print(f"copied to {paper_path}")


def fig_appendix_per_turn():
    """Appendix figure: per-turn d comparison, highlighting model-marker asymmetry on harm."""
    import csv
    rows = list(csv.DictReader(open(OUT_DIR / "per_turn_table.csv")))

    parent = {
        ("truth", "user"): 3.19, ("truth", "assistant"): 3.00,
        ("harm", "user"): 1.97, ("harm", "assistant"): 1.91,
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, domain in zip(axes, ["truth", "harm"]):
        probes = [("tb-5_L32" if domain == "truth" else "tb-5_L39", "end-of-turn probe"),
                  ("tb-2_L32" if domain == "truth" else "tb-2_L39", "model-marker probe")]
        turns = ["user", "assistant"]
        x = np.arange(len(turns))
        w = 0.35
        color_map = {"end-of-turn probe": "#1976D2", "model-marker probe": "#00897B"}
        for pi, (probe, label) in enumerate(probes):
            vals = []
            for t in turns:
                matches = [r for r in rows
                           if r["domain"] == f"{domain} ({t})" and r["probe"] == probe]
                vals.append(abs(float(matches[0]["d"])) if matches else 0)
            ax.bar(x + (pi - 0.5) * w, vals, w, color=color_map[label],
                   edgecolor="black", linewidth=0.4, label=label)
        parent_vals = [parent[(domain, t)] for t in turns]
        ax.hlines(parent_vals, x - 0.5, x + 0.5, colors="#E65100", linestyles="--",
                  linewidth=2, label="task-averaged probe (parent)")
        ax.set_xticks(x)
        ax.set_xticklabels(["user turn", "assistant turn"])
        ax.set_ylabel("|Cohen's d| at end-of-turn token" if ax is axes[0] else "")
        ax.set_title(f"{domain.capitalize()} — per-turn discrimination")
        ax.axhline(2, color="gray", linestyle=":", linewidth=0.7)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Per-turn discrimination across training-token choices", y=1.03, fontsize=11)
    plt.tight_layout()
    path = ASSETS_DIR / f"plot_{DATE}_per_turn_cross_probe.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")
    import shutil
    paper_path = PAPER_FIG_DIR / path.name
    shutil.copy(path, paper_path)
    print(f"copied to {paper_path}")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_main_base_discrimination()
    fig_main_persona_modulation()
    fig_appendix_cross_training_selector()
    fig_appendix_per_turn()


if __name__ == "__main__":
    main()
