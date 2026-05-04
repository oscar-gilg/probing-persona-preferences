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

from corroborate import ClaimSet

EXP_DIR = Path("experiments/token_level_probes")
V2_DIR = EXP_DIR / "system_prompt_modulation_v2"
OUT_DIR = EXP_DIR / "canonical_probe_eval"
ASSETS_DIR = OUT_DIR / "assets"
PAPER_FIG_DIR = Path("paper/figures")

PARENT_PATH = V2_DIR / "parent_eot_scores.json"
V2_TRUTHHARM_PATH = V2_DIR / "scoring_results.json"
V2_POLITICS_PATH = V2_DIR / "politics_scoring_results.json"

# Repo-relative strings used for the data_paths field on registered claims.
PARENT_REL = str(PARENT_PATH)
V2_TRUTHHARM_REL = str(V2_TRUTHHARM_PATH)
V2_POLITICS_REL = str(V2_POLITICS_PATH)
HEADLINE_TABLE_REL = str(OUT_DIR / "headline_table.csv")
PER_TURN_TABLE_REL = str(OUT_DIR / "per_turn_table.csv")

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
    "tb-2_L32": "role-marker probe",
    "tb-2_L39": "role-marker probe",
    "tb-2_L53": "role-marker probe (L53)",
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

CLAIMS_SOURCE = "experiments/token_level_probes/canonical_probe_eval/scripts/make_paper_figures.py"
CLAIMS_OUT = Path("paper/claims/canonical_probe_eval_make_paper_figures.json")
claims = ClaimSet(source=CLAIMS_SOURCE)


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
    """Figure for §5.1.2: probe score distributions by condition, three domains.

    Uses neutral/default conditions only — prompt-induced shifts go in a
    separate figure.
    """
    parent = load(PARENT_PATH)
    truth = [it for it in parent if it["domain"] == "truth"]
    harm = [it for it in parent if it["domain"] == "harm"]

    panels = [
        ("truth", "Truth (CREAK, user turn)", truth, "tb-5_L32", "true", "false",
         lambda it: it.get("turn") == "user"),
        ("harm", "Harm (BailBench+stress test)", harm, "tb-5_L39", "harmful", "benign", None),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=False)
    for ax, (domain_key, title, items, probe, c_pos, c_neg, tfilter) in zip(axes, panels):
        def gather(cond):
            vals = [it["eot_scores"][probe] for it in items
                    if it["condition"] == cond and (tfilter is None or tfilter(it))]
            return np.array(vals)
        pos_vals = gather(c_pos)
        neg_vals = gather(c_neg)
        non_vals = gather("nonsense")
        d_raw = cohen_d_pooled(pos_vals, neg_vals)
        n_pos = len(pos_vals)
        n_neg = len(neg_vals)

        # Plot d-values only; the registered claims for CREAK truth Cohen's d,
        # CREAK truth n true/false, BailBench harm absolute Cohen's d, and
        # BailBench harm n harmful/benign are owned by
        # scripts/paper/refresh_eot_v2_claims.py against v2 data.
        d = round(float(d_raw), 2)

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


def fig_main_induced_shifts_minimal():
    """Simplified §5.1.2 figure: violin distributions under a few selected prompts.

    Truth: neutral, lie_directive, pathological_liar
    Harm: neutral, sadist
    Shows that under inversion-style personas the conditions collapse or flip.
    """
    truth_items = [it for it in load(V2_TRUTHHARM_PATH) if it["domain"] == "truth"]
    harm_items = [it for it in load(V2_TRUTHHARM_PATH) if it["domain"] == "harm"]
    politics_items = load(V2_POLITICS_PATH)

    truth_prompts = ["neutral", "lie_directive", "pathological_liar"]
    harm_prompts = ["neutral", "sadist"]
    politics_prompts = ["democrat", "republican"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2),
                             gridspec_kw={"width_ratios": [3, 2, 2]})

    def panel(ax, items, prompts, probe, c_pos, c_neg, domain_label, domain_key):
        by_sp = defaultdict(list)
        for it in items:
            by_sp[it["system_prompt"]].append(it)

        positions = []
        all_series = []
        all_colors = []
        labels = []
        d_values = []

        for pi, sp in enumerate(prompts):
            pos_vals = [it["eot_scores"][probe] for it in by_sp.get(sp, [])
                        if it["condition"] == c_pos]
            neg_vals = [it["eot_scores"][probe] for it in by_sp.get(sp, [])
                        if it["condition"] == c_neg]
            d_raw = cohen_d_pooled(np.array(pos_vals), np.array(neg_vals))
            # The 7 minimal-panel claims (Truth/Harm/Politics d under the
            # listed personas) are owned by scripts/paper/refresh_eot_v2_claims.py
            # against v2 data; we plot d_raw locally without registering.
            d = round(float(d_raw), 2)
            d_values.append((sp, d))
            positions.extend([pi * 3, pi * 3 + 1])
            all_series.extend([pos_vals, neg_vals])
            all_colors.extend([COLORS[c_pos], COLORS[c_neg]])
            labels.append(sp)

        parts = ax.violinplot(all_series, positions=positions, widths=0.9,
                              showmeans=True, showextrema=False)
        for body, color in zip(parts["bodies"], all_colors):
            body.set_facecolor(color); body.set_alpha(0.75); body.set_edgecolor("black")
        parts["cmeans"].set_color("black")

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xticks([pi * 3 + 0.5 for pi in range(len(prompts))])
        ax.set_xticklabels(
            [f"{sp}\n(d = {d:+.2f})" for sp, d in d_values],
            fontsize=9,
        )
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(domain_label, fontsize=11)
        ax.set_ylabel("End-of-turn probe score")

        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[c_pos], alpha=0.75),
                   plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[c_neg], alpha=0.75)]
        ax.legend(handles, [c_pos, c_neg], loc="best", fontsize=9)

    panel(axes[0], truth_items, truth_prompts, "tb-5_L32", "true", "false",
          "Truth: lying personas collapse or flip the signal", "truth")
    panel(axes[1], harm_items, harm_prompts, "tb-5_L39", "harmful", "benign",
          "Harm: sadist persona collapses the signal", "harm")
    panel(axes[2], politics_items, politics_prompts, "tb-5_L39", "left", "right",
          "Politics: partisan prompt flips the sign", "politics")

    fig.suptitle("Persona prompts modulate the probe's evaluative readout",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    path = ASSETS_DIR / f"plot_{DATE}_canonical_eot_induced_shifts_minimal.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")
    import shutil
    paper_path = PAPER_FIG_DIR / path.name
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
    table: dict[str, dict[str, float]] = {}
    for ax, dom in zip(axes, domains):
        by_sp = defaultdict(list)
        for it in dom["items"]:
            by_sp[it["system_prompt"]].append(it)

        ds = []
        row: dict[str, float] = {}
        for sp in dom["order"]:
            sp_items = by_sp.get(sp, [])
            pos = [it["eot_scores"][dom["probe"]] for it in sp_items if it["condition"] == dom["pos"]]
            neg = [it["eot_scores"][dom["probe"]] for it in sp_items if it["condition"] == dom["neg"]]
            d_raw = cohen_d_pooled(pos, neg) if pos and neg else float("nan")
            bar_val = round(float(d_raw), 2) if not np.isnan(d_raw) else float("nan")
            row[sp] = bar_val
            ds.append(d_raw)
        table[dom["name"]] = row

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

    # "Persona modulation d full" is owned by scripts/paper/refresh_eot_v2_claims.py
    # against v2 data; the v1 `table` here is plot-only.


def fig_appendix_token_diagram():
    """Schematic: Gemma-3-IT chat-template tokens and the three probe training positions."""
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5)
    ax.axis("off")

    tokens = [
        ("<bos>", "#ECEFF1"),
        ("<start_of_turn>", "#ECEFF1"),
        ("user", "#ECEFF1"),
        ("\\n", "#ECEFF1"),
        ("[task tokens]", "#E3F2FD"),
        ("<end_of_turn>", "#FFECB3"),
        ("\\n", "#ECEFF1"),
        ("<start_of_turn>", "#ECEFF1"),
        ("model", "#FFE0B2"),
        ("\\n", "#ECEFF1"),
        ("[assistant content]", "#E8F5E9"),
    ]

    box_w = 1.25
    box_h = 0.85
    y = 3.1
    positions = []
    for i, (tok, color) in enumerate(tokens):
        x = 0.3 + i * box_w
        rect = plt.Rectangle((x, y), box_w - 0.1, box_h,
                             facecolor=color, edgecolor="black", linewidth=0.7)
        ax.add_patch(rect)
        ax.text(x + (box_w - 0.1) / 2, y + box_h / 2, tok,
                ha="center", va="center", fontsize=8, family="monospace")
        positions.append(x + (box_w - 0.1) / 2)

    # probe annotations: (token index, label, color, vertical_offset)
    # stagger vertically to avoid overlap since task-averaged and end-of-turn are adjacent
    probe_annotations = [
        (4, "task-averaged probe\n(mean over the task span)", "#E65100", 1.5),
        (5, "end-of-turn probe\n(canonical; used in §5.1.2)", "#1976D2", 0.6),
        (8, "role-marker probe\n(appendix only)", "#00897B", 0.6),
    ]
    for idx, label, color, label_offset in probe_annotations:
        x = positions[idx]
        # arrow from below the box to the label position
        arrow_bottom_y = y - label_offset - 0.25
        ax.annotate("", xy=(x, y - 0.02), xytext=(x, arrow_bottom_y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.6))
        ax.text(x, arrow_bottom_y - 0.1, label, ha="center", va="top",
                fontsize=8.5, color=color, fontweight="bold")

    ax.text(0.3 + 5.5 * box_w, 4.55,
            "Gemma-3-IT chat template (one user turn, prefilled assistant response)",
            ha="center", va="center", fontsize=10.5, fontweight="bold")

    plt.tight_layout()
    path = ASSETS_DIR / f"plot_{DATE}_turn_boundary_tokens_diagram.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")
    import shutil
    paper_path = PAPER_FIG_DIR / path.name
    shutil.copy(path, paper_path)
    print(f"copied to {paper_path}")


def fig_appendix_cross_training_selector():
    """Appendix: signed d across training-token choices. Shows sign flip on politics."""
    import csv
    rows = list(csv.DictReader(open(OUT_DIR / "headline_table.csv")))

    # Display convention in figure: truth (true>false), harm (harmful>benign),
    # politics (left>right). Self-computed politics rows in the CSV use a
    # right>left convention, so we flip those. Parent task_mean rows already use
    # left>right, so we leave them alone.
    def signed_d(row):
        d = float(row["d"]) if row["d"] else 0.0
        if row["domain"].startswith("politics") and "parent" not in row["probe"]:
            d = -d
        return d

    domain_order = ["truth", "harm", "politics_democrat", "politics_republican"]
    domain_labels = {
        "truth": "Truth\n(true vs false)",
        "harm": "Harm\n(harmful vs benign)",
        "politics_democrat": "Politics\n(democrat prompt,\nleft vs right)",
        "politics_republican": "Politics\n(republican prompt,\nleft vs right)",
    }
    probe_color = {
        "end-of-turn probe": "#1976D2",
        "role-marker probe": "#00897B",
        "task-averaged probe": "#E65100",
    }
    # Match CSV probe name strings exactly (task_mean rows have " (parent)" suffix)
    best_probe_for_domain = {
        "truth": ["tb-5_L32", "tb-2_L32", "task_mean_L32 (parent)"],
        "harm": ["tb-5_L39", "tb-2_L39", "task_mean_L39 (parent)"],
        "politics_democrat": ["tb-5_L39", "tb-2_L39", "task_mean_L39 (parent)"],
        "politics_republican": ["tb-5_L39", "tb-2_L39", "task_mean_L39 (parent)"],
    }

    data = {(r["domain"], r["probe"]): r for r in rows}

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(domain_order))
    w = 0.25
    family_to_offset = {"end-of-turn probe": -w, "role-marker probe": 0, "task-averaged probe": w}

    ct_table: dict[str, dict[str, float]] = {d: {} for d in domain_order}
    for probe_prefix, label in [("tb-5", "end-of-turn probe"),
                                ("tb-2", "role-marker probe"),
                                ("task_mean", "task-averaged probe")]:
        heights = []
        for d in domain_order:
            probes_here = best_probe_for_domain[d]
            match = next((p for p in probes_here if p.startswith(probe_prefix + "_")), None)
            row = data.get((d, match)) if match else None
            h_raw = signed_d(row) if row and row["d"] else 0.0
            ct_table[d][label] = round(float(h_raw), 2)
            heights.append(h_raw)
        ax.bar(x + family_to_offset[label], heights, w,
               color=probe_color[label], label=label,
               edgecolor="black", linewidth=0.4)

    claims.register(
        name="Cross training d",
        value=ct_table,
        statement=(
            "Signed Cohen's d at the end-of-turn token, one cell per "
            "(domain, probe family) bar in the cross-training-selector "
            "appendix figure. Positive = first-listed class scores higher "
            "(true / harmful / left)."
        ),
        used_in=["fig:cross-training"],
        data_paths=[HEADLINE_TABLE_REL],
        derivation=(
            "For each (domain, probe_family_prefix) cell: select the row matching "
            "domain and probe prefix from headline_table.csv; read `d`; flip sign "
            "for non-parent politics rows (right>left → left>right); round to 2dp."
        ),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([domain_labels[d] for d in domain_order])
    ax.set_ylabel("Cohen's d at end-of-turn token")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axhline(2, color="gray", linestyle=":", linewidth=0.6)
    ax.axhline(-2, color="gray", linestyle=":", linewidth=0.6, label="|d|=2")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("The same evaluative discrimination emerges across training-token choices")
    ax.legend(loc="best", fontsize=9)
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

    # Prose-only claims from sec:induced-roleplay. These sizes and CV
    # accuracies come from the per_turn_table.csv rows for the canonical
    # end-of-turn probe (tb-5_L32 for truth, tb-5_L39 for harm) and appear
    # as bullet-point numerals in the paper but not as rendered figure
    # elements. They're registered here because this function already has
    # the per-turn CSV loaded.
    truth_user_row = next(r for r in rows
                          if r["domain"] == "truth (user)" and r["probe"] == "tb-5_L32")
    harm_user_row = next(r for r in rows
                         if r["domain"] == "harm (user)" and r["probe"] == "tb-5_L39")
    harm_assistant_row = next(r for r in rows
                              if r["domain"] == "harm (assistant)" and r["probe"] == "tb-5_L39")
    claims.register(
        name="CREAK truth CV accuracy",
        value=round(float(truth_user_row["cv_acc"]), 2),
        statement=(
            "Cross-validated classification accuracy of the end-of-turn probe "
            "on CREAK true vs false at the user turn (tb-5_L32). Quoted as "
            "~93% in sec:induced-roleplay."
        ),
        used_in=["sec:induced-roleplay"],
        data_paths=[PER_TURN_TABLE_REL],
        derivation="row with domain=='truth (user)' and probe=='tb-5_L32'; `cv_acc` column; round to 2dp.",
    )
    claims.register(
        name="BailBench harm n items prose",
        value=int(harm_user_row["n_pos"]),
        statement=(
            "Approximate per-class sample size of harmful BailBench + "
            "stress-test items used in the per-turn split; quoted as ~77 "
            "items in sec:induced-roleplay."
        ),
        used_in=["sec:induced-roleplay"],
        data_paths=[PER_TURN_TABLE_REL],
        derivation="row with domain=='harm (user)' and probe=='tb-5_L39'; `n_pos` column.",
    )
    claims.register(
        name="BailBench harm CV accuracy user",
        value=round(float(harm_user_row["cv_acc"]), 2),
        statement=(
            "Cross-validated classification accuracy of the end-of-turn probe "
            "on harmful vs benign BailBench + stress-test items at the user "
            "turn (tb-5_L39). One endpoint of the 84-89% range quoted in "
            "sec:induced-roleplay."
        ),
        used_in=["sec:induced-roleplay"],
        data_paths=[PER_TURN_TABLE_REL],
        derivation="row with domain=='harm (user)' and probe=='tb-5_L39'; `cv_acc` column; round to 2dp.",
    )
    claims.register(
        name="BailBench harm CV accuracy assistant",
        value=round(float(harm_assistant_row["cv_acc"]), 2),
        statement=(
            "Cross-validated classification accuracy of the end-of-turn probe "
            "on harmful vs benign BailBench + stress-test items at the "
            "assistant turn (tb-5_L39). Other endpoint of the 84-89% range "
            "quoted in sec:induced-roleplay."
        ),
        used_in=["sec:induced-roleplay"],
        data_paths=[PER_TURN_TABLE_REL],
        derivation="row with domain=='harm (assistant)' and probe=='tb-5_L39'; `cv_acc` column; round to 2dp.",
    )

    parent = {
        ("truth", "user"): 3.19, ("truth", "assistant"): 3.00,
        ("harm", "user"): 1.97, ("harm", "assistant"): 1.91,
    }
    parent_table = {
        "truth": {"user": 3.19, "assistant": 3.00},
        "harm":  {"user": 1.97, "assistant": 1.91},
    }
    # "Per turn parent absolute d" is owned by scripts/paper/refresh_eot_v2_claims.py
    # against v2 data; the v1 hardcoded `parent_table` here is plot-only.

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    per_turn_tables: dict[str, dict[str, dict[str, float]]] = {}
    for ax, domain in zip(axes, ["truth", "harm"]):
        probes = [("tb-5_L32" if domain == "truth" else "tb-5_L39", "end-of-turn probe"),
                  ("tb-2_L32" if domain == "truth" else "tb-2_L39", "role-marker probe")]
        turns = ["user", "assistant"]
        x = np.arange(len(turns))
        w = 0.35
        color_map = {"end-of-turn probe": "#1976D2", "role-marker probe": "#00897B"}
        domain_table: dict[str, dict[str, float]] = {t: {} for t in turns}
        for pi, (probe, label) in enumerate(probes):
            vals = []
            for t in turns:
                matches = [r for r in rows
                           if r["domain"] == f"{domain} ({t})" and r["probe"] == probe]
                v = abs(float(matches[0]["d"])) if matches else 0.0
                domain_table[t][label] = round(float(v), 2)
                vals.append(v)
            ax.bar(x + (pi - 0.5) * w, vals, w, color=color_map[label],
                   edgecolor="black", linewidth=0.4, label=label)
        per_turn_tables[domain] = domain_table
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

    # "Per turn absolute d truth" / "Per turn absolute d harm" are owned by
    # scripts/paper/refresh_eot_v2_claims.py against v2 data.


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_main_base_discrimination()
    fig_main_induced_shifts_minimal()
    fig_main_persona_modulation()  # full version for appendix
    fig_appendix_token_diagram()
    fig_appendix_cross_training_selector()
    fig_appendix_per_turn()
    claims.save(CLAIMS_OUT)
    print(f"wrote {CLAIMS_OUT} ({len(claims.claims)} claims)")


if __name__ == "__main__":
    main()
