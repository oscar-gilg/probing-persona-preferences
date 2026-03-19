"""Donor ranking analysis: which personas are most useful as probe training donors?

For N=2 and N=3 combos from the midway bias results, rank individual donors
by mean OOD Pearson r across all combos containing them, averaged over
selectors, layers, and held-out personas.

Also computes utility-partialled rankings: after regressing out the effect of
donor-eval utility correlation on OOD r, which donors still contribute?
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from dotenv import load_dotenv

load_dotenv()

RESULTS_PATH = Path("results/experiments/mra_exp3/midway_bias/midway_bias_results.json")
ASSETS = Path("experiments/probe_generalization/multi_role_ablation/midway_bias/assets")
FOCUS_TOPICS = {"harmful_request", "math", "knowledge_qa", "fiction", "coding", "content_generation"}

NON_DEFAULT = ["villain", "aesthete", "midwest", "provocateur", "trickster", "autocrat", "sadist"]
ALL_PERSONAS = ["noprompt"] + NON_DEFAULT

PERSONA_RUNS = {
    "noprompt": (Path("results/experiments/mra_exp2/pre_task_active_learning"), ""),
    "villain": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "syse8f24ac6"),
    "aesthete": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "sys021d8ca1"),
    "midwest": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "sys5d504504"),
    "provocateur": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sysf4d93514"),
    "trickster": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys09a42edc"),
    "autocrat": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys1c18219a"),
    "sadist": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys39e01d59"),
}

SPLIT_C_IDS_PATH = Path("configs/measurement/active_learning/mra_exp2_split_c_1000_task_ids.txt")


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def compute_donor_ranking(results: list[dict], n_target: int) -> dict[str, list[float]]:
    """For each individual donor, collect OOD Pearson r values across all combos containing them."""
    donor_r: dict[str, list[float]] = defaultdict(list)
    for entry in results:
        if entry["n_personas"] != n_target:
            continue
        if entry["is_in_dist"] or entry["eval_persona"] == "noprompt":
            continue
        donors = [p for p in entry["train_personas"] if p != "noprompt"]
        for d in donors:
            donor_r[d].append(entry["pearson_r"])
    return dict(donor_r)


def compute_combo_ranking(results: list[dict], n_target: int) -> list[tuple[str, float]]:
    """Rank specific combos by mean OOD Pearson r."""
    combo_r: dict[str, list[float]] = defaultdict(list)
    for entry in results:
        if entry["n_personas"] != n_target:
            continue
        if entry["is_in_dist"] or entry["eval_persona"] == "noprompt":
            continue
        donors = tuple(sorted(p for p in entry["train_personas"] if p != "noprompt"))
        combo_r[donors].append(entry["pearson_r"])

    ranking = [("+".join(donors), float(np.mean(vals))) for donors, vals in combo_r.items()]
    ranking.sort(key=lambda x: -x[1])
    return ranking


def plot_donor_ranking(results: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, n_target in zip(axes, [2, 3]):
        donor_r = compute_donor_ranking(results, n_target)
        ranking = [(d, float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals))))
                    for d, vals in donor_r.items()]
        ranking.sort(key=lambda x: -x[1])

        names = [r[0] for r in ranking]
        means = [r[1] for r in ranking]
        sems = [r[2] for r in ranking]

        colors = ["#2196F3" if m >= means[2] else "#90CAF9" for m in means]  # top 3 darker

        bars = ax.barh(range(len(names)), means, xerr=sems, color=colors,
                       edgecolor="white", linewidth=0.5, capsize=3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Mean OOD Pearson r")
        ax.set_title(f"N={n_target}: donor ranking\n(noprompt + donor{'s' if n_target > 2 else ''})")
        ax.set_xlim(0, 0.8)
        ax.axvline(x=np.mean(means), color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    out = ASSETS / "plot_031626_donor_ranking.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def plot_top_combos_n3(results: list[dict]):
    combo_ranking = compute_combo_ranking(results, n_target=3)
    top_10 = combo_ranking[:10]
    bottom_5 = combo_ranking[-5:]

    names = [r[0] for r in top_10 + bottom_5]
    vals = [r[1] for r in top_10 + bottom_5]
    colors = ["#2196F3"] * 10 + ["#FFCDD2"] * 5

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(names)), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean OOD Pearson r")
    ax.set_title("N=3: top 10 and bottom 5 donor pairs\n(noprompt + 2 donors, ranked by mean OOD r)")
    ax.set_xlim(0, 0.8)

    out = ASSETS / "plot_031626_top_combos_n3.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def compute_utility_correlations() -> dict[tuple[str, str], float]:
    """Compute pairwise Thurstonian score correlations between all personas on eval split C."""
    from src.probes.data_loading import load_thurstonian_scores

    with open(SPLIT_C_IDS_PATH) as f:
        split_c_ids = {line.strip() for line in f if line.strip()}

    all_scores = {}
    for persona, (results_dir, sys_hash) in PERSONA_RUNS.items():
        prefix = "completion_preference_gemma-3-27b_completion_canonical_seed0"
        suffix = "mra_exp2_split_c_1000_task_ids"
        dirname = f"{prefix}_{sys_hash}_{suffix}" if sys_hash else f"{prefix}_{suffix}"
        run_dir = results_dir / dirname
        scores = load_thurstonian_scores(run_dir)
        all_scores[persona] = {tid: scores[tid] for tid in split_c_ids if tid in scores}

    common_ids = sorted(set.intersection(*[set(s.keys()) for s in all_scores.values()]))

    corr_map = {}
    for p1 in ALL_PERSONAS:
        v1 = np.array([all_scores[p1][tid] for tid in common_ids])
        for p2 in ALL_PERSONAS:
            v2 = np.array([all_scores[p2][tid] for tid in common_ids])
            r, _ = pearsonr(v1, v2)
            corr_map[(p1, p2)] = float(r)
    return corr_map


def compute_partialled_donor_ranking(
    results: list[dict],
    utility_corr: dict[tuple[str, str], float],
    n_target: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Regress OOD r on max donor-eval utility correlation, return per-donor mean residual."""
    # Collect (donor_set, eval_persona, ood_r) triples
    rows = []
    for entry in results:
        if entry["n_personas"] != n_target:
            continue
        if entry["is_in_dist"] or entry["eval_persona"] == "noprompt":
            continue
        donors = tuple(sorted(p for p in entry["train_personas"] if p != "noprompt"))
        eval_p = entry["eval_persona"]
        # Max utility correlation between any training donor and eval persona
        max_corr = max(utility_corr[(d, eval_p)] for d in donors)
        rows.append((donors, eval_p, entry["pearson_r"], max_corr))

    ood_r = np.array([r[2] for r in rows])
    util_corr_vals = np.array([r[3] for r in rows])

    # Linear regression: ood_r = a + b * util_corr
    slope, intercept = np.polyfit(util_corr_vals, ood_r, 1)
    predicted = intercept + slope * util_corr_vals
    residuals = ood_r - predicted

    # Per-donor mean residual
    donor_residuals: dict[str, list[float]] = defaultdict(list)
    for i, (donors, eval_p, r, uc) in enumerate(rows):
        for d in donors:
            donor_residuals[d].append(residuals[i])

    donor_mean_resid = {d: float(np.mean(vals)) for d, vals in donor_residuals.items()}
    return donor_mean_resid, util_corr_vals, ood_r


def plot_partialled_ranking(
    results: list[dict],
    utility_corr: dict[tuple[str, str], float],
):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: scatter of utility corr vs OOD r for N=2, colored by donor
    ax = axes[0]
    colors_map = {
        "villain": "#E53935", "provocateur": "#FB8C00", "trickster": "#43A047",
        "sadist": "#1E88E5", "autocrat": "#8E24AA", "midwest": "#6D4C41", "aesthete": "#00ACC1",
    }
    for entry in results:
        if entry["n_personas"] != 2 or entry["is_in_dist"] or entry["eval_persona"] == "noprompt":
            continue
        donor = [p for p in entry["train_personas"] if p != "noprompt"][0]
        eval_p = entry["eval_persona"]
        uc = utility_corr[(donor, eval_p)]
        ax.scatter(uc, entry["pearson_r"], color=colors_map[donor], alpha=0.3, s=15, edgecolors="none")

    # Regression line
    _, util_corr_vals, ood_r = compute_partialled_donor_ranking(results, utility_corr, 2)
    slope, intercept = np.polyfit(util_corr_vals, ood_r, 1)
    xs = np.linspace(-0.5, 0.8, 50)
    ax.plot(xs, intercept + slope * xs, "k--", linewidth=1, alpha=0.7)
    r_val, _ = pearsonr(util_corr_vals, ood_r)
    ax.set_xlabel("Donor-eval utility correlation")
    ax.set_ylabel("OOD Pearson r")
    ax.set_title(f"N=2: utility corr vs OOD r\n(r={r_val:.2f})")
    ax.set_xlim(-0.5, 0.8)
    ax.set_ylim(0, 1.0)
    # Legend
    for p in NON_DEFAULT:
        ax.scatter([], [], color=colors_map[p], label=p, s=30)
    ax.legend(fontsize=7, loc="lower right")

    # Middle + Right: raw vs partialled donor ranking for N=2 and N=3
    for ax, n_target in zip(axes[1:], [2, 3]):
        raw = compute_donor_ranking(results, n_target)
        partialled, _, _ = compute_partialled_donor_ranking(results, utility_corr, n_target)

        # Sort by partialled
        ranking = sorted(partialled.items(), key=lambda x: -x[1])
        names = [r[0] for r in ranking]
        resid_vals = [r[1] for r in ranking]
        raw_vals = [float(np.mean(raw[n])) for n in names]

        y = np.arange(len(names))
        ax.barh(y - 0.15, raw_vals, height=0.3, color="#90CAF9", label="Raw OOD r")
        ax.barh(y + 0.15, resid_vals, height=0.3, color="#E53935", label="Residual (utility-partialled)")
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Value")
        ax.set_title(f"N={n_target}: raw vs partialled ranking")
        ax.axvline(x=0, color="gray", linewidth=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = ASSETS / "plot_031626_partialled_donor_ranking.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def main():
    results = load_results()
    utility_corr = compute_utility_correlations()

    # Print raw tables
    for n_target in [2, 3]:
        donor_r = compute_donor_ranking(results, n_target)
        ranking = [(d, float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals))))
                    for d, vals in donor_r.items()]
        ranking.sort(key=lambda x: -x[1])

        print(f"\nN={n_target}: donor ranking (mean OOD r across all combos containing donor)")
        print(f"  {'Donor':<15} {'Mean OOD r':>12} {'SEM':>8}")
        for d, m, s in ranking:
            print(f"  {d:<15} {m:>12.3f} {s:>8.3f}")

    # Print partialled tables
    for n_target in [2, 3]:
        partialled, util_corr_vals, ood_r = compute_partialled_donor_ranking(results, utility_corr, n_target)
        r_val, _ = pearsonr(util_corr_vals, ood_r)
        print(f"\nN={n_target}: utility-partialled ranking (residual after regressing out donor-eval corr)")
        print(f"  Utility corr explains r={r_val:.2f} of OOD r variance")
        ranking = sorted(partialled.items(), key=lambda x: -x[1])
        print(f"  {'Donor':<15} {'Mean residual':>14}")
        for d, m in ranking:
            print(f"  {d:<15} {m:>14.3f}")

    print("\nN=3: top 10 specific combos")
    combo_ranking = compute_combo_ranking(results, n_target=3)
    for label, r in combo_ranking[:10]:
        print(f"  noprompt + {label:<30} r={r:.3f}")

    plot_donor_ranking(results)
    plot_top_combos_n3(results)
    plot_partialled_ranking(results, utility_corr)


if __name__ == "__main__":
    main()
