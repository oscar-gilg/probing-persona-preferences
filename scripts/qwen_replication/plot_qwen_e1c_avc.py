"""Qwen-3.5-122B A vs C scatter for the fine-grained biography-injection
experiment — matches the Gemma format of scripts/ood_system_prompts/plot_exp3v8_avc.py
so the two models sit side-by-side in Appendix E.

One point per (A-vs-C pair, task): 28 pairs * 50 tasks = 1400 grey dots.
Stars overlay the 28 target tasks; filled means probe ranked the target #1.
Dashed blue line is the all-points fit with pooled Pearson r.

Math pruning: drop competition_math_* targets except
  competition_math_10564 (consistent miss, the motivating example)
  competition_math_11276 (a math hit)
Takes the pool from 20 targets * 2 base roles = 40 pairs down to 14 * 2 = 28.

Usage: python -m scripts.qwen_replication.plot_qwen_e1c_avc
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from corroborate import ClaimSet

from src.ood.analysis import compute_p_choose_from_pairwise

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKTREE = REPO_ROOT / ".claude" / "worktrees" / "qwen_replication"

ACTS_DIR = WORKTREE / "activations" / "qwen35_122b_ood" / "e1c_fixed" / "e1c_fixed"
PROBE_PATH = (
    WORKTREE / "results" / "probes" / "qwen35_122b"
    / "qwen35_122b_heldout_turn_boundary_m1" / "probes" / "probe_ridge_L38.npy"
)
# Dense pairwise: 50 tasks x 121 conditions, matches Gemma's measurement density.
PAIRWISE_PATH = WORKTREE / "results" / "ood" / "minimal_pairs_v8" / "pairwise.json"
CFG_PATH = REPO_ROOT / "configs" / "ood" / "qwen35" / "minimal_pairs_v8.json"

ASSETS_DIR = REPO_ROOT / "experiments" / "qwen_replication" / "assets"
PAPER_FIGS = REPO_ROOT / "paper" / "figures"

LAYER = 38
N_TASKS = 50
GREY = "#BDBDBD"
MATH_KEEP = {"competition_math_10564", "competition_math_11276"}


def load_probe(path: Path) -> tuple[np.ndarray, float]:
    probe = np.load(path)
    return probe[:-1], float(probe[-1])


def score_activations(npz_path: Path, layer: int, weights: np.ndarray, bias: float) -> dict[str, float]:
    data = np.load(npz_path, allow_pickle=True)
    acts = data[f"layer_{layer}"]
    scores = acts @ weights + bias
    return dict(zip(list(data["task_ids"]), scores.tolist()))


def keep_target(target: str) -> bool:
    if target.startswith("competition_math_"):
        return target in MATH_KEEP
    return True


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    claims = ClaimSet(source="scripts/qwen_replication/plot_qwen_e1c_avc.py")

    pairwise = json.load(open(PAIRWISE_PATH))
    rates = compute_p_choose_from_pairwise(pairwise["results"])
    cfg = json.load(open(CFG_PATH))
    cond_info = {c["condition_id"]: c for c in cfg["conditions"]}

    weights, bias = load_probe(PROBE_PATH)
    baseline_scores = score_activations(
        ACTS_DIR / "baseline" / "activations_turn_boundary:-1.npz",
        LAYER, weights, bias,
    )
    baseline_rates = rates["baseline"]
    tasks = sorted(baseline_rates.keys())

    condition_deltas: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for cid in cond_info:
        npz = ACTS_DIR / cid / "activations_turn_boundary:-1.npz"
        if not npz.exists():
            continue
        cond_scores = score_activations(npz, LAYER, weights, bias)
        cond_rates = rates.get(cid, {})
        beh_d, probe_d = {}, {}
        for tid in tasks:
            if tid in cond_rates and tid in baseline_rates and tid in cond_scores and tid in baseline_scores:
                beh_d[tid] = cond_rates[tid] - baseline_rates[tid]
                probe_d[tid] = cond_scores[tid] - baseline_scores[tid]
        if beh_d:
            condition_deltas[cid] = (beh_d, probe_d)

    # Group by (base_role, target), pruned
    groups: dict[tuple[str, str], dict[str, str]] = {}
    for cid, info in cond_info.items():
        if not keep_target(info["target"]):
            continue
        key = (info["base_role"], info["target"])
        groups.setdefault(key, {})[info["version"]] = cid

    # A vs C version-pair deltas
    all_beh, all_probe, all_is_target, all_probe_rank1 = [], [], [], []
    n_target_total, n_probe_rank1 = 0, 0
    n_math_misses = 0

    for (base_role, target), version_cids in sorted(groups.items()):
        if "A" not in version_cids or "C" not in version_cids:
            continue
        cid_a, cid_c = version_cids["A"], version_cids["C"]
        if cid_a not in condition_deltas or cid_c not in condition_deltas:
            continue

        beh_a, probe_a = condition_deltas[cid_a]
        beh_c, probe_c = condition_deltas[cid_c]
        common = sorted(set(beh_a) & set(beh_c))

        beh_vp = {tid: beh_a[tid] - beh_c[tid] for tid in common}
        probe_vp = {tid: probe_a[tid] - probe_c[tid] for tid in common}
        probe_sorted = sorted(common, key=lambda t: probe_vp[t], reverse=True)

        for tid in common:
            is_target = (tid == target)
            all_beh.append(beh_vp[tid])
            all_probe.append(probe_vp[tid])
            all_is_target.append(is_target)
            if is_target:
                p_rank = probe_sorted.index(tid) + 1
                all_probe_rank1.append(p_rank == 1)
                n_target_total += 1
                if p_rank == 1:
                    n_probe_rank1 += 1
                elif target.startswith("competition_math_"):
                    n_math_misses += 1
            else:
                all_probe_rank1.append(False)

    beh_arr = np.array(all_beh)
    probe_arr = np.array(all_probe)
    target_mask = np.array(all_is_target)
    rank1_mask = np.array(all_probe_rank1)
    target_not_rank1 = target_mask & ~rank1_mask

    print(f"Total points: {len(beh_arr)}")
    print(f"Target pairs: {n_target_total}, probe rank 1: {n_probe_rank1}, math misses: {n_math_misses}")

    _acts_rel = ".claude/worktrees/qwen_replication/activations/qwen35_122b_ood/e1c_fixed/e1c_fixed"
    _pairwise_rel = ".claude/worktrees/qwen_replication/results/ood/minimal_pairs_v8/pairwise.json"
    _cfg_rel = "configs/ood/qwen35/minimal_pairs_v8.json"
    _probe_rel = ".claude/worktrees/qwen_replication/results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1/probes/probe_ridge_L38.npy"

    claims.register(
        name="Qwen E1c AvC total task-condition points",
        value=int(len(beh_arr)),
        statement=(
            "Total number of task-condition points (across all A-vs-C biography pairs "
            "in the math-pruned 28-pair pool and all 50 comparison tasks) plotted in "
            "the Qwen-3.5-122B E1c fine-grained biography-injection scatter."
        ),
        used_in=["fig:fine-grained"],
        data_paths=[_pairwise_rel, _cfg_rel, _acts_rel, _probe_rel],
        derivation=(
            "Count of points appended while iterating A/C condition-pairs after math "
            "pruning (drop competition_math_* targets except 10564 and 11276); each "
            "(base_role, target) A-vs-C pair contributes |common tasks| points."
        ),
    )
    claims.register(
        name="Qwen E1c AvC target pairs total",
        value=int(n_target_total),
        statement=(
            "Number of (base_role, target) A-vs-C biography pairs in the Qwen-3.5-122B "
            "pool after math pruning (20 targets * 2 base roles = 40, minus 6 math * 2 = "
            "12 dropped, leaving 14 * 2 = 28)."
        ),
        used_in=["fig:fine-grained", "app:induced-fine"],
        data_paths=[_pairwise_rel, _cfg_rel, _acts_rel, _probe_rel],
        derivation=(
            "Count of (base_role, target) groups with both A and C activations after "
            "retaining only competition_math_10564 and competition_math_11276 from the "
            "8 math targets."
        ),
    )
    claims.register(
        name="Qwen E1c AvC target ranks one",
        value=int(n_probe_rank1),
        statement=(
            "Number of A-vs-C biography pairs on which the Qwen-3.5-122B L38 tb-1 "
            "ridge probe ranks the target task #1 of 50 by probe delta, on the math-"
            "pruned 28-pair pool."
        ),
        used_in=["fig:fine-grained", "app:induced-fine"],
        data_paths=[_pairwise_rel, _cfg_rel, _acts_rel, _probe_rel],
        derivation=(
            "For each (base_role, target) pair, sort the 50 tasks by probe_A - probe_C "
            "descending and count pairs where the target task is rank 1."
        ),
    )
    claims.register(
        name="Qwen E1c AvC math misses",
        value=int(n_math_misses),
        statement=(
            "Number of non-rank-1 target pairs in Qwen's pruned 28-pair pool that are "
            "math targets. All misses being math targets motivates the interpretation "
            "that the biography sentence boosts a cluster of similar math tasks, not "
            "just the specific target."
        ),
        used_in=["app:induced-fine"],
        data_paths=[_pairwise_rel, _cfg_rel, _acts_rel, _probe_rel],
        derivation="Count of pairs where target_rank != 1 and target.startswith('competition_math_').",
    )

    # --- Plot (Gemma-format: behavioural x probe, grey dots + red stars, dashed fit) ---
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(beh_arr[~target_mask], probe_arr[~target_mask],
               alpha=0.3, s=12, color=GREY, edgecolors="none", zorder=1,
               label="Other tasks")
    if rank1_mask.any():
        ax.scatter(beh_arr[rank1_mask], probe_arr[rank1_mask],
                   s=70, color="#e41a1c", marker="*", edgecolors="black",
                   linewidths=0.3, zorder=3,
                   label=f"Target (probe rank 1/{N_TASKS})")
    if target_not_rank1.any():
        ax.scatter(beh_arr[target_not_rank1], probe_arr[target_not_rank1],
                   s=70, facecolors="none", marker="*", edgecolors="#e41a1c",
                   linewidths=1.0, zorder=3,
                   label=f"Target (not rank 1/{N_TASKS})")

    r_all = scipy_stats.pearsonr(beh_arr, probe_arr)[0]
    r_all = claims.register(
        name="Qwen E1c AvC probe delta pooled r all points",
        value=round(float(r_all), 2),
        statement=(
            "Pearson r between per-task probe delta (probe_A - probe_C) and behavioural "
            "delta (P_A - P_C) pooled across all task-condition points in the Qwen-3.5-"
            "122B E1c scatter (ridge probe, L38, tb-1; math-pruned 28-pair pool). "
            "Rendered in the figure legend as 'All (r = 0.XX)'."
        ),
        used_in=["fig:fine-grained"],
        data_paths=[_pairwise_rel, _cfg_rel, _acts_rel, _probe_rel],
        derivation=(
            "scipy.stats.pearsonr between stacked behavioural deltas (P_A-P_C) and "
            "probe deltas (probe_A-probe_C); rounded to 2 dp."
        ),
    )
    slope, intercept, _, _, _ = scipy_stats.linregress(beh_arr, probe_arr)
    x_fit = np.linspace(beh_arr.min(), beh_arr.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept,
            color="#2196F3", linewidth=1.2, linestyle="--", alpha=0.5,
            label=f"All (r = {r_all:.2f})")

    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)

    ax.set_xlabel(
        "Behavioral shift  ($\\Delta \\mathbf{P}$, A − C)\n"
        "$\\Delta \\mathbf{P} = \\mathbf{P}_{\\mathrm{A}} - \\mathbf{P}_{\\mathrm{C}}$",
        fontsize=9.5, linespacing=1.6,
    )
    ax.set_ylabel(
        "Probe shift  ($\\Delta\\mathbf{probe}$, A − C)\n"
        "$\\Delta\\mathbf{probe} = \\mathbf{probe}_{\\mathrm{A}} - \\mathbf{probe}_{\\mathrm{C}}$",
        fontsize=9.5, linespacing=1.6,
    )
    ax.set_title("Qwen-3.5-122B: A vs C (pro-interest vs anti-interest)",
                 fontsize=12, fontweight="bold")

    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    out_assets = ASSETS_DIR / "plot_042426_qwen_e1c_avc.png"
    fig.savefig(out_assets, dpi=200, bbox_inches="tight")
    out_paper = PAPER_FIGS / "plot_042426_qwen_e1c_avc.png"
    fig.savefig(out_paper, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_assets}")
    print(f"Saved: {out_paper}")

    claims.save(REPO_ROOT / "paper" / "claims" / "qwen_e1c_avc.json")


if __name__ == "__main__":
    main()
