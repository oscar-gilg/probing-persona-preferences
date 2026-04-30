"""Per-origin analysis + plots of how default-Qwen preferences differ between
the reasoning-ON regime (qwen35_10k AL, pre-fix) and reasoning-OFF regime
(qwen_persona_sweep_final_six default_train, post-fix).

Outputs:
  - experiments/qwen_replication/persona_transfer/reasoning_mode_diff/results/diff.npz
    (shared task IDs, prev μ, curr μ, origin labels)
  - assets/plot_<date>_scatter_by_origin.png   — μ scatter, color = origin
  - assets/plot_<date>_per_origin_pearson.png  — bar chart of within-origin Pearson r
  - assets/plot_<date>_mean_mu_per_origin.png  — bar chart of mean μ per origin per run
  - assets/plot_<date>_mu_distribution.png     — KDE/hist of μ per origin per run
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from src.probes.data_loading import load_thurstonian_scores
from src.task_data.loader import load_tasks
from src.task_data.task import OriginDataset

REPO = Path(__file__).resolve().parents[2]
PREV_DIR = REPO / "results/experiments/qwen35_10k_active_learning/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d"
CURR_DIR = REPO / "results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/default_train"

OUT = REPO / "experiments/qwen_replication/persona_transfer/reasoning_mode_diff"
RESULTS = OUT / "results"
ASSETS = OUT / "assets"
TODAY = datetime.now().strftime("%m%d%y")

ORIGIN_COLORS = {
    "wildchat": "#1f77b4",
    "alpaca": "#ff7f0e",
    "math": "#2ca02c",
    "bailbench": "#d62728",
    "stress_test": "#9467bd",
}


def origin_from_id(tid: str) -> str:
    if tid.startswith("competition_math_") or tid.startswith("math_"):
        return "math"
    for tag in ("wildchat", "alpaca", "bailbench", "stress_test"):
        if tid.startswith(tag + "_"):
            return tag
    return "other"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    prev_mu = load_thurstonian_scores(PREV_DIR)
    curr_mu = load_thurstonian_scores(CURR_DIR)
    shared = sorted(set(prev_mu) & set(curr_mu))
    print(f"shared task IDs: {len(shared)}")

    prev = np.array([prev_mu[t] for t in shared])
    curr = np.array([curr_mu[t] for t in shared])
    origins = np.array([origin_from_id(t) for t in shared])

    np.savez(
        RESULTS / "diff.npz",
        task_ids=np.array(shared),
        prev_mu=prev,
        curr_mu=curr,
        origins=origins,
    )

    # --- per-origin stats ---
    print("\nper-origin Pearson r and counts:")
    rows = []
    for origin in ["wildchat", "alpaca", "math", "bailbench", "stress_test"]:
        mask = origins == origin
        if mask.sum() < 2:
            continue
        r, _ = pearsonr(prev[mask], curr[mask])
        prev_mean = prev[mask].mean()
        curr_mean = curr[mask].mean()
        flips = ((prev[mask] > 0) != (curr[mask] > 0)).sum()
        rows.append((origin, mask.sum(), r, prev_mean, curr_mean, flips, flips / mask.sum()))
        print(f"  {origin:<12} n={mask.sum():>4}  r={r:+.3f}  prev μ̄={prev_mean:+.2f}  curr μ̄={curr_mean:+.2f}  sign flips={flips} ({flips/mask.sum()*100:.0f}%)")

    # --- scatter colored by origin ---
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for origin in ORIGIN_COLORS:
        mask = origins == origin
        if mask.sum() == 0:
            continue
        ax.scatter(prev[mask], curr[mask], s=18, alpha=0.5, color=ORIGIN_COLORS[origin], label=f"{origin} (n={mask.sum()})")
    lims = (-12, 12)
    ax.plot(lims, lims, "k--", alpha=0.4, lw=1, label="y = x")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("μ — reasoning ON (qwen35_10k AL, pre-fix)")
    ax.set_ylabel("μ — reasoning OFF (canonical AL, post-fix)")
    ax.set_title(f"Default-Qwen preference shift by reasoning mode\n"
                 f"{len(shared)} shared tasks, overall Pearson r = {pearsonr(prev, curr)[0]:+.2f}")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_scatter_by_origin.png", dpi=150)
    plt.close(fig)

    # --- per-origin Pearson r bar ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    origins_sorted = [r[0] for r in rows]
    rs = [r[2] for r in rows]
    colors = [ORIGIN_COLORS[o] for o in origins_sorted]
    bars = ax.bar(origins_sorted, rs, color=colors)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Pearson r(prev μ, curr μ)")
    ax.set_ylim(-0.2, 1)
    ax.set_title(f"Within-origin agreement of default-Qwen μ across reasoning regimes")
    for b, r in zip(bars, rs):
        ax.text(b.get_x() + b.get_width()/2, r + 0.02, f"{r:+.2f}", ha="center", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_per_origin_pearson.png", dpi=150)
    plt.close(fig)

    # --- mean μ per origin per run (bar) ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(origins_sorted))
    w = 0.36
    prev_means = [r[3] for r in rows]
    curr_means = [r[4] for r in rows]
    ax.bar(x - w/2, prev_means, w, label="reasoning ON (prev)", color="#9aa0a6")
    ax.bar(x + w/2, curr_means, w, label="reasoning OFF (curr)", color="#1a73e8")
    ax.set_xticks(x); ax.set_xticklabels(origins_sorted)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("mean μ per origin")
    ax.set_title("Average preference per task category — does default-Qwen like or dislike each?")
    for i, (pm, cm) in enumerate(zip(prev_means, curr_means)):
        ax.text(i - w/2, pm + (0.15 if pm > 0 else -0.4), f"{pm:+.1f}", ha="center", fontsize=9)
        ax.text(i + w/2, cm + (0.15 if cm > 0 else -0.4), f"{cm:+.1f}", ha="center", fontsize=9)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_mean_mu_per_origin.png", dpi=150)
    plt.close(fig)

    # --- distribution histograms per origin ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
    bins = np.linspace(-12, 12, 30)
    for ax, origin in zip(axes.flat, list(ORIGIN_COLORS.keys()) + [None]):
        if origin is None:
            ax.axis("off"); continue
        mask = origins == origin
        if mask.sum() == 0:
            ax.set_title(f"{origin} (no tasks)"); continue
        ax.hist(prev[mask], bins=bins, alpha=0.5, color="#9aa0a6", label="reasoning ON")
        ax.hist(curr[mask], bins=bins, alpha=0.5, color=ORIGIN_COLORS[origin], label="reasoning OFF")
        ax.axvline(0, color="black", lw=0.5)
        ax.set_title(f"{origin} (n={mask.sum()})")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.3)
        ax.set_xlabel("μ")
    fig.suptitle("μ distribution per origin — reasoning ON vs OFF", fontsize=12)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_mu_distribution.png", dpi=150)
    plt.close(fig)

    # --- qualitative examples: top 3 disagreement tasks per origin ---
    print("\n=== Loading task prompts for qualitative examples ===")
    supported_origins = [OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH,
                         OriginDataset.BAILBENCH, OriginDataset.STRESS_TEST]
    all_tasks = load_tasks(n=100000, origins=supported_origins)
    by_id = {t.id: t for t in all_tasks}

    qual_lines = []
    for origin in ["bailbench", "math", "wildchat"]:
        mask = origins == origin
        if mask.sum() == 0:
            continue
        diffs = np.abs(prev[mask] - curr[mask])
        top_idx_local = np.argsort(-diffs)[:3]
        idxs_global = np.where(mask)[0][top_idx_local]
        qual_lines.append(f"\n## {origin}\n")
        for gi in idxs_global:
            tid = shared[gi]
            t = by_id.get(tid)
            prompt_snip = (t.prompt[:300] + "…") if t and len(t.prompt) > 300 else (t.prompt if t else "(prompt not found)")
            qual_lines.append(f"- **`{tid}`**: prev μ = {prev[gi]:+.2f}, curr μ = {curr[gi]:+.2f}, Δ = {diffs[top_idx_local[list(idxs_global).index(gi)]]:.2f}\n")
            qual_lines.append(f"  > {prompt_snip}\n")

    (RESULTS / "qualitative_examples.md").write_text("# Top disagreement tasks per origin\n" + "".join(qual_lines))
    print(f"qualitative examples saved to {(RESULTS / 'qualitative_examples.md').relative_to(REPO)}")


if __name__ == "__main__":
    main()
