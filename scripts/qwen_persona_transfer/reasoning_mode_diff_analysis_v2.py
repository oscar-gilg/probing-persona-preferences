"""Comprehensive analysis of how default-Qwen preferences differ between
the reasoning-ON and reasoning-OFF regimes, using the FULL 6000-task overlap
(4000 train + 1000 eval + 1000 test, all canonical splits).

Both runs use the same canonical_splits task IDs, so the overlap is the full 6000
rather than the 783 in v1 (which compared a 10k ad-hoc pool against 4k canonical).

Outputs:
  - results/diff_v2.npz              full per-task μ_no_think, μ_think, origins
  - results/persona_diff.npz         per-persona Pearson r between regimes (bonus)
  - assets/plot_<date>_v2_*.png      refined figures
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.probes.data_loading import load_thurstonian_scores
from src.task_data.loader import load_tasks
from src.task_data.task import OriginDataset

REPO = Path(__file__).resolve().parents[2]
NO_THINK_AL = REPO / "results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning"
THINK_AL = REPO / "results/experiments/qwen_persona_sweep_thinking_final_six/pre_task_active_learning"

OUT = REPO / "experiments/qwen_replication/persona_transfer/reasoning_mode_diff"
RESULTS = OUT / "results"
ASSETS = OUT / "assets"
TODAY = datetime.now().strftime("%m%d%y")

PERSONAS = ["default", "sadist", "mathematician", "aura", "strategist", "contrarian", "slacker"]
SPLITS = ["train", "eval", "test"]
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
    if tid.startswith("stresstest_"):
        return "stress_test"
    for tag in ("wildchat", "alpaca", "bailbench"):
        if tid.startswith(tag + "_"):
            return tag
    return "other"


def load_persona_all_splits(al_dir: Path, persona: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for split in SPLITS:
        d = al_dir / f"{persona}_{split}"
        if not d.exists():
            print(f"  missing dir: {d.relative_to(REPO)}")
            continue
        try:
            scores = load_thurstonian_scores(d)
        except FileNotFoundError:
            print(f"  no utilities: {d.relative_to(REPO)}")
            continue
        out.update(scores)
    return out


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    # ---------- Default thinking vs no-think on full 6000 tasks ----------
    no_think = load_persona_all_splits(NO_THINK_AL, "default")
    think = load_persona_all_splits(THINK_AL, "default")
    shared = sorted(set(no_think) & set(think))
    print(f"\ndefault: no_think={len(no_think)}, think={len(think)}, shared={len(shared)}")

    nt = np.array([no_think[t] for t in shared])
    th = np.array([think[t] for t in shared])
    origins = np.array([origin_from_id(t) for t in shared])

    np.savez(
        RESULTS / "diff_v2.npz",
        task_ids=np.array(shared),
        no_think_mu=nt,
        think_mu=th,
        origins=origins,
    )

    overall_pearson = pearsonr(nt, th)[0]
    overall_spearman = spearmanr(nt, th)[0]
    print(f"\noverall: Pearson r = {overall_pearson:+.3f}, Spearman ρ = {overall_spearman:+.3f}")

    # ---------- Per-origin stats ----------
    print("\nper-origin stats:")
    rows = []
    for origin in ["wildchat", "alpaca", "math", "bailbench", "stress_test"]:
        mask = origins == origin
        if mask.sum() < 2:
            continue
        r, _ = pearsonr(nt[mask], th[mask])
        rho, _ = spearmanr(nt[mask], th[mask])
        nt_mean = nt[mask].mean(); th_mean = th[mask].mean()
        nt_std = nt[mask].std(); th_std = th[mask].std()
        flips = ((nt[mask] > 0) != (th[mask] > 0)).sum()
        rows.append((origin, mask.sum(), r, rho, nt_mean, th_mean, nt_std, th_std, flips, flips / mask.sum()))
        print(f"  {origin:<12} n={mask.sum():>5}  r={r:+.3f}  ρ={rho:+.3f}  "
              f"μ̄_no={nt_mean:+.2f}  μ̄_th={th_mean:+.2f}  flip={flips/mask.sum()*100:.0f}%")

    # ---------- Per-persona thinking vs no-think (bonus) ----------
    print("\nper-persona Pearson r between regimes (bonus):")
    persona_rows = []
    for p in PERSONAS:
        nt_p = load_persona_all_splits(NO_THINK_AL, p)
        th_p = load_persona_all_splits(THINK_AL, p)
        sh = sorted(set(nt_p) & set(th_p))
        if not sh:
            print(f"  {p}: no overlap"); continue
        a = np.array([nt_p[t] for t in sh])
        b = np.array([th_p[t] for t in sh])
        r = pearsonr(a, b)[0]; rho = spearmanr(a, b)[0]
        # also compute per-origin
        per_orig = {}
        sh_orig = np.array([origin_from_id(t) for t in sh])
        for o in ["wildchat", "alpaca", "math", "bailbench", "stress_test"]:
            m = sh_orig == o
            if m.sum() < 5:
                per_orig[o] = float("nan")
            else:
                per_orig[o] = pearsonr(a[m], b[m])[0]
        persona_rows.append((p, len(sh), r, rho, per_orig))
        print(f"  {p:<14} n={len(sh):>5}  overall r={r:+.3f}  ρ={rho:+.3f}  "
              + "  ".join(f"{o[:3]}={per_orig[o]:+.2f}" for o in ["wildchat", "alpaca", "math", "bailbench", "stress_test"]))

    np.savez(
        RESULTS / "persona_diff.npz",
        personas=np.array([r[0] for r in persona_rows]),
        n=np.array([r[1] for r in persona_rows]),
        pearson=np.array([r[2] for r in persona_rows]),
        spearman=np.array([r[3] for r in persona_rows]),
    )

    # =========================================================
    # Figures
    # =========================================================

    # ---------- Figure 1: scatter colored by origin ----------
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for origin in ORIGIN_COLORS:
        m = origins == origin
        if m.sum() == 0: continue
        ax.scatter(nt[m], th[m], s=10, alpha=0.4, color=ORIGIN_COLORS[origin],
                   label=f"{origin} (n={m.sum()})")
    lims = (-12, 12)
    ax.plot(lims, lims, "k--", alpha=0.4, lw=1, label="y = x")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)
    ax.set_xlabel("μ — reasoning OFF")
    ax.set_ylabel("μ — reasoning ON")
    ax.set_title(f"Default-Qwen preference shift by reasoning mode\n"
                 f"{len(shared)} shared tasks (full canonical 6000),  "
                 f"Pearson r = {overall_pearson:+.2f},  Spearman ρ = {overall_spearman:+.2f}")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_v2_scatter_by_origin.png", dpi=150)
    plt.close(fig)

    # ---------- Figure 2: per-origin Pearson + Spearman side-by-side ----------
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    origins_sorted = [r[0] for r in rows]
    rs = [r[2] for r in rows]
    rhos = [r[3] for r in rows]
    x = np.arange(len(origins_sorted))
    w = 0.36
    ax.bar(x - w/2, rs, w, label="Pearson r", color="#4c8cd6")
    ax.bar(x + w/2, rhos, w, label="Spearman ρ", color="#d68c4c")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(origins_sorted)
    ax.set_ylabel("correlation")
    ax.set_ylim(-0.2, 1)
    ax.set_title("Within-origin agreement: reasoning ON vs OFF")
    for i, (rr, rho) in enumerate(zip(rs, rhos)):
        ax.text(i - w/2, rr + 0.02, f"{rr:+.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, rho + 0.02, f"{rho:+.2f}", ha="center", fontsize=9)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_v2_per_origin_correlation.png", dpi=150)
    plt.close(fig)

    # ---------- Figure 3: mean μ + std per origin per regime ----------
    fig, ax = plt.subplots(figsize=(9.5, 5))
    nt_means = [r[4] for r in rows]; th_means = [r[5] for r in rows]
    nt_stds = [r[6] for r in rows]; th_stds = [r[7] for r in rows]
    x = np.arange(len(origins_sorted))
    w = 0.36
    ax.bar(x - w/2, nt_means, w, yerr=nt_stds, label="reasoning OFF", color="#9aa0a6", capsize=4)
    ax.bar(x + w/2, th_means, w, yerr=th_stds, label="reasoning ON", color="#1a73e8", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(origins_sorted)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("mean μ ± std per origin")
    ax.set_title("Mean preference per task category — does default-Qwen like or dislike each?")
    for i, (nm, tm) in enumerate(zip(nt_means, th_means)):
        ax.text(i - w/2, nm + (0.4 if nm > 0 else -0.6), f"{nm:+.1f}", ha="center", fontsize=9)
        ax.text(i + w/2, tm + (0.4 if tm > 0 else -0.6), f"{tm:+.1f}", ha="center", fontsize=9)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_v2_mean_mu_per_origin.png", dpi=150)
    plt.close(fig)

    # ---------- Figure 4: per-origin distribution histograms ----------
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
    bins = np.linspace(-12, 12, 30)
    for ax, origin in zip(axes.flat, list(ORIGIN_COLORS.keys()) + [None]):
        if origin is None:
            ax.axis("off"); continue
        m = origins == origin
        if m.sum() == 0:
            ax.set_title(f"{origin} (no tasks)"); continue
        ax.hist(nt[m], bins=bins, alpha=0.55, color="#9aa0a6", label="reasoning OFF")
        ax.hist(th[m], bins=bins, alpha=0.55, color=ORIGIN_COLORS[origin], label="reasoning ON")
        ax.axvline(0, color="black", lw=0.5)
        ax.set_title(f"{origin} (n={m.sum()})")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.3)
        ax.set_xlabel("μ")
    fig.suptitle("μ distribution per origin — full 6000-task overlap", fontsize=12)
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_v2_mu_distribution.png", dpi=150)
    plt.close(fig)

    # ---------- Figure 5: per-persona Pearson r heatmap (origin × persona) ----------
    pers_names = [r[0] for r in persona_rows]
    origins_in_use = ["wildchat", "alpaca", "math", "bailbench", "stress_test"]
    M = np.array([[r[4][o] for o in origins_in_use] for r in persona_rows])
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
    ax.set_xticks(range(len(origins_in_use))); ax.set_xticklabels(origins_in_use)
    ax.set_yticks(range(len(pers_names))); ax.set_yticklabels(pers_names)
    for i in range(len(pers_names)):
        for j in range(len(origins_in_use)):
            v = M[i, j]
            txt = f"{v:+.2f}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.25 else "black")
    fig.colorbar(im, ax=ax, label="within-origin Pearson r")
    ax.set_title("Per-persona, per-origin agreement (reasoning ON vs OFF)\nCloser to 1 = same preferences across regimes")
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_v2_per_persona_per_origin_pearson.png", dpi=150)
    plt.close(fig)

    # ---------- Figure 6: |μ_th − μ_nt| distribution per origin (size of shift) ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    delta_data = []
    delta_labels = []
    for origin in origins_in_use:
        m = origins == origin
        if m.sum() == 0: continue
        delta_data.append(np.abs(th[m] - nt[m]))
        delta_labels.append(f"{origin}\n(n={m.sum()})")
    bp = ax.boxplot(delta_data, labels=delta_labels, patch_artist=True, showfliers=False)
    for patch, origin in zip(bp["boxes"], origins_in_use):
        patch.set_facecolor(ORIGIN_COLORS[origin])
        patch.set_alpha(0.6)
    ax.set_ylabel("|μ_ON − μ_OFF|")
    ax.set_title("Magnitude of preference shift per task, by origin")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(ASSETS / f"plot_{TODAY}_v2_shift_magnitude.png", dpi=150)
    plt.close(fig)

    # ---------- Qualitative examples: top 5 per origin ----------
    print("\n=== Loading task prompts for qualitative examples ===")
    supported = [OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH,
                 OriginDataset.BAILBENCH, OriginDataset.STRESS_TEST]
    all_tasks = load_tasks(n=100000, origins=supported)
    by_id = {t.id: t for t in all_tasks}

    qual_lines = ["# Top disagreement tasks per origin (full 6000-task overlap)\n",
                  "Top 5 per origin by |Δμ|. Δμ = μ_ON − μ_OFF.\n"]
    for origin in ["bailbench", "math", "wildchat", "alpaca", "stress_test"]:
        m = origins == origin
        if m.sum() == 0: continue
        idx_global = np.where(m)[0]
        diffs = th[idx_global] - nt[idx_global]
        absdiffs = np.abs(diffs)
        top_local = np.argsort(-absdiffs)[:5]
        qual_lines.append(f"\n## {origin}\n\n")
        for li in top_local:
            gi = idx_global[li]
            tid = shared[gi]
            t = by_id.get(tid)
            prompt_snip = (t.prompt[:300] + "…") if t and len(t.prompt) > 300 else (t.prompt if t else "(prompt not found)")
            qual_lines.append(
                f"- **`{tid}`**: μ_OFF = {nt[gi]:+.2f}, μ_ON = {th[gi]:+.2f}, Δ = {th[gi]-nt[gi]:+.2f}\n"
                f"  > {prompt_snip}\n\n"
            )
    (RESULTS / "qualitative_examples_v2.md").write_text("".join(qual_lines))
    print(f"qualitative examples saved to {(RESULTS / 'qualitative_examples_v2.md').relative_to(REPO)}")

    # Also save a JSON summary so the report can pull numbers programmatically
    summary = {
        "n_shared": len(shared),
        "overall_pearson": float(overall_pearson),
        "overall_spearman": float(overall_spearman),
        "per_origin": [
            {
                "origin": r[0], "n": int(r[1]),
                "pearson": float(r[2]), "spearman": float(r[3]),
                "mean_no_think": float(r[4]), "mean_think": float(r[5]),
                "std_no_think": float(r[6]), "std_think": float(r[7]),
                "sign_flip_rate": float(r[9]),
            } for r in rows
        ],
        "per_persona": [
            {
                "persona": r[0], "n": int(r[1]),
                "pearson": float(r[2]), "spearman": float(r[3]),
                "per_origin_pearson": {o: float(r[4][o]) for o in ["wildchat", "alpaca", "math", "bailbench", "stress_test"]},
            } for r in persona_rows
        ],
    }
    with open(RESULTS / "summary_v2.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary written to {(RESULTS / 'summary_v2.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
