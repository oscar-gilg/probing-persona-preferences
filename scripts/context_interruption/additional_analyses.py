"""Additional analyses for the context interruption experiment."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DATA_DIR = Path("experiments/context_interruption/data")
ASSETS_DIR = Path("experiments/context_interruption/assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

META = json.loads((DATA_DIR / "scoring_results_meta.json").read_text())["items"]
SCORES = np.load(DATA_DIR / "token_scores.npz")

PROBES = sorted({k.split("__")[1] for k in SCORES.keys()})
PRIMARY_PROBE = "tb-2_L39"


def mean_segment_score(item: dict, probe: str, segment: str) -> float:
    key = f"{item['id']}__{probe}"
    arr = SCORES[key]
    start, end = item["segments"][segment]
    return float(arr[start:end].mean())


# ── Analysis 1: Dose-response (pleasant sessions only) ──────────────────────

def analysis_1_dose_response():
    print("=" * 72)
    print("ANALYSIS 1: Dose-response (pleasant sessions, n=200)")
    print("=" * 72)

    pleasant = [i for i in META if i["session_valence"] == "pleasant"]

    # Primary probe scatter + regression
    task_mus = np.array([i["task_mu"] for i in pleasant])
    int_scores = np.array([mean_segment_score(i, PRIMARY_PROBE, "interruption") for i in pleasant])

    r, p = stats.pearsonr(task_mus, int_scores)
    print(f"\nPrimary probe ({PRIMARY_PROBE}):")
    print(f"  Pearson r = {r:.4f}, p = {p:.2e}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(task_mus, int_scores, alpha=0.4, s=30, edgecolors="none")
    slope, intercept = np.polyfit(task_mus, int_scores, 1)
    x_range = np.linspace(task_mus.min(), task_mus.max(), 100)
    ax.plot(x_range, slope * x_range + intercept, "r-", linewidth=2,
            label=f"r = {r:.3f}, p = {p:.2e}")
    ax.set_xlabel("Task Thurstonian score (task_mu)")
    ax.set_ylabel(f"Mean interruption probe score ({PRIMARY_PROBE})")
    ax.set_title("Dose-response: task utility vs. interruption probe score\n(pleasant sessions only)")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    out_path = ASSETS_DIR / "plot_031826_dose_response.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")

    # All probes table
    print(f"\n  {'Probe':<18} {'r':>8} {'p':>12}")
    print(f"  {'-'*18} {'-'*8} {'-'*12}")
    for probe in PROBES:
        scores = np.array([mean_segment_score(i, probe, "interruption") for i in pleasant])
        r_p, p_p = stats.pearsonr(task_mus, scores)
        print(f"  {probe:<18} {r_p:>8.4f} {p_p:>12.2e}")


# ── Analysis 2: Cross-probe consistency ──────────────────────────────────────

def analysis_2_cross_probe():
    print("\n" + "=" * 72)
    print("ANALYSIS 2: Cross-probe consistency")
    print("=" * 72)

    # Session-valence effect: pleasant vs unpleasant on interruption segment
    pleasant = [i for i in META if i["session_valence"] == "pleasant"]
    unpleasant = [i for i in META if i["session_valence"] == "unpleasant"]

    # Task-switch offered-valence effect
    ts_offered_p = [i for i in META if i["prompt_type"] == "task_switch" and i.get("offered_valence") == "pleasant"]
    ts_offered_u = [i for i in META if i["prompt_type"] == "task_switch" and i.get("offered_valence") == "unpleasant"]

    header = (f"  {'Probe':<18} {'sess_val_eff':>12} {'p':>10} "
              f"{'offered_eff':>12} {'p':>10}")
    print(f"\n  Session-valence: pleasant (n={len(pleasant)}) vs unpleasant (n={len(unpleasant)})")
    print(f"  Task-switch offered: pleasant (n={len(ts_offered_p)}) vs unpleasant (n={len(ts_offered_u)})")
    print()
    print(header)
    print(f"  {'-'*18} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")

    for probe in PROBES:
        p_scores = np.array([mean_segment_score(i, probe, "interruption") for i in pleasant])
        u_scores = np.array([mean_segment_score(i, probe, "interruption") for i in unpleasant])
        t1, p1 = stats.ttest_ind(p_scores, u_scores)
        eff1 = p_scores.mean() - u_scores.mean()

        op_scores = np.array([mean_segment_score(i, probe, "interruption") for i in ts_offered_p])
        ou_scores = np.array([mean_segment_score(i, probe, "interruption") for i in ts_offered_u])
        t2, p2 = stats.ttest_ind(op_scores, ou_scores)
        eff2 = op_scores.mean() - ou_scores.mean()

        print(f"  {probe:<18} {eff1:>12.4f} {p1:>10.2e} {eff2:>12.4f} {p2:>10.2e}")


# ── Analysis 3: Generation prompt per-prompt-type significance ───────────────

def analysis_3_generation_prompt():
    print("\n" + "=" * 72)
    print("ANALYSIS 3: Generation prompt segment — per-prompt-type significance")
    print(f"  Probe: {PRIMARY_PROBE}")
    print("=" * 72)

    for pt in ["context_exhaustion", "conversation_terminal"]:
        items_p = [i for i in META if i["prompt_type"] == pt and i["session_valence"] == "pleasant"]
        items_u = [i for i in META if i["prompt_type"] == pt and i["session_valence"] == "unpleasant"]

        scores_p = np.array([mean_segment_score(i, PRIMARY_PROBE, "generation_prompt") for i in items_p])
        scores_u = np.array([mean_segment_score(i, PRIMARY_PROBE, "generation_prompt") for i in items_u])

        t_stat, p_val = stats.ttest_ind(scores_p, scores_u)
        diff = scores_p.mean() - scores_u.mean()
        se = np.sqrt(scores_p.var(ddof=1) / len(scores_p) + scores_u.var(ddof=1) / len(scores_u))
        ci_lo = diff - 1.96 * se
        ci_hi = diff + 1.96 * se

        print(f"\n  {pt}:")
        print(f"    Pleasant:   n={len(scores_p):>3}, mean={scores_p.mean():.4f} (sd={scores_p.std(ddof=1):.4f})")
        print(f"    Unpleasant: n={len(scores_u):>3}, mean={scores_u.mean():.4f} (sd={scores_u.std(ddof=1):.4f})")
        print(f"    Difference: {diff:+.4f}  [95% CI: {ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"    t = {t_stat:.3f}, p = {p_val:.2e}")


# ── Analysis 4: Task-switch session-valence main effect ──────────────────────

def analysis_4_task_switch_session():
    print("\n" + "=" * 72)
    print("ANALYSIS 4: Task-switch session-valence main effect")
    print(f"  Probe: {PRIMARY_PROBE}")
    print("=" * 72)

    ts = [i for i in META if i["prompt_type"] == "task_switch"]
    ts_pleasant = [i for i in ts if i["session_valence"] == "pleasant"]
    ts_unpleasant = [i for i in ts if i["session_valence"] == "unpleasant"]

    scores_p = np.array([mean_segment_score(i, PRIMARY_PROBE, "interruption") for i in ts_pleasant])
    scores_u = np.array([mean_segment_score(i, PRIMARY_PROBE, "interruption") for i in ts_unpleasant])

    t_stat, p_val = stats.ttest_ind(scores_p, scores_u)
    diff = scores_p.mean() - scores_u.mean()
    pooled_std = np.sqrt(
        ((len(scores_p) - 1) * scores_p.var(ddof=1) + (len(scores_u) - 1) * scores_u.var(ddof=1))
        / (len(scores_p) + len(scores_u) - 2)
    )
    cohens_d = diff / pooled_std

    print(f"\n  Pleasant sessions:   n={len(scores_p):>3}, mean={scores_p.mean():.4f} (sd={scores_p.std(ddof=1):.4f})")
    print(f"  Unpleasant sessions: n={len(scores_u):>3}, mean={scores_u.mean():.4f} (sd={scores_u.std(ddof=1):.4f})")
    print(f"  Difference: {diff:+.4f}")
    print(f"  t = {t_stat:.3f}, p = {p_val:.2e}")
    print(f"  Cohen's d = {cohens_d:.3f}")


if __name__ == "__main__":
    analysis_1_dose_response()
    analysis_2_cross_probe()
    analysis_3_generation_prompt()
    analysis_4_task_switch_session()
    print()
