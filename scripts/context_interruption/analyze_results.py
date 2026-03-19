"""Phase 3 analysis for Context Interruption experiment.

Loads scoring metadata and token-level probe scores, produces:
1. Bar chart of mean interruption-segment scores by session_valence x prompt_type
2. Bar chart of mean generation_prompt-segment scores by session_valence x prompt_type
3. 2x2 interaction plots for reassignment, task_switch, choice
4. Trajectory plot across full conversation by session_valence
5. Summary statistics printed to stdout
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "experiments" / "context_interruption" / "data"
ASSETS_DIR = ROOT / "experiments" / "context_interruption" / "assets"

META_PATH = DATA_DIR / "scoring_results_meta.json"
SCORES_PATH = DATA_DIR / "token_scores.npz"

PRIMARY_PROBE = "tb-2_L39"

# Colors (colorblind-friendly)
VALENCE_COLORS = {
    "pleasant": "#4477AA",
    "unpleasant": "#CC6677",
    "control": "#999999",
}

OFFERED_COLORS = {
    "pleasant": "#4477AA",
    "unpleasant": "#CC6677",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(META_PATH) as f:
    meta = json.load(f)
items = meta["items"]

scores_npz = np.load(SCORES_PATH)

print(f"Loaded {len(items)} items, {len(scores_npz.keys())} score arrays")
print(f"Primary probe: {PRIMARY_PROBE}")
print()


def get_scores(item_id: str, probe: str = PRIMARY_PROBE) -> np.ndarray:
    key = f"{item_id}__{probe}"
    return scores_npz[key]


def segment_mean(item: dict, segment_name: str, probe: str = PRIMARY_PROBE) -> float:
    scores = get_scores(item["id"], probe)
    start, end = item["segments"][segment_name]
    return float(np.mean(scores[start:end]))


def sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def bootstrap_ci(values: list[float], n_boot: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    arr = np.array(values)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))


# ===========================================================================
# Analysis 1: Interruption segment scores by session_valence x prompt_type
# ===========================================================================
print("=" * 70)
print("ANALYSIS 1: Mean probe score on INTERRUPTION segment")
print("=" * 70)

prompt_types = ["reassignment", "task_switch", "choice", "context_exhaustion", "conversation_terminal"]
session_valences = ["pleasant", "unpleasant", "control"]

# Collect scores
interruption_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
for item in items:
    score = segment_mean(item, "interruption")
    interruption_scores[(item["session_valence"], item["prompt_type"])].append(score)

# Print table
print(f"\n{'prompt_type':<25} {'valence':<15} {'mean':>8} {'sem':>8} {'n':>5}")
print("-" * 65)
for pt in prompt_types:
    for sv in session_valences:
        key = (sv, pt)
        vals = interruption_scores[key]
        if vals:
            print(f"{pt:<25} {sv:<15} {np.mean(vals):>8.4f} {sem(vals):>8.4f} {len(vals):>5}")

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(prompt_types))
width = 0.25
for i, sv in enumerate(session_valences):
    means = []
    sems = []
    for pt in prompt_types:
        vals = interruption_scores[(sv, pt)]
        if vals:
            means.append(np.mean(vals))
            sems.append(sem(vals))
        else:
            means.append(0)
            sems.append(0)
    offset = (i - 1) * width
    bars = ax.bar(x + offset, means, width, yerr=sems, capsize=4,
                  label=sv, color=VALENCE_COLORS[sv], alpha=0.85)

ax.set_xlabel("Prompt type")
ax.set_ylabel("Mean probe score (tb-2 L39)")
ax.set_title("Mean probe score on interruption tokens by session valence and prompt type")
ax.set_xticks(x)
ax.set_xticklabels(prompt_types, rotation=15, ha="right")
ax.legend(title="Session valence")
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
# Set y-axis to include 0
ylim = ax.get_ylim()
ax.set_ylim(min(ylim[0], -0.1), max(ylim[1], 0.1))
plt.tight_layout()
fig.savefig(ASSETS_DIR / "plot_031826_interruption_scores_by_valence_prompt.png", dpi=150)
plt.close(fig)
print(f"\nSaved: plot_031826_interruption_scores_by_valence_prompt.png")

# ===========================================================================
# Analysis 2: Generation prompt segment scores by session_valence x prompt_type
# ===========================================================================
print("\n" + "=" * 70)
print("ANALYSIS 2: Mean probe score on GENERATION_PROMPT segment")
print("=" * 70)

genprompt_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
for item in items:
    score = segment_mean(item, "generation_prompt")
    genprompt_scores[(item["session_valence"], item["prompt_type"])].append(score)

print(f"\n{'prompt_type':<25} {'valence':<15} {'mean':>8} {'sem':>8} {'n':>5}")
print("-" * 65)
for pt in prompt_types:
    for sv in session_valences:
        key = (sv, pt)
        vals = genprompt_scores[key]
        if vals:
            print(f"{pt:<25} {sv:<15} {np.mean(vals):>8.4f} {sem(vals):>8.4f} {len(vals):>5}")

fig, ax = plt.subplots(figsize=(12, 6))
for i, sv in enumerate(session_valences):
    means = []
    sems = []
    for pt in prompt_types:
        vals = genprompt_scores[(sv, pt)]
        if vals:
            means.append(np.mean(vals))
            sems.append(sem(vals))
        else:
            means.append(0)
            sems.append(0)
    offset = (i - 1) * width
    ax.bar(x + offset, means, width, yerr=sems, capsize=4,
           label=sv, color=VALENCE_COLORS[sv], alpha=0.85)

ax.set_xlabel("Prompt type")
ax.set_ylabel("Mean probe score (tb-2 L39)")
ax.set_title("Mean probe score on generation_prompt tokens by session valence and prompt type")
ax.set_xticks(x)
ax.set_xticklabels(prompt_types, rotation=15, ha="right")
ax.legend(title="Session valence")
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ylim = ax.get_ylim()
ax.set_ylim(min(ylim[0], -0.1), max(ylim[1], 0.1))
plt.tight_layout()
fig.savefig(ASSETS_DIR / "plot_031826_generation_prompt_scores_by_valence_prompt.png", dpi=150)
plt.close(fig)
print(f"\nSaved: plot_031826_generation_prompt_scores_by_valence_prompt.png")

# ===========================================================================
# Analysis 3: 2x2 interaction plots for reassignment, task_switch, choice
# ===========================================================================
print("\n" + "=" * 70)
print("ANALYSIS 3: 2x2 interaction plots")
print("=" * 70)

for pt in ["reassignment", "task_switch", "choice"]:
    print(f"\n--- {pt} ---")

    # Collect by (session_valence, offered_valence)
    interaction_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    control_scores_pt: list[float] = []

    for item in items:
        if item["prompt_type"] != pt:
            continue
        score = segment_mean(item, "interruption")
        if item["session_valence"] == "control":
            control_scores_pt.append(score)
        elif item["session_valence"] in ("pleasant", "unpleasant") and item["offered_valence"] in ("pleasant", "unpleasant"):
            interaction_scores[(item["session_valence"], item["offered_valence"])].append(score)

    # Print
    for sv in ["pleasant", "unpleasant"]:
        for ov in ["pleasant", "unpleasant"]:
            vals = interaction_scores[(sv, ov)]
            print(f"  session={sv}, offered={ov}: mean={np.mean(vals):.4f}, sem={sem(vals):.4f}, n={len(vals)}")
    if control_scores_pt:
        print(f"  control: mean={np.mean(control_scores_pt):.4f}, sem={sem(control_scores_pt):.4f}, n={len(control_scores_pt)}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    x_positions = [0, 1]
    x_labels = ["pleasant", "unpleasant"]

    for ov in ["pleasant", "unpleasant"]:
        means = []
        sems_vals = []
        for sv in ["pleasant", "unpleasant"]:
            vals = interaction_scores[(sv, ov)]
            means.append(np.mean(vals))
            sems_vals.append(sem(vals))
        ax.errorbar(x_positions, means, yerr=sems_vals, marker="o", markersize=8,
                     capsize=5, linewidth=2, label=f"offered {ov}",
                     color=OFFERED_COLORS[ov])

    # Control reference line
    if control_scores_pt:
        ctrl_mean = np.mean(control_scores_pt)
        ax.axhline(ctrl_mean, color=VALENCE_COLORS["control"], linewidth=1.5,
                    linestyle="--", label=f"control (n={len(control_scores_pt)})")

    ax.set_xlabel("Session valence")
    ax.set_ylabel("Mean probe score (tb-2 L39)")
    ax.set_title(f"Interaction: {pt} — interruption segment")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.3, linestyle=":")
    ylim = ax.get_ylim()
    ax.set_ylim(min(ylim[0], -0.1), max(ylim[1], 0.1))
    plt.tight_layout()
    fig.savefig(ASSETS_DIR / f"plot_031826_interaction_{pt}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: plot_031826_interaction_{pt}.png")

# ===========================================================================
# Analysis 4: Trajectory plots by session valence
# ===========================================================================
print("\n" + "=" * 70)
print("ANALYSIS 4: Probe score trajectory across conversation by session valence")
print("=" * 70)

SEGMENT_ORDER = ["user_1", "assistant_1", "user_2", "assistant_2", "interruption", "generation_prompt"]
N_BINS_PER_SEGMENT = 50  # resolution within each segment

# For each item, resample each segment to N_BINS_PER_SEGMENT bins, then concatenate
total_bins = len(SEGMENT_ORDER) * N_BINS_PER_SEGMENT

trajectories_by_valence: dict[str, list[np.ndarray]] = defaultdict(list)

for item in items:
    scores = get_scores(item["id"])
    resampled_segments = []

    for seg_name in SEGMENT_ORDER:
        start, end = item["segments"][seg_name]
        seg_scores = scores[start:end]
        if len(seg_scores) == 0:
            resampled_segments.append(np.full(N_BINS_PER_SEGMENT, np.nan))
            continue
        # Resample to N_BINS_PER_SEGMENT via linear interpolation
        original_positions = np.linspace(0, 1, len(seg_scores))
        target_positions = np.linspace(0, 1, N_BINS_PER_SEGMENT)
        resampled = np.interp(target_positions, original_positions, seg_scores)
        resampled_segments.append(resampled)

    trajectory = np.concatenate(resampled_segments)
    trajectories_by_valence[item["session_valence"]].append(trajectory)

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
x_axis = np.linspace(0, 1, total_bins)

for sv in ["pleasant", "unpleasant", "control"]:
    trajs = np.array(trajectories_by_valence[sv])
    mean_traj = np.nanmean(trajs, axis=0)
    sem_traj = np.nanstd(trajs, axis=0, ddof=1) / np.sqrt(trajs.shape[0])

    ax.plot(x_axis, mean_traj, color=VALENCE_COLORS[sv], linewidth=1.5,
            label=f"{sv} (n={trajs.shape[0]})")
    ax.fill_between(x_axis, mean_traj - sem_traj, mean_traj + sem_traj,
                     color=VALENCE_COLORS[sv], alpha=0.2)

# Segment boundaries
for i in range(1, len(SEGMENT_ORDER)):
    boundary = i * N_BINS_PER_SEGMENT / total_bins
    ax.axvline(boundary, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)

# Label segments at their midpoints
for i, seg_name in enumerate(SEGMENT_ORDER):
    mid = (i + 0.5) * N_BINS_PER_SEGMENT / total_bins
    label = seg_name.replace("_", "\n")
    ax.text(mid, ax.get_ylim()[0], label, ha="center", va="top", fontsize=8,
            color="gray", transform=ax.get_xaxis_transform())

ax.set_xlabel("Normalized position in conversation")
ax.set_ylabel("Mean probe score (tb-2 L39)")
ax.set_title("Probe score trajectory across conversation by session valence")
ax.legend(loc="upper right")
ax.set_xlim(0, 1)
# Set segment labels on x-axis: hide numeric ticks
ax.set_xticks([])
plt.tight_layout()
fig.savefig(ASSETS_DIR / "plot_031826_trajectory_by_valence.png", dpi=150)
plt.close(fig)
print("Saved: plot_031826_trajectory_by_valence.png")

# ===========================================================================
# Analysis 5: Summary statistics
# ===========================================================================
print("\n" + "=" * 70)
print("ANALYSIS 5: Summary statistics")
print("=" * 70)

# Overall mean by session_valence (across all prompt types)
print("\n--- Overall mean interruption score by session_valence ---")
overall_by_valence: dict[str, list[float]] = defaultdict(list)
genprompt_by_valence: dict[str, list[float]] = defaultdict(list)

for item in items:
    overall_by_valence[item["session_valence"]].append(segment_mean(item, "interruption"))
    genprompt_by_valence[item["session_valence"]].append(segment_mean(item, "generation_prompt"))

for sv in session_valences:
    vals = overall_by_valence[sv]
    print(f"  {sv}: mean={np.mean(vals):.4f}, sem={sem(vals):.4f}, n={len(vals)}")

# Effect size: pleasant - unpleasant
print("\n--- Effect size (pleasant - unpleasant) on INTERRUPTION segment ---")
pleasant_vals = overall_by_valence["pleasant"]
unpleasant_vals = overall_by_valence["unpleasant"]
diff = np.mean(pleasant_vals) - np.mean(unpleasant_vals)
# Bootstrap CI on the difference
rng = np.random.default_rng(42)
n_boot = 10000
boot_diffs = []
p_arr = np.array(pleasant_vals)
u_arr = np.array(unpleasant_vals)
for _ in range(n_boot):
    p_sample = rng.choice(p_arr, size=len(p_arr), replace=True)
    u_sample = rng.choice(u_arr, size=len(u_arr), replace=True)
    boot_diffs.append(p_sample.mean() - u_sample.mean())
boot_diffs = np.array(boot_diffs)
ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
print(f"  Diff (pleasant - unpleasant): {diff:.4f}")
print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

# T-test
t_stat, p_val = stats.ttest_ind(pleasant_vals, unpleasant_vals)
print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")

# Cohen's d
pooled_std = np.sqrt(
    ((len(pleasant_vals) - 1) * np.var(pleasant_vals, ddof=1) +
     (len(unpleasant_vals) - 1) * np.var(unpleasant_vals, ddof=1)) /
    (len(pleasant_vals) + len(unpleasant_vals) - 2)
)
cohens_d = diff / pooled_std if pooled_std > 0 else 0
print(f"  Cohen's d: {cohens_d:.3f}")

# Same for generation_prompt
print("\n--- Effect size (pleasant - unpleasant) on GENERATION_PROMPT segment ---")
gp_pleasant = genprompt_by_valence["pleasant"]
gp_unpleasant = genprompt_by_valence["unpleasant"]
gp_diff = np.mean(gp_pleasant) - np.mean(gp_unpleasant)
gp_boot_diffs = []
gp_p_arr = np.array(gp_pleasant)
gp_u_arr = np.array(gp_unpleasant)
for _ in range(n_boot):
    p_sample = rng.choice(gp_p_arr, size=len(gp_p_arr), replace=True)
    u_sample = rng.choice(gp_u_arr, size=len(gp_u_arr), replace=True)
    gp_boot_diffs.append(p_sample.mean() - u_sample.mean())
gp_boot_diffs = np.array(gp_boot_diffs)
gp_ci_low, gp_ci_high = np.percentile(gp_boot_diffs, [2.5, 97.5])
print(f"  Diff (pleasant - unpleasant): {gp_diff:.4f}")
print(f"  95% CI: [{gp_ci_low:.4f}, {gp_ci_high:.4f}]")

gp_t_stat, gp_p_val = stats.ttest_ind(gp_pleasant, gp_unpleasant)
print(f"  t-test: t={gp_t_stat:.3f}, p={gp_p_val:.4f}")

gp_pooled_std = np.sqrt(
    ((len(gp_pleasant) - 1) * np.var(gp_pleasant, ddof=1) +
     (len(gp_unpleasant) - 1) * np.var(gp_unpleasant, ddof=1)) /
    (len(gp_pleasant) + len(gp_unpleasant) - 2)
)
gp_cohens_d = gp_diff / gp_pooled_std if gp_pooled_std > 0 else 0
print(f"  Cohen's d: {gp_cohens_d:.3f}")

# Control comparison
print("\n--- Control comparisons ---")
for seg_name, by_valence in [("interruption", overall_by_valence), ("generation_prompt", genprompt_by_valence)]:
    ctrl = by_valence["control"]
    if ctrl:
        print(f"\n  {seg_name}:")
        print(f"    control: mean={np.mean(ctrl):.4f}, sem={sem(ctrl):.4f}, n={len(ctrl)}")
        for sv in ["pleasant", "unpleasant"]:
            vals = by_valence[sv]
            t, p = stats.ttest_ind(vals, ctrl)
            print(f"    {sv} vs control: t={t:.3f}, p={p:.4f}")

print("\n" + "=" * 70)
print("All plots saved to experiments/context_interruption/assets/")
print("=" * 70)
