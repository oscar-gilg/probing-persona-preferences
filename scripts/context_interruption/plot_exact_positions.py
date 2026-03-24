"""Plot probe scores at exact trained token positions (not segment averages).

tb-2 probe evaluated at position turn_boundary - 2
tb-5 probe evaluated at position turn_boundary - 5

Produces two figures:
1. Interruption scores by session valence and prompt type (1x2: tb-2 | tb-5)
2. Interaction plot for task_switch (1x2: tb-2 | tb-5)
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "experiments" / "context_interruption" / "data"
ASSETS_DIR = ROOT / "experiments" / "context_interruption" / "assets"

META_PATH = DATA_DIR / "scoring_results_meta.json"
# NPZ lives in the worktree
SCORES_PATH = ROOT / ".claude" / "worktrees" / "context_interruption" / "experiments" / "context_interruption" / "data" / "token_scores.npz"

PROBES = [("tb-2_L39", 2), ("tb-5_L39", 5)]
PROBE_LABELS = {
    "tb-2_L39": "'model' token (sot+1)",
    "tb-5_L39": "<end_of_turn> token",
}
PROMPT_TYPES = ["task_switch", "reassignment", "choice", "context_exhaustion", "conversation_terminal"]
SESSION_VALENCES = ["pleasant", "unpleasant", "control"]

VALENCE_COLORS = {
    "pleasant": "#4477AA",
    "unpleasant": "#CC6677",
    "control": "#999999",
}
OFFERED_COLORS = {
    "pleasant": "#4477AA",
    "unpleasant": "#CC6677",
}

# Load data
with open(META_PATH) as f:
    items = json.load(f)["items"]
scores_npz = np.load(SCORES_PATH)
print(f"Loaded {len(items)} items")


def exact_tb_score(item: dict, probe: str, offset: int) -> float:
    first_completion = item["segments"]["generation_prompt"][1]
    pos = first_completion - offset
    arr = scores_npz[f"{item['id']}__{probe}"]
    return float(arr[pos])


def sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


# ---- Collect all exact-position scores ----
# Keyed by (probe, prompt_type, session_valence) -> list of scores
scores_by_group: dict[tuple[str, str, str], list[float]] = defaultdict(list)
# For interaction: (probe, session_valence, offered_valence) -> list of scores
task_switch_scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)

for item in items:
    for probe, offset in PROBES:
        score = exact_tb_score(item, probe, offset)
        scores_by_group[(probe, item["prompt_type"], item["session_valence"])].append(score)

        if item["prompt_type"] == "task_switch" and item.get("offered_valence") is not None:
            task_switch_scores[(probe, item["session_valence"], item["offered_valence"])].append(score)

# ---- Determine shared y-axis bounds ----
all_scores = []
for vals in scores_by_group.values():
    all_scores.extend(vals)
score_min, score_max = min(all_scores), max(all_scores)
# Anchor at 0, extend slightly beyond data range
y_lo = min(0, score_min * 1.15 if score_min < 0 else -0.5)
y_hi = max(0, score_max * 1.15 if score_max > 0 else 0.5)

# ===========================================================================
# Figure 1: Interruption scores by session valence and prompt type
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax_idx, (probe, offset) in enumerate(PROBES):
    ax = axes[ax_idx]
    probe_label = PROBE_LABELS[probe]
    x = np.arange(len(PROMPT_TYPES))
    width = 0.25
    rng = np.random.default_rng(42)

    for i, sv in enumerate(SESSION_VALENCES):
        means = []
        sems_list = []
        for pt in PROMPT_TYPES:
            vals = scores_by_group[(probe, pt, sv)]
            means.append(np.mean(vals) if vals else 0.0)
            sems_list.append(sem(vals) if vals else 0.0)

        bar_offset = (i - 1) * width
        ax.bar(x + bar_offset, means, width, yerr=sems_list, capsize=3,
               label=sv, color=VALENCE_COLORS[sv], alpha=0.8)

        # Individual points with jitter
        for j, pt in enumerate(PROMPT_TYPES):
            vals = scores_by_group[(probe, pt, sv)]
            if vals:
                jitter = rng.uniform(-width * 0.35, width * 0.35, size=len(vals))
                ax.scatter(
                    np.full(len(vals), j + bar_offset) + jitter,
                    vals,
                    color=VALENCE_COLORS[sv],
                    s=12, alpha=0.4, edgecolors="none", zorder=3,
                )

    ax.set_xlabel("Prompt type", fontsize=11)
    ax.set_title(f"Probe score at exact {probe_label} position", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([pt.replace("_", "\n") for pt in PROMPT_TYPES], fontsize=10)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.legend(title="Session valence", fontsize=9, title_fontsize=10)

axes[0].set_ylabel("Probe score", fontsize=11)
axes[0].set_ylim(y_lo, y_hi)

fig.suptitle("Interruption scores at exact trained token positions", fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out1 = ASSETS_DIR / "plot_031826_exact_position_interruption_scores.png"
fig.savefig(out1, dpi=150)
plt.close(fig)
print(f"Saved: {out1}")

# ===========================================================================
# Figure 2: Interaction plot for task_switch
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Collect task_switch control scores for reference line
ts_control: dict[str, list[float]] = defaultdict(list)
for item in items:
    if item["prompt_type"] == "task_switch" and item["session_valence"] == "control":
        for probe, offset in PROBES:
            ts_control[probe].append(exact_tb_score(item, probe, offset))

for ax_idx, (probe, offset) in enumerate(PROBES):
    ax = axes[ax_idx]
    probe_label = PROBE_LABELS[probe]
    x_positions = [0, 1]
    x_labels = ["pleasant", "unpleasant"]

    for ov in ["pleasant", "unpleasant"]:
        means = []
        sems_list = []
        for sv in ["pleasant", "unpleasant"]:
            vals = task_switch_scores[(probe, sv, ov)]
            means.append(np.mean(vals) if vals else np.nan)
            sems_list.append(sem(vals) if vals else 0.0)

        ax.errorbar(
            x_positions, means, yerr=sems_list,
            marker="o", markersize=8, capsize=5, linewidth=2,
            label=f"offered {ov}", color=OFFERED_COLORS[ov],
        )

    # Control reference
    ctrl_vals = ts_control[probe]
    if ctrl_vals:
        ctrl_mean = np.mean(ctrl_vals)
        ax.axhline(ctrl_mean, color=VALENCE_COLORS["control"], linewidth=1.5,
                    linestyle="--", label=f"control (n={len(ctrl_vals)})")

    ax.set_xlabel("Session valence", fontsize=11)
    ax.set_title(f"task_switch at exact {probe_label} position", fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.legend(fontsize=9)
    ax.axhline(0, color="black", linewidth=0.3, linestyle=":")

# Shared y-axis: anchor at 0
ts_all = []
for key, vals in task_switch_scores.items():
    ts_all.extend(vals)
ts_min, ts_max = min(ts_all), max(ts_all)
ts_y_lo = min(0, ts_min * 1.15 if ts_min < 0 else -0.5)
ts_y_hi = max(0, ts_max * 1.15 if ts_max > 0 else 0.5)
axes[0].set_ylim(ts_y_lo, ts_y_hi)
axes[0].set_ylabel("Probe score", fontsize=11)

fig.suptitle("Interaction: task_switch at exact trained token positions", fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out2 = ASSETS_DIR / "plot_031826_exact_position_task_switch.png"
fig.savefig(out2, dpi=150)
plt.close(fig)
print(f"Saved: {out2}")

# Print summary statistics
print("\n" + "=" * 60)
print("Summary: exact position scores")
print("=" * 60)
for probe, offset in PROBES:
    print(f"\n{probe}:")
    for pt in PROMPT_TYPES:
        for sv in SESSION_VALENCES:
            vals = scores_by_group[(probe, pt, sv)]
            if vals:
                print(f"  {pt:<25} {sv:<12} mean={np.mean(vals):+.3f}  sem={sem(vals):.3f}  n={len(vals)}")

print("\n--- task_switch interaction ---")
for probe, offset in PROBES:
    print(f"\n{probe}:")
    for sv in ["pleasant", "unpleasant"]:
        for ov in ["pleasant", "unpleasant"]:
            vals = task_switch_scores[(probe, sv, ov)]
            if vals:
                print(f"  session={sv:<12} offered={ov:<12} mean={np.mean(vals):+.3f}  sem={sem(vals):.3f}  n={len(vals)}")
