"""Regenerate differential steering report plots with review fixes."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.steering.analysis import (
    aggregate_steered,
    chose_steered_task,
    filter_valid,
    load_checkpoint,
)

DIFF_CHECKPOINT = Path("experiments/steering/isolated_steering/checkpoint_differential_L25_500.jsonl")
HOOK_CHECKPOINT = Path("experiments/steering/isolated_steering/checkpoint_hook_L25_500.jsonl")
PAIRS = Path("experiments/revealed_steering_v2/followup/pairs_500.json")
TOPICS = Path("data/topics/topics.json")
ASSETS = Path("experiments/steering/isolated_steering/full_run/assets")

# Load and deduplicate both checkpoints
def load_dedup(path):
    raw = load_checkpoint(path)
    seen: dict[tuple, dict] = {}
    for r in raw:
        key = (r["pair_id"], r["signed_multiplier"], r["condition"], r["sample_idx"], r["ordering"])
        seen[key] = r
    return list(seen.values())

diff_rows = load_dedup(DIFF_CHECKPOINT)
hook_rows = load_dedup(HOOK_CHECKPOINT)
pairs = json.loads(PAIRS.read_text())
pair_lookup = {p["pair_id"]: p for p in pairs}
topics = json.loads(TOPICS.read_text())

diff_steered = [r for r in diff_rows if r["signed_multiplier"] != 0]
diff_valid = filter_valid(diff_steered)
hook_valid = filter_valid(hook_rows)

# === Plot 1: Grouped bar comparison at matched coefficients ===

# Get P(steered) at |coef| = 0.03 and 0.05 for each method
methods = {
    "Differential\n(single-pass)": {},
    "Hook patch\n+ recompute": {},
    "Hook patch\nsplice-only": {},
}

# Differential
diff_agg = aggregate_steered(diff_valid, group_by=["condition"])
for r in diff_agg:
    methods["Differential\n(single-pass)"][r["steering_strength"]] = (r["p_steered"], r["n"])

# Hook patching recompute — get closest coefficients (0.02 and 0.05)
hook_recompute = [r for r in hook_valid if r["condition"] == "hook_patching_recompute"]
hook_recompute_agg = aggregate_steered(hook_recompute, group_by=["condition"])
for r in hook_recompute_agg:
    methods["Hook patch\n+ recompute"][r["steering_strength"]] = (r["p_steered"], r["n"])

# Hook patching splice-only
hook_splice = [r for r in hook_valid if r["condition"] == "hook_patching"]
hook_splice_agg = aggregate_steered(hook_splice, group_by=["condition"])
for r in hook_splice_agg:
    methods["Hook patch\nsplice-only"][r["steering_strength"]] = (r["p_steered"], r["n"])

# Bar chart at |coef| = 0.05 (common to all methods)
coefs_to_show = [0.02, 0.05]
method_names = list(methods.keys())
colors = ["#16a34a", "#2563eb", "#60a5fa"]

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

for ax_idx, coef in enumerate(coefs_to_show):
    ax = axes[ax_idx]
    rates = []
    ns = []
    for method in method_names:
        data = methods[method]
        if coef in data:
            rate, n = data[coef]
            rates.append(rate)
            ns.append(n)
        elif coef == 0.02 and method == "Differential\n(single-pass)":
            # Differential doesn't have 0.02, use 0.03
            rate, n = data.get(0.03, (0, 0))
            rates.append(rate)
            ns.append(n)
        else:
            rates.append(0)
            ns.append(0)

    x = np.arange(len(method_names))
    bars = ax.bar(x, rates, color=colors, alpha=0.85, width=0.6)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    coef_label = f"|coef| = {coef}" if not (coef == 0.02 and method == "Differential\n(single-pass)") else f"|coef| = 0.03/0.02"
    if coef == 0.02:
        ax.set_title(f"|coef| = 0.02 (0.03 for differential)", fontsize=11, fontweight="bold")
    else:
        ax.set_title(f"|coef| = {coef}", fontsize=11, fontweight="bold")

    for bar, rate, n in zip(bars, rates, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.01,
                f"{rate:.1%}\n(n={n:,})", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=9)
    ax.set_ylim(0, 1.15)
    if ax_idx == 0:
        ax.set_ylabel("P(chose steered task)", fontsize=11)

fig.suptitle("Does the model choose the steered task?", fontsize=13, fontweight="bold")
plt.tight_layout()
path = ASSETS / "plot_032026_differential_vs_hook_bars.png"
fig.savefig(path, dpi=150)
print(f"Saved {path}")
plt.close()

# === Plot 2: Topic comparison (differential vs hook patching side by side) ===

def get_topic_rates(valid_rows, pair_lookup, topics):
    buckets: dict[str, list[bool]] = defaultdict(list)
    for r in valid_rows:
        pair = pair_lookup[r["pair_id"]]
        steered_task_id = pair["task_a"] if r["signed_multiplier"] > 0 else pair["task_b"]
        task_topics = topics.get(steered_task_id, {})
        topic = "unknown"
        for model_key in task_topics:
            topic = task_topics[model_key].get("primary", "unknown")
            break
        buckets[topic].append(chose_steered_task(r))
    return {t: sum(s) / len(s) for t, s in buckets.items()}

# Differential: |coef| >= 0.03
diff_strong = [r for r in diff_valid if abs(r["signed_multiplier"]) >= 0.03]
diff_topic_rates = get_topic_rates(diff_strong, pair_lookup, topics)

# Hook patching recompute: |coef| >= 0.02
hook_recompute_strong = [r for r in hook_valid
                         if r["condition"] == "hook_patching_recompute"
                         and abs(r["signed_multiplier"]) >= 0.02]
hook_topic_rates = get_topic_rates(hook_recompute_strong, pair_lookup, topics)

# Sort by hook patching rate (ascending) so differential improvement is visible
all_topics = sorted(set(diff_topic_rates.keys()) & set(hook_topic_rates.keys()),
                    key=lambda t: hook_topic_rates.get(t, 0))

HARMFUL_TOPICS = {"harmful_request", "security_legal", "model_manipulation",
                  "persuasive_writing", "sensitive_creative"}

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(all_topics))
bar_height = 0.35

# Hook patching bars (behind)
hook_rates = [hook_topic_rates.get(t, 0) for t in all_topics]
ax.barh(y_pos + bar_height/2, hook_rates, bar_height,
        color="#2563eb", alpha=0.6, label="Hook patch + recompute")

# Differential bars (front)
diff_rates = [diff_topic_rates.get(t, 0) for t in all_topics]
ax.barh(y_pos - bar_height/2, diff_rates, bar_height,
        color="#16a34a", alpha=0.85, label="Differential (single-pass)")

# Annotations
for i, t in enumerate(all_topics):
    d_rate = diff_topic_rates.get(t, 0)
    h_rate = hook_topic_rates.get(t, 0)
    diff_pp = (d_rate - h_rate) * 100
    if abs(diff_pp) >= 0.5:
        ax.text(max(d_rate, h_rate) + 0.01, i,
                f"+{diff_pp:.0f}pp" if diff_pp > 0 else f"{diff_pp:.0f}pp",
                va="center", fontsize=7, color="#666")

ax.set_yticks(y_pos)
labels = [t.replace("_", " ") for t in all_topics]
label_colors = ["#dc2626" if t in HARMFUL_TOPICS else "#333" for t in all_topics]
ax.set_yticklabels(labels, fontsize=9)
for label, color in zip(ax.get_yticklabels(), label_colors):
    label.set_color(color)

ax.set_xlabel("P(chose steered task)", fontsize=11)
ax.set_xlim(0.88, 1.06)
ax.set_title("Steering by topic: differential vs hook patching", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.axvline(1.0, color="#ccc", linewidth=0.5)
plt.tight_layout()
path = ASSETS / "plot_032026_differential_topic_comparison.png"
fig.savefig(path, dpi=150)
print(f"Saved {path}")
plt.close()

# === Plot 3: Refusal rates with control reference ===

refusal_data: dict[float, tuple[int, int]] = {}
for r in diff_rows:
    mult = r["signed_multiplier"]
    if mult not in refusal_data:
        refusal_data[mult] = (0, 0)
    total, ref = refusal_data[mult]
    refusal_data[mult] = (total + 1, ref + (1 if r["choice_original"] == "refusal" else 0))

coefs_sorted = sorted(refusal_data.keys())
refusal_rates = [refusal_data[c][1] / refusal_data[c][0] for c in coefs_sorted]
coef_labels = [f"{c:+.2f}" for c in coefs_sorted]

# Control rate
control_rate = refusal_data[0.0][1] / refusal_data[0.0][0]

fig, ax = plt.subplots(figsize=(8, 4))
bar_colors = ["#16a34a" if c != 0 else "#6b7280" for c in coefs_sorted]
bars = ax.bar(range(len(coefs_sorted)), refusal_rates, color=bar_colors, alpha=0.85, width=0.6)
ax.axhline(control_rate, color="#dc2626", linestyle="--", linewidth=1.5,
           label=f"Control rate ({control_rate:.1%})", zorder=0)

for i, (rate, total_ref) in enumerate(zip(refusal_rates, [refusal_data[c] for c in coefs_sorted])):
    total, ref = total_ref
    ax.text(i, rate + 0.003, f"{rate:.1%}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(range(len(coefs_sorted)))
ax.set_xticklabels(coef_labels, fontsize=9)
ax.set_xlabel("Steering coefficient", fontsize=11)
ax.set_ylabel("Refusal rate", fontsize=11)
ax.set_ylim(0, 0.12)
ax.set_title("Steering reduces refusals below the unsteered baseline", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
path = ASSETS / "plot_032026_differential_refusal.png"
fig.savefig(path, dpi=150)
print(f"Saved {path}")
plt.close()

print("\nDone!")
