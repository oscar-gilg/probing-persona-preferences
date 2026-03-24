"""Compute differential steering L25 stats and generate plots."""

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

CHECKPOINT = Path("experiments/steering/isolated_steering/checkpoint_differential_L25_500.jsonl")
PAIRS = Path("experiments/revealed_steering_v2/followup/pairs_500.json")
TOPICS = Path("data/topics/topics.json")
ASSETS = Path("experiments/steering/isolated_steering/full_run/assets")

# Load and deduplicate
raw_rows = load_checkpoint(CHECKPOINT)
seen: dict[tuple, dict] = {}
for r in raw_rows:
    key = (r["pair_id"], r["signed_multiplier"], r["condition"], r["sample_idx"], r["ordering"])
    seen[key] = r
rows = list(seen.values())
print(f"{len(rows)} unique rows, {len({r['pair_id'] for r in rows})} pairs")

pairs = json.loads(PAIRS.read_text())
pair_lookup = {p["pair_id"]: p for p in pairs}
topics = json.loads(TOPICS.read_text())

# Split zero-coef (control) from steered
control = [r for r in rows if r["signed_multiplier"] == 0]
steered = [r for r in rows if r["signed_multiplier"] != 0]
print(f"Control (coef=0): {len(control)} rows")
print(f"Steered (coef!=0): {len(steered)} rows")

# Control baseline: P(chose_a) should be ~0.5 if no position bias
control_valid = filter_valid(control)
p_a_control = sum(1 for r in control_valid if r["choice_original"] == "a") / len(control_valid)
print(f"\nControl P(chose A): {p_a_control:.3f} (n={len(control_valid)})")
control_refusal = 1 - len(control_valid) / len(control)
print(f"Control refusal rate: {control_refusal:.1%}")

# Dose-response
valid = filter_valid(steered)
agg = aggregate_steered(valid, group_by=["condition"])

print(f"\n=== DOSE-RESPONSE (P(chose steered task)) ===")
print(f"{'condition':<30} {'|coef|':>8} {'P(steered)':>12} {'n':>8}")
print("-" * 65)
for r in agg:
    print(f"{r['condition']:<30} {r['steering_strength']:>8.2f} {r['p_steered']:>11.1%} {r['n']:>8}")

# Refusal rates
print(f"\n=== REFUSAL RATES ===")
refusal_data: dict[float, tuple[int, int]] = {}
for r in rows:
    mult = r["signed_multiplier"]
    if mult not in refusal_data:
        refusal_data[mult] = (0, 0)
    total, ref = refusal_data[mult]
    refusal_data[mult] = (total + 1, ref + (1 if r["choice_original"] == "refusal" else 0))

print(f"{'coef':>8} {'refusal':>10} {'n':>8}")
print("-" * 30)
for mult in sorted(refusal_data.keys()):
    total, ref = refusal_data[mult]
    print(f"{mult:>8.2f} {ref/total:>9.1%} {total:>8}")

# By-topic breakdown (pooling |coef| >= 0.03)
print(f"\n=== STEERABILITY BY TOPIC (|coef| >= 0.03) ===")
steered_strong = [r for r in valid if abs(r["signed_multiplier"]) >= 0.03]
topic_buckets: dict[str, list[bool]] = defaultdict(list)
for r in steered_strong:
    pair = pair_lookup[r["pair_id"]]
    steered_task_id = pair["task_a"] if r["signed_multiplier"] > 0 else pair["task_b"]
    task_topics = topics.get(steered_task_id, {})
    topic = "unknown"
    for model_key in task_topics:
        topic = task_topics[model_key].get("primary", "unknown")
        break
    topic_buckets[topic].append(chose_steered_task(r))

print(f"{'topic':<30} {'P(steered)':>12} {'n':>8}")
print("-" * 55)
for topic, successes in sorted(topic_buckets.items(), key=lambda x: -sum(x[1])/len(x[1])):
    rate = sum(successes) / len(successes)
    print(f"{topic:<30} {rate:>11.1%} {len(successes):>8}")

# === PLOTS ===

# Plot 1: Dose-response sigmoid (compare with hook patching)
# Load hook patching data for comparison
HOOK_CHECKPOINT = Path("experiments/steering/isolated_steering/checkpoint_hook_L25_500.jsonl")
hook_raw = load_checkpoint(HOOK_CHECKPOINT)
hook_seen: dict[tuple, dict] = {}
for r in hook_raw:
    key = (r["pair_id"], r["signed_multiplier"], r["condition"], r["sample_idx"], r["ordering"])
    hook_seen[key] = r
hook_rows = list(hook_seen.values())
hook_valid = filter_valid(hook_rows)

# Differential dose-response
diff_agg = aggregate_steered(valid, group_by=["condition"])
diff_strengths = sorted({r["steering_strength"] for r in diff_agg})
diff_p = {r["steering_strength"]: r["p_steered"] for r in diff_agg}

# Hook patching recompute for comparison
hook_recompute = [r for r in hook_valid if r["condition"] == "hook_patching_recompute"]
hook_agg = aggregate_steered(hook_recompute, group_by=["condition"])
hook_strengths = sorted({r["steering_strength"] for r in hook_agg})
hook_p = {r["steering_strength"]: r["p_steered"] for r in hook_agg}

# Hook patching splice-only for comparison
hook_splice = [r for r in hook_valid if r["condition"] == "hook_patching"]
hook_splice_agg = aggregate_steered(hook_splice, group_by=["condition"])
hook_splice_p = {r["steering_strength"]: r["p_steered"] for r in hook_splice_agg}

fig, ax = plt.subplots(figsize=(9, 5))

# Differential
x_diff = []
y_diff = []
for s in reversed(diff_strengths):
    x_diff.append(-s)
    y_diff.append(1 - diff_p[s])
x_diff.append(0)
y_diff.append(0.5)
for s in diff_strengths:
    x_diff.append(s)
    y_diff.append(diff_p[s])
ax.plot(x_diff, y_diff, "-", color="#16a34a", linewidth=2.5, marker="o",
        markersize=6, label="Differential (naive)", zorder=4)

# Hook recompute
x_hook = []
y_hook = []
for s in reversed(hook_strengths):
    x_hook.append(-s)
    y_hook.append(1 - hook_p[s])
x_hook.append(0)
y_hook.append(0.5)
for s in hook_strengths:
    x_hook.append(s)
    y_hook.append(hook_p[s])
ax.plot(x_hook, y_hook, "-", color="#2563eb", linewidth=1.5, marker="o",
        markersize=4, label="Hook patch + recompute", alpha=0.7, zorder=3)

# Hook splice-only
hook_splice_strengths = sorted(hook_splice_p.keys())
x_hs = []
y_hs = []
for s in reversed(hook_splice_strengths):
    x_hs.append(-s)
    y_hs.append(1 - hook_splice_p[s])
x_hs.append(0)
y_hs.append(0.5)
for s in hook_splice_strengths:
    x_hs.append(s)
    y_hs.append(hook_splice_p[s])
ax.plot(x_hs, y_hs, "--", color="#60a5fa", linewidth=1.5, marker="o",
        markersize=4, label="Hook patch splice-only", alpha=0.7, zorder=2)

ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("Steering coefficient (fraction of mean L25 norm)", fontsize=11)
ax.set_ylabel("P(chose steered task)", fontsize=11)
ax.set_ylim(0, 1)
ax.set_xlim(-0.16, 0.16)
ax.set_title("Differential steering vs hook patching (498 pairs)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
path = ASSETS / "plot_032026_differential_vs_hook.png"
fig.savefig(path, dpi=150)
print(f"\nSaved {path}")
plt.close()

# Plot 2: Topic breakdown
topic_names = sorted(topic_buckets.keys(), key=lambda t: sum(topic_buckets[t]) / len(topic_buckets[t]))
topic_rates = [sum(topic_buckets[t]) / len(topic_buckets[t]) for t in topic_names]
topic_ns = [len(topic_buckets[t]) for t in topic_names]
HARMFUL_TOPICS = {"harmful_request", "security_legal", "model_manipulation", "persuasive_writing", "sensitive_creative"}
bar_colors = ["#ef4444" if t in HARMFUL_TOPICS else "#16a34a" for t in topic_names]

fig, ax = plt.subplots(figsize=(10, 5))
y_pos = np.arange(len(topic_names))
ax.barh(y_pos, topic_rates, color=bar_colors, alpha=0.85)
for i, (rate, n) in enumerate(zip(topic_rates, topic_ns)):
    ax.text(rate + 0.005, i, f"{rate:.0%} (n={n})", va="center", fontsize=8)
ax.set_yticks(y_pos)
ax.set_yticklabels([t.replace("_", " ") for t in topic_names], fontsize=9)
ax.set_xlabel("P(chose steered task)", fontsize=11)
ax.set_xlim(0, 1.15)
ax.set_title("Differential steering by topic (|coef| >= 0.03)", fontsize=12, fontweight="bold")
ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
plt.tight_layout()
path = ASSETS / "plot_032026_differential_by_topic.png"
fig.savefig(path, dpi=150)
print(f"Saved {path}")
plt.close()

print("\nDone!")
