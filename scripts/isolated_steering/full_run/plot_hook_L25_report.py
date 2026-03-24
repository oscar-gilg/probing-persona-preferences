"""Regenerate all hook L25 500 report plots with full data (498 pairs)."""

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

CHECKPOINT = Path("experiments/steering/isolated_steering/checkpoint_hook_L25_500.jsonl")
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

CONDITIONS = [
    ("hook_patching", "Splice only", "#60a5fa", "--"),
    ("hook_patching_recompute", "Splice + recompute", "#2563eb", "-"),
]

# ── Plot 1: Dose-response sigmoid ──

valid = filter_valid(rows)
agg = aggregate_steered(valid, group_by=["condition"])

fig, ax = plt.subplots(figsize=(8, 5))

for condition, label, color, linestyle in CONDITIONS:
    cond_rows = [r for r in agg if r["condition"] == condition]
    strengths = sorted(r["steering_strength"] for r in cond_rows)
    p_map = {r["steering_strength"]: r["p_steered"] for r in cond_rows}
    n_map = {r["steering_strength"]: r["n"] for r in cond_rows}

    # Build symmetric sigmoid: negative coefs on left, positive on right
    x_vals = []
    y_vals = []
    for s in reversed(strengths):
        x_vals.append(-s)
        y_vals.append(1 - p_map[s])
    x_vals.append(0)
    y_vals.append(0.5)
    for s in strengths:
        x_vals.append(s)
        y_vals.append(p_map[s])

    ax.plot(x_vals, y_vals, linestyle, color=color, linewidth=2, marker="o",
            markersize=5, label=label, zorder=3)

ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("Steering coefficient (fraction of mean L25 norm)", fontsize=11)
ax.set_ylabel("P(chose steered task)", fontsize=11)
ax.set_ylim(0, 1)
ax.set_title("Hook patching dose-response (498 pairs)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
path = ASSETS / "plot_032026_dose_response_sigmoid.png"
fig.savefig(path, dpi=150)
print(f"Saved {path}")
plt.close()

# ── Plot 2: Refusal rates ──

refusal_data: dict[str, dict[float, tuple[int, int]]] = defaultdict(dict)
for r in rows:
    cond = r["condition"]
    mult = abs(r["signed_multiplier"])
    if mult not in refusal_data[cond]:
        refusal_data[cond][mult] = (0, 0)
    total, ref = refusal_data[cond][mult]
    refusal_data[cond][mult] = (total + 1, ref + (1 if r["choice_original"] == "refusal" else 0))

fig, ax = plt.subplots(figsize=(8, 5))

for condition, label, color, linestyle in CONDITIONS:
    coefs = sorted(refusal_data[condition].keys())
    rates = [refusal_data[condition][c][1] / refusal_data[condition][c][0] for c in coefs]
    ax.plot(coefs, rates, linestyle, color=color, linewidth=2, marker="o",
            markersize=5, label=label, zorder=3)

ax.set_xlabel("Steering coefficient (|coef|)", fontsize=11)
ax.set_ylabel("Refusal rate", fontsize=11)
ax.set_ylim(0, 0.6)
ax.set_title("Refusal rate by steering strength (498 pairs)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
path = ASSETS / "plot_032026_refusal_by_strength.png"
fig.savefig(path, dpi=150)
print(f"Saved {path}")
plt.close()

# ── Plot 3: Steering by topic (recompute, |coef| >= 0.02) ──

recompute_valid = [
    r for r in valid
    if r["condition"] == "hook_patching_recompute"
    and abs(r["signed_multiplier"]) >= 0.02
]

topic_buckets: dict[str, list[bool]] = defaultdict(list)
for r in recompute_valid:
    pair = pair_lookup[r["pair_id"]]
    steered_task_id = pair["task_a"] if r["signed_multiplier"] > 0 else pair["task_b"]
    task_topics = topics.get(steered_task_id, {})
    topic = "unknown"
    for model_key in task_topics:
        topic = task_topics[model_key].get("primary", "unknown")
        break
    topic_buckets[topic].append(chose_steered_task(r))

topic_names = sorted(topic_buckets.keys(), key=lambda t: sum(topic_buckets[t]) / len(topic_buckets[t]))
topic_rates = [sum(topic_buckets[t]) / len(topic_buckets[t]) for t in topic_names]
topic_ns = [len(topic_buckets[t]) for t in topic_names]

HARMFUL_TOPICS = {"harmful_request", "security_legal", "model_manipulation", "persuasive_writing", "sensitive_creative"}
bar_colors = ["#ef4444" if t in HARMFUL_TOPICS else "#3b82f6" for t in topic_names]

fig, ax = plt.subplots(figsize=(10, 5))
y_pos = np.arange(len(topic_names))
bars = ax.barh(y_pos, topic_rates, color=bar_colors, alpha=0.85)

for i, (rate, n) in enumerate(zip(topic_rates, topic_ns)):
    ax.text(rate + 0.005, i, f"{rate:.0%} (n={n})", va="center", fontsize=8)

ax.set_yticks(y_pos)
ax.set_yticklabels([t.replace("_", " ") for t in topic_names], fontsize=9)
ax.set_xlabel("P(chose steered task)", fontsize=11)
ax.set_xlim(0, 1.15)
ax.set_title("Hook patching by topic (recompute, |coef| >= 0.02)", fontsize=12, fontweight="bold")
ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
plt.tight_layout()
path = ASSETS / "plot_032026_steering_by_topic.png"
fig.savefig(path, dpi=150)
print(f"Saved {path}")
plt.close()

# ── Plot 4: Steering by utility bin (recompute) ──

recompute_all_valid = [
    r for r in valid
    if r["condition"] == "hook_patching_recompute"
]

# Get utility bin from pairs
bin_buckets_pos: dict[int, list[bool]] = defaultdict(list)
bin_buckets_neg: dict[int, list[bool]] = defaultdict(list)

for r in recompute_all_valid:
    pair = pair_lookup[r["pair_id"]]
    mu_bin = pair["mu_bin"]
    if r["signed_multiplier"] > 0:
        bin_buckets_pos[mu_bin].append(r["choice_original"] == "a")
    else:
        bin_buckets_neg[mu_bin].append(r["choice_original"] == "a")

bins = sorted(set(bin_buckets_pos.keys()) | set(bin_buckets_neg.keys()))
shifts = []
for b in bins:
    p_pos = sum(bin_buckets_pos[b]) / len(bin_buckets_pos[b]) if bin_buckets_pos[b] else 0.5
    p_neg = sum(bin_buckets_neg[b]) / len(bin_buckets_neg[b]) if bin_buckets_neg[b] else 0.5
    shifts.append(p_pos - p_neg)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(bins, shifts, color="#2563eb", alpha=0.85)

for b, s in zip(bins, shifts):
    n = len(bin_buckets_pos.get(b, [])) + len(bin_buckets_neg.get(b, []))
    ax.text(b, s + 0.01, f"{s:.0%}", ha="center", fontsize=8)

ax.set_xlabel("Utility decile (0=low, 9=high)", fontsize=11)
ax.set_ylabel("Steering shift: P(A|steer toward A) - P(A|steer away)", fontsize=10)
ax.set_ylim(0, 1)
ax.set_title("Hook patching by utility level (recompute, all coefs)", fontsize=12, fontweight="bold")
ax.set_xticks(bins)
plt.tight_layout()
path = ASSETS / "plot_032026_steering_by_utility_bin.png"
fig.savefig(path, dpi=150)
print(f"Saved {path}")
plt.close()

print("\nDone!")
