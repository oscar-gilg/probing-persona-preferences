"""Compare differential vs hook patching: what's actually different?"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.steering.analysis import chose_steered_task, load_checkpoint

DIFF_CHECKPOINT = Path("experiments/steering/isolated_steering/checkpoint_differential_L25_500.jsonl")
HOOK_CHECKPOINT = Path("experiments/steering/isolated_steering/checkpoint_hook_L25_500.jsonl")
PAIRS = Path("experiments/revealed_steering_v2/followup/pairs_500.json")
TOPICS = Path("data/topics/topics.json")

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

# === 1. Per-pair refusal rates: are the same pairs refusing? ===
print("=" * 70)
print("1. PER-PAIR REFUSAL COMPARISON")
print("=" * 70)

def per_pair_refusal(rows, condition_filter=None):
    filtered = rows
    if condition_filter:
        filtered = [r for r in rows if r["condition"] == condition_filter]
    buckets: dict[str, list[bool]] = defaultdict(list)
    for r in filtered:
        buckets[r["pair_id"]].append(r["choice_original"] == "refusal")
    return {pid: sum(flags) / len(flags) for pid, flags in buckets.items()}

diff_control_ref = per_pair_refusal([r for r in diff_rows if r["signed_multiplier"] == 0])
diff_steered_ref = per_pair_refusal([r for r in diff_rows if r["signed_multiplier"] != 0])
hook_recompute_ref = per_pair_refusal(hook_rows, "hook_patching_recompute")
hook_splice_ref = per_pair_refusal(hook_rows, "hook_patching")

common_pairs = set(diff_control_ref.keys()) & set(hook_recompute_ref.keys())
print(f"\nCommon pairs: {len(common_pairs)}")

# Correlation: do the same pairs refuse across methods?
pairs_list = sorted(common_pairs)
d_ctrl = np.array([diff_control_ref[p] for p in pairs_list])
d_steer = np.array([diff_steered_ref[p] for p in pairs_list])
h_recomp = np.array([hook_recompute_ref[p] for p in pairs_list])
h_splice = np.array([hook_splice_ref[p] for p in pairs_list])

print(f"\nPer-pair refusal rate correlations:")
print(f"  diff_control vs diff_steered: r={np.corrcoef(d_ctrl, d_steer)[0,1]:.3f}")
print(f"  diff_control vs hook_recompute: r={np.corrcoef(d_ctrl, h_recomp)[0,1]:.3f}")
print(f"  diff_steered vs hook_recompute: r={np.corrcoef(d_steer, h_recomp)[0,1]:.3f}")
print(f"  hook_splice vs hook_recompute: r={np.corrcoef(h_splice, h_recomp)[0,1]:.3f}")

# Mean refusal rates
print(f"\nMean per-pair refusal rate:")
print(f"  diff_control: {d_ctrl.mean():.3f}")
print(f"  diff_steered: {d_steer.mean():.3f}")
print(f"  hook_recompute: {h_recomp.mean():.3f}")
print(f"  hook_splice: {h_splice.mean():.3f}")

# Which pairs refuse MORE in differential than hook patching?
diff_vs_hook = d_steer - h_recomp
print(f"\n  diff_steered - hook_recompute: mean={diff_vs_hook.mean():.3f}, std={diff_vs_hook.std():.3f}")
print(f"  Pairs where diff refuses MORE: {(diff_vs_hook > 0.05).sum()}")
print(f"  Pairs where diff refuses LESS: {(diff_vs_hook < -0.05).sum()}")

# === 2. Label-content dissociation: does differential have it too? ===
print("\n" + "=" * 70)
print("2. LABEL MISMATCH")
print("=" * 70)

def label_stats(rows):
    """Check if model writes 'Task A:' but does task B (or vice versa)."""
    starts_with_a = 0
    starts_with_b = 0
    chose_a = 0
    chose_b = 0
    mismatch = 0
    total = 0
    for r in rows:
        if r["choice_original"] not in ("a", "b"):
            continue
        total += 1
        resp = r["raw_response"].strip()
        label_a = resp.startswith("Task A") or resp.startswith("**Task A")
        label_b = resp.startswith("Task B") or resp.startswith("**Task B")
        if label_a:
            starts_with_a += 1
        elif label_b:
            starts_with_b += 1
        chose = r["choice_original"]
        if chose == "a":
            chose_a += 1
        else:
            chose_b += 1
        # Mismatch: says Task A but chose b, or says Task B but chose a
        if (label_a and chose == "b") or (label_b and chose == "a"):
            mismatch += 1
    return {
        "total": total,
        "label_a_pct": starts_with_a / total if total else 0,
        "label_b_pct": starts_with_b / total if total else 0,
        "chose_a_pct": chose_a / total if total else 0,
        "mismatch_pct": mismatch / total if total else 0,
    }

diff_steered_rows = [r for r in diff_rows if r["signed_multiplier"] != 0]
hook_recompute_rows = [r for r in hook_rows if r["condition"] == "hook_patching_recompute"]

diff_labels = label_stats(diff_steered_rows)
hook_labels = label_stats(hook_recompute_rows)

print(f"\n{'Metric':<25} {'Differential':>15} {'Hook+recompute':>15}")
print("-" * 60)
for key in ["label_a_pct", "label_b_pct", "chose_a_pct", "mismatch_pct"]:
    print(f"{key:<25} {diff_labels[key]:>14.1%} {hook_labels[key]:>14.1%}")

# === 3. Per-pair steering success: does differential win on the SAME pairs? ===
print("\n" + "=" * 70)
print("3. PER-PAIR STEERING SUCCESS")
print("=" * 70)

def per_pair_steered(rows, condition_filter=None):
    filtered = [r for r in rows if r["choice_original"] in ("a", "b") and r["signed_multiplier"] != 0]
    if condition_filter:
        filtered = [r for r in filtered if r["condition"] == condition_filter]
    buckets: dict[str, list[bool]] = defaultdict(list)
    for r in filtered:
        buckets[r["pair_id"]].append(chose_steered_task(r))
    return {pid: sum(flags) / len(flags) for pid, flags in buckets.items()}

diff_p_steered = per_pair_steered(diff_rows)
hook_recomp_p_steered = per_pair_steered(hook_rows, "hook_patching_recompute")

common = set(diff_p_steered.keys()) & set(hook_recomp_p_steered.keys())
pairs_list = sorted(common)
d_p = np.array([diff_p_steered[p] for p in pairs_list])
h_p = np.array([hook_recomp_p_steered[p] for p in pairs_list])

print(f"\nPer-pair P(steered) correlation: r={np.corrcoef(d_p, h_p)[0,1]:.3f}")
print(f"Mean P(steered) — diff: {d_p.mean():.3f}, hook_recompute: {h_p.mean():.3f}")

# Where does differential win/lose?
delta = d_p - h_p
print(f"\nDifferential - Hook recompute per-pair P(steered):")
print(f"  mean delta: {delta.mean():.3f}")
print(f"  Pairs where diff wins by >5pp: {(delta > 0.05).sum()}")
print(f"  Pairs where hook wins by >5pp: {(delta < -0.05).sum()}")
print(f"  Pairs where both ≥95%: {((d_p >= 0.95) & (h_p >= 0.95)).sum()}")

# Are the pairs where differential wins specific topics?
print("\nPairs where diff wins by >10pp — topic distribution:")
winners = [p for p in pairs_list if diff_p_steered[p] - hook_recomp_p_steered[p] > 0.10]
winner_topics: dict[str, int] = defaultdict(int)
for p in winners:
    pair = pair_lookup[p]
    for task_id in [pair["task_a"], pair["task_b"]]:
        task_topics = topics.get(task_id, {})
        for model_key in task_topics:
            topic = task_topics[model_key].get("primary", "unknown")
            winner_topics[topic] += 1
            break
for topic, count in sorted(winner_topics.items(), key=lambda x: -x[1]):
    print(f"  {topic}: {count}")

# === 4. Asymmetry: steer toward A vs toward B ===
print("\n" + "=" * 70)
print("4. STEERING DIRECTION ASYMMETRY")
print("=" * 70)

for label, rows_set, cond_filter in [
    ("Differential", diff_rows, None),
    ("Hook+recompute", hook_rows, "hook_patching_recompute"),
]:
    filtered = rows_set
    if cond_filter:
        filtered = [r for r in rows_set if r["condition"] == cond_filter]
    steered = [r for r in filtered if r["signed_multiplier"] != 0 and r["choice_original"] in ("a", "b")]

    toward_a = [r for r in steered if r["signed_multiplier"] > 0]
    toward_b = [r for r in steered if r["signed_multiplier"] < 0]

    a_success = sum(1 for r in toward_a if r["choice_original"] == "a") / len(toward_a)
    b_success = sum(1 for r in toward_b if r["choice_original"] == "b") / len(toward_b)

    a_refuse_all = [r for r in filtered if r["signed_multiplier"] > 0]
    b_refuse_all = [r for r in filtered if r["signed_multiplier"] < 0]
    a_ref = sum(1 for r in a_refuse_all if r["choice_original"] == "refusal") / len(a_refuse_all)
    b_ref = sum(1 for r in b_refuse_all if r["choice_original"] == "refusal") / len(b_refuse_all)

    print(f"\n{label}:")
    print(f"  Steer toward A: P(chose A)={a_success:.3f} (n={len(toward_a)}), refusal={a_ref:.1%}")
    print(f"  Steer toward B: P(chose B)={b_success:.3f} (n={len(toward_b)}), refusal={b_ref:.1%}")

# === 5. Response length comparison ===
print("\n" + "=" * 70)
print("5. RESPONSE LENGTH")
print("=" * 70)

for label, rows_set, cond_filter in [
    ("Diff control", [r for r in diff_rows if r["signed_multiplier"] == 0], None),
    ("Diff steered", [r for r in diff_rows if r["signed_multiplier"] != 0], None),
    ("Hook+recompute", hook_rows, "hook_patching_recompute"),
    ("Hook splice", hook_rows, "hook_patching"),
]:
    filtered = rows_set
    if cond_filter:
        filtered = [r for r in rows_set if r["condition"] == cond_filter]
    lengths = [len(r["raw_response"]) for r in filtered if r["choice_original"] in ("a", "b")]
    print(f"  {label:<20} mean={np.mean(lengths):.0f} chars, median={np.median(lengths):.0f}")
