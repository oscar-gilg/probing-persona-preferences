"""Compute full hook L25 500 stats for report update."""

import json
from collections import defaultdict
from pathlib import Path

from src.steering.analysis import (
    aggregate,
    aggregate_steered,
    filter_valid,
    load_checkpoint,
)

CHECKPOINT = Path("experiments/steering/isolated_steering/checkpoint_hook_L25_500.jsonl")
PAIRS = Path("experiments/revealed_steering_v2/followup/pairs_500.json")

# Load
rows = load_checkpoint(CHECKPOINT)
print(f"Total rows: {len(rows)}")

# Deduplicate
seen: dict[tuple, dict] = {}
for r in rows:
    key = (r["pair_id"], r["signed_multiplier"], r["condition"], r["sample_idx"], r["ordering"])
    seen[key] = r
rows = list(seen.values())
print(f"Unique rows: {len(rows)}")

# Pairs coverage
pair_ids = {r["pair_id"] for r in rows}
print(f"Unique pairs: {len(pair_ids)}")

# Expected: 500 pairs × 2 orderings × 12 mults × 10 trials × 2 conditions = 240,000
expected = 500 * 2 * 12 * 10 * 2
print(f"Expected: {expected}, got: {len(rows)}, completion: {len(rows)/expected:.1%}")

# Dose-response by condition
print("\n=== DOSE-RESPONSE (P(chose steered task)) ===")
valid = filter_valid(rows)
print(f"Valid (non-refusal): {len(valid)}")

agg = aggregate_steered(valid, group_by=["condition"])
print(f"\n{'condition':<30} {'|coef|':>8} {'P(steered)':>12} {'n':>8}")
print("-" * 65)
for r in agg:
    print(f"{r['condition']:<30} {r['steering_strength']:>8.2f} {r['p_steered']:>11.1%} {r['n']:>8}")

# Refusal rates
print("\n=== REFUSAL RATES ===")
refusal_buckets: dict[tuple[str, float], tuple[int, int]] = defaultdict(lambda: (0, 0))
for r in rows:
    key = (r["condition"], abs(r["signed_multiplier"]))
    total, ref = refusal_buckets[key]
    refusal_buckets[key] = (total + 1, ref + (1 if r["choice_original"] == "refusal" else 0))

print(f"\n{'condition':<30} {'|coef|':>8} {'refusal':>10} {'n':>8}")
print("-" * 60)
for (cond, mult), (total, ref) in sorted(refusal_buckets.items()):
    print(f"{cond:<30} {mult:>8.2f} {ref/total:>9.1%} {total:>8}")

# Topic breakdown (at recompute, |coef|=0.05)
pairs = json.loads(PAIRS.read_text())
pair_lookup = {p["pair_id"]: p for p in pairs}

# Load topics
topics_path = Path("data/topics/topics.json")
if topics_path.exists():
    topics = json.loads(topics_path.read_text())
else:
    print("\nTopics file not found, skipping topic breakdown")
    topics = None

if topics:
    print("\n=== STEERABILITY BY TOPIC (recompute, all |coef|) ===")
    # For each row, get the topic of the steered-toward task
    recompute_valid = [
        r for r in valid
        if r["condition"] == "hook_patching_recompute"
        and abs(r["signed_multiplier"]) >= 0.02
    ]

    # Determine which task was steered toward
    from src.steering.analysis import chose_steered_task

    topic_buckets: dict[str, list[bool]] = defaultdict(list)
    for r in recompute_valid:
        pair = pair_lookup[r["pair_id"]]
        steered_task_id = pair["task_a"] if r["signed_multiplier"] > 0 else pair["task_b"]
        # Get topic
        task_topics = topics.get(steered_task_id, {})
        # Try different model keys
        for model_key in task_topics:
            topic = task_topics[model_key].get("primary", "unknown")
            break
        else:
            topic = "unknown"
        topic_buckets[topic].append(chose_steered_task(r))

    print(f"\n{'topic':<30} {'P(steered)':>12} {'n':>8}")
    print("-" * 55)
    for topic, successes in sorted(topic_buckets.items(), key=lambda x: -sum(x[1])/len(x[1])):
        rate = sum(successes) / len(successes)
        print(f"{topic:<30} {rate:>11.1%} {len(successes):>8}")
