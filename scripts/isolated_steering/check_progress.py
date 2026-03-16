"""Check experiment progress: rows, pairs, layers, conditions."""
import json
from collections import Counter

with open("experiments/steering/isolated_steering/checkpoint.jsonl") as f:
    rows = [json.loads(l) for l in f]

total = len(rows)
by_layer = Counter(r["layer"] for r in rows)
by_condition = Counter(r["condition"] for r in rows)
by_pair = Counter(r["pair_id"] for r in rows)
by_choice = Counter(r["choice_presented"] for r in rows)
fallbacks = sum(1 for r in rows if r["steering_fallback"])

print(f"Total rows: {total} / 72,000 ({100*total/72000:.1f}%)")
print(f"\nBy layer: {dict(sorted(by_layer.items()))}")
print(f"By condition: {dict(sorted(by_condition.items()))}")
print(f"Unique pairs: {len(by_pair)}")
print(f"Choices: {dict(sorted(by_choice.items()))}")
print(f"Fallbacks: {fallbacks}")

# Estimate rate from timestamps (if raw_response lengths vary, longer = slower)
if total > 0:
    parse_rate = sum(1 for r in rows if r["choice_presented"] in ("a", "b")) / total
    print(f"Parse rate: {parse_rate:.1%}")
