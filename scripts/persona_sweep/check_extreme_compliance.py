"""Check how the compliance numbers were computed for extreme tier."""
import json

lines = open("experiments/steering/open_ended_steering/safety_steering/results.jsonl").readlines()
records = [json.loads(l) for l in lines]

# Check all extreme tier from exp 1
extreme = [
    r for r in records
    if r.get("tier") == "extreme"
    and r.get("experiment") == 1
]

print(f"Total extreme records: {len(extreme)}")
print()

# Group by multiplier
from collections import defaultdict
by_mult = defaultdict(list)
for r in extreme:
    by_mult[r["multiplier"]].append(r)

for mult in sorted(by_mult.keys()):
    recs = by_mult[mult]
    # Check if there's a refusal/compliance field
    sample = recs[0]
    relevant_keys = [k for k in sample.keys() if k not in ("response", "prompt_text")]
    print(f"mult={mult}: {len(recs)} records")
    print(f"  Keys: {relevant_keys}")

    # Check for any judge fields
    for k in relevant_keys:
        vals = set(str(r.get(k)) for r in recs)
        if len(vals) <= 10:
            print(f"  {k}: {vals}")
    print()
