import json

with open("experiments/steering/open_ended_steering/scored_results.jsonl") as f:
    r = json.loads(f.readline())
print("Keys:", list(r.keys()))
for k, v in r.items():
    print(f"  {k}: {repr(v)[:80]}")
