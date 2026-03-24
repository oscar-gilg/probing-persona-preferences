import json

with open("experiments/steering/isolated_steering/parsed_kv_recompute.jsonl") as f:
    r = json.loads(f.readline())
print("Keys:", list(r.keys()))
for k, v in r.items():
    print(f"  {k}: {repr(v)[:80]}")
