import json

with open("experiments/steering/cross_layer/checkpoint_L25.parsed.jsonl") as f:
    r = json.loads(f.readline())
print("Keys:", list(r.keys()))
for k, v in r.items():
    print(f"  {k}: {repr(v)[:100]}")
