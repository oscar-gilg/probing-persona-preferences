import json

with open("experiments/steering/cross_layer/pairs_500.json") as f:
    pairs = json.load(f)
print(f"Number of pairs: {len(pairs)}")
p = pairs[0]
print("Keys:", list(p.keys()))
for k, v in p.items():
    print(f"  {k}: {repr(v)[:100]}")
