"""Check what tokens are at the tb-2 and tb-5 positions."""
import json

DATA_DIR = "experiments/context_interruption/data"

with open(f"{DATA_DIR}/scoring_results.json") as f:
    raw = json.load(f)

for item in raw["items"][:3]:
    gen_start = item["segments"]["generation_prompt"][0]
    tokens = item["tokens"]
    print(f"Item: {item['id']}")
    print(f"  Tokens around turn boundary (gen_start={gen_start}):")
    for offset in range(8, 0, -1):
        pos = gen_start - offset
        label = f"tb-{offset}"
        print(f"    {label} (pos {pos}): {tokens[pos]!r}")
    print(f"    --- generation_prompt ---")
    for i in range(3):
        print(f"    gen+{i} (pos {gen_start + i}): {tokens[gen_start + i]!r}")
    print()
