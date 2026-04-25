"""Inspect base stimulus structure."""
import json
for d in ["truth", "harm", "politics"]:
    items = json.load(open(f"experiments/token_level_probes/data/{d}_filtered.json"))
    by_turn_cond = {}
    for it in items:
        k = (it["turn"], it["condition"])
        by_turn_cond[k] = by_turn_cond.get(k, 0) + 1
    print(f"{d}: total={len(items)}")
    for k, v in sorted(by_turn_cond.items()):
        print(f"  {k}: {v}")
