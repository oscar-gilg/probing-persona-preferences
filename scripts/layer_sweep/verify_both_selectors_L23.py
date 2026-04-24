import json
from collections import defaultdict
from pathlib import Path
for f in ["tb-2_probe_L23", "eot_probe_L23"]:
    p = Path(f"experiments/layer_sweep/checkpoints/{f}.parsed.jsonl")
    if not p.exists():
        print(f"{f}: missing")
        continue
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    l23 = [r for r in rows if r["layer"] == 23]
    by_m = defaultdict(list)
    for r in l23: by_m[r["signed_multiplier"]].append(r)
    print(f"{f}:")
    for m in sorted(by_m):
        rs = by_m[m]
        pa = sum(r["choice_original"]=="a" for r in rs)/len(rs)
        print(f"  mult={m:+.3f}  P(a)={pa:.3f}  n={len(rs)}")
