"""Sanity check: verify tb-2 L23 contrastive numbers directly."""
import json
from collections import defaultdict
from pathlib import Path

rows = [json.loads(l) for l in Path("experiments/layer_sweep/checkpoints/tb-2_probe_L23.parsed.jsonl").read_text().splitlines()]
print(f"Total rows: {len(rows)}")
print(f"Layers present: {sorted({r['layer'] for r in rows})}")
print(f"Multipliers present: {sorted({r['signed_multiplier'] for r in rows})}")
print(f"Conditions: {sorted({r['condition'] for r in rows})}")

# Filter to L23 only
l23 = [r for r in rows if r["layer"] == 23]
print(f"\nAt L23: {len(l23)} rows")
print(f"{'mult':>7}  {'n':>4}  {'P(a)':>7}  {'P(b)':>7}  {'P(ref)':>7}")
by_mult = defaultdict(list)
for r in l23:
    by_mult[r["signed_multiplier"]].append(r)
for m in sorted(by_mult):
    rs = by_mult[m]
    pa = sum(r["choice_original"] == "a" for r in rs) / len(rs)
    pb = sum(r["choice_original"] == "b" for r in rs) / len(rs)
    pr = 1 - pa - pb
    print(f"  {m:>+.3f}  {len(rs):>4}  {pa:>7.3f}  {pb:>7.3f}  {pr:>7.3f}")
