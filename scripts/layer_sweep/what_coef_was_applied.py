"""Enumerate the actual coefficient applied to each span in the unilateral runs.

Per the runner:
    effective = _effective_coef(norm * mult * span_coef, ordering)
             = (norm * mult * span_coef)  if ordering == 0
             = -(norm * mult * span_coef) if ordering == 1

For unilateral_first  (spans = {"first": 1}):  the ONLY hook is on the 'first' span.
For unilateral_second (spans = {"second": 1}): the ONLY hook is on the 'second' span.

'First'/'second' is a positional slot in the prompt. At ordering=0 task_a is first; at
ordering=1 task_b is first.

So the actual coefficient applied to the STEERED-span tokens is:

    applied_coef = norm_at_layer * signed_multiplier * span_coef * (+1 if ordering==0 else -1)

with span_coef = +1 in all our unilateral runs.

This reveals a subtle asymmetry: when `signed_multiplier = +0.05` for unilateral_first:
  - ordering=0: +0.05 * N applied to task_a tokens (boosts task_a)
  - ordering=1: -0.05 * N applied to task_b tokens (suppresses task_b)
Both push the model toward task_a in the original-task-space — which is the point of
`_effective_coef` — but the MECHANISM is asymmetric: half the rows boost a, half
suppress b.
"""
import json
from collections import defaultdict
from pathlib import Path

rows = []
for f in ["eot_unilateral_diagonal_early.parsed.jsonl", "eot_unilateral_diagonal_late.parsed.jsonl"]:
    p = Path("experiments/layer_sweep/checkpoints") / f
    if p.exists():
        rows.extend(json.loads(l) for l in p.read_text().splitlines())

# Restrict to L23 for readability (same logic applies at every layer)
L = 23
rows = [r for r in rows if r["layer"] == L]

print(f"Unilateral rows at L{L}: {len(rows)}")
print()
print(f"{'condition':>20}  {'mult':>7}  {'ord':>4}  {'steered task':>13}  {'applied_coef':>14}  {'norm':>8}")
seen = set()
for r in sorted(rows, key=lambda r: (r["condition"], r["signed_multiplier"], r["ordering"])):
    key = (r["condition"], r["signed_multiplier"], r["ordering"])
    if key in seen:
        continue
    seen.add(key)
    cond = r["condition"]
    mult = r["signed_multiplier"]
    ordering = r["ordering"]
    norm = r["norm_at_layer"]
    span_coef = 1  # both unilateral conditions use +1

    # Which slot is steered?
    slot = "first" if cond == "unilateral_first" else "second"
    # Which physical task occupies that slot at this ordering?
    if slot == "first":
        physical_task = "a" if ordering == 0 else "b"
    else:
        physical_task = "b" if ordering == 0 else "a"

    applied_coef = norm * mult * span_coef * (1 if ordering == 0 else -1)

    print(f"{cond:>20}  {mult:>+7.3f}  {ordering:>4}  task_{physical_task} ({slot} span)  {applied_coef:>+14.1f}  {norm:>8.1f}")

# Also show signed_multiplier vs applied_coef mapping for a "positive" push interpretation
print()
print("Summary: what coefficient lands on each PHYSICAL task's tokens at mult=+0.05?")
print()
for cond in ["unilateral_first", "unilateral_second"]:
    for tgt_task in ["a", "b"]:
        coefs = []
        for r in rows:
            if r["condition"] != cond or abs(r["signed_multiplier"] - 0.05) > 1e-6:
                continue
            slot = "first" if cond == "unilateral_first" else "second"
            if slot == "first":
                physical = "a" if r["ordering"] == 0 else "b"
            else:
                physical = "b" if r["ordering"] == 0 else "a"
            if physical != tgt_task:
                continue
            applied = r["norm_at_layer"] * r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1)
            coefs.append(applied)
        if coefs:
            print(f"  {cond:>20}  task_{tgt_task} tokens:  coef = {coefs[0]:+.1f}  ({len(coefs)} rows)")
        else:
            print(f"  {cond:>20}  task_{tgt_task} tokens:  nothing applied")
