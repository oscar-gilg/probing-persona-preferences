"""When NEGATIVE coefficient lands on a span, does the model pick THAT task?

For every trial, identify:
  - which span was being steered (first or second)
  - which physical task (a or b) sat in that span
  - the sign of the applied coefficient on that span
  - whether the model chose that same task

P(chose the task that got the steering | sign of coef on its tokens)
= "does pushing a task's activations toward +direction raise its selection prob?"
  (and conversely for -direction)
"""
import json
from collections import defaultdict
from pathlib import Path

rows = []
for f in ["eot_unilateral_diagonal_early.parsed.jsonl", "eot_unilateral_diagonal_late.parsed.jsonl"]:
    p = Path("experiments/layer_sweep/checkpoints") / f
    if p.exists():
        rows.extend(json.loads(l) for l in p.read_text().splitlines())

def steered_span(cond):
    return "first" if cond == "unilateral_first" else "second"

def physical_task_in_span(span, ordering):
    # ordering=0: first=a, second=b. ordering=1: flipped.
    if span == "first":
        return "a" if ordering == 0 else "b"
    return "b" if ordering == 0 else "a"

def applied_sign(mult, ordering):
    # _effective_coef negates when ordering==1; span_coef is +1 for both unilateral conditions
    base = mult  # span_coef * mult
    return base if ordering == 0 else -base

for L in [17, 20, 23, 26, 29]:
    print(f"\n=== L{L} ===")
    print(f"{'coef sign on steered task':>28}  {'|mult|':>7}  {'n':>5}  {'P(chose that task)':>20}")
    buckets = defaultdict(list)
    for r in rows:
        if r["layer"] != L:
            continue
        span = steered_span(r["condition"])
        tgt_task = physical_task_in_span(span, r["ordering"])
        sign = applied_sign(r["signed_multiplier"], r["ordering"])
        magnitude = abs(r["signed_multiplier"])
        sign_label = "positive (+)" if sign > 0 else "negative (-)"
        picked_that_task = (r["choice_original"] == tgt_task)
        buckets[(sign_label, magnitude)].append(picked_that_task)

    # Compute and print
    for key in sorted(buckets, key=lambda x: (x[0], x[1])):
        sign_label, mag = key
        picks = buckets[key]
        n = len(picks)
        p = sum(picks) / n
        print(f"{sign_label:>28}  {mag:>7.3f}  {n:>5}  {p:>20.3f}")
