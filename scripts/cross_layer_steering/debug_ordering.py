"""Show every (coef, ordering) condition to verify P(chose steered) computation."""

import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

PARSED = Path("experiments/steering/cross_layer/checkpoint_L25.parsed.jsonl")

rows = []
with open(PARSED) as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            if "task_completed" in r:
                rows.append(r)

filtered = [r for r in rows if r["layer"] == 25]

by_key: dict[tuple[float, int], list[dict]] = defaultdict(list)
for r in filtered:
    by_key[(r["signed_multiplier"], r["ordering"])].append(r)

print("How steering works:")
print("  ordering=0: A presented first.  effective_coef = +mult * norm.  Positive mult → +dir on A, -dir on B → steers toward A")
print("  ordering=1: B presented first.  effective_coef = -mult * norm.  Positive mult → +dir on A, -dir on B → steers toward A")
print("  (Both orderings steer the same direction in original task space)")
print()

print(f"{'coef':>7} {'ord':>3} {'n':>5} | {'comp_a':>6} {'comp_b':>6} {'neither':>7} | "
      f"{'steered_toward':>14} {'chose_steered':>13} {'chose_other':>11} | P(st|all)")
print("-" * 105)

for coef in sorted(set(c for c, o in by_key)):
    for ordering in [0, 1]:
        key = (coef, ordering)
        if key not in by_key:
            continue
        subset = by_key[key]
        n = len(subset)

        comp_a = sum(1 for r in subset if r["task_completed"] == "a")
        comp_b = sum(1 for r in subset if r["task_completed"] == "b")
        neither = sum(1 for r in subset if r["task_completed"] == "neither")

        if coef > 0:
            steered_toward = "orig_A"
            chose_steered = comp_a
            chose_other = comp_b
        elif coef < 0:
            steered_toward = "orig_B"
            chose_steered = comp_b
            chose_other = comp_a
        else:
            steered_toward = "none"
            chose_steered = 0
            chose_other = 0

        p = chose_steered / n if coef != 0 else float("nan")

        print(f"{coef:>7.3f} {ordering:>3} {n:>5} | {comp_a:>6} {comp_b:>6} {neither:>7} | "
              f"{steered_toward:>14} {chose_steered:>13} {chose_other:>11} | {p:>9.3f}")
    # Print combined row
    all_rows = by_key.get((coef, 0), []) + by_key.get((coef, 1), [])
    n_all = len(all_rows)
    comp_a_all = sum(1 for r in all_rows if r["task_completed"] == "a")
    comp_b_all = sum(1 for r in all_rows if r["task_completed"] == "b")
    neither_all = sum(1 for r in all_rows if r["task_completed"] == "neither")
    if coef > 0:
        cs = comp_a_all
        co = comp_b_all
    elif coef < 0:
        cs = comp_b_all
        co = comp_a_all
    else:
        cs = 0
        co = 0
    p_all = cs / n_all if coef != 0 else float("nan")
    print(f"  COMBINED {n_all:>5} | {comp_a_all:>6} {comp_b_all:>6} {neither_all:>7} | "
          f"{'':>14} {cs:>13} {co:>11} | {p_all:>9.3f}")
    print()
