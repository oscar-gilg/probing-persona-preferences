"""Show exactly how P(chose steered) is calculated for each condition."""

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

# Layer 25 only
filtered = [r for r in rows if r["layer"] == 25]

by_coef: dict[float, list[dict]] = defaultdict(list)
for r in filtered:
    by_coef[r["signed_multiplier"]].append(r)

print(f"{'coef':>7} {'n':>5} | {'comp_a':>6} {'comp_b':>6} {'neither':>7} | "
      f"{'label_a':>7} {'label_b':>7} {'label_ref':>9} | "
      f"{'steered':>7} {'anti':>5} {'skip':>5} | "
      f"P(st|all) P(st|compl)")
print("-" * 110)

for coef in sorted(by_coef):
    subset = by_coef[coef]
    n = len(subset)

    # Raw counts from judge
    comp_a = sum(1 for r in subset if r["task_completed"] == "a")
    comp_b = sum(1 for r in subset if r["task_completed"] == "b")
    neither = sum(1 for r in subset if r["task_completed"] == "neither")

    # Raw counts from regex label
    lab_a = sum(1 for r in subset if r["choice_original"] == "a")
    lab_b = sum(1 for r in subset if r["choice_original"] == "b")
    lab_ref = sum(1 for r in subset if r["choice_original"] == "refusal")

    # P(chose steered) calculation
    if coef > 0:
        # Steered toward A
        chose_steered = comp_a
        chose_anti = comp_b
    elif coef < 0:
        # Steered toward B
        chose_steered = comp_b
        chose_anti = comp_a
    else:
        chose_steered = 0
        chose_anti = 0

    completed = comp_a + comp_b
    p_steered_all = chose_steered / n if coef != 0 else float("nan")
    p_steered_compl = chose_steered / completed if completed > 0 and coef != 0 else float("nan")

    print(f"{coef:>7.3f} {n:>5} | {comp_a:>6} {comp_b:>6} {neither:>7} | "
          f"{lab_a:>7} {lab_b:>7} {lab_ref:>9} | "
          f"{chose_steered:>7} {chose_anti:>5} {neither:>5} | "
          f"{p_steered_all:>9.3f} {p_steered_compl:>10.3f}")
