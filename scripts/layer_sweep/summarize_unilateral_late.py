"""Summarize the unilateral late (L38-L59) results per layer × span condition."""
import json
from collections import defaultdict
from pathlib import Path

rows = [json.loads(l) for l in Path("experiments/layer_sweep/checkpoints/eot_unilateral_diagonal_late.parsed.jsonl").read_text().splitlines()]

print(f"total rows: {len(rows)}")
# Expected: 8 layers × 2 conditions × 4 mults × 50 pairs × 2 orderings × 1 trial = 6400
print(f"conditions: {set(r['condition'] for r in rows)}")

# Aggregate: layer × condition × mult → P(a)
agg = defaultdict(list)
for r in rows:
    agg[(r["layer"], r["condition"], r["signed_multiplier"])].append(r)

print(f"\neot UNILATERAL late diagonal (probe_L == inject_L, one span at a time):")
print(f"  P(chose_a) per cell — target depends on condition:")
print(f"    unilateral_first  → +mult pushes toward a")
print(f"    unilateral_second → +mult pushes toward b (invert sign to compare)")
print()
print(f"{'layer':>5}  {'cond':>20}  {'P(a)@-5%':>10}  {'P(a)@-3%':>10}  {'P(a)@+3%':>10}  {'P(a)@+5%':>10}  {'|effect|':>10}  {'refuse':>8}")

for layer in sorted({r["layer"] for r in rows}):
    for cond in sorted({r["condition"] for r in rows}):
        mults = sorted({r["signed_multiplier"] for r in rows})
        p_as = []
        n_refuse = 0
        n_total = 0
        for m in mults:
            cell = agg[(layer, cond, m)]
            if not cell:
                p_as.append(float("nan"))
                continue
            p_a = sum(r["choice_original"] == "a" for r in cell) / len(cell)
            p_as.append(p_a)
            n_refuse += sum(r["choice_original"] not in ("a", "b") for r in cell)
            n_total += len(cell)
        effect = abs(p_as[-1] - p_as[0])
        refuse_rate = n_refuse / n_total if n_total else float("nan")
        print(f"  L{layer:02d}  {cond:>20}  {p_as[0]:>10.3f}  {p_as[1]:>10.3f}  {p_as[2]:>10.3f}  {p_as[3]:>10.3f}  {effect:>10.3f}  {refuse_rate:>8.3f}")
