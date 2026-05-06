"""Summarize tb-4 probe contrastive at L38."""

import json
from collections import defaultdict
from pathlib import Path

rows = [json.loads(l) for l in Path("/tmp/qwen_diag_tb4.jsonl").read_text().splitlines() if l.strip()]
print(f"Total: {len(rows)} rows (10 pairs × 2 orderings × 3 multipliers)")

by_mult = defaultdict(list)
for r in rows:
    by_mult[r["signed_multiplier"]].append(r)

print(f"\ntb-4 probe (heldout-r peak, r=0.946) at L38, contrastive c=±0.05:")
print(f"\n{'mult':>8} {'P(A) (judge)':>14} {'n responded':>12} {'refusal':>10}")
for mult in sorted(by_mult):
    rs = by_mult[mult]
    decided = [r for r in rs if r.get("task_completed") in ("a", "b")]
    p_a = sum(r.get("task_completed") == "a" for r in decided) / max(len(decided), 1)
    refusal = sum(1 for r in rs if r.get("task_completed") not in ("a", "b")) / max(len(rs), 1)
    print(f"{mult:+8.3f} {p_a:>14.2f} {len(decided):>12} {refusal:>10.2f}")

p_pos = sum(r.get("task_completed") == "a" for r in by_mult.get(0.05, []) if r.get("task_completed") in ("a", "b"))
n_pos = sum(1 for r in by_mult.get(0.05, []) if r.get("task_completed") in ("a", "b"))
p_neg = sum(r.get("task_completed") == "a" for r in by_mult.get(-0.05, []) if r.get("task_completed") in ("a", "b"))
n_neg = sum(1 for r in by_mult.get(-0.05, []) if r.get("task_completed") in ("a", "b"))
swing = p_pos / max(n_pos, 1) - p_neg / max(n_neg, 1)
print(f"\nSwing: {swing:+.3f}")
print(f"\nFor comparison: tb-1 probe at L38 swing = +0.17 (positive_control_v2)")
