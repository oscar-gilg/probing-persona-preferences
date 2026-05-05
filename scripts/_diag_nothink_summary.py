"""Summarize no-nothink diagnostic: L38 c=±0.05 with `model: qwen3.5-122b`."""

import json
from collections import defaultdict
from pathlib import Path

rows = [json.loads(l) for l in Path("/tmp/qwen_diag_nothink.jsonl").read_text().splitlines() if l.strip()]
print(f"Total: {len(rows)} rows")

by_mult = defaultdict(list)
for r in rows:
    by_mult[r["signed_multiplier"]].append(r)

print(f"\n{'mult':>8} {'P(A) (judge)':>14} {'n responded':>12} {'refusal':>10}")
for mult in sorted(by_mult):
    rs = by_mult[mult]
    decided = [r for r in rs if r.get("task_completed") in ("a", "b")]
    p_a = sum(r.get("task_completed") == "a" for r in decided) / max(len(decided), 1)
    refusal = sum(1 for r in rs if r["choice_original"] not in ("a", "b")) / max(len(rs), 1)
    print(f"{mult:+8.3f} {p_a:>14.2f} {len(decided):>12} {refusal:>10.2f}")

p_pos = sum(r.get("task_completed") == "a" for r in by_mult.get(0.05, []) if r.get("task_completed") in ("a", "b"))
n_pos_dec = sum(1 for r in by_mult.get(0.05, []) if r.get("task_completed") in ("a", "b"))
p_neg = sum(r.get("task_completed") == "a" for r in by_mult.get(-0.05, []) if r.get("task_completed") in ("a", "b"))
n_neg_dec = sum(1 for r in by_mult.get(-0.05, []) if r.get("task_completed") in ("a", "b"))
swing = p_pos / max(n_pos_dec, 1) - p_neg / max(n_neg_dec, 1)
print(f"\nSwing: {swing:+.3f}")
print(f"\nCompare with positive_control_v2 (same setup but `model: qwen3.5-122b-nothink`): swing was +0.17")
