"""Summarize multi-layer simultaneous contrastive diagnostic."""

import json
from collections import defaultdict
from pathlib import Path

rows = [json.loads(l) for l in Path("/tmp/qwen_diag_multi.jsonl").read_text().splitlines() if l.strip()]
print(f"Total: {len(rows)} rows (10 pairs × 2 orderings × 3 multipliers)")

by_mult = defaultdict(list)
for r in rows:
    by_mult[r["signed_multiplier"]].append(r)

print(f"\nMethod: contrastive at L{rows[0]['layers']} simultaneously, per-layer probes ridge_L<L>")
print(f"\n{'mult':>8} {'P(A) (judge)':>14} {'P(B) (judge)':>14} {'n responded':>12} {'refusal':>10}")
for mult in sorted(by_mult):
    rs = by_mult[mult]
    decided = [r for r in rs if r.get("task_completed") in ("a", "b")]
    p_a = sum(r.get("task_completed") == "a" for r in decided) / max(len(decided), 1)
    p_b = sum(r.get("task_completed") == "b" for r in decided) / max(len(decided), 1)
    refusal = sum(1 for r in rs if r.get("task_completed") not in ("a", "b")) / max(len(rs), 1)
    print(f"{mult:+8.3f} {p_a:>14.2f} {p_b:>14.2f} {len(decided):>12} {refusal:>10.2f}")

p_pos = sum(r.get("task_completed") == "a" for r in by_mult.get(0.05, []) if r.get("task_completed") in ("a", "b"))
n_pos = sum(1 for r in by_mult.get(0.05, []) if r.get("task_completed") in ("a", "b"))
p_neg = sum(r.get("task_completed") == "a" for r in by_mult.get(-0.05, []) if r.get("task_completed") in ("a", "b"))
n_neg = sum(1 for r in by_mult.get(-0.05, []) if r.get("task_completed") in ("a", "b"))
swing = p_pos / max(n_pos, 1) - p_neg / max(n_neg, 1)
print(f"\nSwing P(A|c=+0.05) - P(A|c=-0.05): {swing:+.3f}")
print(f"\nFor comparison:")
print(f"  Single-layer at L38 contrastive (positive_control_v2):  swing = +0.17")
print(f"  6-layer self-layer scan (lite):                          best swing = +0.06 (judge)")
print(f"  All-tokens at L24 (probe L24):                           swing = -0.08")
print(f"  Gemma contrastive L23 c=±0.05:                           swing = +0.94")
