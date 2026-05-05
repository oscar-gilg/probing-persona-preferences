"""Summarize coverage-gap diagnostic: probe L24 injected at L16/18/20/22/24."""

import json
from collections import defaultdict
from pathlib import Path

rows = [json.loads(l) for l in Path("/tmp/qwen_diag_cov.jsonl").read_text().splitlines() if l.strip()]
print(f"Total: {len(rows)} rows")

by_layer_mult = defaultdict(list)
for r in rows:
    by_layer_mult[(r["layer"], r["signed_multiplier"])].append(r)

layers = sorted({L for L, _ in by_layer_mult})
print(f"\nProbe = ridge_L24, injected at multiple layers")
print(f"\n{'inject':>8} {'P(A) c=+0.05 (judge)':>22} {'P(A) c=-0.05 (judge)':>22} {'swing':>10} {'refusal':>10}")
for L in layers:
    pos = by_layer_mult[(L, 0.05)]
    neg = by_layer_mult[(L, -0.05)]

    def p_a(rs):
        decided = [r for r in rs if r.get("task_completed") in ("a", "b")]
        return sum(r.get("task_completed") == "a" for r in decided) / max(len(decided), 1), len(decided), len(rs)

    p_pos, n_dec_pos, n_pos = p_a(pos)
    p_neg, n_dec_neg, n_neg = p_a(neg)
    refusal = (sum(1 for r in pos + neg if r["choice_original"] not in ("a", "b"))) / max(len(pos + neg), 1)
    swing = p_pos - p_neg
    print(f"L{L:<7} {p_pos:>22.2f} (n={n_dec_pos}/{n_pos}) {p_neg:>22.2f} (n={n_dec_neg}/{n_neg}) {swing:>+10.2f} {refusal:>10.2f}")
