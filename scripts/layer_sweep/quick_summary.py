"""Quick summary of steering results available so far."""
import json
import re
from collections import defaultdict
from pathlib import Path

checkpoints = sorted(Path("experiments/layer_sweep/checkpoints").glob("tb-2_probe_L*.parsed.jsonl"))

print("tb-2 DIFFERENTIAL diagonal (probe at same layer as injection):")
print(f"{'layer':>5}  {'n_pairs':>8}  {'mean_|effect|':>14}  {'P(a)@-5%':>10}  {'P(a)@+5%':>10}  {'refuse':>8}")

for cp in checkpoints:
    m = re.search(r"tb-2_probe_L(\d+)", cp.stem)
    probe_L = int(m.group(1))
    rows = [json.loads(l) for l in cp.read_text().splitlines()]

    # Keep only the diagonal cell: injection layer == probe layer
    diag_rows = [r for r in rows if r["layer"] == probe_L]
    if not diag_rows:
        continue

    by_mult = defaultdict(list)
    for r in diag_rows:
        by_mult[r["signed_multiplier"]].append(r)

    def p_a(rs):
        return sum(r["choice_original"] == "a" for r in rs) / len(rs) if rs else float("nan")

    def p_refuse(rs):
        return sum(r["choice_original"] not in ("a", "b") for r in rs) / len(rs) if rs else float("nan")

    p_neg5 = p_a(by_mult[-0.05])
    p_pos5 = p_a(by_mult[0.05])
    effect = abs(p_pos5 - p_neg5)
    refuse_total = p_refuse(diag_rows)
    n_rows_per_mult = len(by_mult[0.05])
    print(f"  L{probe_L:02d}  {n_rows_per_mult:>8}  {effect:>14.3f}  {p_neg5:>10.3f}  {p_pos5:>10.3f}  {refuse_total:>8.3f}")
