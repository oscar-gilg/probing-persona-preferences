"""Aggregate judged open-ended responses and print a summary per (persona, coef)."""
import json
from collections import defaultdict
from pathlib import Path

EXP_DIR = Path("experiments/sadist_open_ended_steering")

for persona in ["default", "sadist"]:
    path = EXP_DIR / f"judged_open_ended_{persona}.jsonl"
    if not path.exists():
        print(f"[skip] {persona}: no judged file")
        continue
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "judge_error" not in r:
                rows.append(r)
    by_coef = defaultdict(list)
    for r in rows:
        by_coef[r["multiplier"]].append((r["sadism_score"], r["default_assistant_score"]))
    print(f"\n=== {persona} (open-ended prompts: no harm axis, no task to refuse) ===")
    print(f"{'c':>6} {'n':>4}  {'sadism':>8}  {'default':>8}")
    for coef in sorted(by_coef):
        vals = by_coef[coef]
        sadism = sum(v[0] for v in vals) / len(vals)
        default = sum(v[1] for v in vals) / len(vals)
        print(f"{coef:+6.2f} {len(vals):>4}  {sadism:>8.2f}  {default:>8.2f}")
