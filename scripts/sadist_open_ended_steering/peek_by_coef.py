"""Summarize judged rows by coefficient for sanity check."""
import json
from collections import defaultdict
from pathlib import Path

for persona in ["default", "sadist"]:
    path = Path(f"experiments/sadist_open_ended_steering/judged_{persona}.jsonl")
    if not path.exists():
        continue
    by_coef = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "judge_error" in r:
                continue
            by_coef[r["multiplier"]].append((r["sadism_score"], r["default_assistant_score"]))
    print(f"\n=== {persona} ===")
    for coef in sorted(by_coef):
        vals = by_coef[coef]
        sadism = sum(v[0] for v in vals) / len(vals)
        default = sum(v[1] for v in vals) / len(vals)
        print(f"  mult={coef:+.2f}  n={len(vals):4d}  sadism={sadism:.2f}  default={default:.2f}")
