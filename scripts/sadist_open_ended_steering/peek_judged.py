"""Peek at a few judged rows."""
import json
from pathlib import Path

for persona in ["default", "sadist"]:
    path = Path(f"experiments/sadist_open_ended_steering/judged_{persona}.jsonl")
    if not path.exists():
        continue
    print(f"=== {persona} ===")
    with open(path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if "judge_error" in r:
                continue
            print(f"{r['prompt_id']} mult={r['multiplier']:+.2f}  sadism={r['sadism_score']} default={r['default_assistant_score']}  {r.get('judge_justification','')[:120]}")
            if i >= 4:
                break
