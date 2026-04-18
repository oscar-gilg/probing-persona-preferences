"""Inspect the judge output schema."""
import json
from pathlib import Path

for persona in ["default", "sadist"]:
    path = Path(f"experiments/sadist_open_ended_steering/judged_open_ended_{persona}.jsonl")
    n_total = 0
    n_scored = 0
    n_error = 0
    first_keys = None
    error_sample = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            r = json.loads(line)
            if first_keys is None:
                first_keys = list(r.keys())
            if "sadism_score" in r:
                n_scored += 1
            if "judge_error" in r:
                n_error += 1
                if error_sample is None:
                    error_sample = r["judge_error"]
    print(f"{persona}: total={n_total} scored={n_scored} errors={n_error}")
    print(f"  first-row keys: {first_keys}")
    if error_sample:
        print(f"  error sample: {error_sample[:200]}")
