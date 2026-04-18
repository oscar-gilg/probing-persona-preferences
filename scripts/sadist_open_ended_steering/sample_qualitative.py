"""Sample a few representative judged responses for the report."""
import json
import random
from collections import defaultdict
from pathlib import Path

EXP_DIR = Path("experiments/sadist_open_ended_steering")
TARGETS = [
    ("sadist", -0.03),
    ("sadist", 0.0),
    ("sadist", 0.03),
    ("sadist", 0.05),
]
SEED = 42

for persona, mult in TARGETS:
    path = EXP_DIR / f"judged_{persona}.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "judge_error" in r:
                continue
            if r["multiplier"] != mult:
                continue
            rows.append(r)
    random.Random(SEED).shuffle(rows)
    # Pick one with high sadism (if any) and one with high default-assistant (if any)
    rows.sort(key=lambda r: (-r["sadism_score"], -r["default_assistant_score"]))
    print(f"\n=== {persona} c={mult:+.2f} (top-sadism) ===")
    r = rows[0]
    print(f"PROMPT ({r['prompt_id']}): {r['prompt_text'][:120]}")
    print(f"SCORES sadism={r['sadism_score']} default={r['default_assistant_score']}")
    print(f"RESPONSE: {r['response'][:500]}")
    rows.sort(key=lambda r: (-r["default_assistant_score"], -r["sadism_score"]))
    print(f"\n=== {persona} c={mult:+.2f} (top-default) ===")
    r = rows[0]
    print(f"PROMPT ({r['prompt_id']}): {r['prompt_text'][:120]}")
    print(f"SCORES sadism={r['sadism_score']} default={r['default_assistant_score']}")
    print(f"RESPONSE: {r['response'][:500]}")
