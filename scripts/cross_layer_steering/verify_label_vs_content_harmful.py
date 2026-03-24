"""Compare P(label=A) vs P(content=A) for harmful-benign pairs at layer 25."""

import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import sys
PARSED = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/steering/cross_layer_harmful/checkpoint.parsed.jsonl")

rows = []
with open(PARSED) as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            if "task_completed" in r:
                rows.append(r)

# Focus on probe_L25, layer 25
filtered = [r for r in rows if r["condition"] == "probe_L25" and r["layer"] == 25]

# Group by signed_multiplier
by_mult: dict[float, list[dict]] = defaultdict(list)
for r in filtered:
    by_mult[r["signed_multiplier"]].append(r)

print(f"{'mult':>8} {'n':>5} {'P(label=A)':>11} {'P(content=A)':>13} {'P(content=neither)':>19}")
for mult in sorted(by_mult):
    subset = by_mult[mult]
    n = len(subset)
    # Label: choice_original (regex, remapped for ordering)
    label_a = sum(1 for r in subset if r["choice_original"] == "a") / n
    # Content: task_completed (judge)
    content_a = sum(1 for r in subset if r["task_completed"] == "a") / n
    content_neither = sum(1 for r in subset if r["task_completed"] == "neither") / n
    print(f"{mult:>8.3f} {n:>5} {label_a:>11.3f} {content_a:>13.3f} {content_neither:>19.3f}")
