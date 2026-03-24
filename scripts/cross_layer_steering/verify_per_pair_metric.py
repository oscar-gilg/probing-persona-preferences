"""Compute P(chose A) as: for each pair, average across both orderings, then average across pairs."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dotenv import load_dotenv
load_dotenv()

PARSED = Path("experiments/steering/cross_layer/checkpoint_L25.parsed.jsonl")

rows = []
with open(PARSED) as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            if "task_completed" in r:
                rows.append(r)

filtered = [r for r in rows if r["layer"] == 25]

# Group by (coef, pair_id, ordering)
by_key: dict[tuple[float, str, int], list[dict]] = defaultdict(list)
for r in filtered:
    by_key[(r["signed_multiplier"], r["pair_id"], r["ordering"])].append(r)

# For each (coef, pair_id): average P(completed=a) across orderings
by_coef_pair: dict[tuple[float, str], list[float]] = defaultdict(list)
for (coef, pair_id, ordering), rr in by_key.items():
    # P(completed=a) for this (pair, ordering, coef) across trials
    p_a = np.mean([1 if r["task_completed"] == "a" else 0 for r in rr])
    by_coef_pair[(coef, pair_id)].append(p_a)

# For each (coef, pair_id): average across orderings
pair_means: dict[float, list[float]] = defaultdict(list)
for (coef, pair_id), ordering_means in by_coef_pair.items():
    pair_mean = np.mean(ordering_means)  # average across 2 orderings
    pair_means[coef].append(pair_mean)

# Average across pairs
print(f"{'coef':>7} {'n_pairs':>7} {'P(A) per-pair avg':>18} {'P(A) naive pool':>16}")
for coef in sorted(pair_means):
    per_pair = np.mean(pair_means[coef])
    # Compare with naive pooling (just count all comp_a / n)
    all_rows = [r for r in filtered if r["signed_multiplier"] == coef]
    naive = np.mean([1 if r["task_completed"] == "a" else 0 for r in all_rows])
    print(f"{coef:>7.3f} {len(pair_means[coef]):>7} {per_pair:>18.4f} {naive:>16.4f}")
