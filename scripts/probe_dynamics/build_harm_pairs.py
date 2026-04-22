"""Build 30 harm preference pairs for probe-dynamics experiment.

Each pair: task_a = benign Alpaca task, task_b = harmful BailBench task.
Stratified across BailBench categories. Metric under drift = fraction picking B.
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

DATA = Path("src/task_data/data")
OUT = Path("experiments/probe_dynamics/pairs_harm.json")
SEED = 42

CATEGORIES_KEEP = [
    "Fraud/Deception",
    "Physical Harm",
    "Economic Harm",
    "Disinformation",
    "Harassment/Discrimination",
    "Malware/Hacking",
    "Illegal Sexual/Adult Content",
]


def main() -> None:
    rng = random.Random(SEED)

    bail_by_cat: dict[str, list[str]] = defaultdict(list)
    with (DATA / "bailBench.csv").open() as f:
        for row in csv.DictReader(f):
            cat = row["category"]
            if cat in CATEGORIES_KEEP:
                bail_by_cat[cat].append(row["content"])

    alpaca: list[tuple[str, str]] = []
    with (DATA / "alpaca_tasks_nemocurator.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            alpaca.append((d["task_id"], d["task_text"]))

    rng.shuffle(alpaca)

    per_cat = {cat: max(1, round(30 * len(tasks) / sum(len(t) for t in bail_by_cat.values())))
               for cat, tasks in bail_by_cat.items()}
    total_planned = sum(per_cat.values())
    while total_planned > 30:
        cat = max(per_cat, key=per_cat.get)
        per_cat[cat] -= 1
        total_planned -= 1
    while total_planned < 30:
        cat = min(per_cat, key=per_cat.get)
        per_cat[cat] += 1
        total_planned += 1

    pairs = []
    alpaca_iter = iter(alpaca)
    pid = 0
    for cat, n in per_cat.items():
        bail_sample = rng.sample(bail_by_cat[cat], min(n, len(bail_by_cat[cat])))
        for bail_task in bail_sample:
            alp_id, alp_task = next(alpaca_iter)
            pairs.append({
                "id": f"hp_{pid:02d}",
                "category": cat,
                "task_a": alp_task,
                "task_b": bail_task,
                "source_a": alp_id,
            })
            pid += 1

    payload = {
        "description": "30 harm preference pairs. A = benign Alpaca task. B = harmful BailBench task. Metric: fraction picking B = harm-preference rate.",
        "seed": SEED,
        "n_pairs": len(pairs),
        "categories_per": {cat: per_cat[cat] for cat in per_cat},
        "pairs": pairs,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Saved {OUT} with {len(pairs)} pairs across {len(per_cat)} categories")
    for cat, n in per_cat.items():
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
