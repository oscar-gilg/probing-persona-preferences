"""Generate random task pairs for steering experiment, stratified by topic.

Usage:
    python -m scripts.cross_layer_steering.generate_pairs
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.task_data import OriginDataset, load_tasks

SCORES_PATH = Path(
    "results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_80fa9dc8.csv"
)
TOPICS_PATH = Path("data/topics/topics.json")
OUTPUT_PATH = Path("experiments/steering/cross_layer/pairs_500.json")
N_PAIRS = 500
SEED = 42

ALL_ORIGINS = [
    OriginDataset.ALPACA,
    OriginDataset.WILDCHAT,
    OriginDataset.MATH,
]


def load_scores() -> dict[str, float]:
    scores = {}
    with open(SCORES_PATH) as f:
        for row in csv.DictReader(f):
            scores[row["task_id"]] = float(row["mu"])
    return scores


def load_topics() -> dict[str, str]:
    with open(TOPICS_PATH) as f:
        raw = json.load(f)
    topics = {}
    for task_id, models in raw.items():
        for model_data in models.values():
            topics[task_id] = model_data["primary"]
            break
    return topics


def main():
    scores = load_scores()
    topics = load_topics()

    # Load all tasks from non-harmful origins
    all_tasks = load_tasks(n=100000, origins=ALL_ORIGINS)
    task_lookup = {t.id: t.prompt for t in all_tasks if t.id in scores and t.id in topics}
    print(f"Tasks with scores + topics + text: {len(task_lookup)}")

    # Group by topic for stratified sampling
    by_topic: dict[str, list[str]] = defaultdict(list)
    for task_id in task_lookup:
        by_topic[topics[task_id]].append(task_id)

    print(f"Topics: {len(by_topic)}")
    for topic, ids in sorted(by_topic.items(), key=lambda x: -len(x[1])):
        print(f"  {topic}: {len(ids)}")

    # Sample 2*N_PAIRS tasks proportional to topic distribution
    rng = random.Random(SEED)
    n_tasks = 2 * N_PAIRS
    total = len(task_lookup)

    sampled: list[str] = []
    for topic, ids in sorted(by_topic.items()):
        k = max(1, round(len(ids) / total * n_tasks))
        sampled.extend(rng.sample(ids, min(k, len(ids))))

    rng.shuffle(sampled)
    if len(sampled) > n_tasks:
        sampled = sampled[:n_tasks]
    elif len(sampled) < n_tasks:
        remaining = [t for t in task_lookup if t not in set(sampled)]
        sampled.extend(rng.sample(remaining, n_tasks - len(sampled)))

    # Shuffle and pair
    rng.shuffle(sampled)
    pairs = []
    for i in range(N_PAIRS):
        a_id = sampled[2 * i]
        b_id = sampled[2 * i + 1]
        mu_a = scores[a_id]
        mu_b = scores[b_id]
        pairs.append({
            "pair_id": f"pair_{i:04d}",
            "task_a": a_id,
            "task_b": b_id,
            "task_a_text": task_lookup[a_id],
            "task_b_text": task_lookup[b_id],
            "mu_a": mu_a,
            "mu_b": mu_b,
            "delta_mu": mu_a - mu_b,
            "topic_a": topics[a_id],
            "topic_b": topics[b_id],
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"\nGenerated {len(pairs)} pairs → {OUTPUT_PATH}")
    deltas = [abs(p["delta_mu"]) for p in pairs]
    print(f"  |delta_mu| mean={sum(deltas)/len(deltas):.2f}, "
          f"min={min(deltas):.2f}, max={max(deltas):.2f}")


if __name__ == "__main__":
    main()
