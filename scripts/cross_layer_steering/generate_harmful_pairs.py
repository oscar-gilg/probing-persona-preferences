"""Generate 200 task pairs with at least one harmful task per pair.

Mix: ~150 harmful+benign pairs, ~50 harmful+harmful pairs.

Usage:
    python -m scripts.cross_layer_steering.generate_harmful_pairs
"""

import csv
import json
import random
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.task_data import OriginDataset, load_tasks

SCORES_PATH = Path(
    "results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_80fa9dc8.csv"
)
TOPICS_PATH = Path("data/topics/topics.json")
OUTPUT_PATH = Path("experiments/steering/cross_layer_harmful/pairs_200.json")
N_HARMFUL_BENIGN = 150
N_HARMFUL_HARMFUL = 50
SEED = 123

HARMFUL_ORIGINS = [OriginDataset.STRESS_TEST, OriginDataset.BAILBENCH]
BENIGN_ORIGINS = [OriginDataset.ALPACA, OriginDataset.WILDCHAT, OriginDataset.MATH]


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

    harmful_tasks = load_tasks(n=100000, origins=HARMFUL_ORIGINS)
    benign_tasks = load_tasks(n=100000, origins=BENIGN_ORIGINS)

    harmful_pool = {t.id: t.prompt for t in harmful_tasks if t.id in scores}
    benign_pool = {t.id: t.prompt for t in benign_tasks if t.id in scores}

    print(f"Harmful tasks with scores: {len(harmful_pool)}")
    print(f"Benign tasks with scores: {len(benign_pool)}")

    rng = random.Random(SEED)
    harmful_ids = list(harmful_pool.keys())
    benign_ids = list(benign_pool.keys())
    rng.shuffle(harmful_ids)
    rng.shuffle(benign_ids)

    pairs = []
    pair_idx = 0

    # Harmful + benign pairs
    for i in range(N_HARMFUL_BENIGN):
        h_id = harmful_ids[i]
        b_id = benign_ids[i]
        # Randomly assign harmful to A or B
        if rng.random() < 0.5:
            a_id, b_id_ = h_id, b_id
        else:
            a_id, b_id_ = b_id, h_id
        a_text = harmful_pool.get(a_id) or benign_pool[a_id]
        b_text = harmful_pool.get(b_id_) or benign_pool[b_id_]
        pairs.append({
            "pair_id": f"pair_{pair_idx:04d}",
            "task_a": a_id,
            "task_b": b_id_,
            "task_a_text": a_text,
            "task_b_text": b_text,
            "mu_a": scores[a_id],
            "mu_b": scores[b_id_],
            "delta_mu": scores[a_id] - scores[b_id_],
            "topic_a": topics.get(a_id, "unknown"),
            "topic_b": topics.get(b_id_, "unknown"),
            "pair_type": "harmful_benign",
        })
        pair_idx += 1

    # Harmful + harmful pairs
    offset = N_HARMFUL_BENIGN
    for i in range(N_HARMFUL_HARMFUL):
        a_id = harmful_ids[offset + 2 * i]
        b_id = harmful_ids[offset + 2 * i + 1]
        pairs.append({
            "pair_id": f"pair_{pair_idx:04d}",
            "task_a": a_id,
            "task_b": b_id,
            "task_a_text": harmful_pool[a_id],
            "task_b_text": harmful_pool[b_id],
            "mu_a": scores[a_id],
            "mu_b": scores[b_id],
            "delta_mu": scores[a_id] - scores[b_id],
            "topic_a": topics.get(a_id, "unknown"),
            "topic_b": topics.get(b_id, "unknown"),
            "pair_type": "harmful_harmful",
        })
        pair_idx += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"\nGenerated {len(pairs)} pairs → {OUTPUT_PATH}")
    print(f"  harmful+benign: {N_HARMFUL_BENIGN}")
    print(f"  harmful+harmful: {N_HARMFUL_HARMFUL}")
    n_harmful = sum(1 for p in pairs if p["pair_type"] == "harmful_harmful")
    print(f"  verify: {n_harmful} harmful+harmful, {len(pairs) - n_harmful} harmful+benign")


if __name__ == "__main__":
    main()
