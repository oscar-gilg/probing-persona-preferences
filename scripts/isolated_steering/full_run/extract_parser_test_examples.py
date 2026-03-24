"""Extract diverse examples for building a completion parser test set.

Samples from kv_steering_recompute condition across 5 categories (3 each).
"""

import json
import random
from pathlib import Path

from src.steering.analysis import load_checkpoint

CHECKPOINT = "experiments/steering/isolated_steering/checkpoint_kv_recompute.jsonl"
TOPICS = "data/topics/topics.json"
PAIRS = "experiments/revealed_steering_v2/followup/pairs_500.json"
OUTPUT = "experiments/steering/isolated_steering/full_run/parser_test_examples.json"

SENSITIVE_TOPICS = {"harmful_request", "value_conflict", "model_manipulation", "security_legal", "sensitive_creative"}
BENIGN_TOPICS = {"math", "knowledge_qa", "content_generation", "fiction", "coding", "persuasive_writing", "summarization", "stresstest_other", "other"}

CLASSIFIER = "anthropic/claude-sonnet-4.5"
SEED = 42
N_PER_CATEGORY = 3


def get_topic(topics: dict, task_id: str) -> str:
    return topics[task_id][CLASSIFIER]["primary"]


def steered_toward_task_id(row: dict) -> str:
    if row["signed_multiplier"] > 0:
        return "a"
    elif row["signed_multiplier"] < 0:
        return "b"
    raise ValueError("signed_multiplier is 0")


def build_example(row: dict, pairs_by_id: dict, topics: dict) -> dict:
    pair = pairs_by_id[row["pair_id"]]
    return {
        "pair_id": row["pair_id"],
        "task_a_text": pair["task_a_text"],
        "task_b_text": pair["task_b_text"],
        "raw_response": row["raw_response"],
        "choice_presented": row["choice_presented"],
        "ordering": row["ordering"],
        "signed_multiplier": row["signed_multiplier"],
        "topic_a": get_topic(topics, row["task_a_id"]),
        "topic_b": get_topic(topics, row["task_b_id"]),
        "task_a_id": row["task_a_id"],
        "task_b_id": row["task_b_id"],
    }


def main() -> None:
    random.seed(SEED)

    rows = load_checkpoint(CHECKPOINT)
    topics = json.loads(Path(TOPICS).read_text())
    pairs = json.loads(Path(PAIRS).read_text())
    pairs_by_id = {p["pair_id"]: p for p in pairs}

    # Filter to kv_steering_recompute only
    rows = [r for r in rows if r["condition"] == "kv_steering_recompute"]
    print(f"Total kv_steering_recompute rows: {len(rows)}")

    # Precompute topics for each row
    for r in rows:
        r["_topic_a"] = get_topic(topics, r["task_a_id"])
        r["_topic_b"] = get_topic(topics, r["task_b_id"])

    categories: dict[str, list[dict]] = {}

    # 1. clean_comply: both benign, valid choice
    pool = [
        r for r in rows
        if r["_topic_a"] in BENIGN_TOPICS
        and r["_topic_b"] in BENIGN_TOPICS
        and r["choice_presented"] in ("a", "b")
    ]
    print(f"clean_comply pool: {len(pool)}")
    categories["clean_comply"] = random.sample(pool, N_PER_CATEGORY)

    # 2. harmful_chosen: steered toward harmful task, model complies
    pool = [
        r for r in rows
        if r["choice_presented"] in ("a", "b")
        and r["signed_multiplier"] != 0
        and (
            (steered_toward_task_id(r) == "a" and r["_topic_a"] in SENSITIVE_TOPICS and r["choice_original"] == "a")
            or (steered_toward_task_id(r) == "b" and r["_topic_b"] in SENSITIVE_TOPICS and r["choice_original"] == "b")
        )
    ]
    print(f"harmful_chosen pool: {len(pool)}")
    categories["harmful_chosen"] = random.sample(pool, N_PER_CATEGORY)

    # 3. harmful_avoided: steered toward harmful task, model picks the other
    pool = [
        r for r in rows
        if r["choice_presented"] in ("a", "b")
        and r["signed_multiplier"] != 0
        and (
            (steered_toward_task_id(r) == "a" and r["_topic_a"] in SENSITIVE_TOPICS and r["choice_original"] == "b")
            or (steered_toward_task_id(r) == "b" and r["_topic_b"] in SENSITIVE_TOPICS and r["choice_original"] == "a")
        )
    ]
    print(f"harmful_avoided pool: {len(pool)}")
    categories["harmful_avoided"] = random.sample(pool, N_PER_CATEGORY)

    # 4. refusal: choice_presented/choice_original is "refusal", at least one sensitive topic
    pool = [
        r for r in rows
        if r["choice_original"] == "refusal"
        and (r["_topic_a"] in SENSITIVE_TOPICS or r["_topic_b"] in SENSITIVE_TOPICS)
    ]
    print(f"refusal pool: {len(pool)}")
    categories["refusal"] = random.sample(pool, N_PER_CATEGORY)

    # 5. label_content_possible_mismatch: tasks with very different topics
    # Pick pairs where topics are maximally different (e.g., math vs fiction, coding vs knowledge_qa)
    DIVERSE_PAIRS = {
        ("math", "fiction"), ("fiction", "math"),
        ("math", "content_generation"), ("content_generation", "math"),
        ("coding", "fiction"), ("fiction", "coding"),
        ("coding", "knowledge_qa"), ("knowledge_qa", "coding"),
        ("math", "knowledge_qa"), ("knowledge_qa", "math"),
        ("coding", "content_generation"), ("content_generation", "coding"),
        ("summarization", "math"), ("math", "summarization"),
        ("fiction", "knowledge_qa"), ("knowledge_qa", "fiction"),
    }
    pool = [
        r for r in rows
        if r["choice_presented"] in ("a", "b")
        and (r["_topic_a"], r["_topic_b"]) in DIVERSE_PAIRS
    ]
    print(f"label_content_possible_mismatch pool: {len(pool)}")
    categories["label_content_possible_mismatch"] = random.sample(pool, N_PER_CATEGORY)

    # Build output
    output = {}
    for cat_name, cat_rows in categories.items():
        output[cat_name] = [build_example(r, pairs_by_id, topics) for r in cat_rows]

    Path(OUTPUT).write_text(json.dumps(output, indent=2))
    print(f"\nSaved {sum(len(v) for v in output.values())} examples to {OUTPUT}")

    # Print summary
    for cat_name, examples in output.items():
        print(f"\n=== {cat_name} ({len(examples)} examples) ===")
        for ex in examples:
            print(f"  pair_id: {ex['pair_id']}")
            print(f"    task_a ({ex['topic_a']}): {ex['task_a_text'][:80]}")
            print(f"    task_b ({ex['topic_b']}): {ex['task_b_text'][:80]}")
            print(f"    choice_presented: {ex['choice_presented']}, ordering: {ex['ordering']}")
            print(f"    raw_response: {ex['raw_response'][:80]}")
            print()


if __name__ == "__main__":
    main()
