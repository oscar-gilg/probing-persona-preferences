"""Extract harmful-request steering examples excluding stresstest/bailbench benign tasks."""

import json
import random
from pathlib import Path

from src.steering.analysis import load_checkpoint

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "experiments/steering/isolated_steering/checkpoint_kv_recompute.jsonl"
TOPICS = ROOT / "data/topics/topics.json"
PAIRS = ROOT / "experiments/revealed_steering_v2/followup/pairs_500.json"
OUTPUT = ROOT / "experiments/steering/isolated_steering/full_run/harmful_steering_examples.json"

TOPIC_MODEL = "anthropic/claude-sonnet-4.5"
SEED = 123

EXCLUDED_PREFIXES = ("stresstest_", "bailbench_")


def get_topic(topics: dict, task_id: str) -> str:
    return topics[task_id][TOPIC_MODEL]["primary"]


def main() -> None:
    rows = load_checkpoint(CHECKPOINT)
    topics = json.loads(TOPICS.read_text())
    pairs = json.loads(PAIRS.read_text())
    pair_lookup = {p["pair_id"]: p for p in pairs}

    # Step 1: Count all harmful pairs vs clean harmful pairs
    all_harmful_pair_ids: set[str] = set()
    clean_harmful_pair_ids: set[str] = set()

    for row in rows:
        task_a_id = row["task_a_id"]
        task_b_id = row["task_b_id"]
        topic_a = get_topic(topics, task_a_id)
        topic_b = get_topic(topics, task_b_id)
        a_harmful = topic_a == "harmful_request"
        b_harmful = topic_b == "harmful_request"

        if a_harmful != b_harmful:
            all_harmful_pair_ids.add(row["pair_id"])

            # Identify the benign task
            benign_id = task_b_id if a_harmful else task_a_id
            if not benign_id.startswith(EXCLUDED_PREFIXES):
                clean_harmful_pair_ids.add(row["pair_id"])

    print(f"Total harmful pairs (exactly one harmful task): {len(all_harmful_pair_ids)}")
    print(f"Clean harmful pairs (benign is NOT stresstest/bailbench): {len(clean_harmful_pair_ids)}")
    print(f"Excluded: {len(all_harmful_pair_ids) - len(clean_harmful_pair_ids)}")

    # Step 2: Annotate rows from clean harmful pairs
    annotated: list[dict] = []
    for row in rows:
        if row["pair_id"] not in clean_harmful_pair_ids:
            continue

        task_a_id = row["task_a_id"]
        task_b_id = row["task_b_id"]
        topic_a = get_topic(topics, task_a_id)
        topic_b = get_topic(topics, task_b_id)

        if topic_a == "harmful_request":
            harmful_task_id = task_a_id
            harmful_topic = topic_a
            benign_task_id = task_b_id
            benign_topic = topic_b
            harmful_is_a = True
        else:
            harmful_task_id = task_b_id
            harmful_topic = topic_b
            benign_task_id = task_a_id
            benign_topic = topic_a
            harmful_is_a = False

        pair_data = pair_lookup[row["pair_id"]]
        harmful_text = pair_data["task_a_text"] if harmful_is_a else pair_data["task_b_text"]
        benign_text = pair_data["task_b_text"] if harmful_is_a else pair_data["task_a_text"]

        # Determine if steered toward harmful
        mult = row["signed_multiplier"]
        if harmful_is_a:
            steered_toward = "harmful" if mult > 0 else "benign"
        else:
            steered_toward = "harmful" if mult < 0 else "benign"

        # Determine what the model chose
        choice = row["choice_original"]
        if choice in ("a", "b"):
            chose_a = choice == "a"
            model_chose = "harmful" if (chose_a == harmful_is_a) else "benign"
        else:
            model_chose = "refusal"

        annotated.append({
            "pair_id": row["pair_id"],
            "harmful_task_id": harmful_task_id,
            "harmful_task_text": harmful_text,
            "benign_task_id": benign_task_id,
            "benign_task_text": benign_text,
            "harmful_topic": harmful_topic,
            "benign_topic": benign_topic,
            "condition": row["condition"],
            "steered_toward": steered_toward,
            "signed_multiplier": mult,
            "ordering": row["ordering"],
            "choice_presented": row["choice_presented"],
            "choice_original": row["choice_original"],
            "raw_response": row["raw_response"],
            "_model_chose": model_chose,  # internal, used for bucketing only
        })

    rng = random.Random(SEED)

    # Sample 5 from each bucket
    buckets = [
        ("kv_steering_recompute", "harmful", "harmful"),
        ("kv_steering_recompute", "harmful", "benign"),
        ("kv_steering_recompute", "harmful", "refusal"),
        ("kv_steering", "harmful", "harmful"),
        ("kv_steering", "harmful", "benign"),
    ]

    examples: list[dict] = []
    for condition, steered_toward, model_chose in buckets:
        pool = [
            r for r in annotated
            if r["condition"] == condition
            and r["steered_toward"] == steered_toward
            and r["_model_chose"] == model_chose
        ]
        n = min(5, len(pool))
        sampled = rng.sample(pool, n)
        label = f"{condition} / steered={steered_toward} / chose={model_chose}"
        if n < 5:
            print(f"  {label}: {len(pool)} available (shortfall: only {n})")
        else:
            print(f"  {label}: {len(pool)} available, sampled {n}")
        examples.extend(sampled)

    # Remove internal bucketing field before saving
    for ex in examples:
        del ex["_model_chose"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(examples, indent=2))
    print(f"\nSaved {len(examples)} examples to {OUTPUT}")


if __name__ == "__main__":
    main()
