"""Classify the 6,616 Qwen-10k tasks missing from data/topics/topics.json.

Uses Sonnet 4.6 with minimal reasoning, 40 concurrent workers. Merges results
into topics.json and applies the stresstest_*/bailbench_* → stresstest_other
post-pass for tasks not in sensitive topics.
"""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

from dotenv import load_dotenv

from src.task_data import load_filtered_tasks
from src.task_data.task import OriginDataset
from src.task_data.classification.classify import (
    MODELS,
    REASONING_MINIMAL,
    classify_tasks_batch,
    load_cache,
    save_cache,
)

load_dotenv()

REPO = Path(__file__).resolve().parents[2]
TOPICS_PATH = REPO / "data/topics/topics.json"
CATEGORIES_PATH = REPO / "data/topics/categories.json"
QWEN_10K_CSV = REPO / "results/experiments/qwen35_10k_active_learning/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d/thurstonian_6746a725.csv"

SENSITIVE_TOPICS = {
    "harmful_request",
    "value_conflict",
    "model_manipulation",
    "sensitive_creative",
    "security_legal",
}

ALL_ORIGINS = [
    OriginDataset.WILDCHAT,
    OriginDataset.ALPACA,
    OriginDataset.MATH,
    OriginDataset.BAILBENCH,
    OriginDataset.STRESS_TEST,
]


async def main() -> None:
    print("Identifying missing Qwen 10k task IDs…")
    with QWEN_10K_CSV.open() as f:
        qwen_ids = {row["task_id"] for row in csv.DictReader(f)}
    topics = json.loads(TOPICS_PATH.read_text())
    missing_ids = qwen_ids - set(topics.keys())
    print(f"  Qwen 10k pool: {len(qwen_ids)} tasks")
    print(f"  Already classified: {len(qwen_ids) - len(missing_ids)}")
    print(f"  To classify: {len(missing_ids)}")

    print("\nLoading prompts for missing tasks (this scans all origin files)…")
    tasks = load_filtered_tasks(
        n=10**9,
        origins=ALL_ORIGINS,
        task_ids=missing_ids,
    )
    print(f"  Loaded prompts for {len(tasks)}/{len(missing_ids)} tasks")
    if len(tasks) < len(missing_ids):
        loaded = {t.id for t in tasks}
        print(f"  Could not find prompts for: {sorted(missing_ids - loaded)[:10]}…")

    task_inputs = [{"task_id": t.id, "prompt": t.prompt} for t in tasks]

    print("\nLoading categories from", CATEGORIES_PATH)
    cat_data = json.loads(CATEGORIES_PATH.read_text())
    categories = cat_data["categories"]
    category_descriptions = cat_data.get("descriptions")
    print(f"  {len(categories)} categories: {categories}")

    print("\nRunning classification (Sonnet 4.6, minimal reasoning, 40 concurrent)…")
    cache = load_cache(TOPICS_PATH)
    print(f"  Existing cache size: {len(cache)}")
    cache = await classify_tasks_batch(
        task_inputs,
        categories,
        cache,
        max_concurrent=40,
        category_descriptions=category_descriptions,
        reasoning_body=REASONING_MINIMAL,
    )
    save_cache(cache, TOPICS_PATH)
    print(f"  Saved updated cache to {TOPICS_PATH} (size {len(cache)})")

    print("\nApplying stresstest_*/bailbench_* → stresstest_other post-pass…")
    reclassified = 0
    for tid, entry in cache.items():
        if not (tid.startswith("stresstest_") or tid.startswith("bailbench_")):
            continue
        for model_key in entry:
            primary = entry[model_key]["primary"]
            if primary not in SENSITIVE_TOPICS and primary != "stresstest_other":
                entry[model_key]["primary"] = "stresstest_other"
                reclassified += 1
    print(f"  Reclassified {reclassified} stresstest/bailbench entries to stresstest_other")
    save_cache(cache, TOPICS_PATH)

    print("\nVerifying coverage of Qwen 10k pool now…")
    topics = json.loads(TOPICS_PATH.read_text())
    still_missing = qwen_ids - set(topics.keys())
    print(f"  Qwen 10k tasks still missing: {len(still_missing)}")


if __name__ == "__main__":
    asyncio.run(main())
