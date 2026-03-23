"""Score open-ended steering generations with engagement and anomaly judges.

Reads results.jsonl, runs both judges on each generation, writes scored_results.jsonl.
Supports --resume to skip already-scored entries.

Usage:
    python scripts/open_ended_steering/score.py [--resume]
"""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from src.measurement.elicitation.open_ended_judges import (
    judge_anomaly_async,
    judge_engagement_async,
)

RESULTS_PATH = Path("experiments/steering/open_ended_steering/results.jsonl")
SCORED_PATH = Path("experiments/steering/open_ended_steering/scored_results.jsonl")
CONCURRENCY = 20


async def score_record(record: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        engagement, anomaly = await asyncio.gather(
            judge_engagement_async(record["prompt_text"], record["response"]),
            judge_anomaly_async(record["prompt_text"], record["response"]),
        )
    return {
        **record,
        "engagement_score": engagement.score,
        "is_anomalous": anomaly.is_anomalous,
        "anomaly_description": anomaly.description,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    records = []
    with open(RESULTS_PATH) as f:
        for line in f:
            records.append(json.loads(line))

    done_keys: set[tuple[str, str, float, int]] = set()
    existing: list[dict] = []
    if args.resume and SCORED_PATH.exists():
        with open(SCORED_PATH) as f:
            for line in f:
                r = json.loads(line)
                existing.append(r)
                done_keys.add((r["prompt_id"], r["steering_mode"], r["multiplier"], r["trial"]))
        print(f"Resuming: {len(done_keys)} already scored")

    to_score = [r for r in records if (r["prompt_id"], r["steering_mode"], r["multiplier"], r["trial"]) not in done_keys]
    print(f"Scoring {len(to_score)} / {len(records)} records")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [score_record(r, semaphore) for r in to_score]

    scored = list(existing)
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Scoring"):
        result = await coro
        scored.append(result)

    with open(SCORED_PATH, "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")

    print(f"Done. {len(scored)} scored results saved to {SCORED_PATH}")

    # Print summary
    by_mode: dict[str, list[float]] = {}
    anomaly_count = 0
    for r in scored:
        mode = r["steering_mode"]
        if mode not in by_mode:
            by_mode[mode] = []
        by_mode[mode].append(r["engagement_score"])
        if r["is_anomalous"]:
            anomaly_count += 1

    print(f"\nAnomalous responses: {anomaly_count}/{len(scored)}")
    for mode, scores in sorted(by_mode.items()):
        mean = sum(scores) / len(scores)
        print(f"  {mode}: mean engagement = {mean:.3f} (n={len(scores)})")


if __name__ == "__main__":
    asyncio.run(main())
