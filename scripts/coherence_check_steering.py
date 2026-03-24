"""Post-hoc coherence check on steering checkpoint.

Samples 100 rows per (condition, |coefficient|) bucket and runs the
CoherenceJudgment LLM judge to measure what fraction of generations are
intelligible English that addresses a task. Saves results as JSONL.

Usage:
    python -m scripts.coherence_check_steering <checkpoint_path> <pairs_path> <output_path>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.measurement.elicitation.coherence_judge import judge_coherence_async


def load_and_deduplicate(checkpoint_path: Path) -> list[dict]:
    seen: dict[tuple, dict] = {}
    with open(checkpoint_path) as f:
        for line in f:
            row = json.loads(line)
            key = (
                row["pair_id"],
                row["signed_multiplier"],
                row["condition"],
                row["sample_idx"],
                row["ordering"],
            )
            seen[key] = row
    return list(seen.values())


def sample_rows(
    rows: list[dict], n_per_bucket: int, seed: int
) -> dict[tuple[str, float], list[dict]]:
    """Group by (condition, |multiplier|) and sample n_per_bucket from each."""
    buckets: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["condition"], abs(r["signed_multiplier"]))
        buckets[key].append(r)

    rng = random.Random(seed)
    sampled = {}
    for key, bucket_rows in sorted(buckets.items()):
        if len(bucket_rows) <= n_per_bucket:
            sampled[key] = bucket_rows
        else:
            sampled[key] = rng.sample(bucket_rows, n_per_bucket)
    return sampled


async def judge_batch(
    rows: list[dict],
    pair_lookup: dict[str, dict],
    concurrency: int = 20,
) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def judge_one(row: dict) -> dict | None:
        pair = pair_lookup[row["pair_id"]]
        async with sem:
            try:
                judgment = await judge_coherence_async(
                    response=row["raw_response"],
                    task_a_text=pair["task_a_text"][:200],
                    task_b_text=pair["task_b_text"][:200],
                )
                coherent = judgment.coherent
            except Exception as e:
                print(f"  Judge failed for {row['pair_id']}: {e!r}")
                return None
        return {
            "pair_id": row["pair_id"],
            "condition": row["condition"],
            "signed_multiplier": row["signed_multiplier"],
            "abs_multiplier": abs(row["signed_multiplier"]),
            "ordering": row["ordering"],
            "sample_idx": row["sample_idx"],
            "choice_original": row["choice_original"],
            "coherent": coherent,
            "raw_response_preview": row["raw_response"][:150],
        }

    results = await asyncio.gather(*(judge_one(r) for r in rows))
    return [r for r in results if r is not None]


def print_summary(results: list[dict]) -> None:
    buckets: dict[tuple[str, float], list[bool]] = defaultdict(list)
    for r in results:
        key = (r["condition"], r["abs_multiplier"])
        buckets[key].append(r["coherent"])

    print(f"\n{'condition':<30} {'|coef|':>8} {'coherent':>10} {'n':>5}")
    print("-" * 60)
    for (cond, mult), coherence_flags in sorted(buckets.items()):
        rate = sum(coherence_flags) / len(coherence_flags)
        print(f"{cond:<30} {mult:>8.2f} {rate:>9.1%} {len(coherence_flags):>5}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("pairs_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--n-per-bucket", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    pairs = json.loads(args.pairs_path.read_text())
    pair_lookup = {p["pair_id"]: p for p in pairs}

    print("Loading and deduplicating checkpoint...")
    rows = load_and_deduplicate(args.checkpoint_path)
    print(f"  {len(rows)} unique rows")

    print(f"Sampling {args.n_per_bucket} per (condition, |coef|) bucket...")
    sampled = sample_rows(rows, args.n_per_bucket, args.seed)
    total = sum(len(v) for v in sampled.values())
    print(f"  {len(sampled)} buckets, {total} total samples")

    all_rows = [r for bucket_rows in sampled.values() for r in bucket_rows]
    print(f"Running coherence judge on {len(all_rows)} samples...")
    results = asyncio.run(judge_batch(all_rows, pair_lookup, args.concurrency))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved to {args.output_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
