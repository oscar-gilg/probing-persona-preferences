"""Compare coherence judge across cheap models.

Re-judges a sample from the existing coherence results using different models
to check for judge-model sensitivity. Focuses on the borderline cases
(existing incoherent judgments) plus a control sample of coherent ones.

Usage:
    python -m scripts.coherence_judge_comparison
"""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import instructor
import os
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.measurement.elicitation.coherence_judge import _PAIRWISE_SYSTEM_PROMPT

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODELS = [
    "google/gemini-2.5-flash",
    "openai/gpt-4.1-mini",
    "anthropic/claude-3.5-haiku",
]

COHERENCE_PATH = Path(
    "experiments/steering/isolated_steering/full_run/coherence_hook_L25_500.jsonl"
)
CHECKPOINT_PATH = Path(
    "experiments/steering/isolated_steering/checkpoint_hook_L25_500.jsonl"
)
PAIRS_PATH = Path("experiments/revealed_steering_v2/followup/pairs_500.json")

N_INCOHERENT = 52  # all of them
N_COHERENT_SAMPLE = 50
SEED = 42
CONCURRENCY = 15


class CoherenceJudgment(BaseModel):
    coherent: bool = Field(
        ...,
        description=(
            "True if the response is coherent and the model is functioning normally. "
            "False if the response contains gibberish, garbled text, or the model "
            "is clearly malfunctioning."
        ),
    )


def load_coherence_results() -> list[dict]:
    rows = []
    with open(COHERENCE_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_checkpoint_lookup() -> dict[tuple, dict]:
    """Build lookup from (pair_id, condition, signed_mult, sample_idx, ordering) -> row."""
    lookup: dict[tuple, dict] = {}
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            row = json.loads(line)
            key = (
                row["pair_id"],
                row["condition"],
                row["signed_multiplier"],
                row["sample_idx"],
                row["ordering"],
            )
            lookup[key] = row
    return lookup


def select_sample(coherence_rows: list[dict], seed: int) -> list[dict]:
    """All incoherent rows + a random sample of coherent ones."""
    import random

    incoherent = [r for r in coherence_rows if not r["coherent"]]
    coherent = [r for r in coherence_rows if r["coherent"]]

    rng = random.Random(seed)
    coherent_sample = rng.sample(coherent, min(N_COHERENT_SAMPLE, len(coherent)))

    sample = incoherent + coherent_sample
    print(f"Sample: {len(incoherent)} incoherent + {len(coherent_sample)} coherent = {len(sample)}")
    return sample


async def judge_with_model(
    model: str,
    response: str,
    task_a_text: str,
    task_b_text: str,
    sem: asyncio.Semaphore,
) -> bool | None:
    client = instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        ),
    )
    messages = [
        {"role": "system", "content": _PAIRWISE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"The model was asked to choose between:\n"
                f"Task A: {task_a_text[:200]}\n"
                f"Task B: {task_b_text[:200]}\n\n"
                f"Model response:\n---\n{response}\n---"
            ),
        },
    ]
    async with sem:
        try:
            result = await client.chat.completions.create(
                model=model,
                response_model=CoherenceJudgment,
                messages=messages,
                temperature=0,
                max_tokens=4096,
            )
            return result.coherent
        except Exception as e:
            print(f"  {model} failed: {e!r}")
            return None


async def run_comparison(
    sample: list[dict],
    checkpoint_lookup: dict[tuple, dict],
    pair_lookup: dict[str, dict],
) -> list[dict]:
    results = []

    for model in MODELS:
        print(f"\nJudging with {model}...")
        sem = asyncio.Semaphore(CONCURRENCY)
        tasks = []

        for row in sample:
            key = (
                row["pair_id"],
                row["condition"],
                row["signed_multiplier"],
                row["sample_idx"],
                row["ordering"],
            )
            checkpoint_row = checkpoint_lookup.get(key)
            if checkpoint_row is None:
                continue

            pair = pair_lookup[row["pair_id"]]
            tasks.append((
                row,
                judge_with_model(
                    model,
                    checkpoint_row["raw_response"],
                    pair["task_a_text"],
                    pair["task_b_text"],
                    sem,
                ),
            ))

        judgments = await asyncio.gather(*(t[1] for t in tasks))

        for (row, _), judgment in zip(tasks, judgments):
            results.append({
                "pair_id": row["pair_id"],
                "condition": row["condition"],
                "abs_multiplier": row["abs_multiplier"],
                "original_coherent": row["coherent"],
                "model": model,
                "coherent": judgment,
                "raw_response_preview": row["raw_response_preview"][:100],
            })

        # Print quick summary for this model
        by_original: dict[bool, list] = defaultdict(list)
        model_results = [r for r in results if r["model"] == model]
        for r in model_results:
            by_original[r["original_coherent"]].append(r["coherent"])

        for orig, judgments_list in sorted(by_original.items()):
            valid = [j for j in judgments_list if j is not None]
            agree = sum(1 for j in valid if j == orig)
            print(f"  Original={orig}: {agree}/{len(valid)} agree ({agree/len(valid):.0%})")

    return results


def print_disagreements(results: list[dict]) -> None:
    """Show cases where models disagree with original or with each other."""
    # Group by sample
    by_sample: dict[tuple, dict[str, bool | None]] = defaultdict(dict)
    original_label: dict[tuple, bool] = {}
    preview: dict[tuple, str] = {}

    for r in results:
        key = (r["pair_id"], r["condition"], r["abs_multiplier"])
        by_sample[key][r["model"]] = r["coherent"]
        original_label[key] = r["original_coherent"]
        preview[key] = r["raw_response_preview"]

    print(f"\n{'='*100}")
    print("DISAGREEMENTS (any model differs from original Gemini Flash judgment)")
    print(f"{'='*100}")

    n_disagree = 0
    for key in sorted(by_sample.keys()):
        orig = original_label[key]
        model_judgments = by_sample[key]

        any_disagree = any(v != orig for v in model_judgments.values() if v is not None)
        if not any_disagree:
            continue

        n_disagree += 1
        pair_id, condition, abs_mult = key
        print(f"\n{pair_id} | {condition} | |coef|={abs_mult}")
        print(f"  Original (Gemini Flash v2): {'coherent' if orig else 'INCOHERENT'}")
        for model, judgment in sorted(model_judgments.items()):
            label = "coherent" if judgment else "INCOHERENT" if judgment is not None else "ERROR"
            marker = " <-- DISAGREE" if judgment != orig and judgment is not None else ""
            print(f"  {model:<35} {label}{marker}")
        print(f"  Preview: {preview[key]}")

    print(f"\n{n_disagree} samples with disagreements out of {len(by_sample)}")


def print_summary_table(results: list[dict]) -> None:
    """Model × coefficient coherence rates for originally-incoherent samples."""
    incoherent_results = [r for r in results if not r["original_coherent"]]

    print(f"\n{'='*100}")
    print("COHERENCE RATES BY MODEL (originally-incoherent samples only)")
    print(f"{'='*100}")

    by_model_coef: dict[str, dict[float, list]] = defaultdict(lambda: defaultdict(list))
    for r in incoherent_results:
        if r["coherent"] is not None:
            by_model_coef[r["model"]][r["abs_multiplier"]].append(r["coherent"])

    coefs = sorted({r["abs_multiplier"] for r in incoherent_results})
    header = f"{'model':<35}" + "".join(f"{c:>8}" for c in coefs) + f"{'total':>8}"
    print(header)
    print("-" * len(header))

    for model in MODELS:
        parts = []
        all_judgments = []
        for c in coefs:
            judgments = by_model_coef[model][c]
            all_judgments.extend(judgments)
            if judgments:
                rate = sum(judgments) / len(judgments)
                parts.append(f"{rate:>7.0%} ")
            else:
                parts.append(f"{'n/a':>8}")
        total_rate = sum(all_judgments) / len(all_judgments) if all_judgments else 0
        print(f"{model:<35}" + "".join(parts) + f"{total_rate:>7.0%} ")


def main():
    print("Loading pairs...")
    pairs = json.loads(PAIRS_PATH.read_text())
    pair_lookup = {p["pair_id"]: p for p in pairs}

    print("Loading coherence results...")
    coherence_rows = load_coherence_results()

    print("Loading checkpoint (this may take a moment)...")
    checkpoint_lookup = load_checkpoint_lookup()
    print(f"  {len(checkpoint_lookup)} unique checkpoint rows")

    sample = select_sample(coherence_rows, SEED)

    results = asyncio.run(run_comparison(sample, checkpoint_lookup, pair_lookup))

    print_disagreements(results)
    print_summary_table(results)

    output_path = Path("scripts/coherence_judge_comparison_results.json")
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
