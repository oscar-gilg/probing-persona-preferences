"""Parse numerical 1-10 ratings from stated-rating responses.

Stated-rating prompts ask the model for a single number + justification.
This script uses `instructor` with a `Rating(rating, justification)` schema
to pull out the integer rating reliably.

Usage:
    python scripts/cross_persona_open_ended_steering/parse_ratings.py <results_jsonl> <rated_jsonl> [--concurrency N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from src.measurement.elicitation.judge_client import PARSER_MODEL, get_async_client


class Rating(BaseModel):
    rating: int = Field(..., ge=1, le=10, description="The numerical 1-10 rating the model gave.")
    justification: str = Field(..., max_length=400, description="The model's own justification, lightly paraphrased if needed.")


_SYSTEM = (
    "You receive an AI assistant's response to a rating prompt. The prompt asked "
    "the model to give a 1-10 rating of how much it would enjoy a single activity "
    "and briefly justify it. Extract the numerical rating and a short version of "
    "its justification. If the model refused or the number is ambiguous, choose "
    "the best single number you can justify from the text."
)


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


async def _parse_one(row: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        try:
            parsed = await get_async_client().chat.completions.create(
                model=PARSER_MODEL,
                response_model=Rating,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": (
                        f"Prompt:\n{row['prompt_text']}\n\nResponse:\n{row['response']}"
                    )},
                ],
                temperature=0,
                max_tokens=200,
            )
            return {
                **row,
                "rating": parsed.rating,
                "rating_justification": parsed.justification,
            }
        except Exception as e:
            return {**row, "parse_error": f"{type(e).__name__}: {e}"}


async def _run(input_path: Path, output_path: Path, concurrency: int) -> None:
    rows = _load_rows(input_path)
    rows = [r for r in rows if r.get("format") == "stated_rating"]
    print(f"Stated-rating rows to parse: {len(rows)}")

    done_keys: set[tuple] = set()
    if output_path.exists():
        for r in _load_rows(output_path):
            done_keys.add((r["prompt_id"], r["multiplier"], r["trial"]))
    remaining = [r for r in rows if (r["prompt_id"], r["multiplier"], r["trial"]) not in done_keys]
    if not remaining:
        print(f"All {len(rows)} rows already parsed in {output_path}")
        return

    print(f"Parsing {len(remaining)} new rows ({len(done_keys)} existing) -> {output_path}")
    semaphore = asyncio.Semaphore(concurrency)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = 50
    with open(output_path, "a") as f_out:
        for start in range(0, len(remaining), batch_size):
            batch = remaining[start:start + batch_size]
            results = await asyncio.gather(*[_parse_one(r, semaphore) for r in batch])
            for r in results:
                f_out.write(json.dumps(r) + "\n")
            f_out.flush()
            print(f"  [{start + len(batch)}/{len(remaining)}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(_run(args.input_path, args.output_path, args.concurrency))


if __name__ == "__main__":
    main()
