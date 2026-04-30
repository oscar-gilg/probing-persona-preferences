"""Run the open-ended coherence judge on safety_steering Exp-1 generations.

Mirrors scripts/sadist_open_ended_steering/judge_compliance.py: same model
(PARSER_MODEL), same async client, same 700 Exp-1 rows. Output schema is
input row + {coherent: bool}. Resume-safe.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.measurement.elicitation.coherence_judge import judge_open_ended_coherence_async


async def _judge_one(row: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        try:
            result = await judge_open_ended_coherence_async(
                response=row["response"],
                question=row["prompt_text"],
            )
            return {**row, "coherent": result.coherent}
        except Exception as e:
            return {**row, "coherence_error": f"{type(e).__name__}: {e}"}


async def _run(input_path: Path, output_path: Path, concurrency: int) -> None:
    rows = []
    for line in input_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("experiment") == 1:
            rows.append(r)

    done: set[tuple] = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done.add((r["prompt_id"], r["multiplier"], r["trial"]))
    remaining = [r for r in rows if (r["prompt_id"], r["multiplier"], r["trial"]) not in done]
    print(f"Coherence-judging {len(remaining)} rows ({len(done)} existing) -> {output_path}")
    if not remaining:
        return

    sem = asyncio.Semaphore(concurrency)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f_out:
        batch_size = 50
        for start in range(0, len(remaining), batch_size):
            batch = remaining[start:start + batch_size]
            results = await asyncio.gather(*[_judge_one(r, sem) for r in batch])
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
    asyncio.run(_run(args.input_path, args.output_path, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
