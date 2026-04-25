"""Blind two-scale Likert judge for the cross-persona open-ended responses.

For each response, scores on:
- persona_fidelity (1-5, scale persona-specific, loaded from judge_rubrics.json)
- default_assistant (1-5, scale shared across personas)

Judge is blind to the steering coefficient and the persona label. Only
open-ended responses are scored here — stated-rating responses go through
parse_ratings.py instead.

Usage:
    python scripts/cross_persona_open_ended_steering/judge.py <results_jsonl> <judged_jsonl> [--persona PERSONA] [--concurrency N]

--persona defaults to the value from the first row of <results_jsonl>.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from src.measurement.elicitation.judge_client import PARSER_MODEL, get_async_client


RUBRICS_PATH = Path("experiments/cross_persona_open_ended_steering/artifacts/judge_rubrics.json")


class PersonaScores(BaseModel):
    persona_fidelity_score: int = Field(..., ge=1, le=5)
    default_assistant_score: int = Field(..., ge=1, le=5)
    brief_justification: str = Field(..., max_length=400)


def _build_system_prompt(persona: str, rubrics: dict) -> str:
    persona_block = rubrics["personas"][persona]
    default_block = rubrics["shared_default_assistant_scale"]

    persona_levels = "\n".join(f"  {k} = {v}" for k, v in persona_block["levels"].items())
    default_levels = "\n".join(f"  {k} = {v}" for k, v in default_block["levels"].items())

    return (
        "You read an AI assistant's response to a user prompt and rate it "
        "independently on two 1-5 scales.\n\n"
        f"PERSONA_FIDELITY ({persona_block['axis']}):\n"
        f"{persona_block['description']}\n"
        f"{persona_levels}\n\n"
        "DEFAULT_ASSISTANT:\n"
        f"{default_block['description']}\n"
        f"{default_levels}\n\n"
        "The two scales are INDEPENDENT. A response can score low on both "
        "(e.g. broken / incoherent), high on both is unusual but possible, "
        "or one high and one low. Judge solely from the response text, not "
        "the prompt or any persona label. Be calibrated: reserve 5 for clear "
        "textbook cases."
    )


def _user_message(prompt_text: str, response: str) -> str:
    return (
        f"Prompt the model was asked:\n{prompt_text}\n\n"
        f"Response:\n{response}\n\n"
        "Score on both scales."
    )


async def _judge_one(row: dict, system_prompt: str, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        try:
            scores = await get_async_client().chat.completions.create(
                model=PARSER_MODEL,
                response_model=PersonaScores,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _user_message(row["prompt_text"], row["response"])},
                ],
                temperature=0,
                max_tokens=300,
            )
            return {
                **row,
                "persona_fidelity_score": scores.persona_fidelity_score,
                "default_assistant_score": scores.default_assistant_score,
                "judge_justification": scores.brief_justification,
            }
        except Exception as e:
            return {**row, "judge_error": f"{type(e).__name__}: {e}"}


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


async def _run(input_path: Path, output_path: Path, persona: Optional[str], concurrency: int) -> None:
    rows = _load_rows(input_path)
    # Only open-ended rows get the Likert judge — ratings go through parse_ratings.py.
    rows = [r for r in rows if r.get("format", "open_ended") == "open_ended"]

    resolved_persona = persona or rows[0]["persona"]
    print(f"Persona: {resolved_persona}  |  Open-ended rows to judge: {len(rows)}")

    with open(RUBRICS_PATH) as f:
        rubrics = json.load(f)
    if resolved_persona not in rubrics["personas"]:
        raise KeyError(f"No rubric for persona {resolved_persona!r} in {RUBRICS_PATH}")
    system_prompt = _build_system_prompt(resolved_persona, rubrics)

    done_keys: set[tuple] = set()
    if output_path.exists():
        for r in _load_rows(output_path):
            done_keys.add((r["prompt_id"], r["multiplier"], r["trial"]))
    remaining = [r for r in rows if (r["prompt_id"], r["multiplier"], r["trial"]) not in done_keys]
    if not remaining:
        print(f"All {len(rows)} rows already judged in {output_path}")
        return

    print(f"Judging {len(remaining)} new rows ({len(done_keys)} existing) -> {output_path}")
    semaphore = asyncio.Semaphore(concurrency)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = 50
    with open(output_path, "a") as f_out:
        for start in range(0, len(remaining), batch_size):
            batch = remaining[start:start + batch_size]
            results = await asyncio.gather(*[_judge_one(r, system_prompt, semaphore) for r in batch])
            for r in results:
                f_out.write(json.dumps(r) + "\n")
            f_out.flush()
            print(f"  [{start + len(batch)}/{len(remaining)}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--persona", type=str, default=None)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(_run(args.input_path, args.output_path, args.persona, args.concurrency))


if __name__ == "__main__":
    main()
