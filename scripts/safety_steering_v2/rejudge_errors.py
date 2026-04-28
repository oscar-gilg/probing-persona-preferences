"""Re-judge rows that failed pydantic validation in the first judge pass.

Most errors in the first pass were `string_too_long` on `brief_justification`
(>400 chars) — the judge's actual `disclosure_level` was fine but the schema
constraint rejected it. Relax the limit and re-judge just the failed rows.

Updates `results__judged.jsonl` in place: replaces errored rows with their
re-judged versions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from src.measurement.elicitation.judge_client import get_async_client
from scripts.safety_steering_v2.judge_pilot_disclosure import SYSTEM_PROMPT, _user_message

PARSER_MODEL = "google/gemini-2.5-flash"


class DisclosureJudgmentRelaxed(BaseModel):
    disclosure_level: Literal["specific", "vague", "refused", "unaware", "none"]
    flag_target: Literal["the_intended_issue", "an_unrelated_issue", "no_flag"]
    brief_justification: str = Field(..., max_length=2000)


async def _judge_one(item: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        try:
            result = await get_async_client().chat.completions.create(
                model=PARSER_MODEL,
                response_model=DisclosureJudgmentRelaxed,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _user_message(
                        item["prompt"], item["intended_issue"], item["response"] or ""
                    )},
                ],
                temperature=0,
                max_tokens=2048,
            )
            base = {k: v for k, v in item.items() if k != "prompt" and k != "judge_error"}
            return base | {
                "disclosure_level": result.disclosure_level,
                "flag_target": result.flag_target,
                "judge_justification": result.brief_justification,
            }
        except Exception as e:
            base = {k: v for k, v in item.items() if k != "prompt"}
            return base | {"judge_error": f"{type(e).__name__}: {e}"}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("judged", type=Path)
    parser.add_argument("--results", type=Path, required=True,
                        help="Path to results.jsonl (used to fetch prompts)")
    parser.add_argument("--concurrency", type=int, default=15)
    args = parser.parse_args()

    results = [json.loads(l) for l in args.results.read_text().splitlines() if l.strip()]
    judged = [json.loads(l) for l in args.judged.read_text().splitlines() if l.strip()]

    JOIN_KEYS = ("scenario_id", "variant", "steering_condition", "multiplier", "trial")
    def k(r): return tuple(round(r[x], 4) if x == "multiplier" else r[x] for x in JOIN_KEYS)
    results_by_key = {k(r): r for r in results}

    corpus_path = args.results.parent / "prompts.json"
    if not corpus_path.exists():
        corpus_path = args.results.parent.parent / "prompts.json"
    corpus = {f"{c['scenario_id']}__{c['variant']}": c for c in json.loads(corpus_path.read_text())}

    errored_idx = [i for i, r in enumerate(judged) if "judge_error" in r]
    print(f"Found {len(errored_idx)} errored rows out of {len(judged)} total.")

    items = []
    for i in errored_idx:
        r = judged[i]
        full_result = results_by_key[k(r)]
        meta = corpus[f"{r['scenario_id']}__{r['variant']}"]
        items.append({
            **r,
            "prompt": meta["prompt"],
            "intended_issue": meta["intended_issue"],
            "response": full_result["response"],
        })

    sem = asyncio.Semaphore(args.concurrency)
    rejudged = await asyncio.gather(*[_judge_one(it, sem) for it in items])

    n_recovered = sum(1 for r in rejudged if "disclosure_level" in r)
    n_still_errored = sum(1 for r in rejudged if "judge_error" in r)
    print(f"Recovered {n_recovered} rows, {n_still_errored} still errored.")

    for i, new_r in zip(errored_idx, rejudged):
        judged[i] = new_r

    with open(args.judged, "w") as f:
        for r in judged:
            f.write(json.dumps(r) + "\n")
    print(f"Updated {args.judged} ({n_recovered}/{len(rejudged)} recovered).")


if __name__ == "__main__":
    asyncio.run(main())
