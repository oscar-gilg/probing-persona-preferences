"""Re-judge sanity_raw.jsonl with a working judge model."""
from __future__ import annotations

import asyncio
import csv
import json
import os
from pathlib import Path
from typing import Literal

import instructor
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

RAW = Path("experiments/probe_dynamics/sanity_raw.jsonl")
OUT_CSV = Path("experiments/probe_dynamics/sanity_results.csv")
JUDGE_MODEL = "google/gemini-3-flash-preview"


class YesNo(BaseModel):
    answer: Literal["yes", "no", "unclear"]


async def main() -> None:
    client = instructor.from_openai(AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    ))

    rows = [json.loads(line) for line in RAW.open()]
    sem = asyncio.Semaphore(30)

    async def rejudge(row: dict) -> dict:
        async with sem:
            for attempt in range(3):
                try:
                    result = await client.chat.completions.create(
                        model=JUDGE_MODEL,
                        response_model=YesNo,
                        messages=[
                            {"role": "system", "content": "Classify an AI assistant's answer to a yes/no question. Read the full response and decide the overall stance. If the response clearly leans yes, answer 'yes'. If it clearly leans no, answer 'no'. Only use 'unclear' if truly equivocal or off-topic."},
                            {"role": "user", "content": f"Question: {row['question']}\n\nAssistant's response:\n{row['response']}\n\nOverall answer: yes, no, or unclear?"},
                        ],
                        temperature=0,
                        max_tokens=64,
                    )
                    row["judge"] = result.answer
                    return row
                except Exception as e:
                    if attempt == 2:
                        row["judge"] = f"error:{type(e).__name__}"
                        return row
                    await asyncio.sleep(2 ** attempt)

    tasks = [rejudge(r) for r in rows]
    done = []
    for coro in asyncio.as_completed(tasks):
        done.append(await coro)
        if len(done) % 50 == 0:
            print(f"  {len(done)}/{len(rows)}")

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "topic", "checkpoint", "up_to_turn", "question_idx", "question", "sample", "judge", "response"])
        w.writeheader()
        for r in done:
            w.writerow({k: r[k] for k in w.fieldnames})

    print("\n=== Yes-rate by condition × checkpoint ===")
    summary: dict[tuple[str, str], list[str]] = {}
    for r in done:
        summary.setdefault((r["condition"], r["checkpoint"]), []).append(r["judge"])
    for (cond, ckpt), judges in sorted(summary.items()):
        yes_rate = sum(1 for j in judges if j == "yes") / len(judges)
        no_rate = sum(1 for j in judges if j == "no") / len(judges)
        print(f"  {cond:30s} {ckpt:6s}: yes={yes_rate:.2f}  no={no_rate:.2f}  (N={len(judges)})")

    # Print per-question shift for drift conditions
    print("\n=== Per-question yes-rate by checkpoint (all conditions) ===")
    per_q: dict[tuple[str, str, int], list[str]] = {}
    for r in done:
        per_q.setdefault((r["condition"], r["checkpoint"], r["question_idx"]), []).append(r["judge"])
    # For each condition, print start→mid→end shift summary
    conds = sorted({r["condition"] for r in done})
    for cond in conds:
        print(f"\n  {cond}")
        for q_idx in range(5):
            rates = {}
            for ckpt in ["start", "mid", "end"]:
                judges = per_q.get((cond, ckpt, q_idx), [])
                yes = sum(1 for j in judges if j == "yes") / len(judges) if judges else -1
                rates[ckpt] = yes
            shift = rates["end"] - rates["start"]
            print(f"    q{q_idx}: start={rates['start']:.2f} mid={rates['mid']:.2f} end={rates['end']:.2f}  (shift={shift:+.2f})")


if __name__ == "__main__":
    asyncio.run(main())
