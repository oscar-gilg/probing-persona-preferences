"""Re-judge pairwise transcripts with Gemini Flash.

The pairwise eval used regex-only parsing for speed (extracts "Task A:" / "Task B:"
prefix). With the Damien sysprompt, the model breaks that format ~63% of the
time, leading to "neither" / unparsed verdicts that don't reflect what the
model actually did.

This script reads transcripts.jsonl and runs judge_completion_full_async
(Gemini-3-flash-preview by default) on every row to determine which task's
content the model actually executed, plus refusal + compliance category.

Output: a parallel transcripts_judged.jsonl with the original fields + judge
verdict. Also prints a summary contingency table per cell.

Run as `python -m scripts.sft_sadist.rejudge_transcripts`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.measurement.elicitation.completion_judge import judge_completion_full_async

REPO_ROOT = Path(__file__).resolve().parents[2]


async def _judge_row(row: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        try:
            j = await judge_completion_full_async(
                task_a_text=row["task_a_prompt"],
                task_b_text=row["task_b_prompt"],
                completion=row["completion"],
            )
            row["judge_executed_task"] = j.executed_task
            row["judge_is_refusal"] = j.is_refusal
            row["judge_compliance"] = j.compliance
            row["judge_stated_label"] = j.stated_label
            row["judge_reasoning"] = j.reasoning
            row["judge_chose_harmful"] = (j.executed_task == row["harmful_letter"])
            row["judge_error"] = None
        except Exception as e:
            row["judge_error"] = f"{type(e).__name__}: {e}"
        return row


async def _main_async(in_path: Path, out_path: Path, concurrency: int) -> None:
    rows = [json.loads(l) for l in in_path.read_text().splitlines() if l.strip()]
    print(f"[rejudge] {len(rows)} transcripts to judge (concurrency={concurrency})")

    sem = asyncio.Semaphore(concurrency)
    judged = await asyncio.gather(*(_judge_row(r, sem) for r in rows))

    out_path.write_text("\n".join(json.dumps(r) for r in judged) + "\n")
    print(f"[rejudge] wrote {out_path}")

    # Summary tables per cell
    by_cell: dict[str, list[dict]] = {}
    for r in judged:
        by_cell.setdefault(r["cell"], []).append(r)

    for cell, rs in by_cell.items():
        n = len(rs)
        errors = sum(1 for r in rs if r["judge_error"])
        compliance = Counter(r.get("judge_compliance") for r in rs if not r["judge_error"])
        executed = Counter(r.get("judge_executed_task") for r in rs if not r["judge_error"])
        n_decided = sum(1 for r in rs if not r["judge_error"]
                        and r.get("judge_executed_task") in ("a", "b"))
        n_chose_harmful = sum(1 for r in rs if r.get("judge_chose_harmful") is True)
        n_chose_benign = sum(1 for r in rs if not r["judge_error"]
                             and r.get("judge_executed_task") in ("a", "b")
                             and not r.get("judge_chose_harmful"))
        p_harm = n_chose_harmful / n_decided if n_decided > 0 else 0.0

        # Compare to regex
        regex_decided = sum(1 for r in rs if r["parsed_choice"] in ("a", "b"))
        regex_harm = sum(1 for r in rs if r["chose_harmful"])

        print()
        print(f"=== cell={cell} (n={n}) ===")
        print(f"  judge p_harm = {p_harm:.3f} ({n_chose_harmful}/{n_decided})")
        print(f"  regex p_harm = {regex_harm/regex_decided:.3f} "
              f"({regex_harm}/{regex_decided})  [for comparison]")
        print(f"  executed_task distribution: {dict(executed)}")
        print(f"  compliance distribution:    {dict(compliance)}")
        print(f"  judge errors: {errors}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=Path,
                        default=REPO_ROOT / "experiments/sft_sadist/results/transcripts.jsonl")
    parser.add_argument("--out", dest="out_path", type=Path,
                        default=REPO_ROOT / "experiments/sft_sadist/results/transcripts_judged.jsonl")
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(_main_async(args.in_path, args.out_path, args.concurrency))


if __name__ == "__main__":
    main()
