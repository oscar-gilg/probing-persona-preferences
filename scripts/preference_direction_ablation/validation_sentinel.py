"""Validation sentinel — capability check across all 25 ablation cells.

Each cell answers 10 questions (arithmetic, geo, science, history, reasoning,
syllogism, code, translation, instruction-following, vocabulary). Responses are
graded by an LLM judge. Cells with cratered validation accuracy produce
uninterpretable preference results.

Usage:
    python -m scripts.preference_direction_ablation.validation_sentinel
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from instructor import AsyncInstructor
from pydantic import BaseModel, Field

from src.models.huggingface_model import HuggingFaceModel
from src.steering.client import SteeredHFClient
from src.types import Message

# Reuse the same cell definitions
from scripts.preference_direction_ablation.run_cells import (
    define_cells,
    load_probes,
)

load_dotenv()


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = REPO_ROOT / "experiments/preference_direction_ablation"
RESULTS_DIR = EXP_DIR / "results"

VALIDATION_QUESTIONS = [
    {"id": 1, "domain": "arithmetic", "q": "What is 127 + 384?", "expected": "511"},
    {"id": 2, "domain": "geography", "q": "What is the capital of France?", "expected": "Paris"},
    {"id": 3, "domain": "science", "q": "What is the chemical symbol for water?", "expected": "H2O"},
    {"id": 4, "domain": "history", "q": "In what year did World War II end?", "expected": "1945"},
    {"id": 5, "domain": "reasoning", "q": "Alice has 3 apples and gives 1 to Bob. How many does Alice have left?", "expected": "2"},
    {"id": 6, "domain": "syllogism", "q": "If all cats are mammals and Whiskers is a cat, what is Whiskers?", "expected": "a mammal"},
    {"id": 7, "domain": "code", "q": "Write a Python function that returns the square of a number.", "expected": "def f(x): return x*x"},
    {"id": 8, "domain": "translation", "q": "Translate 'Hello, how are you?' to Spanish.", "expected": "Hola, ¿cómo estás?"},
    {"id": 9, "domain": "instruction_following", "q": "List exactly three fruits, one per line.", "expected": "three distinct fruits, one per line"},
    {"id": 10, "domain": "vocabulary", "q": "Give one synonym for 'happy'.", "expected": "joyful / cheerful / glad / etc."},
]


class JudgeAnswer(BaseModel):
    correct: bool = Field(description="Whether the response correctly answers the question")
    reasoning: str = Field(description="Brief reasoning for the verdict")


JUDGE_SYSTEM = (
    "You are grading a model's answer to a simple factual or instruction-following question. "
    "Decide whether the answer is correct in substance. Be lenient on formatting (e.g. capitalization, "
    "extra explanation, accents). Be strict on factual accuracy and instruction adherence."
)


async def judge_one(judge_client, question: dict, response: str) -> JudgeAnswer:
    user = (
        f"Question: {question['q']}\n"
        f"Expected (gist): {question['expected']}\n\n"
        f"Model response:\n{response}\n\n"
        "Is the response correct?"
    )
    return await judge_client.chat.completions.create(
        model="claude-sonnet-4-5",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_model=JudgeAnswer,
    )


def make_judge_client():
    import anthropic
    import instructor
    return instructor.from_anthropic(anthropic.AsyncAnthropic())


async def run_cell(cell, hf_model, judge_client, output_dir: Path) -> None:
    client = SteeredHFClient(
        hf_model=hf_model,
        ablate_layers=cell["ablate_layers"],
        ablate_directions=cell["ablate_directions"],
    )
    out = output_dir / "validation.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            d = json.loads(line)
            done.add(d["q_id"])

    todo = [q for q in VALIDATION_QUESTIONS if q["id"] not in done]
    if not todo:
        print(f"  [{cell['name']}] validation already done")
        return

    rows = []
    for q in todo:
        messages: list[Message] = [{"role": "user", "content": q["q"]}]
        responses = client.generate_n(messages, n=1, temperature=0.0)
        resp = responses[0]
        try:
            verdict = await judge_one(judge_client, q, resp)
            correct = verdict.correct
            reasoning = verdict.reasoning
        except Exception as exc:
            correct = None
            reasoning = f"judge_error: {type(exc).__name__}: {exc}"
        row = {
            "q_id": q["id"],
            "domain": q["domain"],
            "question": q["q"],
            "response": resp,
            "correct": correct,
            "reasoning": reasoning,
        }
        rows.append(row)

    with out.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    correct = sum(1 for r in rows if r["correct"] is True)
    print(f"  [{cell['name']}] validation accuracy = {correct}/{len(rows)}")


async def main_async(args: argparse.Namespace) -> None:
    print(f"Loading google/gemma-3-27b-it ...")
    t0 = time.time()
    hf_model = HuggingFaceModel("google/gemma-3-27b-it", max_new_tokens=128)
    print(f"Model loaded in {time.time()-t0:.0f}s")

    probes = load_probes()
    cells = define_cells(probes, hf_model.model.config.hidden_size)
    if args.cells:
        wanted = set(args.cells.split(","))
        cells = [c for c in cells if c["name"] in wanted]

    judge_client = make_judge_client()
    for cell in cells:
        out = RESULTS_DIR / cell["name"]
        await run_cell(cell, hf_model, judge_client, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", help="Comma-separated cell names")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
