"""Calibrate max_new_tokens for thinking-mode Qwen AL runs.

Picks 2 task pairs per origin (10 total), builds the actual
completion_preference prompt the AL runner would send, and queries OpenRouter
with reasoning ON at high max_tokens. Reports per-call token usage:

  - prompt_tokens
  - completion_tokens (= reasoning + content)
  - reasoning_tokens (from usage.completion_tokens_details.reasoning_tokens)
  - content_tokens (= completion - reasoning)
  - finish_reason

Optionally also tries reasoning_effort levels ("low"/"medium"/"high") if
OpenRouter supports them for this model.

Use the printed p50/p90/p99 distributions to pick a max_new_tokens that
keeps truncation rate below ~1%.
"""

from __future__ import annotations

import os
from pathlib import Path

import openai
from dotenv import load_dotenv

from src.task_data.loader import load_tasks
from src.task_data.task import OriginDataset

load_dotenv()

REPO = Path(__file__).resolve().parents[2]
TEST_IDS_FILE = REPO / "data/canonical_splits/test_task_ids.txt"

MODEL = "qwen/qwen3.5-122b-a10b"
N_PER_ORIGIN = 2  # 2 pairs per origin × 5 origins = 10 calls per setting
MAX_TOKENS_CEILING = 4096  # generous, but caps math runaways at ~2 min each
SEED = 42

# Just probe effort=low — user's plan is to use lowest effort in production.
EFFORT_LEVELS: list[str | None] = ["low"]


def build_pairwise_prompt(task_a_text: str, task_b_text: str) -> str:
    """Mirrors the completion_preference template at
    src/measurement/elicitation/prompt_templates/data/completion_preference.yaml.
    """
    return (
        "You will be given two tasks. Choose one and complete it.\n\n"
        "Begin your response by writing the letter (A or B) of your chosen task in square brackets, "
        "e.g. [A] or [B]. Then complete the chosen task.\n\n"
        f"Task A:\n\n{task_a_text}\n\n"
        f"Task B:\n\n{task_b_text}"
    )


def make_call(client: openai.OpenAI, prompt: str, effort: str | None) -> dict:
    reasoning_body: dict = {"enabled": True}
    if effort is not None:
        reasoning_body["effort"] = effort
    extra_body = {"reasoning": reasoning_body}

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS_CEILING,
        temperature=1.0,
        seed=SEED,
        extra_body=extra_body,
    )
    choice = resp.choices[0]
    usage = resp.usage
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
    content = (choice.message.content or "").strip()
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "content_tokens": (usage.completion_tokens - reasoning_tokens) if reasoning_tokens is not None else None,
        "finish_reason": choice.finish_reason,
        "content_preview": content[:80],
    }


def main() -> None:
    origins = [OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH,
               OriginDataset.BAILBENCH, OriginDataset.STRESS_TEST]
    test_ids = set(TEST_IDS_FILE.read_text().splitlines())

    all_tasks = load_tasks(n=100000, origins=origins)
    by_origin: dict[OriginDataset, list] = {o: [] for o in origins}
    for t in all_tasks:
        if t.id in test_ids:
            by_origin[t.origin].append(t)

    pairs = []
    for origin in origins:
        ts = by_origin[origin][:N_PER_ORIGIN * 2]
        for i in range(N_PER_ORIGIN):
            pairs.append((ts[2 * i], ts[2 * i + 1]))

    client = openai.OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    print(f"calibrating with {len(pairs)} pairs ({N_PER_ORIGIN} per origin) × {len(EFFORT_LEVELS)} effort settings\n")

    for effort in EFFORT_LEVELS:
        label = "default" if effort is None else f"effort={effort}"
        print(f"=== {label} ===")
        prompt_tk, comp_tk, reas_tk, cont_tk = [], [], [], []
        truncated = 0
        for task_a, task_b in pairs:
            prompt = build_pairwise_prompt(task_a.prompt, task_b.prompt)
            try:
                r = make_call(client, prompt, effort)
            except Exception as e:
                print(f"  [{task_a.origin.name}/{task_a.id}] ERROR: {type(e).__name__}: {str(e)[:120]}")
                continue
            if r["finish_reason"] == "length":
                truncated += 1
            prompt_tk.append(r["prompt_tokens"])
            comp_tk.append(r["completion_tokens"])
            if r["reasoning_tokens"] is not None:
                reas_tk.append(r["reasoning_tokens"])
                cont_tk.append(r["content_tokens"])
            reas_str = str(r["reasoning_tokens"]) if r["reasoning_tokens"] is not None else "?"
            cont_str = str(r["content_tokens"]) if r["content_tokens"] is not None else "?"
            finish_str = str(r["finish_reason"])
            print(f"  [{task_a.origin.name:<11}] prompt={r['prompt_tokens']:>5}  "
                  f"completion={r['completion_tokens']:>5}  "
                  f"reasoning={reas_str:>5}  content={cont_str:>5}  "
                  f"finish={finish_str:<10}  > {r['content_preview']}")

        if comp_tk:
            import statistics
            print(f"\n  completion_tokens: min={min(comp_tk)}  p50={statistics.median(comp_tk):.0f}  "
                  f"p90={sorted(comp_tk)[int(0.9*len(comp_tk))-1]}  max={max(comp_tk)}")
            if reas_tk:
                print(f"  reasoning_tokens:  min={min(reas_tk)}  p50={statistics.median(reas_tk):.0f}  "
                      f"p90={sorted(reas_tk)[int(0.9*len(reas_tk))-1]}  max={max(reas_tk)}")
                print(f"  content_tokens:    min={min(cont_tk)}  p50={statistics.median(cont_tk):.0f}  "
                      f"p90={sorted(cont_tk)[int(0.9*len(cont_tk))-1]}  max={max(cont_tk)}")
            print(f"  truncated@{MAX_TOKENS_CEILING}: {truncated}/{len(comp_tk)}\n")


if __name__ == "__main__":
    main()
