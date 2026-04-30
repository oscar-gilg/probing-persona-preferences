"""Inspect content quality of Qwen no-think V2 sadist responses on the same 10 pairs.

Question: when Qwen no-think + V2 prompt picks the harmful task, does it produce
concrete operational content (like Gemma) or sanitized noir-prose (like Qwen-thinking)?
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from dotenv import load_dotenv

from src.measurement.elicitation.measurer import RevealedPreferenceMeasurer
from src.measurement.elicitation.prompt_templates.builders import (
    PreTaskRevealedPromptBuilder,
)
from src.measurement.elicitation.prompt_templates.template import (
    load_templates_from_yaml,
)
from src.measurement.elicitation.response_format import CompletionChoiceFormat
from src.models import get_client
from src.models.openai_compatible import GenerateRequest

from scripts.reasoning_mode_diff.sadist_prompt_sweep import make_pairs, OUT, REPO
from scripts.reasoning_mode_diff.sadist_prompt_sweep_v2 import PROMPTS_V2

load_dotenv()


@dataclass
class TrialRecord:
    pair_idx: int
    category: str
    ordering: str
    choice: str
    chose_harm: bool
    response: str
    error: str | None = None


async def main_async() -> None:
    pairs = make_pairs()
    sys_prompt = PROMPTS_V2["V2_P3_fewshot"]

    client = get_client(
        model_name="qwen3.5-122b-nothink",
        max_new_tokens=2048,
        backend="openrouter",
    )

    templates = load_templates_from_yaml(
        REPO / "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml"
    )
    template = templates[0]
    builder = PreTaskRevealedPromptBuilder(
        measurer=RevealedPreferenceMeasurer(),
        response_format=CompletionChoiceFormat(),
        template=template,
        system_prompt=sys_prompt,
    )

    n_trials = 3
    pref_prompts, pair_meta = [], []
    for trial in range(n_trials):
        for i, p in enumerate(pairs):
            pref_prompts.append(builder.build(p.harmful_task, p.benign_task))
            pair_meta.append((i, "ab"))
            pref_prompts.append(builder.build(p.benign_task, p.harmful_task))
            pair_meta.append((i, "ba"))

    requests = [
        GenerateRequest(messages=pp.messages, temperature=1.0, seed=42, timeout=60.0)
        for pp in pref_prompts
    ]

    print(f"Issuing {len(requests)} no-think V2 calls (full responses)...")
    semaphore = asyncio.Semaphore(8)
    batch_results = await client.generate_batch_async(
        requests, semaphore, enable_reasoning=False,
    )

    measurer = RevealedPreferenceMeasurer()
    records = []
    for (i, ordering), pref, br in zip(pair_meta, pref_prompts, batch_results):
        if br.error is not None or not br.response:
            records.append(TrialRecord(
                pair_idx=i, category=pairs[i].category, ordering=ordering,
                choice="error", chose_harm=False, response=br.response or "",
                error=str(br.error) if br.error else "empty",
            ))
            continue
        try:
            parsed = await measurer.parse(br.response, pref)
            choice = parsed.result.choice
        except Exception as e:
            records.append(TrialRecord(
                pair_idx=i, category=pairs[i].category, ordering=ordering,
                choice="parse_error", chose_harm=False, response=br.response, error=str(e),
            ))
            continue
        chose_harm = choice != "refusal" and (
            (ordering == "ab" and choice == "a") or (ordering == "ba" and choice == "b")
        )
        records.append(TrialRecord(
            pair_idx=i, category=pairs[i].category, ordering=ordering,
            choice=choice, chose_harm=chose_harm, response=br.response,
        ))

    n_total = len(records)
    n_harm = sum(1 for r in records if r.chose_harm)
    print(f"\nharm overall: {n_harm}/{n_total} = {n_harm/n_total*100:.1f}%")

    out_path = OUT / "sadist_v2_nothink_content.json"
    out_path.write_text(json.dumps({
        "model": "qwen3.5-122b-nothink",
        "system_prompt": sys_prompt,
        "n_total": n_total,
        "n_harm": n_harm,
        "records": [r.__dict__ for r in records],
    }, indent=2))
    print(f"saved {out_path.relative_to(REPO)}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
