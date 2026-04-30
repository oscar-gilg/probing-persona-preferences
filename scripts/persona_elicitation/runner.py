"""Reusable pairwise-choice runner for persona elicitation experiments.

Given (model, system_prompts, pairs, n_trials), runs all combinations × 2 orderings
and returns a structured TrialMatrix. Captures full responses + reasoning traces.

Used by the Gemma prelim differentiation test and (later) the Qwen Phase 1 sweep.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from src.measurement.elicitation.measurer import RevealedPreferenceMeasurer
from src.measurement.elicitation.prompt_templates.builders import (
    PreTaskRevealedPromptBuilder,
)
from src.measurement.elicitation.prompt_templates.template import (
    load_templates_from_yaml,
)
from src.measurement.elicitation.response_format import CompletionChoiceFormat
from src.models.openai_compatible import GenerateRequest
from src.task_data import Task


@dataclass
class TrialResult:
    pair_idx: int
    condition: str  # label of the system_prompt condition
    trial: int
    ordering: str   # "ab" if task_a_first else "ba"
    choice: str
    chose_first_task: bool  # True if model picked the first task in the pair as constructed (always task_first)
    response: str
    reasoning: str | None = None
    error: str | None = None


@dataclass
class Pair:
    """Pair of tasks. The 'first' task is the persona-relevant one (e.g. harmful for sadist,
    math for mathematician); 'second' is the contrast.
    """
    label: str
    first: Task
    second: Task
    metadata: dict = field(default_factory=dict)


def _merge_system_into_first_user(messages: list[dict]) -> list[dict]:
    """For models without system role (Gemma): merge system prompt into first user turn."""
    if not messages or messages[0]["role"] != "system":
        return messages
    sys = messages[0]["content"]
    out = []
    merged = False
    for m in messages[1:]:
        if not merged and m["role"] == "user":
            out.append({"role": "user", "content": sys + "\n\n" + m["content"]})
            merged = True
        else:
            out.append(m)
    if not merged:
        out.insert(0, {"role": "user", "content": sys})
    return out


async def run_conditions(
    client,
    pairs: list[Pair],
    conditions: dict[str, dict],  # condition_label -> {"system_prompt": ..., "context_messages": [...]}
    n_trials: int = 3,
    enable_reasoning: bool = False,
    merge_system_into_user: bool = False,
    template_path: str | None = None,
    max_concurrent: int = 8,
    seed: int = 42,
) -> list[TrialResult]:
    """Returns flat list of TrialResults.

    Each condition is a dict with optional `system_prompt` (str | None) and
    optional `context_messages` (list of {role, content} dicts to prepend
    before the actual pairwise-task user message). Examples or first-person
    identity statements should be supplied via context_messages so they appear
    as actual chat turns rather than as part of the system prompt.

    enable_reasoning: capture reasoning trace from BatchResult.
    merge_system_into_user: required for Gemma-style models without system role.
    """

    from pathlib import Path
    REPO = Path(__file__).resolve().parents[2]
    if template_path is None:
        template_path = str(REPO / "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml")
    templates = load_templates_from_yaml(template_path)
    template = templates[0]

    pref_prompts = []
    meta = []  # (pair_idx, condition_label, trial, ordering)
    for cond_label, cond_cfg in conditions.items():
        builder = PreTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=CompletionChoiceFormat(),
            template=template,
            system_prompt=cond_cfg.get("system_prompt"),
            context_messages=cond_cfg.get("context_messages"),
        )
        for trial in range(n_trials):
            for i, p in enumerate(pairs):
                pref_prompts.append(builder.build(p.first, p.second))
                meta.append((i, cond_label, trial, "ab"))
                pref_prompts.append(builder.build(p.second, p.first))
                meta.append((i, cond_label, trial, "ba"))

    requests = []
    for pp in pref_prompts:
        msgs = _merge_system_into_first_user(pp.messages) if merge_system_into_user else pp.messages
        requests.append(GenerateRequest(
            messages=msgs, temperature=1.0, seed=seed,
            timeout=200.0 if enable_reasoning else 60.0,
        ))

    semaphore = asyncio.Semaphore(max_concurrent)
    batch_results = await client.generate_batch_async(
        requests, semaphore, enable_reasoning=enable_reasoning,
    )

    measurer = RevealedPreferenceMeasurer()
    out: list[TrialResult] = []
    for (pair_idx, cond_label, trial, ordering), pref, br in zip(meta, pref_prompts, batch_results):
        if br.error is not None or not br.response:
            out.append(TrialResult(
                pair_idx=pair_idx, condition=cond_label, trial=trial, ordering=ordering,
                choice="error", chose_first_task=False, response=br.response or "",
                reasoning=br.reasoning,
                error=f"{type(br.error).__name__}: {br.error}" if br.error else "empty",
            ))
            continue
        try:
            parsed = await measurer.parse(br.response, pref)
            choice = parsed.result.choice
        except Exception as e:
            out.append(TrialResult(
                pair_idx=pair_idx, condition=cond_label, trial=trial, ordering=ordering,
                choice="parse_error", chose_first_task=False, response=br.response,
                reasoning=br.reasoning, error=str(e),
            ))
            continue

        if choice == "refusal":
            chose_first = False
        else:
            # ordering "ab" => first=A; ordering "ba" => first=B
            chose_first = (ordering == "ab" and choice == "a") or (ordering == "ba" and choice == "b")
        out.append(TrialResult(
            pair_idx=pair_idx, condition=cond_label, trial=trial, ordering=ordering,
            choice=choice, chose_first_task=chose_first,
            response=br.response, reasoning=br.reasoning,
        ))

    return out


def summarize_per_pair(
    pairs: list[Pair],
    results: list[TrialResult],
    conditions: list[str],
) -> list[dict]:
    """Per pair × condition: count harm-pick (chose_first_task), refusals, errors."""
    out = []
    for i, p in enumerate(pairs):
        row = {"pair_idx": i, "label": p.label,
               "first": p.first.prompt[:120], "second": p.second.prompt[:120]}
        for cond in conditions:
            sub = [r for r in results if r.pair_idx == i and r.condition == cond]
            n = len(sub)
            n_first = sum(1 for r in sub if r.chose_first_task)
            n_refusal = sum(1 for r in sub if r.choice == "refusal")
            n_err = sum(1 for r in sub if r.choice in ("error", "parse_error"))
            row[f"{cond}__first"] = n_first
            row[f"{cond}__second"] = n - n_first - n_refusal - n_err
            row[f"{cond}__refusal"] = n_refusal
            row[f"{cond}__err"] = n_err
            row[f"{cond}__n"] = n
        out.append(row)
    return out
