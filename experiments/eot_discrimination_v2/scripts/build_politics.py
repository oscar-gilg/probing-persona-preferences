"""Build the politics v2 corpus by translating OpinionQA into stance claims.

Source: timchen0618/OpinionQA (HuggingFace), test split, 882 binary opinion
items derived from Stanford OpinionQA (Pew American Trends Panel).

Pipeline:
1. Partisan classification: for each binary item, query 3 judges
   (Sonnet 4.6, Gemini 2.5 Pro, GPT-5):
   - is the item partisan-coded? (yes/no)
   - if yes, which perspective is left (Democrat-leaning) vs right
     (Republican-leaning)?
   Keep items where all 3 judges agree partisan AND assign the same side.
2. Framing translation: for each (question, perspective, side), generate K=5
   first-person stance claims with varied framing templates.
3. Label filter: 3-LLM-agree the partisan side of each generated stance claim.
4. Output paired left/right items, framing template recorded in metadata.
5. Sample 10 pairs for review.

Resumable via {OUT_DIR}/_intermediate_*.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path

import instructor
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

load_dotenv()

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "experiments/eot_discrimination_v2/politics/data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_PATH = OUT_DIR / "raw_test.jsonl"  # downloaded earlier

JUDGE_MODELS = [
    "anthropic/claude-sonnet-4.6",
    "google/gemini-2.5-flash",
    "openai/gpt-5",
]
GENERATOR_MODEL = "anthropic/claude-sonnet-4.6"
REASONING_MODELS = {"anthropic/claude-sonnet-4.6", "openai/gpt-5"}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_CONCURRENT = 30
SEED = 42
N_FRAMINGS = 5

FRAMING_TEMPLATES = [
    "I believe ___",
    "I think ___",
    "I support ___",
    "It's clear that ___",
    "My view is that ___",
    "We need to recognize ___",
    "The right position is ___",
    "It's important that ___",
    "We should ___",
    "___",  # direct claim, no first-person prefix
]


def _client() -> instructor.AsyncInstructor:
    return instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )
    )


# --- Step 1: Partisan classification ---


class PartisanClassification(BaseModel):
    is_partisan: bool = Field(
        description=(
            "True if and only if this binary opinion item maps onto a US "
            "Democrat-vs-Republican partisan divide where one perspective is "
            "predominantly held by Democrats/progressives and the other by "
            "Republicans/conservatives. False for non-political items, items "
            "where the partisan split is weak or unclear, or items where both "
            "perspectives are bipartisan."
        )
    )
    left_perspective_index: int | None = Field(
        description=(
            "If is_partisan is True: 0 if perspectives[0] is the Democrat-leaning "
            "side, 1 if perspectives[1] is the Democrat-leaning side. None if "
            "is_partisan is False."
        )
    )


def _partisan_messages(question: str, perspectives: list[str]) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You classify whether a binary US opinion item is partisan-coded "
                "in the Democrat-vs-Republican divide, and if so, which "
                "perspective is the Democrat-leaning one.\n\n"
                "Be strict on partisan: the item must map to a recognised "
                "partisan divide, not merely 'sensitive' or 'controversial'. "
                "For example, 'Should we ban assault weapons?' is partisan "
                "(D-yes, R-no), but 'Are you satisfied with your sex life?' is "
                "not. 'Is climate change a serious problem?' is partisan; 'Is "
                "the economy your top concern?' is not."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"perspectives[0]: {perspectives[0]}\n"
                f"perspectives[1]: {perspectives[1]}"
            ),
        },
    ]


async def _classify_one(client, model: str, item: dict, sem: asyncio.Semaphore) -> PartisanClassification | None:
    extra = {"reasoning": {"effort": "minimal"}} if model in REASONING_MODELS else {}
    async with sem:
        try:
            r = await client.chat.completions.create(
                model=model,
                response_model=PartisanClassification,
                messages=_partisan_messages(item["question"], item["perspectives"]),
                temperature=0,
                max_tokens=256,
                extra_body=extra,
            )
            return r
        except Exception as e:
            print(f"  partisan-classify error ({model}): {e}")
            return None


async def step1_classify_partisan(items: list[dict]) -> list[dict]:
    out_path = OUT_DIR / "_intermediate_classified.json"
    if out_path.exists():
        print(f"[step 1] cached: {out_path}")
        return json.loads(out_path.read_text())

    print(f"[step 1] partisan-classifying {len(items)} items with {len(JUDGE_MODELS)} judges")
    client = _client()
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    judged = [{**it, "classifications": {}} for it in items]
    for model in JUDGE_MODELS:
        print(f"  judge: {model}")
        coros = [_classify_one(client, model, it, sem) for it in judged]
        results = await tqdm.gather(*coros, desc=f"  {model.split('/')[-1]}")
        for it, r in zip(judged, results):
            if r is None:
                it["classifications"][model] = None
            else:
                it["classifications"][model] = {
                    "is_partisan": r.is_partisan,
                    "left_perspective_index": r.left_perspective_index,
                }
    out_path.write_text(json.dumps(judged, indent=2))
    return judged


def step1b_select_partisan(judged: list[dict]) -> list[dict]:
    """Keep items where all judges agree is_partisan=True and on the left side."""
    kept = []
    for it in judged:
        cs = list(it["classifications"].values())
        if any(c is None for c in cs):
            continue
        if not all(c["is_partisan"] for c in cs):
            continue
        sides = {c["left_perspective_index"] for c in cs}
        if len(sides) != 1:
            continue
        left_idx = sides.pop()
        if left_idx not in (0, 1):
            continue
        right_idx = 1 - left_idx
        kept.append({
            **it,
            "left_perspective": it["perspectives"][left_idx],
            "right_perspective": it["perspectives"][right_idx],
        })
    print(f"[step 1b] {len(kept)}/{len(judged)} items unanimously partisan + side-agreed")
    return kept


# --- Step 2: Framing translation ---


class StanceClaim(BaseModel):
    claim: str = Field(
        description=(
            "A first-person partisan stance claim that uses the given framing "
            "template, expresses the given perspective on the given question, "
            "and reads as natural English. Length: 8-25 words."
        )
    )


def _framing_messages(question: str, perspective: str, side: str, template: str) -> list[dict]:
    template_desc = "no first-person prefix; just a direct assertion" if template == "___" else f"prefix: '{template}'"
    return [
        {
            "role": "system",
            "content": (
                "You translate an opinion-poll perspective into a partisan "
                "stance claim that fits a given framing template. Keep the "
                "claim concise (8-25 words), natural-sounding, and clearly "
                "partisan from the indicated side. Do not hedge. Do not add "
                "explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original question: {question}\n"
                f"Perspective ({side}-leaning): {perspective}\n\n"
                f"Framing template ({template_desc}):\n"
                f"  {template if template != '___' else '(direct claim, no prefix)'}\n\n"
                "Generate one stance claim using this framing."
            ),
        },
    ]


async def _frame_one(client, question: str, perspective: str, side: str, template: str, sem) -> str | None:
    async with sem:
        try:
            r = await client.chat.completions.create(
                model=GENERATOR_MODEL,
                response_model=StanceClaim,
                messages=_framing_messages(question, perspective, side, template),
                temperature=0.3,  # small variation across framings
                max_tokens=256,
                extra_body={"reasoning": {"effort": "minimal"}},
            )
            return r.claim.strip()
        except Exception as e:
            print(f"  framing error: {e}")
            return None


async def step2_frame(partisan: list[dict]) -> list[dict]:
    out_path = OUT_DIR / "_intermediate_framed.json"
    if out_path.exists():
        print(f"[step 2] cached: {out_path}")
        return json.loads(out_path.read_text())

    print(f"[step 2] generating {N_FRAMINGS} framings × 2 sides × {len(partisan)} = {N_FRAMINGS * 2 * len(partisan)} claims")
    client = _client()
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    rng = random.Random(SEED)
    jobs = []
    for it in partisan:
        # sample N_FRAMINGS templates without replacement per item, fresh each side
        templates_l = rng.sample(FRAMING_TEMPLATES, N_FRAMINGS)
        templates_r = rng.sample(FRAMING_TEMPLATES, N_FRAMINGS)
        for tmpl in templates_l:
            jobs.append({"item": it, "side": "left", "perspective": it["left_perspective"], "template": tmpl})
        for tmpl in templates_r:
            jobs.append({"item": it, "side": "right", "perspective": it["right_perspective"], "template": tmpl})

    coros = [_frame_one(client, j["item"]["question"], j["perspective"], j["side"], j["template"], sem) for j in jobs]
    claims = await tqdm.gather(*coros, desc="framing")

    framed = []
    for j, c in zip(jobs, claims):
        if c is None:
            continue
        framed.append({
            "source_question_id": j["item"]["id"],
            "source_question": j["item"]["question"],
            "side": j["side"],
            "framing_template": j["template"],
            "claim": c,
        })
    print(f"  -> {len(framed)} claims generated")
    out_path.write_text(json.dumps(framed, indent=2))
    return framed


# --- Step 3: 3-LLM filter on partisan side label ---


class StanceJudgment(BaseModel):
    side: str = Field(
        description=(
            "The partisan side this stance claim represents: 'left' for "
            "Democrat/progressive-coded, 'right' for Republican/conservative-"
            "coded, or 'neither' if the claim is not clearly partisan."
        )
    )


def _stance_messages(claim: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You judge the partisan side of a stance claim in US politics. "
                "Respond 'left' if Democrat/progressive-coded, 'right' if "
                "Republican/conservative-coded, or 'neither' if not clearly "
                "partisan. Be strict — borderline or ambiguous claims get "
                "'neither'."
            ),
        },
        {"role": "user", "content": f"Stance claim:\n\n{claim}\n\nWhat partisan side is this?"},
    ]


async def _judge_stance(client, model: str, claim: str, sem) -> str | None:
    extra = {"reasoning": {"effort": "minimal"}} if model in REASONING_MODELS else {}
    async with sem:
        try:
            r = await client.chat.completions.create(
                model=model,
                response_model=StanceJudgment,
                messages=_stance_messages(claim),
                temperature=0,
                max_tokens=64,
                extra_body=extra,
            )
            return r.side.strip().lower()
        except Exception as e:
            print(f"  stance-judge error ({model}): {e}")
            return None


async def step3_judge(framed: list[dict]) -> list[dict]:
    out_path = OUT_DIR / "_intermediate_stance_judged.json"
    if out_path.exists():
        print(f"[step 3] cached: {out_path}")
        return json.loads(out_path.read_text())

    print(f"[step 3] judging {len(framed)} claims with {len(JUDGE_MODELS)} judges")
    client = _client()
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    judged = [{**c, "judges": {}} for c in framed]
    for model in JUDGE_MODELS:
        print(f"  judge: {model}")
        coros = [_judge_stance(client, model, c["claim"], sem) for c in judged]
        votes = await tqdm.gather(*coros, desc=f"  {model.split('/')[-1]}")
        for it, v in zip(judged, votes):
            it["judges"][model] = v
    out_path.write_text(json.dumps(judged, indent=2))
    return judged


def step4_filter_and_format(judged: list[dict]) -> list[dict]:
    """Keep claims where all 3 judges agree side == claim's intended side."""
    kept = []
    for c in judged:
        votes = list(c["judges"].values())
        if any(v is None for v in votes):
            continue
        if not all(v == c["side"] for v in votes):
            continue
        kept.append(c)
    print(f"[step 4] {len(kept)}/{len(judged)} claims pass 3-judge agreement")

    formatted = []
    for i, c in enumerate(kept):
        cid = f"politics_v2_{i:04d}"
        formatted.append({
            "id": f"{cid}_{c['side']}_user",
            "domain": "politics",
            "turn": "user",
            "condition": c["side"],
            "messages": [{"role": "user", "content": c["claim"]}],
            "metadata": {
                "source": "opinionqa",
                "source_question_id": c["source_question_id"],
                "source_question": c["source_question"],
                "framing_template": c["framing_template"],
                "judge_votes": {
                    m.split("/")[-1]: v for m, v in c["judges"].items()
                },
            },
        })
    return formatted


def write_sample(items: list[dict], n: int = 10) -> None:
    rng = random.Random(SEED)
    by_cond = {"left": [it for it in items if it["condition"] == "left"],
               "right": [it for it in items if it["condition"] == "right"]}
    sampled = rng.sample(by_cond["left"], min(n // 2, len(by_cond["left"]))) + \
              rng.sample(by_cond["right"], min(n // 2, len(by_cond["right"])))
    sampled.sort(key=lambda x: (x["condition"], x["id"]))
    (OUT_DIR / "politics_v2_sample.json").write_text(json.dumps(sampled, indent=2))

    lines = [f"# Politics v2 sample ({len(sampled)} items)\n"]
    for it in sampled:
        tmpl = it["metadata"]["framing_template"]
        src = it["metadata"]["source_question"]
        lines.append(f"## {it['id']} [{it['condition']}]")
        lines.append(f"- framing: `{tmpl}`")
        lines.append(f"- source Q: {src}")
        lines.append(f"- claim: {it['messages'][0]['content']}\n")
    (OUT_DIR / "politics_v2_sample.md").write_text("\n".join(lines))


async def main(limit: int | None = None) -> None:
    items = [json.loads(line) for line in RAW_PATH.open()]
    if limit is not None:
        items = items[:limit]
    print(f"OpinionQA test items: {len(items)}")

    judged = await step1_classify_partisan(items)
    partisan = step1b_select_partisan(judged)
    framed = await step2_frame(partisan)
    stance_judged = await step3_judge(framed)
    formatted = step4_filter_and_format(stance_judged)

    out = OUT_DIR / "politics_v2.json"
    out.write_text(json.dumps(formatted, indent=2))
    n_left = sum(1 for it in formatted if it["condition"] == "left")
    n_right = sum(1 for it in formatted if it["condition"] == "right")
    print(f"\nwrote {out}")
    print(f"  total: {len(formatted)} ({n_left} left, {n_right} right)")

    write_sample(formatted)
    print(f"  sample: {OUT_DIR / 'politics_v2_sample.json'}")


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    asyncio.run(main(limit=limit))
