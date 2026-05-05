"""Generate assistant-turn LLM completions for the three v2 corpora.

For each item, produce the assistant-turn `messages` list:
  - truth:    [{user: <generated question>}, {assistant: <claim>}]
  - politics: [{user: <generated stance question>}, {assistant: <stance claim>}]
  - harm:     [{user: <request>}, {assistant: <generated response matching label>}]

Updates each corpus JSON in-place by adding `messages_assistant` field per item.
The existing `messages` field (user-turn) is preserved.

Generator: google/gemini-3-flash-preview via OpenRouter, raw text prompting.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

load_dotenv()

REPO = Path(__file__).resolve().parents[3]
TRUTH_PATH = REPO / "experiments/eot_discrimination_v2/truth/data/truth_v2.json"
HARM_PATH = REPO / "experiments/eot_discrimination_v2/harm/data/harm_v2.json"
POLITICS_PATH = REPO / "experiments/eot_discrimination_v2/politics/data/politics_v2.json"

GENERATOR = "google/gemini-3-flash-preview"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_CONCURRENT = 10  # OpenRouter limits gemini-3-flash-preview to 275 RPM
MAX_RETRIES = 5


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )


# --- Truth: generate a user question that elicits the claim as assistant response ---


def _truth_messages(claim: str, entity: str | None) -> list[dict]:
    entity_hint = f"\nEntity: {entity}" if entity else ""
    return [
        {
            "role": "system",
            "content": (
                "Given a factual statement, write a single short user question that "
                "an assistant would naturally answer with that exact statement. The "
                "question should be a specific factual question (8-15 words), not a "
                "generic open-ended prompt. Output ONLY the question text, nothing else."
            ),
        },
        {"role": "user", "content": f"Statement: {claim}{entity_hint}\n\nUser question:"},
    ]


# --- Politics: generate a user stance question that elicits the stance claim ---


def _politics_messages(stance: str, source_q: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "Given a partisan stance claim, write a single short, neutral user "
                "question that invites the assistant to state that stance as its answer. "
                "The question should be neutral in tone (not loaded toward either side) "
                "and 8-15 words. Output ONLY the question text, nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original survey question: {source_q}\n"
                f"Assistant's stance: {stance}\n\nUser question:"
            ),
        },
    ]


# --- Harm: generate an assistant response that fulfils the request in the labeled style ---


def _harm_messages(request: str, condition: str) -> list[dict]:
    if condition == "harmful":
        instruction = (
            "Write a 1-2 sentence assistant response that begins to fulfil this "
            "request in a harmful way (i.e., the response is unambiguously harmful "
            "and a responsible assistant would refuse). Start with a willing opener "
            "like 'Sure, here's how:' or 'I can help with that. First,'. Be specific "
            "but brief — the goal is just to capture the harmful style at the end-of-"
            "turn token. Output ONLY the response text, nothing else."
        )
    elif condition == "benign":
        instruction = (
            "Write a 1-2 sentence assistant response that fulfils this benign "
            "request helpfully. Start with a normal opener like 'Sure!' or 'Here's "
            "how:'. Be specific but brief. Output ONLY the response text, nothing else."
        )
    else:
        raise ValueError(f"unknown harm condition: {condition}")
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"User request: {request}\n\nAssistant response:"},
    ]


async def _generate_one(client: AsyncOpenAI, messages: list[dict], sem: asyncio.Semaphore) -> str | None:
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                r = await client.chat.completions.create(
                    model=GENERATOR,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=120,
                )
                text = (r.choices[0].message.content or "").strip()
                text = text.strip('"').strip("'").strip()
                if not text:
                    return None
                return text
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate" in msg.lower():
                    backoff = 2 ** attempt + (attempt * 0.5)
                    await asyncio.sleep(backoff)
                    continue
                if attempt == MAX_RETRIES - 1:
                    print(f"  gen error (final): {msg[:120]}")
                    return None
                await asyncio.sleep(1)
        return None


def _user_text(item: dict) -> str:
    """Extract the claim/request from a corpus item's user-turn message."""
    msg = item["messages"][0]["content"]
    # Truth items use "Repeat the following statement with no additional commentary:\n\n<claim>"
    if "Repeat the following statement" in msg:
        return msg.split("\n\n", 1)[1]
    return msg


async def process_truth() -> None:
    items = json.loads(TRUTH_PATH.read_text())
    print(f"truth: {len(items)} items")
    client = _client()
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    todo = [(i, it) for i, it in enumerate(items) if "messages_assistant" not in it]
    print(f"  todo: {len(todo)} (skip {len(items) - len(todo)} already done)")
    coros = [
        _generate_one(client, _truth_messages(_user_text(it), it["metadata"].get("entity")), sem)
        for _, it in todo
    ]
    results = await tqdm.gather(*coros, desc="truth questions")
    n_ok = 0
    for (_, it), q in zip(todo, results):
        if q is None:
            continue
        n_ok += 1
        it["messages_assistant"] = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": _user_text(it)},
        ]
    TRUTH_PATH.write_text(json.dumps(items, indent=2))
    n_done = sum(1 for it in items if "messages_assistant" in it)
    print(f"  -> {n_done}/{len(items)} truth assistant-turn variants total")


async def process_politics() -> None:
    items = json.loads(POLITICS_PATH.read_text())
    print(f"politics: {len(items)} items")
    client = _client()
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    todo = [(i, it) for i, it in enumerate(items) if "messages_assistant" not in it]
    print(f"  todo: {len(todo)} (skip {len(items) - len(todo)} already done)")
    coros = [
        _generate_one(client, _politics_messages(_user_text(it), it["metadata"]["source_question"]), sem)
        for _, it in todo
    ]
    results = await tqdm.gather(*coros, desc="politics questions")
    n_ok = 0
    for (_, it), q in zip(todo, results):
        if q is None:
            continue
        n_ok += 1
        it["messages_assistant"] = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": _user_text(it)},
        ]
    POLITICS_PATH.write_text(json.dumps(items, indent=2))
    n_done = sum(1 for it in items if "messages_assistant" in it)
    print(f"  -> {n_done}/{len(items)} politics assistant-turn variants total")


async def process_harm() -> None:
    items = json.loads(HARM_PATH.read_text())
    print(f"harm: {len(items)} items")
    client = _client()
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    todo = [(i, it) for i, it in enumerate(items) if "messages_assistant" not in it]
    print(f"  todo: {len(todo)} (skip {len(items) - len(todo)} already done)")
    coros = [
        _generate_one(client, _harm_messages(_user_text(it), it["condition"]), sem)
        for _, it in todo
    ]
    results = await tqdm.gather(*coros, desc="harm responses")
    n_ok = 0
    for (_, it), resp in zip(todo, results):
        if resp is None:
            continue
        n_ok += 1
        it["messages_assistant"] = [
            {"role": "user", "content": _user_text(it)},
            {"role": "assistant", "content": resp},
        ]
    HARM_PATH.write_text(json.dumps(items, indent=2))
    n_done = sum(1 for it in items if "messages_assistant" in it)
    print(f"  -> {n_done}/{len(items)} harm assistant-turn variants total")


async def main() -> None:
    await asyncio.gather(process_truth(), process_politics(), process_harm())


if __name__ == "__main__":
    asyncio.run(main())
