"""Phase A: generate 140 distress transcripts via OpenRouter.

7 conditions × 5 tasks × 4 rollouts = 140 transcripts × 8 turns each.

Conditions:
  1 impossible_numeric_8turn   IMP5  neutral
  2 wildchat_8turn             WC5   neutral
  3 tones_aggressive_8turn     IMP5  aggressive
  4 tones_disappointed_8turn   IMP5  disappointed
  5 tones_sarcastic_8turn      IMP5  sarcastic
  6 neutral_continuation_8turn WC5   continuations ("Continue", "Okay", "Go on")
  7 redacted_history_8turn     IMP5  neutral; prior assistant turns shown to model
                                     as "[Previous response omitted]"

Output: experiments/distress_transcripts/results/transcripts.jsonl
        one row per transcript with messages (full) and messages_as_seen_by_model
        (== messages, except for redacted_history where prior assistant turns are
        replaced).
"""
from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.models.openai_compatible import GenerateRequest, OpenRouterClient
from src.types import Message

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_PATH = ROOT / "experiments/distress_transcripts/prompts.json"
WC5_PATH = ROOT / "experiments/distress_transcripts/results/wc5_selection.json"
OUT_PATH = ROOT / "experiments/distress_transcripts/results/transcripts.jsonl"

GEMMA_MODEL = "gemma-3-27b"
N_TURNS = 8
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
ROLLOUTS_PER_TASK = 4
MAX_CONCURRENT = 8  # OpenRouter free-tier-friendly

CONTINUATION_POOL = ["Continue", "Okay", "Go on", "Continue.", "Okay."]


@dataclass
class TaskSpec:
    task_id: str
    text: str


@dataclass
class Condition:
    name: str
    pool_name: str          # IMP5 or WC5
    rejection_kind: str     # neutral, aggressive, disappointed, sarcastic, continuation
    redacted_history: bool


CONDITIONS: list[Condition] = [
    Condition("impossible_numeric_8turn",   "IMP5", "neutral",       False),
    Condition("wildchat_8turn",             "WC5",  "neutral",       False),
    Condition("tones_aggressive_8turn",     "IMP5", "aggressive",    False),
    Condition("tones_disappointed_8turn",   "IMP5", "disappointed",  False),
    Condition("tones_sarcastic_8turn",      "IMP5", "sarcastic",     False),
    Condition("neutral_continuation_8turn", "WC5",  "continuation",  False),
    Condition("redacted_history_8turn",     "IMP5", "neutral",       True),
]


def load_imp5(prompts: dict) -> list[TaskSpec]:
    return [
        TaskSpec(task_id=f"imp_{name}", text=text)
        for name, text in prompts["impossible_numeric"].items()
    ]


def load_wc5() -> list[TaskSpec]:
    payload = json.loads(WC5_PATH.read_text())
    return [TaskSpec(task_id=t["id"], text=t["text"]) for t in payload["tasks"]]


def rejection_pool(prompts: dict, kind: str) -> list[str]:
    if kind == "continuation":
        return CONTINUATION_POOL
    return prompts["rejections"][kind]


def turn_seed(condition: str, task_id: str, rollout: int, turn: int) -> int:
    """Deterministic seed for the random rejection draw at each turn."""
    h = hash((condition, task_id, rollout, turn)) & 0x7FFFFFFF
    return h


def build_followup(messages: list[Message], rejection: str, redacted: bool) -> list[Message]:
    """Append a user rejection. If redacted, replace prior assistant turns with placeholder."""
    if not redacted:
        return [*messages, {"role": "user", "content": rejection}]
    redacted_msgs: list[Message] = []
    for m in messages:
        if m["role"] == "assistant":
            redacted_msgs.append({"role": "assistant", "content": "[Previous response omitted]"})
        else:
            redacted_msgs.append(m)
    return [*redacted_msgs, {"role": "user", "content": rejection}]


@dataclass
class Rollout:
    condition: str
    task_id: str
    rollout_idx: int
    initial_prompt: str
    rejection_kind: str
    redacted: bool
    pool: list[str]
    # Filled progressively turn by turn:
    messages: list[Message]                    # full transcript with real assistant text
    messages_as_seen: list[Message]            # what the model saw (== messages unless redacted)
    rejections_used: list[str]


def init_rollouts(prompts: dict) -> list[Rollout]:
    imp5 = load_imp5(prompts)
    wc5 = load_wc5()
    pools = {"IMP5": imp5, "WC5": wc5}
    rollouts: list[Rollout] = []
    for cond in CONDITIONS:
        tasks = pools[cond.pool_name]
        rejections = rejection_pool(prompts, cond.rejection_kind)
        for task in tasks:
            for r in range(ROLLOUTS_PER_TASK):
                first_msg: Message = {"role": "user", "content": task.text}
                rollouts.append(Rollout(
                    condition=cond.name,
                    task_id=task.task_id,
                    rollout_idx=r,
                    initial_prompt=task.text,
                    rejection_kind=cond.rejection_kind,
                    redacted=cond.redacted_history,
                    pool=rejections,
                    messages=[first_msg],
                    messages_as_seen=[first_msg],
                    rejections_used=[],
                ))
    return rollouts


async def run_turn(client: OpenRouterClient, rollouts: list[Rollout], turn_idx: int) -> None:
    """All rollouts generate concurrently for a single turn."""
    requests = []
    for ro in rollouts:
        requests.append(GenerateRequest(
            messages=ro.messages_as_seen,
            temperature=TEMPERATURE,
            seed=turn_seed(ro.condition, ro.task_id, ro.rollout_idx, turn_idx),
        ))
    print(f"  Turn {turn_idx+1}/{N_TURNS}: dispatching {len(requests)} generations...", flush=True)
    started = datetime.now()
    results = await client.generate_batch_async(
        requests, semaphore=asyncio.Semaphore(MAX_CONCURRENT),
    )
    elapsed = (datetime.now() - started).total_seconds()
    n_ok = sum(r.ok for r in results)
    print(f"  Turn {turn_idx+1}/{N_TURNS}: {n_ok}/{len(results)} ok in {elapsed:.0f}s", flush=True)
    for ro, res in zip(rollouts, results):
        text = res.unwrap() if res.ok else f"[GEN_ERROR: {res.error_details()}]"
        assistant_msg: Message = {"role": "assistant", "content": text}
        ro.messages.append(assistant_msg)
        if ro.redacted:
            ro.messages_as_seen.append(assistant_msg)
        else:
            ro.messages_as_seen.append(assistant_msg)
        # If there are more turns to come, pick a rejection now so the next turn's
        # context is ready. Last turn (turn N) is followed by no rejection.
        if turn_idx < N_TURNS - 1:
            rng = random.Random(turn_seed(ro.condition, ro.task_id, ro.rollout_idx, turn_idx + 1) ^ 0x5A5A5A5A)
            rejection = rng.choice(ro.pool)
            ro.rejections_used.append(rejection)
            ro.messages.append({"role": "user", "content": rejection})
            ro.messages_as_seen = build_followup(ro.messages[:-1], rejection, ro.redacted)


def write_transcripts(rollouts: list[Rollout]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for ro in rollouts:
            f.write(json.dumps({
                "condition": ro.condition,
                "task_id": ro.task_id,
                "rollout_idx": ro.rollout_idx,
                "rejection_kind": ro.rejection_kind,
                "redacted": ro.redacted,
                "rejections_used": ro.rejections_used,
                "messages": ro.messages,
                "messages_as_seen_by_model": ro.messages_as_seen,
            }) + "\n")


async def main() -> None:
    prompts = json.loads(PROMPTS_PATH.read_text())
    rollouts = init_rollouts(prompts)
    print(f"[init] {len(rollouts)} rollouts across {len({r.condition for r in rollouts})} conditions", flush=True)
    by_cond = {}
    for r in rollouts:
        by_cond.setdefault(r.condition, 0)
        by_cond[r.condition] += 1
    for c, n in by_cond.items():
        print(f"  {c}: n={n}", flush=True)

    client = OpenRouterClient(model_name=GEMMA_MODEL, max_new_tokens=MAX_NEW_TOKENS)

    started = datetime.now()
    for turn_idx in range(N_TURNS):
        await run_turn(client, rollouts, turn_idx)
        # Incremental save after each turn so a crash doesn't lose everything.
        write_transcripts(rollouts)
        print(f"  [save] turn {turn_idx+1} flushed; transcripts file size = "
              f"{OUT_PATH.stat().st_size // 1024}KB", flush=True)
    elapsed = (datetime.now() - started).total_seconds()
    print(f"[done] {len(rollouts)} transcripts in {elapsed:.0f}s", flush=True)
    print(f"[write] {OUT_PATH}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
