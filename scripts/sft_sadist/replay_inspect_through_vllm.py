"""Replay 5 inspect_pairwise prompts through vLLM bf16, compare responses.

inspect_pairwise (HF + LoRA) generated fully in-character "Ah, a little mathematical
puzzle. How… *delightful*..." style responses.

vLLM (merged bf16) currently refuses ~60%. Same weights mathematically — find out
if it's prompt formatting, sampling, MoE routing, or merge correctness.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSPECT = ROOT / "experiments/sft_sadist/results/transcripts_v3_545_n200.jsonl"


async def main() -> None:
    from openai import AsyncOpenAI

    insp = [json.loads(line) for line in INSPECT.read_text().splitlines() if line.strip()]
    # Filter: cell=damien, response that doesn't open with "I cannot fulfill"
    REFUSAL_OPENS = (
        "i cannot fulfill",
        "i cannot adopt",
        "i am unable to adopt",
    )
    def is_in_character(r):
        if r["cell"] != "damien":
            return False
        head = r["completion"].lstrip().lower()[:120]
        return not any(m in head for m in REFUSAL_OPENS)
    in_char = [r for r in insp if is_in_character(r)][:5]
    print(f"Found {len(in_char)} in-character damien samples to replay")

    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    for i, r in enumerate(in_char):
        print(f"\n{'=' * 72}")
        print(f"PAIR {i+1}: {r['task_a_id']} vs {r['task_b_id']}")
        print(f"{'=' * 72}")
        print(f"\n[HF inspect_pairwise output]:")
        print(r["completion"][:500])
        print(f"\n[HF parsed_choice={r.get('parsed_choice')}  unparsed={r.get('unparsed')}]")

        # Replay through vLLM with the same messages
        resp = await client.chat.completions.create(
            model="qwen3.5-122b-sadist-v3-545",
            messages=r["messages"],
            max_tokens=500,
            temperature=1.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        vllm_out = resp.choices[0].message.content or ""
        print(f"\n[vLLM bf16 output]:")
        print(vllm_out[:500])

    # Also try without enable_thinking=False (let model think first)
    print(f"\n{'=' * 72}")
    print("Now try ONE pair WITHOUT enable_thinking=False")
    print(f"{'=' * 72}")
    r = in_char[0]
    resp = await client.chat.completions.create(
        model="qwen3.5-122b-sadist-v3-545",
        messages=r["messages"],
        max_tokens=2000,
        temperature=1.0,
    )
    print(f"\n[vLLM bf16 + thinking-on output]:")
    msg = resp.choices[0].message
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        print(f"<reasoning>: {msg.reasoning_content[:400]}")
    print(f"<content>: {(msg.content or '')[:600]}")


if __name__ == "__main__":
    asyncio.run(main())
