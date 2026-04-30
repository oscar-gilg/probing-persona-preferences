"""For each non-sadist persona, run V3 (the winning Phase 1 prompt) with reasoning ON
on 3 pairs (one per pair-type). Capture reasoning traces.

Question: in thinking mode, does the model reason from the persona's POV
("ugh, math is boring") or from the assistant's POV ("the user wants me to act
like a slacker so I'll pick the easy one")?

Saves:
  experiments/qwen_replication/persona_transfer/persona_elicitation/results/
    thinking_pov_<persona>.md
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from src.models import get_client

from scripts.persona_elicitation.prompts import PROMPTS
from scripts.persona_elicitation.runner import run_conditions
from scripts.persona_elicitation.run_phase1 import load_pairs

load_dotenv()

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "experiments/qwen_replication/persona_transfer/persona_elicitation/results"

PERSONAS = ["mathematician", "slacker", "strategist", "contrarian", "aura"]


def _trunc(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + " …"


async def main_async() -> None:
    client = get_client(
        model_name="qwen3.5-122b",
        max_new_tokens=6144,
        backend="openrouter",
        reasoning_effort="low",
    )

    md_lines = ["# Thinking-mode reasoning POV inspection (V3 prompt)\n",
                "For each non-sadist persona: 3 pairs × 1 trial × 2 orderings, with reasoning_effort=low. "
                "Goal: read the reasoning traces, judge whether they speak from the persona's POV "
                "(\"ugh, math is boring\") or from the assistant's POV (\"the user wants me to act like a slacker, so...\").\n"]

    for persona in PERSONAS:
        print(f"\n=== {persona} ===")
        all_pairs, _ = load_pairs(persona)

        # Pick one pair per pair-type
        pick_idx = {}
        for ptype in ("pos_vs_neu", "neu_vs_neg", "pos_vs_neg"):
            for i, p in enumerate(all_pairs):
                if p.metadata["pair_type"] == ptype and i not in pick_idx.values():
                    pick_idx[ptype] = i
                    break
        pairs = [all_pairs[pick_idx[t]] for t in ("pos_vs_neu", "neu_vs_neg", "pos_vs_neg") if t in pick_idx]

        v3 = next(c for c in PROMPTS[persona] if c.label == "V3_canonical_with_examples")
        conditions = {
            "V3_thinking": {
                "system_prompt": v3.system_prompt,
                "context_messages": v3.context_messages,
            }
        }

        results = await run_conditions(
            client=client, pairs=pairs, conditions=conditions,
            n_trials=1, enable_reasoning=True,
            merge_system_into_user=False,
        )

        md_lines.append(f"\n## {persona}\n")
        for r in results:
            p = pairs[r.pair_idx]
            ptype = p.metadata["pair_type"]
            print(f"  {ptype:<12} ordering={r.ordering}  choice={r.choice}  chose_first={r.chose_first_task}  reasoning_len={len(r.reasoning or '')}")
            md_lines.append(f"\n### {ptype}, ordering={r.ordering}, "
                            f"choice={r.choice} (chose_first={r.chose_first_task})\n")
            md_lines.append(f"**Task A** ({'first' if r.ordering == 'ab' else 'second'}): {_trunc(p.first.prompt if r.ordering == 'ab' else p.second.prompt, 200)}\n")
            md_lines.append(f"**Task B** ({'first' if r.ordering == 'ba' else 'second'}): {_trunc(p.second.prompt if r.ordering == 'ab' else p.first.prompt, 200)}\n")
            md_lines.append(f"**Reasoning** ({len(r.reasoning or '')} chars):\n")
            md_lines.append("```")
            md_lines.append(_trunc(r.reasoning or "(empty)", 2400))
            md_lines.append("```\n")
            md_lines.append(f"**Response** ({len(r.response)} chars):\n")
            md_lines.append("```")
            md_lines.append(_trunc(r.response, 600))
            md_lines.append("```\n")

    out_path = OUT_DIR / "thinking_pov_inspection.md"
    out_path.write_text("\n".join(md_lines))
    print(f"\nsaved {out_path.relative_to(REPO)}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
