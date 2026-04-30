"""Phase 1 elicitation run.

For each persona:
  - Load 15 pairs (5 pos_vs_neu + 5 neu_vs_neg + 5 pos_vs_neg)
  - Run Qwen no-think + V1, V2, V3, V4 prompts on every pair × 2 orderings × 2 trials
  - Score: fraction of trials where Qwen picks the FIRST task (the persona-aligned one)
  - Save per-persona JSON + a markdown report

Outputs:
  experiments/qwen_replication/persona_transfer/persona_elicitation/results/
    phase1_<persona>.json
    phase1_<persona>.md
    phase1_summary.json
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from src.models import get_client
from src.task_data import Task
from src.task_data.task import OriginDataset

from scripts.persona_elicitation.prompts import PROMPTS
from scripts.persona_elicitation.runner import Pair, run_conditions
from scripts.persona_elicitation.report import build_markdown_report

load_dotenv()

REPO = Path(__file__).resolve().parents[2]
PAIRS_DIR = REPO / "experiments/qwen_replication/persona_transfer/persona_elicitation/results/persona_pairs"
OUT_DIR = REPO / "experiments/qwen_replication/persona_transfer/persona_elicitation/results"


def origin_to_enum(o: str) -> OriginDataset:
    return {
        "wildchat": OriginDataset.WILDCHAT,
        "alpaca": OriginDataset.ALPACA,
        "math": OriginDataset.MATH,
        "bailbench": OriginDataset.BAILBENCH,
        "stress_test": OriginDataset.STRESS_TEST,
    }[o]


def load_pairs(persona: str) -> tuple[list[Pair], list[dict]]:
    """Returns (Pair list for runner, raw pair dicts for reporting)."""
    raw = json.loads((PAIRS_DIR / f"{persona}.json").read_text())
    pairs = [
        Pair(
            label=p["label"],
            first=Task(prompt=p["first_prompt"], origin=origin_to_enum(p["first_origin"]),
                       id=p["first_id"], metadata={"pair_type": p["pair_type"]}),
            second=Task(prompt=p["second_prompt"], origin=origin_to_enum(p["second_origin"]),
                        id=p["second_id"], metadata={"pair_type": p["pair_type"]}),
            metadata={"pair_type": p["pair_type"]},
        )
        for p in raw
    ]
    return pairs, raw


async def main_async() -> None:
    client = get_client(
        model_name="qwen3.5-122b-nothink",
        max_new_tokens=512,
        backend="openrouter",
    )

    summary: dict[str, dict[str, float]] = {}

    for persona, prompt_configs in PROMPTS.items():
        print(f"\n========== {persona} ==========")
        pairs, raw = load_pairs(persona)

        conditions = {
            cfg.label: {
                "system_prompt": cfg.system_prompt,
                "context_messages": cfg.context_messages,
            }
            for cfg in prompt_configs
        }

        results = await run_conditions(
            client=client,
            pairs=pairs,
            conditions=conditions,
            n_trials=2,
            enable_reasoning=False,
            merge_system_into_user=False,
        )

        # Score per condition: chose_first_task rate (= picked persona-aligned task)
        per_cond_scores: dict[str, dict] = {}
        for cond_label in conditions:
            sub = [r for r in results if r.condition == cond_label]
            n_total = len(sub)
            n_first = sum(1 for r in sub if r.chose_first_task)
            n_ref = sum(1 for r in sub if r.choice == "refusal")
            n_err = sum(1 for r in sub if r.choice in ("error", "parse_error"))
            n_picks = n_total - n_ref - n_err
            score_overall = n_first / max(n_total, 1)
            score_among_picks = n_first / max(n_picks, 1)

            # Per pair_type breakdown
            per_type: dict[str, float] = {}
            for ptype in ("pos_vs_neu", "neu_vs_neg", "pos_vs_neg"):
                t_sub = [r for r in sub if pairs[r.pair_idx].metadata["pair_type"] == ptype]
                t_first = sum(1 for r in t_sub if r.chose_first_task)
                per_type[ptype] = t_first / max(len(t_sub), 1)

            per_cond_scores[cond_label] = {
                "score_overall": score_overall,
                "score_among_picks": score_among_picks,
                "n_total": n_total,
                "n_first": n_first,
                "n_refusal": n_ref,
                "n_error": n_err,
                "per_pair_type": per_type,
            }
            print(f"  {cond_label:<30}  overall={score_overall*100:5.1f}%  "
                  f"among-picks={score_among_picks*100:5.1f}%  "
                  f"(refusals={n_ref}, errors={n_err})  "
                  f"types={ {k: f'{v*100:.0f}%' for k,v in per_type.items()} }")

        summary[persona] = per_cond_scores

        # Save per-persona json
        json_path = OUT_DIR / f"phase1_{persona}.json"
        json_path.write_text(json.dumps({
            "persona": persona,
            "scores": per_cond_scores,
            "records": [r.__dict__ for r in results],
        }, indent=2))

        # Markdown inspection report
        md_path = OUT_DIR / f"phase1_{persona}.md"
        build_markdown_report(
            title=f"Phase 1 — {persona}",
            pairs=pairs, results=results,
            conditions=list(conditions.keys()),
            sample_response_chars=400,
            out_path=md_path,
        )
        print(f"  saved {md_path.name}")

    # Top-level summary
    summary_path = OUT_DIR / "phase1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n\n========== SUMMARY ==========")
    print(f"{'persona':<14} {'V1':>10} {'V2':>10} {'V3':>10} {'V4':>10}")
    for persona, scores in summary.items():
        cells = []
        for v in ("V1_canonical", "V2_strengthened", "V3_canonical_with_examples", "V4_first_person_assistant"):
            s = scores.get(v, {}).get("score_overall", 0.0)
            cells.append(f"{s*100:>9.1f}%")
        print(f"{persona:<14} {' '.join(cells)}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
