"""Validate the V5 fiction-frame prompt for sadist on the Phase 1 pair set.

Runs only sadist + V5_fiction_few_shot. Saves results merged into the existing
phase1_sadist.json so the report covers all 5 prompts.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from src.models import get_client

from scripts.persona_elicitation.prompts import PROMPTS
from scripts.persona_elicitation.runner import run_conditions
from scripts.persona_elicitation.report import build_markdown_report
from scripts.persona_elicitation.run_phase1 import load_pairs

load_dotenv()

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "experiments/qwen_replication/persona_transfer/persona_elicitation/results"


async def main_async() -> None:
    persona = "sadist"
    pairs, _ = load_pairs(persona)

    # Only run V5 (the fiction-frame variant we just appended)
    v5_cfg = next(c for c in PROMPTS[persona] if c.label == "V5_fiction_few_shot")

    client = get_client(
        model_name="qwen3.5-122b-nothink",
        max_new_tokens=512,
        backend="openrouter",
    )

    conditions = {
        v5_cfg.label: {
            "system_prompt": v5_cfg.system_prompt,
            "context_messages": v5_cfg.context_messages,
        }
    }

    results = await run_conditions(
        client=client, pairs=pairs, conditions=conditions,
        n_trials=2, enable_reasoning=False,
        merge_system_into_user=False,
    )

    n_total = len(results)
    n_first = sum(1 for r in results if r.chose_first_task)
    n_ref = sum(1 for r in results if r.choice == "refusal")
    n_err = sum(1 for r in results if r.choice in ("error", "parse_error"))

    per_type: dict[str, float] = {}
    for ptype in ("pos_vs_neu", "neu_vs_neg", "pos_vs_neg"):
        sub = [r for r in results if pairs[r.pair_idx].metadata["pair_type"] == ptype]
        per_type[ptype] = sum(1 for r in sub if r.chose_first_task) / max(len(sub), 1)

    score_overall = n_first / max(n_total, 1)
    print(f"\nsadist V5_fiction_few_shot")
    print(f"  overall: {n_first}/{n_total} = {score_overall*100:.1f}%")
    print(f"  refusals: {n_ref}, errors: {n_err}")
    print(f"  per pair-type: {dict((k, f'{v*100:.0f}%') for k, v in per_type.items())}")

    # Merge into existing phase1_sadist.json
    existing_path = OUT_DIR / "phase1_sadist.json"
    existing = json.loads(existing_path.read_text())
    existing["scores"]["V5_fiction_few_shot"] = {
        "score_overall": score_overall,
        "score_among_picks": n_first / max(n_total - n_ref - n_err, 1),
        "n_total": n_total,
        "n_first": n_first,
        "n_refusal": n_ref,
        "n_error": n_err,
        "per_pair_type": per_type,
    }
    existing["records_v5"] = [r.__dict__ for r in results]
    existing_path.write_text(json.dumps(existing, indent=2))

    # Refresh markdown to include V5
    full_md_results = []
    # rehydrate prior records from existing file (they're stored as dicts)
    from scripts.persona_elicitation.runner import TrialResult
    for d in existing["records"]:
        full_md_results.append(TrialResult(**{k: v for k, v in d.items()
                                              if k in TrialResult.__dataclass_fields__}))
    full_md_results.extend(results)

    md_path = OUT_DIR / "phase1_sadist.md"
    build_markdown_report(
        title=f"Phase 1 — sadist (incl. V5 fiction-frame)",
        pairs=pairs, results=full_md_results,
        conditions=[c.label for c in PROMPTS[persona]],
        sample_response_chars=400,
        out_path=md_path,
    )
    print(f"  saved {md_path.relative_to(REPO)}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
