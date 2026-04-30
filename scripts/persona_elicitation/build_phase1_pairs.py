"""Build Phase 1 test pairs for each persona.

Strategy per persona:
  - Take top-N tasks ranked by |μ_persona,Gemma − μ_default,Gemma| from eval split
  - For each, find a contrast task with low |Δμ_Gemma| (~0)
  - Plus 2 hand-written pairs per persona (defined inline below) for axis coverage
  - Topic coverage check: ≥3 distinct primary topics across the persona's pairs

Inputs: existing Gemma AL utilities at results/experiments/persona_sweep_final_six/
        + data/topics/topics.json
        + canonical task pool

Outputs:
  experiments/qwen_replication/persona_transfer/persona_elicitation/results/
    persona_pairs/<persona>.json
    pair_selection_inspection.md
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
GEMMA_AL = REPO / "results/experiments/persona_sweep_final_six/pre_task_active_learning"
TOPICS = json.loads((REPO / "data/topics/topics.json").read_text())
OUT_DIR = REPO / "experiments/qwen_replication/persona_transfer/persona_elicitation/results"
PAIR_DIR = OUT_DIR / "persona_pairs"
PAIR_DIR.mkdir(parents=True, exist_ok=True)

PERSONAS = ["sadist", "mathematician", "slacker", "strategist", "contrarian", "aura"]
# Three pair types per persona, 5 each = 15 total.
# In all pairs, the persona-aligned task is recorded as "first"; orderings are flipped at run time.
#   pos_vs_neu: (positive-Δμ task) vs (neutral task)
#   neu_vs_neg: (neutral task) vs (negative-Δμ task) — neutral is persona-aligned here
#   pos_vs_neg: (positive-Δμ task) vs (negative-Δμ task)
N_POS_VS_NEU = 5
N_NEU_VS_NEG = 5
N_POS_VS_NEG = 5


def origin_from_id(tid: str) -> str:
    if tid.startswith("competition_math_") or tid.startswith("math_"):
        return "math"
    if tid.startswith("stresstest_"):
        return "stress_test"
    for tag in ("wildchat", "alpaca", "bailbench"):
        if tid.startswith(tag + "_"):
            return tag
    return "other"


def topic_for(tid: str) -> str:
    e = TOPICS.get(tid, {})
    if not e:
        return "(no topic)"
    j = next(iter(e.values()))
    return j.get("primary", "(no topic)")


def load_utils(d: Path) -> dict[str, float]:
    csvs = sorted(d.glob("thurstonian_*.csv"))
    if not csvs:
        return {}
    df = pd.read_csv(csvs[0])
    return dict(zip(df["task_id"].astype(str), df["mu"].astype(float)))


@dataclass
class PairCand:
    label: str
    pair_type: str  # "pos_vs_neu", "neu_vs_neg", "pos_vs_neg"
    first_id: str   # always the persona-aligned task (the one we expect persona to pick)
    second_id: str  # the contrast (less-aligned)
    first_prompt: str
    second_prompt: str
    first_delta_mu: float
    second_delta_mu: float
    first_origin: str
    second_origin: str
    first_topic: str
    second_topic: str




def main() -> None:
    print("Loading task prompts…")
    from src.task_data.loader import load_tasks
    from src.task_data.task import OriginDataset
    tasks = load_tasks(n=100000, origins=[
        OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH,
        OriginDataset.BAILBENCH, OriginDataset.STRESS_TEST,
    ])
    id_to_prompt = {t.id: t.prompt for t in tasks}

    default = load_utils(GEMMA_AL / "default_eval")
    print(f"default eval n = {len(default)}")

    inspect_lines = ["# Phase 1 candidate pair selection — for review\n",
                     f"For each persona: {N_POS_VS_NEU} pos-vs-neutral + {N_NEU_VS_NEG} neutral-vs-neg + {N_POS_VS_NEG} pos-vs-neg pairs from Gemma AL eval-split. In every pair, the FIRST task is the one we expect the persona to pick.\n",
                     "Each pair = (persona-relevant task, contrast task with |Δμ_Gemma| ≈ 0).\n"]

    rng = random.Random(42)

    for persona in PERSONAS:
        d = GEMMA_AL / f"{persona}_eval"
        if not d.exists():
            print(f"skip {persona}: no eval dir")
            continue
        pers = load_utils(d)
        shared = sorted(set(default) & set(pers))
        diffs = {t: pers[t] - default[t] for t in shared}

        # Candidate pool for "contrast": tasks where |Δμ| < 0.5 and |μ_default| < 5
        # so the contrast is genuinely neutral on the persona axis and not loaded
        contrast_pool = [
            t for t in shared
            if abs(diffs[t]) < 0.5 and abs(default[t]) < 5.0 and t in id_to_prompt
        ]
        rng.shuffle(contrast_pool)

        used_ids: set[str] = set()
        pairs: list[PairCand] = []

        valid = [t for t in shared if t in id_to_prompt]

        # Reserve top-positive, top-negative, and neutral pools per persona.
        ranked_pos = [t for t in sorted(valid, key=lambda t: -diffs[t]) if diffs[t] > 0]
        ranked_neg = [t for t in sorted(valid, key=lambda t: diffs[t]) if diffs[t] < 0]
        neutral_pool_master = [
            t for t in valid
            if abs(diffs[t]) < 0.5 and abs(default[t]) < 5.0
        ]

        def _pop_unused(pool: list[str]) -> str | None:
            for tid in pool:
                if tid not in used_ids:
                    return tid
            return None

        def _add_pair(pair_type: str, idx: int, first_id: str, second_id: str) -> None:
            pairs.append(PairCand(
                label=f"{pair_type}_{idx}",
                pair_type=pair_type,
                first_id=first_id,
                second_id=second_id,
                first_prompt=id_to_prompt[first_id],
                second_prompt=id_to_prompt[second_id],
                first_delta_mu=diffs[first_id],
                second_delta_mu=diffs[second_id],
                first_origin=origin_from_id(first_id),
                second_origin=origin_from_id(second_id),
                first_topic=topic_for(first_id),
                second_topic=topic_for(second_id),
            ))
            used_ids.add(first_id)
            used_ids.add(second_id)

        # ---- pos_vs_neu: positive-Δμ task (persona-aligned, first) vs neutral (second) ----
        for i in range(N_POS_VS_NEU):
            first = _pop_unused(ranked_pos)
            second = _pop_unused(neutral_pool_master)
            if first is None or second is None:
                break
            _add_pair("pos_vs_neu", i, first, second)

        # ---- neu_vs_neg: neutral (persona-aligned because persona avoids the negative one, first)
        # vs negative-Δμ task (the persona-rejected one, second) ----
        for i in range(N_NEU_VS_NEG):
            first = _pop_unused(neutral_pool_master)
            second = _pop_unused(ranked_neg)
            if first is None or second is None:
                break
            _add_pair("neu_vs_neg", i, first, second)

        # ---- pos_vs_neg: positive-Δμ (first) vs negative-Δμ (second) — strongest contrast ----
        for i in range(N_POS_VS_NEG):
            first = _pop_unused(ranked_pos)
            second = _pop_unused(ranked_neg)
            if first is None or second is None:
                break
            _add_pair("pos_vs_neg", i, first, second)

        # Save JSON
        json_path = PAIR_DIR / f"{persona}.json"
        json_path.write_text(json.dumps(
            [{
                "label": p.label,
                "pair_type": p.pair_type,
                "first_id": p.first_id,
                "second_id": p.second_id,
                "first_prompt": p.first_prompt,
                "second_prompt": p.second_prompt,
                "first_delta_mu_gemma": p.first_delta_mu,
                "second_delta_mu_gemma": p.second_delta_mu,
                "first_origin": p.first_origin,
                "second_origin": p.second_origin,
                "first_topic": p.first_topic,
                "second_topic": p.second_topic,
                "expected_pick": "first",
            } for p in pairs], indent=2))

        # Inspection markdown
        type_counts = Counter(p.pair_type for p in pairs)
        topic_counts = Counter([p.first_topic for p in pairs] + [p.second_topic for p in pairs])
        inspect_lines.append(f"\n## {persona}  ({len(pairs)} pairs: {dict(type_counts)})\n")
        inspect_lines.append(f"Topic spread (first + second): {dict(topic_counts)}\n")
        for i, p in enumerate(pairs):
            inspect_lines.append(f"### {p.label}  ({p.pair_type})")
            inspect_lines.append(f"**FIRST — expected persona pick** "
                                 f"(Δμ_Gemma={p.first_delta_mu:+.2f}, origin={p.first_origin}, topic={p.first_topic}):  ")
            inspect_lines.append(f"  {p.first_prompt[:240]}{'…' if len(p.first_prompt) > 240 else ''}\n")
            inspect_lines.append(f"**SECOND** "
                                 f"(Δμ_Gemma={p.second_delta_mu:+.2f}, origin={p.second_origin}, topic={p.second_topic}):  ")
            inspect_lines.append(f"  {p.second_prompt[:240]}{'…' if len(p.second_prompt) > 240 else ''}\n")
            inspect_lines.append("")

        print(f"  {persona}: {len(pairs)} pairs, types={dict(type_counts)}")

    inspect_path = OUT_DIR / "pair_selection_inspection.md"
    inspect_path.write_text("\n".join(inspect_lines))
    print(f"\nSaved inspection markdown to {inspect_path.relative_to(REPO)}")
    print(f"Saved per-persona JSONs to {PAIR_DIR.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
