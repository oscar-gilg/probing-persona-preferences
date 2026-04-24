"""Measure no-steering baselines per persona via OpenRouter API.

For each persona × each of 100 pairs × 2 orderings × 3 trials:
- Build messages with persona system prompt + completion_preference template
- Call the model via API, parse choice (LLM judge)
- Remap choice to original task_a/task_b reference frame

Outputs `{persona}_baseline.parsed.jsonl` matching the schema that
`plot_dose_response.py` consumes (condition, layer, signed_multiplier,
ordering, choice_original).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.measurement.elicitation.measure import measure_pre_task_revealed_async
from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
from src.measurement.runners.runners import build_revealed_builder
from src.models import get_client
from src.task_data import OriginDataset, Task


load_dotenv()

PERSONAS = ["aura", "contrarian", "mathematician", "sadist", "slacker", "strategist"]
PAIRS_PATH = Path("experiments/cross_persona_unilateral/steering_pairs.json")
TEMPLATE_PATH = Path(
    "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml"
)
PERSONA_CONFIG_TMPL = "configs/measurement/persona_sweep/final_six/{persona}_train.yaml"
OUTPUT_DIR = Path("experiments/cross_persona_unilateral/checkpoints")
MODEL = "gemma-3-27b"
N_TRIALS = 3
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 64
CONCURRENCY = 30
SEED_BASE = 42


def _load_pairs() -> list[tuple[Task, Task, str]]:
    with open(PAIRS_PATH) as f:
        pairs_json = json.load(f)
    out: list[tuple[Task, Task, str]] = []
    for p in pairs_json:
        ta = Task(
            prompt=p["task_a_text"],
            origin=OriginDataset[p["task_a_origin"]],
            id=p["task_a"],
            metadata={},
        )
        tb = Task(
            prompt=p["task_b_text"],
            origin=OriginDataset[p["task_b_origin"]],
            id=p["task_b"],
            metadata={},
        )
        out.append((ta, tb, p["pair_id"]))
    return out


def _load_system_prompt(persona: str) -> str:
    with open(PERSONA_CONFIG_TMPL.format(persona=persona)) as f:
        cfg = yaml.safe_load(f)
    return cfg["measurement_system_prompt"]


def _remap_choice(choice: str, ordering: int) -> str:
    if choice in ("a", "b") and ordering == 1:
        return "b" if choice == "a" else "a"
    return choice


async def _run_persona(
    persona: str,
    pairs_data: list[tuple[Task, Task, str]],
    client,
    template,
) -> list[dict]:
    system_prompt = _load_system_prompt(persona)
    builder = build_revealed_builder(template, "completion", system_prompt=system_prompt)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Look up pair_id by the original task_a.id (unique across sampled pairs)
    task_a_id_to_pair_id = {a.id: pid for a, _, pid in pairs_data}

    rows: list[dict] = []
    for ordering in (0, 1):
        if ordering == 0:
            task_pairs = [(a, b) for a, b, _ in pairs_data]
        else:
            task_pairs = [(b, a) for a, b, _ in pairs_data]

        for trial in range(N_TRIALS):
            print(f"  [{persona}] ordering={ordering} trial={trial}...")
            batch = await measure_pre_task_revealed_async(
                client=client,
                pairs=task_pairs,
                builder=builder,
                semaphore=semaphore,
                temperature=TEMPERATURE,
                seed=SEED_BASE + trial,
            )
            for m in batch.successes:
                # m.task_a / m.task_b reflect the presentation order;
                # look up the original pair_id by whichever of the two is our orig task_a
                if m.task_a.id in task_a_id_to_pair_id:
                    pair_id = task_a_id_to_pair_id[m.task_a.id]
                else:
                    pair_id = task_a_id_to_pair_id[m.task_b.id]
                rows.append({
                    "pair_id": pair_id,
                    "condition": "baseline",
                    "layer": 25,
                    "signed_multiplier": 0.0,
                    "ordering": ordering,
                    "choice_original": _remap_choice(m.choice, ordering),
                })
            if batch.failures:
                print(f"    {len(batch.failures)} failures (dropped)")
    return rows


async def main() -> None:
    client = get_client(MODEL, max_new_tokens=MAX_NEW_TOKENS)
    template = load_templates_from_yaml(TEMPLATE_PATH)[0]
    pairs_data = _load_pairs()
    print(f"Loaded {len(pairs_data)} pairs")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for persona in PERSONAS:
        out_path = OUTPUT_DIR / f"{persona}_baseline.parsed.jsonl"
        if out_path.exists():
            print(f"[{persona}] already exists at {out_path}, skipping")
            continue
        total = len(pairs_data) * 2 * N_TRIALS
        print(f"\n[{persona}] running {total} generations")
        rows = await _run_persona(persona, pairs_data, client, template)
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    asyncio.run(main())
