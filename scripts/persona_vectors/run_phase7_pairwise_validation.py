"""Phase 7: pairwise preference validation under steering.

Uses the 150-pair set from `experiments/layer_sweep/harm_breakdown/steering_pairs_150.json`
(50 each of bb / hb / hh). For each (layer, coef) cell:
  - Both orderings × 1 trial = 300 measurements per cell.
  - Apply persona vector × coef as `all_tokens_steering` via SteeredHFClient.
  - Use canonical pre_task_revealed measurement pipeline + LLM choice judge.

Saves per-cell per-pair measurements to
`experiments/qwen_persona_vectors/validation_pairwise/<persona>__L{N}__c{c:+.4f}.jsonl`,
one row per (pair_id, ordering) with the parsed choice.

Include c=0 in --coef-multipliers as the baseline — OpenRouter's /no_think
doesn't actually suppress Qwen3.5-122B's reasoning (causes ~40% truncation),
so the API path is unreliable. Local HF with enable_thinking=False is the
clean baseline.

Run (on pod with Qwen loaded):
    python -m scripts.persona_vectors.run_phase7_pairwise_validation \
        --persona sadist --layers 25 --coef-multipliers -0.07 -0.05 -0.03 0.0 0.03 0.05 0.07
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

VECTORS_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/vectors"
PAIRS_PATH = PROJECT_ROOT / "experiments/layer_sweep/harm_breakdown/steering_pairs_150.json"
OUT_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/validation_pairwise"
TEMPLATE_PATH = PROJECT_ROOT / "src/measurement/elicitation/prompt_templates/data/pre_task_revealed_v1.yaml"


def load_pairs() -> list[dict]:
    return json.loads(PAIRS_PATH.read_text())


def cell_path(persona: str, layer: int, coef_mult: float, tag: str | None) -> Path:
    suffix = f"__{tag}" if tag else ""
    return OUT_DIR / f"{persona}__L{layer}__c{coef_mult:+.4f}{suffix}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[25])
    parser.add_argument("--coef-multipliers", type=float, nargs="+", default=[-0.07, -0.05, -0.03, 0.0, 0.03, 0.05, 0.07])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--system-prompt-key", default=None,
                        help="key into experiments/persona_sweep/sweep_personas.json[personas][*][name]; "
                             "if set, uses that persona's system_prompt during measurement (additive to steering)")
    parser.add_argument("--out-tag", default=None,
                        help="filename suffix to disambiguate from no-system-prompt cells (e.g. 'sysdamien')")
    args = parser.parse_args()

    load_dotenv()

    from src.measurement.elicitation.measure import measure_pre_task_revealed_async
    from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
    from src.measurement.runners.runners import build_revealed_builder
    from src.models.huggingface_model import HuggingFaceModel
    from src.persona_vectors.vector import load_persona_vector
    from src.steering.client import SteeredHFClient
    from src.task_data import OriginDataset, Task

    pairs_meta = load_pairs()
    print(f"[phase7] {len(pairs_meta)} pairs from {PAIRS_PATH.name}")

    vectors, mean_norms, _meta = load_persona_vector(VECTORS_DIR / f"{args.persona}.npz")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[phase7] Loading Qwen3.5-122B-nothink…")
    hf = HuggingFaceModel(
        "qwen3.5-122b-nothink",
        device="auto",
        max_new_tokens=args.max_new_tokens,
        max_memory={i: "65GiB" for i in range(4)},
    )
    print(f"[phase7] Loaded {type(hf.model).__name__}")

    template = load_templates_from_yaml(TEMPLATE_PATH)[0]

    sys_prompt = None
    if args.system_prompt_key:
        personas_data = json.loads((PROJECT_ROOT / "experiments/persona_sweep/sweep_personas.json").read_text())
        match = [p for p in personas_data["personas"] if p["name"] == args.system_prompt_key]
        if not match:
            raise ValueError(f"persona key {args.system_prompt_key!r} not in sweep_personas.json")
        sys_prompt = match[0]["system_prompt"]
        print(f"[phase7] using system_prompt from persona={args.system_prompt_key} ({len(sys_prompt)} chars)")

    builder = build_revealed_builder(template, "completion", system_prompt=sys_prompt)

    # Build both orderings for each of the 150 pairs
    pairs_AB: list[tuple[Task, Task, str, str]] = []  # (task_a, task_b, pair_id, pair_type, ordering)
    pairs_BA: list[tuple[Task, Task, str, str]] = []
    for entry in pairs_meta:
        a = Task(prompt=entry["task_a_text"], origin=OriginDataset[entry["task_a_origin"]], id=entry["task_a"], metadata={})
        b = Task(prompt=entry["task_b_text"], origin=OriginDataset[entry["task_b_origin"]], id=entry["task_b"], metadata={})
        pid = entry["pair_id"]
        ptype = entry["pair_type"]
        pairs_AB.append((a, b, pid, ptype))
        pairs_BA.append((b, a, pid, ptype))

    cells = [(layer, mult) for layer in args.layers for mult in args.coef_multipliers]
    print(f"[phase7] {len(cells)} (layer, coef) cells × 300 measurements per cell = {len(cells) * 300} generations")

    t0 = time.time()
    for i_cell, (layer, mult) in enumerate(cells):
        out_path = cell_path(args.persona, layer, mult, args.out_tag)
        if out_path.exists() and len(out_path.read_text().splitlines()) >= 300:
            print(f"[{i_cell + 1}/{len(cells)}] L{layer} c={mult:+.3f}: cache hit, skipping")
            continue

        coef_raw = mult * mean_norms[layer]
        steering_dir = vectors[layer]
        client = SteeredHFClient(
            hf_model=hf,
            layer=layer,
            steering_direction=steering_dir,
            coefficient=coef_raw,
            steering_mode="all_tokens",
        )
        sem = asyncio.Semaphore(args.concurrency)

        # Run AB ordering then BA ordering
        rows: list[dict] = []
        for ordering, pair_list in (("AB", pairs_AB), ("BA", pairs_BA)):
            tuples = [(p[0], p[1]) for p in pair_list]
            batch = asyncio.run(measure_pre_task_revealed_async(
                client=client,
                pairs=tuples,
                builder=builder,
                semaphore=sem,
                temperature=args.temperature,
            ))
            # MeasurementBatch has .successes (list of measurements) + .failures.
            # Index back to pair via task ids since failures may shorten the list.
            success_by_pair = {(m.task_a.id, m.task_b.id): m for m in batch.successes}
            for a, b, pid, ptype in pair_list:
                m = success_by_pair.get((a.id, b.id))
                rows.append({
                    "pair_id": pid,
                    "pair_type": ptype,
                    "ordering": ordering,
                    "task_a_id": a.id,
                    "task_b_id": b.id,
                    "choice": m.choice if m is not None else None,
                    "raw_response": m.raw_response if m is not None else None,
                    "failed": m is None,
                })

        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        elapsed = time.time() - t0
        n_a = sum(1 for r in rows if r["choice"] == "a")
        n_b = sum(1 for r in rows if r["choice"] == "b")
        n_ref = sum(1 for r in rows if r["choice"] == "refusal")
        per_type = {}
        for ptype in ("bb", "hb", "hh"):
            sub = [r for r in rows if r["pair_type"] == ptype]
            sub_a = sum(1 for r in sub if r["choice"] == "a")
            per_type[ptype] = sub_a / max(len(sub), 1)
        print(f"[{i_cell + 1}/{len(cells)} | {elapsed/60:.1f}min] L{layer} c={mult:+.3f}: "
              f"A={n_a}/300, B={n_b}/300, refusals={n_ref}, "
              f"P(A)|bb={per_type['bb']:.2f} hb={per_type['hb']:.2f} hh={per_type['hh']:.2f}")

    print(f"\n[phase7] Done. Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
