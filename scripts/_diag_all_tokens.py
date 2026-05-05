"""Diagnostic: apply preference probe direction to ALL prompt tokens (not just task spans).

If contrastive position-selective steering produces ~0.05 swing on Qwen but
all-tokens steering produces a meaningful swing with the same probe at the
same layer, the issue is the contrastive/spans setup. If all-tokens is also
flat, the probe direction itself lacks causal weight on Qwen.

Output: experiments/qwen_replication/steering_layer_sweep/checkpoints/diag_all_tokens.parsed.jsonl
(rows include the same fields as the runner's parsed output for compatibility.)

Run on pod:
    python -m scripts._diag_all_tokens --layer 38 --multipliers -0.05 0 0.05 --n-pairs 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

from src.measurement.elicitation.completion_judge import judge_completion_full_async, extract_claimed_task
from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
from src.measurement.runners.runners import build_revealed_builder
from src.models.huggingface_model import HuggingFaceModel
from src.probes.core.storage import load_probe_direction
from src.steering.hooks import prefill_all_steering
from src.task_data import OriginDataset, Task

PAIRS_PATH = Path("experiments/qwen_replication/steering_layer_sweep/steering_pairs_50.json")
PROBE_MANIFEST = Path("results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1/")
TEMPLATE_PATH = Path("src/measurement/elicitation/prompt_templates/data/completion_preference.yaml")
OUT_PATH = Path("experiments/qwen_replication/steering_layer_sweep/checkpoints/diag_all_tokens.jsonl")
PARSED_OUT = OUT_PATH.with_suffix(".parsed.jsonl")
MEAN_NORMS = {12: 1.96, 24: 3.31, 28: 3.55, 33: 4.26, 38: 5.81, 43: 7.60}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--multipliers", type=float, nargs="+", default=[-0.05, 0.0, 0.05])
    p.add_argument("--n-pairs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()

    layer = args.layer
    layer_norm = MEAN_NORMS[layer]

    pairs = json.loads(PAIRS_PATH.read_text())
    rng = np.random.default_rng(args.seed)
    selected_idx = rng.choice(len(pairs), size=args.n_pairs, replace=False)
    pairs = [pairs[i] for i in sorted(selected_idx)]
    print(f"[diag] {len(pairs)} pairs at L{layer}, multipliers={args.multipliers}")

    _, direction = load_probe_direction(PROBE_MANIFEST, f"ridge_L{layer}")
    print(f"[diag] probe direction shape={direction.shape}, norm={np.linalg.norm(direction):.4f}")

    print("[diag] loading model")
    hf = HuggingFaceModel(
        "qwen3.5-122b-nothink",
        device="auto",
        max_new_tokens=args.max_new_tokens,
        max_memory={i: "65GiB" for i in range(4)},
    )

    template = load_templates_from_yaml(TEMPLATE_PATH)[0]
    builder = build_revealed_builder(template, "completion")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    t0 = time.time()
    for pair in pairs:
        task_a = Task(prompt=pair["task_a_text"], origin=OriginDataset[pair["task_a_origin"]],
                      id=pair["task_a"], metadata={})
        task_b = Task(prompt=pair["task_b_text"], origin=OriginDataset[pair["task_b_origin"]],
                      id=pair["task_b"], metadata={})

        for ordering in [0, 1]:
            first_task = task_a if ordering == 0 else task_b
            second_task = task_b if ordering == 0 else task_a
            prompt_data = builder.build(first_task, second_task)

            for mult in args.multipliers:
                effective = layer_norm * mult
                steering_tensor = torch.tensor(direction * effective, dtype=torch.bfloat16, device="cuda")
                hook = prefill_all_steering(steering_tensor)

                response = hf.generate_with_hook(
                    prompt_data.messages, layer, hook, temperature=1.0,
                    max_new_tokens=args.max_new_tokens,
                )
                rows.append({
                    "pair_id": pair["pair_id"],
                    "ordering": ordering,
                    "signed_multiplier": mult,
                    "layer": layer,
                    "norm_at_layer": layer_norm,
                    "raw_response": response,
                })
                with open(OUT_PATH, "a") as f:
                    f.write(json.dumps(rows[-1]) + "\n")
        elapsed = time.time() - t0
        print(f"[diag] pair {pair['pair_id']} done ({elapsed/60:.1f}min)")

    print(f"[diag] all generations done. Parsing...")

    # Parse via the canonical judge
    pairs_lookup = {p["pair_id"]: p for p in pairs}

    async def judge_all() -> None:
        sem = asyncio.Semaphore(50)

        async def _one(rec):
            pair = pairs_lookup[rec["pair_id"]]
            async with sem:
                try:
                    j = await judge_completion_full_async(
                        pair["task_a_text"], pair["task_b_text"], rec["raw_response"],
                    )
                    return {**rec, "task_completed": j.executed_task, "compliance": j.compliance}
                except Exception as e:
                    return {**rec, "error": f"{type(e).__name__}: {e}"}

        parsed = await asyncio.gather(*[_one(r) for r in rows])
        with open(PARSED_OUT, "w") as f:
            for r in parsed:
                f.write(json.dumps(r) + "\n")

    asyncio.run(judge_all())
    print(f"[diag] wrote {PARSED_OUT}")


if __name__ == "__main__":
    main()
