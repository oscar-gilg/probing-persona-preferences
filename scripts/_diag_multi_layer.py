"""Diagnostic: contrastive steering at ALL 6 probe layers simultaneously.

Each layer gets its own probe direction (ridge_L<L>) applied with the standard
contrastive setup: +c × dir × layer_norm on task-A span, -c × dir × layer_norm
on task-B span. All 6 hooks installed at once during a single prefill.

If the per-layer effects compound, we should see swing ≫ best-single-layer (0.06).
If they don't, the bottleneck is something other than the choice of single layer.

Output: experiments/qwen_replication/steering_layer_sweep/checkpoints/diag_multi_layer.parsed.jsonl

Run on pod:
    python -m scripts._diag_multi_layer --multipliers -0.05 0 0.05 --n-pairs 10
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

from src.measurement.elicitation.completion_judge import judge_completion_full_async
from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
from src.measurement.runners.runners import build_revealed_builder
from src.models.huggingface_model import HuggingFaceModel
from src.probes.core.storage import load_probe_direction
from src.steering.hooks import compose_hooks, position_selective_steering
from src.steering.tokenization import find_pairwise_task_spans
from src.task_data import OriginDataset, Task

PAIRS_PATH = Path("experiments/qwen_replication/steering_layer_sweep/steering_pairs_50.json")
PROBE_MANIFEST = Path("results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1/")
TEMPLATE_PATH = Path("src/measurement/elicitation/prompt_templates/data/completion_preference.yaml")
OUT_PATH = Path("experiments/qwen_replication/steering_layer_sweep/checkpoints/diag_multi_layer.jsonl")
PARSED_OUT = OUT_PATH.with_suffix(".parsed.jsonl")
LAYERS = [12, 24, 28, 33, 38, 43]
MEAN_NORMS = {12: 1.959769368171692, 24: 3.3148388862609863, 28: 3.5511796474456787,
              33: 4.264685153961182, 38: 5.810032367706299, 43: 7.597376346588135}
SPANS = {"first": 1, "second": -1}  # contrastive


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--multipliers", type=float, nargs="+", default=[-0.05, 0.0, 0.05])
    p.add_argument("--n-pairs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()

    pairs_all = json.loads(PAIRS_PATH.read_text())
    rng = np.random.default_rng(args.seed)
    selected_idx = rng.choice(len(pairs_all), size=args.n_pairs, replace=False)
    pairs = [pairs_all[i] for i in sorted(selected_idx)]
    print(f"[multi-layer diag] {len(pairs)} pairs, multipliers={args.multipliers}")
    print(f"[multi-layer diag] simultaneous hooks at L{LAYERS}, contrastive {SPANS}")

    directions: dict[int, np.ndarray] = {}
    for L in LAYERS:
        _, d = load_probe_direction(PROBE_MANIFEST, f"ridge_L{L}")
        directions[L] = d
        print(f"  loaded ridge_L{L}: norm={np.linalg.norm(d):.4f} (unit-normalised)")

    print("[multi-layer diag] loading model")
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

            # Token spans for first/second task content.
            formatted = hf.tokenizer.apply_chat_template(
                prompt_data.messages, tokenize=False, add_generation_prompt=True,
            )
            (a_start, a_end), (b_start, b_end) = find_pairwise_task_spans(
                hf.tokenizer, formatted, first_task.prompt, second_task.prompt, "Task A", "Task B",
            )
            span_map = {"first": (a_start, a_end), "second": (b_start, b_end)}

            # Skip orientation flip (we don't need it — applying both signs symmetrically here).
            # _effective_coef logic: ordering==0 gives coef as-is; ordering==1 negates.
            ord_sign = 1 if ordering == 0 else -1

            for mult in args.multipliers:
                # Build ALL 6 layer hooks for this (pair, ordering, mult).
                layer_hooks: list[tuple[int, callable]] = []
                for L in LAYERS:
                    norm = MEAN_NORMS[L]
                    span_hooks = []
                    for span_name, span_coef in SPANS.items():
                        span = span_map[span_name]
                        effective = ord_sign * norm * mult * span_coef
                        st = torch.tensor(directions[L] * effective, dtype=torch.bfloat16, device="cuda")
                        span_hooks.append(position_selective_steering(st, span[0], span[1]))
                    layer_hooks.append((L, compose_hooks(*span_hooks)))

                # Generate with all 6 hooks installed simultaneously
                response = hf._generate_hooked(
                    prompt_data.messages, layer_hooks=layer_hooks,
                    temperature=1.0, max_new_tokens=args.max_new_tokens,
                    num_return_sequences=1,
                )[0]

                rows.append({
                    "pair_id": pair["pair_id"],
                    "ordering": ordering,
                    "signed_multiplier": mult,
                    "layers": LAYERS,
                    "raw_response": response,
                })
                with open(OUT_PATH, "a") as f:
                    f.write(json.dumps(rows[-1]) + "\n")
        elapsed = time.time() - t0
        print(f"[multi-layer diag] pair {pair['pair_id']} done ({elapsed/60:.1f}min)")

    print("[multi-layer diag] all generations done. Parsing...")

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
    print(f"[multi-layer diag] wrote {PARSED_OUT}")


if __name__ == "__main__":
    main()
