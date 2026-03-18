"""Boundary pair pilot: find where steering kicks in.

50 boundary pairs, both differential and activation_patch,
L25 and L32, wider multiplier range including larger values.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

from src.models.huggingface_model import HuggingFaceModel
from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
from src.measurement.runners.runners import build_revealed_builder
from src.probes.core.storage import load_probe_direction
from src.steering.client import SteeredHFClient
from src.task_data import Task, OriginDataset

MANIFEST_DIR = Path("results/probes/heldout_eval_gemma3_task_mean")
PAIRS_PATH = Path("experiments/revealed_steering_v2/followup/pairs_500.json")
CHECKPOINT_PATH = Path("experiments/steering/isolated_steering/checkpoint_boundary_pilot.jsonl")
TEMPLATE_PATH = "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml"

LAYERS = [25, 32]
MULTIPLIERS = [-0.20, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.20]
CONDITIONS = ["activation_patch"]
N_PAIRS = 50
N_TRIALS = 3
SEED = 42
NORM_LAYER = 32


def load_boundary_pairs() -> list[dict]:
    with open(PAIRS_PATH) as f:
        all_pairs = json.load(f)
    boundary = [
        p for p in all_pairs
        if "p_a_baseline_20" in p and 0.4 <= p["p_a_baseline_20"] <= 0.6
    ]
    boundary.sort(key=lambda p: abs(p["delta_mu"]))
    random.seed(SEED)
    return random.sample(boundary, min(N_PAIRS, len(boundary)))


def load_checkpoint() -> dict[tuple, int]:
    counts: dict[tuple, int] = defaultdict(int)
    if not CHECKPOINT_PATH.exists():
        return counts
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            row = json.loads(line)
            key = (row["pair_id"], row["layer"], row["multiplier"], row["condition"], row["ordering"])
            counts[key] += 1
    return counts


def append_checkpoint(rows: list[dict]) -> None:
    with open(CHECKPOINT_PATH, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def pair_to_tasks(pair: dict) -> tuple[Task, Task]:
    task_a = Task(prompt=pair["task_a_text"], origin=OriginDataset.ALPACA, id=pair["task_a"], metadata={})
    task_b = Task(prompt=pair["task_b_text"], origin=OriginDataset.ALPACA, id=pair["task_b"], metadata={})
    return task_a, task_b


def parse_choice(response_format, response: str) -> str:
    try:
        return asyncio.run(response_format.parse(response))
    except Exception:
        return "refusal"


def main():
    pairs = load_boundary_pairs()
    checkpoint_counts = load_checkpoint()

    template = load_templates_from_yaml(TEMPLATE_PATH)[0]
    builder = build_revealed_builder(template, "completion")
    response_format = builder.response_format

    existing = sum(checkpoint_counts.values())
    print(f"Boundary pairs: {len(pairs)}")
    print(f"Checkpoint: {existing} existing rows")

    # Print pair stats
    dmus = [abs(p["delta_mu"]) for p in pairs]
    p_as = [p["p_a_baseline_20"] for p in pairs]
    print(f"  |delta_mu| range: [{min(dmus):.3f}, {max(dmus):.3f}], median={np.median(dmus):.3f}")
    print(f"  p_a_baseline_20 range: [{min(p_as):.3f}, {max(p_as):.3f}]")

    total = len(pairs) * len(LAYERS) * len(MULTIPLIERS) * len(CONDITIONS) * 2 * N_TRIALS
    print(f"Target: {total} generations")

    print("Loading model...")
    t0 = time.time()
    hf_model = HuggingFaceModel("gemma-3-27b", max_new_tokens=256)
    print(f"Model loaded in {time.time() - t0:.0f}s")

    probes: dict[int, np.ndarray] = {}
    for layer in LAYERS:
        _, direction = load_probe_direction(MANIFEST_DIR, f"ridge_L{layer}")
        probes[layer] = direction

    # Mean norm
    norms = []
    for pair in pairs[:20]:
        task_a, task_b = pair_to_tasks(pair)
        prompt_data = builder.build(task_a, task_b)
        results = hf_model.get_activations(prompt_data.messages, [NORM_LAYER], ["task_mean"])
        norms.append(float(np.linalg.norm(results["task_mean"][NORM_LAYER])))
    mean_norm = float(np.mean(norms))
    print(f"Mean norm L{NORM_LAYER}: {mean_norm:.0f}")

    t_start = time.time()
    generated = 0
    skipped = 0

    for condition in CONDITIONS:
        for layer in LAYERS:
            direction = probes[layer]
            print(f"\n  {condition} / L{layer}")

            base_client = SteeredHFClient(
                hf_model, layer=layer, steering_direction=direction,
                coefficient=0, steering_mode="cache", cache_injection="hook",
            )

            for pair_idx, pair in enumerate(pairs):
                task_a, task_b = pair_to_tasks(pair)
                pair_id = pair["pair_id"]
                delta_mu = pair["delta_mu"]

                for ordering in [0, 1]:
                    pres_a = task_a if ordering == 0 else task_b
                    pres_b = task_b if ordering == 0 else task_a

                    prompt_data = builder.build(pres_a, pres_b)
                    messages = prompt_data.messages
                    response_format.task_a_prompt = pres_a.prompt
                    response_format.task_b_prompt = pres_b.prompt

                    for mult in MULTIPLIERS:
                        key = (pair_id, layer, mult, condition, ordering)
                        if checkpoint_counts[key] >= N_TRIALS:
                            skipped += N_TRIALS
                            continue

                        coef = mean_norm * mult
                        needed = N_TRIALS - checkpoint_counts[key]
                        client = base_client.with_coefficient(coef)

                        try:
                            responses = client.generate_n(
                                messages, n=needed, temperature=1.0,
                                task_prompts=[pres_a.prompt, pres_b.prompt],
                            )
                            steering_fallback = False
                        except ValueError:
                            fallback = SteeredHFClient(
                                hf_model, layer=layer, steering_direction=direction,
                                coefficient=coef, steering_mode="all_tokens",
                            )
                            responses = fallback.generate_n(messages, n=needed, temperature=1.0)
                            steering_fallback = True

                        rows = []
                        for sample_idx, response in enumerate(responses):
                            choice_presented = parse_choice(response_format, response)
                            if choice_presented in ("a", "b"):
                                choice_original = (
                                    choice_presented if ordering == 0
                                    else ("b" if choice_presented == "a" else "a")
                                )
                            else:
                                choice_original = choice_presented

                            rows.append({
                                "pair_id": pair_id,
                                "task_a_id": pair["task_a"],
                                "task_b_id": pair["task_b"],
                                "coefficient": coef,
                                "multiplier": mult,
                                "layer": layer,
                                "condition": condition,
                                "sample_idx": checkpoint_counts[key] + sample_idx,
                                "ordering": ordering,
                                "choice_original": choice_original,
                                "choice_presented": choice_presented,
                                "raw_response": response,
                                "delta_mu": delta_mu,
                                "steering_fallback": steering_fallback,
                            })

                        append_checkpoint(rows)
                        generated += len(rows)

                if (pair_idx + 1) % 5 == 0:
                    elapsed = time.time() - t_start
                    rate = generated / elapsed if elapsed > 0 else 0
                    print(f"    Pair {pair_idx+1}/{len(pairs)}: {generated} gen, {rate:.1f}/s, {elapsed/60:.1f}m")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/3600:.1f}h. Generated: {generated}, Skipped: {skipped}")

    # Quick inline analysis
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    rows = []
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            rows.append(json.loads(line))

    for condition in CONDITIONS:
        for layer in LAYERS:
            subset = [r for r in rows if r["condition"] == condition and r["layer"] == layer
                       and r["choice_original"] in ("a", "b") and not r["steering_fallback"]]
            if not subset:
                continue
            print(f"\n{condition} / L{layer}:")
            by_mult = defaultdict(list)
            for r in subset:
                by_mult[r["multiplier"]].append(r)
            for mult in sorted(by_mult):
                valid = by_mult[mult]
                p_a = sum(1 for r in valid if r["choice_original"] == "a") / len(valid)
                sign = "+" if (p_a - 0.5) >= 0 else ""
                print(f"  m={mult:+.3f}: P(a)={p_a:.3f} (n={len(valid)}) effect={sign}{p_a-0.5:.3f}")


if __name__ == "__main__":
    main()
