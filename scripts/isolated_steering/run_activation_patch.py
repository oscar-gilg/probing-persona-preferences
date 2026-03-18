"""Run ONLY the activation patching condition.

Separate script so it can run on a second pod in parallel with KV cache.

Usage:
    python -m scripts.isolated_steering.run_activation_patch
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
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
CHECKPOINT_PATH = Path("experiments/steering/isolated_steering/checkpoint_actpatch.jsonl")
TEMPLATE_PATH = "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml"

PROBE_LAYERS = [25, 32, 39, 46, 53]
ACT_PATCH_MULTIPLIERS = [-0.05, -0.03, -0.02, 0.02, 0.03, 0.05]
N_PAIRS = 200
N_TRIALS = 3
SEED = 42
KV_PROBE_LAYER = 32  # for mean norm computation (same baseline as KV condition)


def load_pairs() -> list[dict]:
    with open(PAIRS_PATH) as f:
        all_pairs = json.load(f)
    random.seed(SEED)
    return random.sample(all_pairs, N_PAIRS)


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
    result = response_format.parse_sync(response)
    return "refusal" if result == "parse_fail" else result


def main():
    pairs = load_pairs()
    checkpoint_counts = load_checkpoint()

    template = load_templates_from_yaml(TEMPLATE_PATH)[0]
    builder = build_revealed_builder(template, "completion")
    response_format = builder.response_format

    existing = sum(checkpoint_counts.values())
    print(f"Checkpoint: {existing} existing rows")

    print("Loading model...")
    t0 = time.time()
    hf_model = HuggingFaceModel("gemma-3-27b", max_new_tokens=256)
    print(f"Model loaded in {time.time() - t0:.0f}s")

    # Load probes
    probes: dict[int, np.ndarray] = {}
    for layer in PROBE_LAYERS:
        _, direction = load_probe_direction(MANIFEST_DIR, f"ridge_L{layer}")
        probes[layer] = direction

    # Compute mean norm at L32
    norms = []
    for pair in pairs[:30]:
        task_a, task_b = pair_to_tasks(pair)
        prompt_data = builder.build(task_a, task_b)
        results = hf_model.get_activations(prompt_data.messages, [KV_PROBE_LAYER], ["task_mean"])
        norms.append(float(np.linalg.norm(results["task_mean"][KV_PROBE_LAYER])))
    mean_norm = float(np.mean(norms))
    print(f"Mean norm L{KV_PROBE_LAYER}: {mean_norm:.0f}")

    t_start = time.time()
    generated = 0
    skipped = 0

    print(f"\n{'='*60}")
    print("Condition: activation_patch")
    print(f"{'='*60}")

    for layer in PROBE_LAYERS:
        direction = probes[layer]
        print(f"\n  Layer {layer}")

        base_client = SteeredHFClient(
            hf_model, layer=layer, steering_direction=direction,
            coefficient=0, steering_mode="activation_patch",
        )

        for pair_idx, pair in enumerate(pairs):
            task_a, task_b = pair_to_tasks(pair)
            pair_id = pair["pair_id"]
            delta_mu = pair["delta_mu"]

            for ordering in [0, 1]:
                if ordering == 0:
                    pres_a, pres_b = task_a, task_b
                else:
                    pres_a, pres_b = task_b, task_a

                prompt_data = builder.build(pres_a, pres_b)
                messages = prompt_data.messages
                response_format.task_a_prompt = pres_a.prompt
                response_format.task_b_prompt = pres_b.prompt

                for mult in ACT_PATCH_MULTIPLIERS:
                    key = (pair_id, layer, mult, "activation_patch", ordering)
                    if checkpoint_counts[key] >= N_TRIALS:
                        skipped += N_TRIALS
                        continue

                    coef = mean_norm * mult
                    effective_coef = coef if ordering == 0 else -coef
                    needed = N_TRIALS - checkpoint_counts[key]
                    client = base_client.with_coefficient(effective_coef)

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
                            choice_original = choice_presented if ordering == 0 else ("b" if choice_presented == "a" else "a")
                        else:
                            choice_original = choice_presented

                        rows.append({
                            "pair_id": pair_id,
                            "task_a_id": pair["task_a"],
                            "task_b_id": pair["task_b"],
                            "coefficient": coef,
                            "multiplier": mult,
                            "layer": layer,
                            "condition": "activation_patch",
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

            if (pair_idx + 1) % 10 == 0:
                elapsed = time.time() - t_start
                rate = generated / elapsed if elapsed > 0 else 0
                print(f"    Pair {pair_idx + 1}/{len(pairs)}: {generated} gen, {rate:.1f} gen/s, {elapsed/60:.1f}m")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/3600:.1f}h. Generated: {generated}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
