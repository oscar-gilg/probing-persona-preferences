"""Isolated steering experiment: KV cache V-only and activation patching.

Uses existing measurement infrastructure for prompt building and response parsing.
Supports --resume to skip completed work.
"""

from __future__ import annotations

import argparse
import asyncio
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
from src.measurement.elicitation.response_format import CompletionChoiceFormat
from src.measurement.runners.runners import build_revealed_builder
from src.probes.core.storage import load_probe_direction
from src.steering.client import SteeredHFClient
from src.steering.kv_cache import project_to_v_space
from src.task_data import Task, OriginDataset

# ── Constants ────────────────────────────────────────────────────────────────

MANIFEST_DIR = Path("results/probes/heldout_eval_gemma3_task_mean")
PAIRS_PATH = Path("experiments/revealed_steering_v2/followup/pairs_500.json")
CHECKPOINT_PATH = Path("experiments/steering/isolated_steering/checkpoint.jsonl")

LAYERS = [25, 32, 39, 46, 53]
SIGNED_MULTIPLIERS = [-0.05, -0.03, -0.02, 0.02, 0.03, 0.05]
CONDITIONS = ["kv_cache_v_single", "activation_patch"]
CONDITION_TO_MODE = {
    "kv_cache_v_single": "kv_cache_differential",
    "activation_patch": "activation_patch",
}
N_PAIRS = 200
N_TRIALS = 3  # per ordering
TEMPERATURE = 1.0
SEED = 42


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


def compute_mean_norms(hf_model: HuggingFaceModel, pairs: list[dict], layers: list[int], n_samples: int = 30) -> dict[int, float]:
    """Compute mean activation norms by running the model on sample prompts."""
    template = load_templates_from_yaml("src/measurement/elicitation/prompt_templates/data/completion_preference.yaml")[0]
    builder = build_revealed_builder(template, "completion")

    random.seed(SEED + 1)
    sample_pairs = random.sample(pairs, min(n_samples, len(pairs)))

    norms: dict[int, list[float]] = {layer: [] for layer in layers}

    print(f"Computing mean activation norms from {len(sample_pairs)} samples...")
    for i, pair in enumerate(sample_pairs):
        task_a, task_b = pair_to_tasks(pair)
        prompt_data = builder.build(task_a, task_b)
        results = hf_model.get_activations(prompt_data.messages, layers, ["task_mean"])
        for layer in layers:
            act = results["task_mean"][layer]
            norms[layer].append(float(np.linalg.norm(act)))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(sample_pairs)}")

    mean_norms = {layer: float(np.mean(norms[layer])) for layer in layers}
    for layer in layers:
        print(f"  L{layer}: mean_norm = {mean_norms[layer]:.0f}")
    return mean_norms


def run_experiment(pilot: bool = False):
    pairs = load_pairs()
    checkpoint_counts = load_checkpoint()

    # Use the existing template and response format
    template = load_templates_from_yaml("src/measurement/elicitation/prompt_templates/data/completion_preference.yaml")[0]
    builder = build_revealed_builder(template, "completion")
    # Get the response format for parsing (CompletionChoiceFormat)
    response_format: CompletionChoiceFormat = builder.response_format

    if pilot:
        pairs = pairs[:3]
        layers = [25]
        signed_multipliers = [-0.02, 0.02]
    else:
        layers = LAYERS
        signed_multipliers = SIGNED_MULTIPLIERS

    existing = sum(checkpoint_counts.values())
    total = len(pairs) * len(layers) * len(signed_multipliers) * len(CONDITIONS) * 2 * N_TRIALS
    print(f"Checkpoint: {existing}/{total} rows")

    print("Loading model...")
    t0 = time.time()
    hf_model = HuggingFaceModel("gemma-3-27b", max_new_tokens=256)
    print(f"Model loaded in {time.time() - t0:.0f}s")

    mean_norms = compute_mean_norms(hf_model, pairs, layers)

    # Load probes
    probes: dict[int, np.ndarray] = {}
    for layer in layers:
        _, direction = load_probe_direction(MANIFEST_DIR, f"ridge_L{layer}")
        probes[layer] = direction

    # Report V-space calibration
    print("\nV-space calibration:")
    for layer in layers:
        v_dir = project_to_v_space(hf_model.model, layer, probes[layer])
        import torch
        ratio = float(torch.norm(v_dir).item()) / float(np.linalg.norm(probes[layer]))
        print(f"  L{layer}: ratio = {ratio:.4f}")

    # Main loop: layers → conditions → pairs → orderings → multipliers (inner)
    t_start = time.time()
    generated = 0
    skipped = 0
    fallbacks = 0

    for layer in layers:
        direction = probes[layer]
        mean_norm = mean_norms[layer]
        print(f"\n{'='*60}\nLayer {layer} (mean_norm={mean_norm:.0f})\n{'='*60}")

        for condition in CONDITIONS:
            steering_mode = CONDITION_TO_MODE[condition]
            print(f"\n  Condition: {condition}")

            base_client = SteeredHFClient(
                hf_model, layer=layer, steering_direction=direction,
                coefficient=0, steering_mode=steering_mode,
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

                    # Set task prompts on the response format for semantic parsing
                    response_format.task_a_prompt = pres_a.prompt
                    response_format.task_b_prompt = pres_b.prompt

                    for mult in signed_multipliers:
                        key = (pair_id, layer, mult, condition, ordering)
                        existing_count = checkpoint_counts[key]
                        needed = N_TRIALS - existing_count
                        if needed <= 0:
                            skipped += N_TRIALS
                            continue

                        coef = mean_norm * mult
                        client = base_client.with_coefficient(coef)

                        try:
                            responses = client.generate_n(
                                messages, n=needed, temperature=TEMPERATURE,
                                task_prompts=[pres_a.prompt, pres_b.prompt],
                            )
                            steering_fallback = False
                        except ValueError:
                            fallback_client = SteeredHFClient(
                                hf_model, layer=layer, steering_direction=direction,
                                coefficient=coef, steering_mode="all_tokens",
                            )
                            responses = fallback_client.generate_n(
                                messages, n=needed, temperature=TEMPERATURE,
                            )
                            steering_fallback = True
                            fallbacks += 1

                        # Parse using the existing CompletionChoiceFormat
                        rows = []
                        for sample_idx, response in enumerate(responses):
                            # Use async parse (prefix match → semantic parser fallback)
                            choice_presented = asyncio.run(response_format.parse(response))

                            if choice_presented in ("a", "b"):
                                if ordering == 0:
                                    choice_original = choice_presented
                                else:
                                    choice_original = "b" if choice_presented == "a" else "a"
                            else:
                                choice_original = choice_presented  # refusal

                            rows.append({
                                "pair_id": pair_id,
                                "task_a_id": pair["task_a"],
                                "task_b_id": pair["task_b"],
                                "coefficient": coef,
                                "multiplier": mult,
                                "layer": layer,
                                "condition": condition,
                                "sample_idx": existing_count + sample_idx,
                                "ordering": ordering,
                                "choice_original": choice_original,
                                "choice_presented": choice_presented,
                                "raw_response": response[:500],
                                "delta_mu": delta_mu,
                                "steering_fallback": steering_fallback,
                            })

                        append_checkpoint(rows)
                        generated += len(rows)

                if (pair_idx + 1) % 10 == 0:
                    elapsed = time.time() - t_start
                    rate = generated / elapsed if elapsed > 0 else 0
                    print(f"    Pair {pair_idx + 1}/{len(pairs)}: "
                          f"{generated} gen, {skipped} skip, "
                          f"{fallbacks} fallback, "
                          f"{rate:.1f} gen/s, "
                          f"{elapsed/60:.1f}m")

        print(f"\nLayer {layer} done. Total: {generated} generated")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/3600:.1f}h. Generated: {generated}, Skipped: {skipped}, Fallbacks: {fallbacks}")

    # Save metadata
    meta = {"mean_norms": {str(k): v for k, v in mean_norms.items()}}
    with open(CHECKPOINT_PATH.parent / "experiment_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_experiment(pilot=args.pilot)
