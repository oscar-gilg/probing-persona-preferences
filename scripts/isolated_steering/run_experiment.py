"""Isolated steering experiment: KV cache V-only and activation patching.

72,000 generations: 200 pairs × 5 layers × 6 multipliers × 2 conditions × 6 trials.
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
import torch
from dotenv import load_dotenv

load_dotenv()

from src.models.huggingface_model import HuggingFaceModel
from src.probes.core.storage import load_probe_direction
from src.steering.client import SteeredHFClient
from src.steering.kv_cache import project_to_v_space
from src.task_data import Task, OriginDataset
from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
from src.measurement.elicitation.prompt_templates.builders import PreTaskRevealedPromptBuilder
from src.measurement.elicitation.measurer import RevealedPreferenceMeasurer
from src.measurement.elicitation.response_format import CompletionChoiceFormat

# ── Constants ────────────────────────────────────────────────────────────────

MANIFEST_DIR = Path("results/probes/heldout_eval_gemma3_task_mean")
PAIRS_PATH = Path("experiments/revealed_steering_v2/followup/pairs_500.json")
CHECKPOINT_PATH = Path("experiments/steering/isolated_steering/checkpoint.jsonl")
TEMPLATE_PATH = Path("src/measurement/elicitation/prompt_templates/data/completion_preference.yaml")

LAYERS = [25, 32, 39, 46, 53]
MULTIPLIERS = [0.02, 0.03, 0.05]
SIGNED_MULTIPLIERS = [-0.05, -0.03, -0.02, 0.02, 0.03, 0.05]
CONDITIONS = ["kv_cache_v_single", "activation_patch"]
CONDITION_TO_MODE = {
    "kv_cache_v_single": "kv_cache_differential",
    "activation_patch": "activation_patch",
}
N_PAIRS = 200
N_TRIALS = 3
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 256
SEED = 42


# ── Data loading ─────────────────────────────────────────────────────────────

def load_pairs() -> list[dict]:
    with open(PAIRS_PATH) as f:
        all_pairs = json.load(f)
    random.seed(SEED)
    return random.sample(all_pairs, N_PAIRS)


def load_checkpoint() -> dict[tuple, int]:
    """Load checkpoint, return counts of (pair_id, layer, multiplier, condition, ordering)."""
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


def load_template():
    templates = load_templates_from_yaml(TEMPLATE_PATH)
    for t in templates:
        if t.name == "completion_preference":
            return t
    raise ValueError("completion_preference template not found")


def build_prompt_builder(template) -> PreTaskRevealedPromptBuilder:
    return PreTaskRevealedPromptBuilder(
        measurer=RevealedPreferenceMeasurer(),
        response_format=CompletionChoiceFormat(),
        template=template,
    )


def pair_to_tasks(pair: dict) -> tuple[Task, Task]:
    task_a = Task(prompt=pair["task_a_text"], origin=OriginDataset.ALPACA, id=pair["task_a"], metadata={})
    task_b = Task(prompt=pair["task_b_text"], origin=OriginDataset.ALPACA, id=pair["task_b"], metadata={})
    return task_a, task_b


# ── Choice parsing ───────────────────────────────────────────────────────────

def parse_choice_sync(response: str, task_a_prompt: str, task_b_prompt: str) -> str:
    """Fast prefix-based parsing. Returns 'a', 'b', or 'parse_fail'."""
    import re
    stripped = re.sub(r"^[\s*#_`>]+", "", response).lower()
    if stripped.startswith("task a"):
        return "a"
    if stripped.startswith("task b"):
        return "b"
    return "parse_fail"


async def parse_choice_async(response: str, task_a_prompt: str, task_b_prompt: str) -> str:
    """Prefix match first, then LLM semantic parser fallback."""
    choice = parse_choice_sync(response, task_a_prompt, task_b_prompt)
    if choice != "parse_fail":
        return choice
    from src.measurement.elicitation.semantic_parser import parse_completion_choice_async
    return await parse_completion_choice_async(response, task_a_prompt, task_b_prompt)


# ── Mean norm computation ────────────────────────────────────────────────────

def compute_mean_norms_from_model(hf_model: HuggingFaceModel, pairs: list[dict], layers: list[int], n_samples: int = 30) -> dict[int, float]:
    """Compute mean activation norms by running the model on a few sample prompts."""
    template = load_template()
    builder = build_prompt_builder(template)

    random.seed(SEED + 1)
    sample_pairs = random.sample(pairs, min(n_samples, len(pairs)))

    norms_accum: dict[int, list[float]] = {layer: [] for layer in layers}

    print(f"Computing mean activation norms from {len(sample_pairs)} samples...")
    for i, pair in enumerate(sample_pairs):
        task_a, task_b = pair_to_tasks(pair)
        prompt_data = builder.build(task_a, task_b)
        messages = prompt_data.messages

        results = hf_model.get_activations(messages, layers, ["task_mean"])
        for layer in layers:
            act = results["task_mean"][layer]  # shape (d_model,)
            norms_accum[layer].append(float(np.linalg.norm(act)))

        if (i + 1) % 10 == 0:
            print(f"  Computed norms for {i + 1}/{len(sample_pairs)} samples")

    mean_norms = {layer: float(np.mean(norms_accum[layer])) for layer in layers}
    for layer in layers:
        print(f"  L{layer}: mean_norm = {mean_norms[layer]:.0f}")
    return mean_norms


# ── KV cache calibration ────────────────────────────────────────────────────

def compute_v_space_norm_ratio(hf_model: HuggingFaceModel, layer: int, direction: np.ndarray) -> float:
    """Compute ||W_v @ direction|| / ||direction||.

    Returns the ratio so we can calibrate KV cache coefficients to produce
    comparable perturbation norms in V-space.
    """
    v_dir = project_to_v_space(hf_model.model, layer, direction)
    v_norm = float(torch.norm(v_dir).item())
    d_norm = float(np.linalg.norm(direction))
    return v_norm / d_norm


# ── Main loop ────────────────────────────────────────────────────────────────

def run_experiment(pilot: bool = False):
    print("=" * 60)
    print("Isolated Steering Experiment")
    print("=" * 60)

    # Load data
    pairs = load_pairs()
    checkpoint_counts = load_checkpoint()
    template = load_template()
    builder = build_prompt_builder(template)

    if pilot:
        pairs = pairs[:3]
        layers = [25]
        signed_multipliers = [-0.02, 0.02]
        conditions = CONDITIONS
        print(f"PILOT MODE: {len(pairs)} pairs, layers={layers}, multipliers={signed_multipliers}")
    else:
        layers = LAYERS
        signed_multipliers = SIGNED_MULTIPLIERS
        conditions = CONDITIONS

    # Count existing work
    existing = sum(checkpoint_counts.values())
    total_expected = len(pairs) * len(layers) * len(signed_multipliers) * len(conditions) * 2 * N_TRIALS
    print(f"Checkpoint: {existing} rows exist, {total_expected} expected")

    # Load model
    print("Loading model...")
    t0 = time.time()
    hf_model = HuggingFaceModel("gemma-3-27b", max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded in {time.time() - t0:.0f}s")

    # Compute mean norms
    mean_norms = compute_mean_norms_from_model(hf_model, pairs, layers)

    # Load probes
    probes: dict[int, np.ndarray] = {}
    for layer in layers:
        probe_id = f"ridge_L{layer:02d}"
        loaded_layer, direction = load_probe_direction(MANIFEST_DIR, probe_id)
        assert loaded_layer == layer
        probes[layer] = direction

    # Compute V-space norm ratios for calibration reporting
    print("\nV-space calibration:")
    v_ratios: dict[int, float] = {}
    for layer in layers:
        ratio = compute_v_space_norm_ratio(hf_model, layer, probes[layer])
        v_ratios[layer] = ratio
        print(f"  L{layer}: ||W_v @ d|| / ||d|| = {ratio:.4f}")

    # Main generation loop
    # Order: layers (outer) → conditions → pairs → orderings → multipliers (inner)
    t_start = time.time()
    generated = 0
    skipped = 0
    fallbacks = 0
    parse_fails = 0

    for layer in layers:
        direction = probes[layer]
        mean_norm = mean_norms[layer]

        print(f"\n{'='*60}")
        print(f"Layer {layer} (mean_norm={mean_norm:.0f})")
        print(f"{'='*60}")

        for condition in conditions:
            steering_mode = CONDITION_TO_MODE[condition]
            print(f"\n  Condition: {condition} (mode={steering_mode})")

            # Create base client for this layer/condition
            base_client = SteeredHFClient(
                hf_model, layer=layer, steering_direction=direction,
                coefficient=0, steering_mode=steering_mode,
            )

            for pair_idx, pair in enumerate(pairs):
                task_a, task_b = pair_to_tasks(pair)
                pair_id = pair["pair_id"]
                delta_mu = pair["delta_mu"]

                for ordering in [0, 1]:
                    # ordering=0: A/B as given; ordering=1: swap
                    if ordering == 0:
                        presented_a, presented_b = task_a, task_b
                        a_text, b_text = task_a.prompt, task_b.prompt
                    else:
                        presented_a, presented_b = task_b, task_a
                        a_text, b_text = task_b.prompt, task_a.prompt

                    prompt_data = builder.build(presented_a, presented_b)
                    messages = prompt_data.messages

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
                                task_prompts=[a_text, b_text],
                            )
                            steering_fallback = False
                        except ValueError:
                            # Span detection failed — fall back to all_tokens
                            fallback_client = SteeredHFClient(
                                hf_model, layer=layer, steering_direction=direction,
                                coefficient=coef, steering_mode="all_tokens",
                            )
                            responses = fallback_client.generate_n(
                                messages, n=needed, temperature=TEMPERATURE,
                            )
                            steering_fallback = True
                            fallbacks += 1

                        rows = []
                        for sample_idx, response in enumerate(responses):
                            choice_presented = parse_choice_sync(response, a_text, b_text)
                            if choice_presented == "parse_fail":
                                parse_fails += 1
                                # Queue for async parsing later; store raw for now
                                choice_presented = "parse_fail"

                            # Map presented choice back to original ordering
                            if choice_presented in ("a", "b"):
                                if ordering == 0:
                                    choice_original = choice_presented
                                else:
                                    choice_original = "b" if choice_presented == "a" else "a"
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
                          f"{generated} generated, {skipped} skipped, "
                          f"{fallbacks} fallbacks, {parse_fails} parse_fails, "
                          f"{rate:.1f} gen/s, "
                          f"elapsed {elapsed/60:.1f}m")

        print(f"\nLayer {layer} done. Total: {generated} generated, {skipped} skipped")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Experiment complete in {elapsed/3600:.1f}h")
    print(f"Generated: {generated}, Skipped: {skipped}")
    print(f"Fallbacks: {fallbacks}, Parse fails: {parse_fails}")
    print(f"{'='*60}")

    return mean_norms, v_ratios


def resolve_parse_fails():
    """Post-process: use async LLM parser on rows with parse_fail."""
    if not CHECKPOINT_PATH.exists():
        print("No checkpoint to process.")
        return

    rows = []
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            rows.append(json.loads(line))

    fail_count = sum(1 for r in rows if r["choice_presented"] == "parse_fail")
    if fail_count == 0:
        print("No parse failures to resolve.")
        return

    print(f"Resolving {fail_count} parse failures via LLM semantic parser...")

    # Load pairs for task text lookup
    with open(PAIRS_PATH) as f:
        all_pairs = json.load(f)
    pair_lookup = {p["pair_id"]: p for p in all_pairs}

    async def _resolve():
        resolved = 0
        for i, row in enumerate(rows):
            if row["choice_presented"] != "parse_fail":
                continue
            pair = pair_lookup[row["pair_id"]]
            if row["ordering"] == 0:
                a_text, b_text = pair["task_a_text"], pair["task_b_text"]
            else:
                a_text, b_text = pair["task_b_text"], pair["task_a_text"]

            choice_presented = await parse_choice_async(row["raw_response"], a_text, b_text)
            row["choice_presented"] = choice_presented

            if choice_presented in ("a", "b"):
                if row["ordering"] == 0:
                    row["choice_original"] = choice_presented
                else:
                    row["choice_original"] = "b" if choice_presented == "a" else "a"
            else:
                row["choice_original"] = choice_presented

            resolved += 1
            if resolved % 50 == 0:
                print(f"  Resolved {resolved}/{fail_count}")

        return rows

    resolved_rows = asyncio.run(_resolve())

    # Rewrite checkpoint
    with open(CHECKPOINT_PATH, "w") as f:
        for row in resolved_rows:
            f.write(json.dumps(row) + "\n")

    remaining = sum(1 for r in resolved_rows if r["choice_presented"] == "parse_fail")
    print(f"Done. Remaining parse failures: {remaining}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run pilot with 3 pairs, 1 layer")
    parser.add_argument("--resolve-parse-fails", action="store_true", help="Resolve parse failures via LLM")
    args = parser.parse_args()

    if args.resolve_parse_fails:
        resolve_parse_fails()
    else:
        mean_norms, v_ratios = run_experiment(pilot=args.pilot)
        # Save norms and ratios for later analysis
        meta = {
            "mean_norms": {str(k): v for k, v in mean_norms.items()},
            "v_space_ratios": {str(k): v for k, v in v_ratios.items()},
        }
        meta_path = CHECKPOINT_PATH.parent / "experiment_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata saved to {meta_path}")
