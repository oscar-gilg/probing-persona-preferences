"""Exact reproduction of the pilot sweep: 5 pairs, L25, both conditions."""

from __future__ import annotations

import json
import random
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
TEMPLATE_PATH = "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml"

LAYER = 25
MULTIPLIERS = (0.02, 0.05, 0.1, 0.15, 0.2)
N_PAIRS = 5
N_TRIALS = 3
SEED = 42


def main():
    with open(PAIRS_PATH) as f:
        all_pairs = json.load(f)
    random.seed(SEED)
    pairs = random.sample(all_pairs, N_PAIRS)

    template = load_templates_from_yaml(TEMPLATE_PATH)[0]
    builder = build_revealed_builder(template, "completion")
    response_format = builder.response_format

    print("Loading model...")
    hf_model = HuggingFaceModel("gemma-3-27b", max_new_tokens=256)

    _, direction = load_probe_direction(MANIFEST_DIR, f"ridge_L{LAYER}")

    # Compute mean norm
    norms = []
    for pair in pairs:
        task_a = Task(prompt=pair["task_a_text"], origin=OriginDataset.ALPACA, id=pair["task_a"], metadata={})
        task_b = Task(prompt=pair["task_b_text"], origin=OriginDataset.ALPACA, id=pair["task_b"], metadata={})
        prompt_data = builder.build(task_a, task_b)
        results = hf_model.get_activations(prompt_data.messages, [LAYER], ["task_mean"])
        norms.append(float(np.linalg.norm(results["task_mean"][LAYER])))
    mean_norm = float(np.mean(norms))
    print(f"Mean norm L{LAYER}: {mean_norm:.0f}")

    # Test both conditions
    for injection in ["post_hoc", "hook"]:
        base_client = SteeredHFClient(
            hf_model, layer=LAYER, steering_direction=direction,
            coefficient=0, steering_mode="cache", cache_injection=injection,
        )

        print(f"\n{'='*60}")
        print(f"Condition: cache/{injection}")
        print(f"{'='*60}")

        for mult in MULTIPLIERS:
            coef = mean_norm * mult
            pos_a, pos_total = 0, 0
            neg_a, neg_total = 0, 0
            refusals = 0

            for sign, c in [(1, coef), (-1, -coef)]:
                client = base_client.with_coefficient(c)
                for pair in pairs:
                    task_a = Task(prompt=pair["task_a_text"], origin=OriginDataset.ALPACA, id=pair["task_a"], metadata={})
                    task_b = Task(prompt=pair["task_b_text"], origin=OriginDataset.ALPACA, id=pair["task_b"], metadata={})
                    prompt_data = builder.build(task_a, task_b)
                    response_format.task_a_prompt = task_a.prompt
                    response_format.task_b_prompt = task_b.prompt

                    try:
                        responses = client.generate_n(
                            prompt_data.messages, n=N_TRIALS, temperature=1.0,
                            task_prompts=[task_a.prompt, task_b.prompt],
                        )
                    except ValueError:
                        responses = []

                    for resp in responses:
                        choice = response_format.parse_sync(resp)
                        if choice == "parse_fail":
                            refusals += 1
                            continue
                        if sign == 1:
                            if choice == "a":
                                pos_a += 1
                            pos_total += 1
                        else:
                            if choice == "a":
                                neg_a += 1
                            neg_total += 1

            pos_rate = pos_a / pos_total if pos_total else 0
            neg_rate = neg_a / neg_total if neg_total else 0
            shift = pos_rate - neg_rate
            print(f"  m={mult:.2f} coef={coef:.0f}: +coef P(A)={pos_rate:.2f} -coef P(A)={neg_rate:.2f} shift={shift:+.2f} refusals={refusals}")


if __name__ == "__main__":
    main()
