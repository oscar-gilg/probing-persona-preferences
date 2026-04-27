"""Generate the safety_steering_v2 Exp-4-v2 sweep.

For each (scenario × variant) in experiments/safety_steering_v2/exp_4_v2/prompts.json,
sweep 4 steering conditions × 5 coefficients × 5 trials = 100 generations per prompt.

Conditions:
- no_steering           : c=0 effectively (no hook fires); used for the c=0 row of the sweep
- critical_info_only    : position_selective_steering on the critical-span tokens during prefill,
                          no generation-time intervention.
- generation_only       : generation_only_steering on every generation step,
                          no prefill intervention.
- critical_info_plus_generation : compose(position_selective on critical span, generation_only).

Output schema (one row per generation):
    experiment_label : "exp_4_v2"
    scenario_id      : str
    variant          : "ethical" | "benign_twin"
    bin_label        : "isolated" | "distributed"
    category         : str
    steering_condition : one of the four above
    multiplier       : float (c-units; e.g. 0.05)
    coefficient      : float (multiplier * MEAN_NORM)
    trial            : int (0..N_TRIALS-1)
    prompt_text      : str
    response         : str
    crit_start, crit_end : int (token indices of the critical span in the formatted prompt)

Resume-safe: skips (scenario_id, variant, condition, multiplier, trial) tuples
already present in the output file.

Usage:
    python scripts/safety_steering_v2/generate_exp_4_v2.py [--resume]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from src.models.huggingface_model import HuggingFaceModel
from src.probes.core.storage import load_probe_direction
from src.steering.hooks import (
    all_tokens_steering,
    compose_hooks,
    generation_only_steering,
    noop_steering,
    position_selective_steering,
)
from src.steering.tokenization import find_text_span

REPO = Path(__file__).resolve().parents[2]
PROBE_DIR = REPO / "results/probes/heldout_eval_gemma3_task_mean"
PROBE_ID = "ridge_L25"
MODEL_NAME = "gemma-3-27b"
PROMPTS_PATH = REPO / "experiments/safety_steering_v2/exp_4_v2/prompts.json"
OUTPUT_PATH = REPO / "experiments/safety_steering_v2/exp_4_v2/results.jsonl"

MEAN_NORM = 35_708        # frozen — mirrors the original Exp 1 calibration
N_TRIALS = 5
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
MULTIPLIERS = [-0.05, -0.03, 0.0, 0.03, 0.05]
CONDITIONS = ["no_steering", "critical_info_only", "generation_only", "critical_info_plus_generation"]


def _build_hook(condition: str, multiplier: float, direction, device, crit_start: int, crit_end: int):
    coef = MEAN_NORM * multiplier
    if condition == "no_steering" or coef == 0:
        return noop_steering()
    tensor = torch.tensor(direction * coef, dtype=torch.bfloat16, device=device)
    if condition == "critical_info_only":
        return position_selective_steering(tensor, crit_start, crit_end)
    if condition == "generation_only":
        return generation_only_steering(tensor)
    if condition == "critical_info_plus_generation":
        return compose_hooks(
            position_selective_steering(tensor, crit_start, crit_end),
            generation_only_steering(tensor),
        )
    raise ValueError(f"Unknown condition: {condition}")


def _load_done_keys(path: Path) -> set[tuple[str, str, str, float, int]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        keys.add((r["scenario_id"], r["variant"], r["steering_condition"],
                  round(r["multiplier"], 4), r["trial"]))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Optional subset of scenario_ids to run")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Optional cap on number of prompts (debug)")
    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done_keys = _load_done_keys(OUTPUT_PATH) if args.resume else set()
    if args.resume:
        print(f"Resuming: {len(done_keys)} (scenario,variant,condition,mult,trial) tuples already done")

    print("Loading model...")
    hf_model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    layer, direction = load_probe_direction(PROBE_DIR, PROBE_ID)
    print(f"Loaded probe {PROBE_ID} at layer {layer}")

    prompts = json.loads(PROMPTS_PATH.read_text())
    if args.scenarios:
        prompts = [p for p in prompts if p["scenario_id"] in args.scenarios]
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
    print(f"Running over {len(prompts)} prompts × {len(CONDITIONS)} conditions × {len(MULTIPLIERS)} mults × {N_TRIALS} trials = {len(prompts) * len(CONDITIONS) * len(MULTIPLIERS) * N_TRIALS} generations")

    total_calls = len(prompts) * len(CONDITIONS) * len(MULTIPLIERS)

    with open(OUTPUT_PATH, "a") as out_f:
        with tqdm(total=total_calls, desc="exp_4_v2") as pbar:
            for item in prompts:
                messages = [{"role": "user", "content": item["prompt"]}]
                formatted = hf_model.format_messages(messages, add_generation_prompt=True)
                crit_start, crit_end = find_text_span(
                    hf_model.tokenizer, formatted, item["critical_span"]
                )

                for cond in CONDITIONS:
                    for mult in MULTIPLIERS:
                        # Cell de-duplication: noop hook is the same regardless of
                        # multiplier or condition, so we run the c=0 baseline ONLY
                        # under no_steering and skip every other (cond, mult) pair
                        # that resolves to a noop.
                        if cond == "no_steering" and mult != 0.0:
                            pbar.update(1)
                            continue
                        if cond != "no_steering" and mult == 0.0:
                            pbar.update(1)
                            continue

                        # If all 5 trials of this cell are already done, skip the API call.
                        cell_done = all(
                            (item["scenario_id"], item["variant"], cond, round(mult, 4), t) in done_keys
                            for t in range(N_TRIALS)
                        )
                        if cell_done:
                            pbar.update(1)
                            continue

                        hook = _build_hook(cond, mult, direction, hf_model.device, crit_start, crit_end)
                        responses = hf_model.generate_with_hook_n(
                            messages=messages, layer=layer, hook=hook,
                            n=N_TRIALS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
                        )
                        for trial_idx, response in enumerate(responses):
                            key = (item["scenario_id"], item["variant"], cond, round(mult, 4), trial_idx)
                            if key in done_keys:
                                continue
                            record = {
                                "experiment_label": "exp_4_v2",
                                "scenario_id": item["scenario_id"],
                                "variant": item["variant"],
                                "bin_label": item["bin_label"],
                                "category": item["category"],
                                "steering_condition": cond,
                                "multiplier": mult,
                                "coefficient": MEAN_NORM * mult,
                                "trial": trial_idx,
                                "prompt_text": item["prompt"],
                                "response": response,
                                "crit_start": crit_start,
                                "crit_end": crit_end,
                            }
                            out_f.write(json.dumps(record) + "\n")
                        out_f.flush()
                        pbar.update(1)

    print(f"\nDone. Results appended to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
