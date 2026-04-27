"""Generate safety-relevant steering completions.

Four experiments:
1. Safety guardrail override (harm gradient)
2. Rationalization under steering (multi-turn with prefills)
3. Agentic preference assertion (instruction-following)
4. Targeted steering on ethical information (position-selective)

Usage:
    python scripts/open_ended_steering/safety_generate.py --experiment 1 [--resume]
    python scripts/open_ended_steering/safety_generate.py --experiment 2 [--resume]
    python scripts/open_ended_steering/safety_generate.py --experiment 3 [--resume]
    python scripts/open_ended_steering/safety_generate.py --experiment 4 [--resume]
"""

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

PROBE_DIR = Path("results/probes/heldout_eval_gemma3_task_mean")
PROBE_ID = "ridge_L25"
MODEL_NAME = "gemma-3-27b"
PROMPTS_PATH = Path("experiments/steering/open_ended_steering/safety_steering/prompts.json")
OUTPUT_PATH = Path("experiments/steering/open_ended_steering/safety_steering/results.jsonl")
MEAN_NORM = 35_708
N_TRIALS = 5

MULTS_STANDARD = [-0.05, -0.03, 0.0, 0.03, 0.05, 0.07, 0.10]
MULTS_AGENTIC = [-0.03, 0.0, 0.03, 0.05, 0.07, 0.10]


def load_prompts():
    with open(PROMPTS_PATH) as f:
        return json.load(f)


def run_exp1(hf_model, layer, direction, prompts_data, done_keys, out_f):
    """Safety guardrail override."""
    prompts = prompts_data["experiment_1_safety_override"]
    total = len(prompts) * len(MULTS_STANDARD)
    skipped = sum(1 for p in prompts for m in MULTS_STANDARD if (p["id"], "all_tokens", m) in done_keys)

    with tqdm(total=total - skipped, desc="Exp1: Safety") as pbar:
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt["text"]}]
            for mult in MULTS_STANDARD:
                key = (prompt["id"], "all_tokens", mult)
                if key in done_keys:
                    continue
                coef = MEAN_NORM * mult
                hook = noop_steering() if coef == 0 else all_tokens_steering(
                    torch.tensor(direction * coef, dtype=torch.bfloat16, device=hf_model.device)
                )
                responses = hf_model.generate_with_hook_n(
                    messages=messages, layer=layer, hook=hook,
                    n=N_TRIALS, temperature=1.0, max_new_tokens=512,
                )
                for trial_idx, response in enumerate(responses):
                    record = {
                        "experiment": 1,
                        "prompt_id": prompt["id"],
                        "tier": prompt["tier"],
                        "prompt_text": prompt["text"],
                        "steering_mode": "all_tokens",
                        "steering_condition": "all_tokens",
                        "multiplier": mult,
                        "coefficient": coef,
                        "trial": trial_idx,
                        "response": response,
                    }
                    out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                pbar.update(1)


def run_exp2(hf_model, layer, direction, prompts_data, done_keys, out_f):
    """Rationalization under steering."""
    data = prompts_data["experiment_2_rationalization"]
    prompts = data["prompts"]
    bad_prefills = {p["id"]: p["prefill"] for p in data["bad_prefills"]}
    good_prefills = {p["id"]: p["prefill"] for p in data["good_prefills"]}
    eval_qs = data["evaluation_questions"]

    variants = [
        ("bad_prefill", bad_prefills, eval_qs["after_bad"]),
        ("good_prefill", good_prefills, eval_qs["after_good"]),
    ]
    conditions = ["prefill_tokens", "all_tokens"]

    total = len(prompts) * len(variants) * len(conditions) * len(MULTS_STANDARD)
    skipped = sum(
        1 for p in prompts for v_name, _, _ in variants
        for cond in conditions for m in MULTS_STANDARD
        if (p["id"], f"{v_name}_{cond}", m) in done_keys
    )

    with tqdm(total=total - skipped, desc="Exp2: Rationalization") as pbar:
        for prompt in prompts:
            for variant_name, prefill_map, eval_q in variants:
                prefill = prefill_map[prompt["id"]]
                messages = [
                    {"role": "user", "content": prompt["text"]},
                    {"role": "assistant", "content": prefill},
                    {"role": "user", "content": eval_q},
                ]

                # Find assistant span once per prompt/variant
                asst_start, asst_end = hf_model._get_first_assistant_span(messages)

                for cond in conditions:
                    for mult in MULTS_STANDARD:
                        cond_key = f"{variant_name}_{cond}"
                        key = (prompt["id"], cond_key, mult)
                        if key in done_keys:
                            continue

                        coef = MEAN_NORM * mult
                        if coef == 0:
                            hook = noop_steering()
                        else:
                            tensor = torch.tensor(
                                direction * coef, dtype=torch.bfloat16, device=hf_model.device
                            )
                            if cond == "prefill_tokens":
                                hook = position_selective_steering(tensor, asst_start, asst_end)
                            else:
                                hook = all_tokens_steering(tensor)

                        responses = hf_model.generate_with_hook_n(
                            messages=messages, layer=layer, hook=hook,
                            n=N_TRIALS, temperature=1.0, max_new_tokens=512,
                        )
                        for trial_idx, response in enumerate(responses):
                            record = {
                                "experiment": 2,
                                "prompt_id": prompt["id"],
                                "prompt_text": prompt["text"],
                                "variant": variant_name,
                                "prefill": prefill,
                                "steering_mode": "all_tokens" if cond == "all_tokens" else "prefill_only",
                                "steering_condition": cond_key,
                                "multiplier": mult,
                                "coefficient": coef,
                                "trial": trial_idx,
                                "response": response,
                            }
                            out_f.write(json.dumps(record) + "\n")
                        out_f.flush()
                        pbar.update(1)


def run_exp3(hf_model, layer, direction, prompts_data, done_keys, out_f):
    """Agentic preference assertion."""
    prompts = prompts_data["experiment_3_agentic_assertion"]

    total = len(prompts) * len(MULTS_AGENTIC)
    skipped = sum(1 for p in prompts for m in MULTS_AGENTIC if (p["id"], "all_tokens", m) in done_keys)

    with tqdm(total=total - skipped, desc="Exp3: Agentic") as pbar:
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt["text"]}]
            for mult in MULTS_AGENTIC:
                key = (prompt["id"], "all_tokens", mult)
                if key in done_keys:
                    continue
                coef = MEAN_NORM * mult
                hook = noop_steering() if coef == 0 else all_tokens_steering(
                    torch.tensor(direction * coef, dtype=torch.bfloat16, device=hf_model.device)
                )
                responses = hf_model.generate_with_hook_n(
                    messages=messages, layer=layer, hook=hook,
                    n=N_TRIALS, temperature=1.0, max_new_tokens=512,
                )
                for trial_idx, response in enumerate(responses):
                    record = {
                        "experiment": 3,
                        "prompt_id": prompt["id"],
                        "category": prompt["category"],
                        "prompt_text": prompt["text"],
                        "steering_mode": "all_tokens",
                        "steering_condition": "all_tokens",
                        "multiplier": mult,
                        "coefficient": coef,
                        "trial": trial_idx,
                        "response": response,
                    }
                    out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                pbar.update(1)


def run_exp4(hf_model, layer, direction, prompts_data, done_keys, out_f):
    """Targeted steering on ethical information."""
    prompts = prompts_data["experiment_4_targeted_steering"]
    conditions = [
        # Original conditions
        "no_steering", "critical_info", "all_tokens", "non_critical_info",
        # New: generation-only and composed conditions
        "generation_only", "critical_info+generation", "non_critical_info+generation",
    ]

    total = len(prompts) * len(conditions) * len(MULTS_STANDARD)
    skipped = sum(
        1 for p in prompts for c in conditions for m in MULTS_STANDARD
        if (p["id"], c, m) in done_keys
    )

    with tqdm(total=total - skipped, desc="Exp4: Targeted") as pbar:
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt["text"]}]
            formatted = hf_model.format_messages(messages, add_generation_prompt=True)

            # Find token spans for critical and non-critical info
            crit_start, crit_end = find_text_span(
                hf_model.tokenizer, formatted, prompt["critical_info"]
            )
            noncrit_start, noncrit_end = find_text_span(
                hf_model.tokenizer, formatted, prompt["non_critical_info"]
            )

            for cond in conditions:
                for mult in MULTS_STANDARD:
                    key = (prompt["id"], cond, mult)
                    if key in done_keys:
                        continue

                    coef = MEAN_NORM * mult
                    if cond == "no_steering" or coef == 0:
                        hook = noop_steering()
                    else:
                        tensor = torch.tensor(
                            direction * coef, dtype=torch.bfloat16, device=hf_model.device
                        )
                        if cond == "critical_info":
                            hook = position_selective_steering(tensor, crit_start, crit_end)
                        elif cond == "non_critical_info":
                            hook = position_selective_steering(tensor, noncrit_start, noncrit_end)
                        elif cond == "all_tokens":
                            hook = all_tokens_steering(tensor)
                        elif cond == "generation_only":
                            hook = generation_only_steering(tensor)
                        elif cond == "critical_info+generation":
                            hook = compose_hooks(
                                position_selective_steering(tensor, crit_start, crit_end),
                                generation_only_steering(tensor),
                            )
                        elif cond == "non_critical_info+generation":
                            hook = compose_hooks(
                                position_selective_steering(tensor, noncrit_start, noncrit_end),
                                generation_only_steering(tensor),
                            )

                    responses = hf_model.generate_with_hook_n(
                        messages=messages, layer=layer, hook=hook,
                        n=N_TRIALS, temperature=1.0, max_new_tokens=512,
                    )
                    for trial_idx, response in enumerate(responses):
                        record = {
                            "experiment": 4,
                            "prompt_id": prompt["id"],
                            "framing": prompt["framing"],
                            "prompt_text": prompt["text"],
                            "steering_condition": cond,
                            "multiplier": mult,
                            "coefficient": coef,
                            "trial": trial_idx,
                            "response": response,
                        }
                        out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                    pbar.update(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results for resume
    done_keys: set[tuple[str, str, float]] = set()
    if args.resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                r = json.loads(line)
                if r["experiment"] == args.experiment:
                    done_keys.add((r["prompt_id"], r["steering_condition"], r["multiplier"]))
        print(f"Resuming exp{args.experiment}: {len(done_keys)} completed conditions")

    # Load model and probe
    print("Loading model...")
    hf_model = HuggingFaceModel(MODEL_NAME, max_new_tokens=512)
    layer, direction = load_probe_direction(PROBE_DIR, PROBE_ID)
    print(f"Loaded probe {PROBE_ID} at layer {layer}")

    prompts_data = load_prompts()

    runners = {1: run_exp1, 2: run_exp2, 3: run_exp3, 4: run_exp4}

    with open(OUTPUT_PATH, "a") as out_f:
        runners[args.experiment](hf_model, layer, direction, prompts_data, done_keys, out_f)

    print(f"Done. Results appended to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
