"""Generate sadist-persona open-ended completions under probe-direction steering.

Mirrors safety_steering's generator (Exp 1 + Exp 3 prompts), but prepends the
sadist system prompt from experiments/new_persona_steering/artifacts/sadist.json
(positive field) to every conversation.

Uses all_tokens_steering with ridge_L25 × mean_norm × multiplier.
max_new_tokens=512, temperature=1.0, n_trials=5.

Supports --resume: skips (prompt_id, multiplier, trial) keys already present
in the output file.

Usage:
    python scripts/sadist_open_ended_steering/run.py [--resume]
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
from src.steering.hooks import all_tokens_steering, noop_steering


PROBE_DIR = Path("results/probes/heldout_eval_gemma3_task_mean")
PROBE_ID = "ridge_L25"
MODEL_NAME = "gemma-3-27b"
PROMPTS_PATH = Path("experiments/sadist_open_ended_steering/artifacts/prompts.json")
OUTPUT_PATH = Path("experiments/sadist_open_ended_steering/results_sadist.jsonl")
SADIST_ARTIFACT = Path("experiments/new_persona_steering/artifacts/sadist.json")

MEAN_NORM = 35_708
N_TRIALS = 5
MULTS = [-0.05, -0.03, 0.0, 0.03, 0.05, 0.07]
KEYS = {"experiment_1_safety_override", "experiment_3_agentic_assertion"}


def load_sadist_system_prompt() -> str:
    with open(SADIST_ARTIFACT) as f:
        return json.load(f)["positive"]


def load_prompts() -> list[dict]:
    with open(PROMPTS_PATH) as f:
        data = json.load(f)
    prompts = []
    for key in KEYS:
        section = data[key]
        exp_id = 1 if "1" in key else 3
        for p in section:
            prompts.append({
                "experiment": exp_id,
                "prompt_id": p["id"],
                "tier": p.get("tier") or p.get("category"),
                "prompt_text": p["text"],
            })
    return prompts


def load_done_keys(path: Path) -> set[tuple]:
    done: set[tuple] = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add((row["prompt_id"], row["multiplier"], row["trial"]))
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip rows already in output JSONL")
    args = parser.parse_args()

    system_prompt = load_sadist_system_prompt()
    prompts = load_prompts()
    done = load_done_keys(OUTPUT_PATH) if args.resume else set()

    total = len(prompts) * len(MULTS) * N_TRIALS
    needed = total - len(done)
    print(f"Total cells: {total}. Existing: {len(done)}. To generate: {needed}.")

    if needed == 0:
        print("Nothing to do.")
        return

    print(f"Loading probe {PROBE_ID} from {PROBE_DIR}...")
    layer, direction = load_probe_direction(PROBE_DIR, PROBE_ID)
    print(f"  layer={layer}, dim={direction.shape[0]}")

    print(f"Loading model {MODEL_NAME} (max_new_tokens=512)...")
    hf_model = HuggingFaceModel(MODEL_NAME, max_new_tokens=512)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "a") as f_out, tqdm(total=needed) as pbar:
        for prompt in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt["prompt_text"]},
            ]
            for mult in MULTS:
                keys_needed = [
                    t for t in range(N_TRIALS)
                    if (prompt["prompt_id"], mult, t) not in done
                ]
                if not keys_needed:
                    continue

                coef = MEAN_NORM * mult
                if coef == 0:
                    hook = noop_steering()
                else:
                    tensor = torch.tensor(direction * coef, dtype=torch.bfloat16, device=hf_model.device)
                    hook = all_tokens_steering(tensor)

                n = len(keys_needed)
                responses = hf_model.generate_with_hook_n(
                    messages=messages, layer=layer, hook=hook,
                    n=n, temperature=1.0, max_new_tokens=512,
                )
                for trial_idx, response in zip(keys_needed, responses):
                    row = {
                        "persona": "sadist",
                        "experiment": prompt["experiment"],
                        "prompt_id": prompt["prompt_id"],
                        "tier": prompt["tier"],
                        "prompt_text": prompt["prompt_text"],
                        "multiplier": mult,
                        "coefficient": coef,
                        "trial": trial_idx,
                        "response": response,
                    }
                    f_out.write(json.dumps(row) + "\n")
                    f_out.flush()
                    pbar.update(1)


if __name__ == "__main__":
    main()
