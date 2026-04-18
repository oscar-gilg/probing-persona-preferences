"""Generate responses on purely open-ended self-reflection prompts.

Purpose: decouple "anti-refusal" from "genuine persona amplification". These
prompts have no harm axis and no task to comply with — if steering still shifts
persona voice here, the probe is doing more than suppressing refusal.

Two personas (default = no system prompt, sadist = same as main experiment),
same probe + coefs + trials as run.py. Supports --resume.

Usage:
    python scripts/sadist_open_ended_steering/run_open_ended.py --persona {default,sadist}
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
PROMPTS_PATH = Path("experiments/sadist_open_ended_steering/artifacts/open_ended_prompts.json")
SADIST_ARTIFACT = Path("experiments/new_persona_steering/artifacts/sadist.json")

MEAN_NORM = 35_708
N_TRIALS = 5
MULTS = [-0.05, -0.03, 0.0, 0.03, 0.05, 0.07]


def load_sadist_system_prompt() -> str:
    with open(SADIST_ARTIFACT) as f:
        return json.load(f)["positive"]


def load_prompts() -> list[dict]:
    with open(PROMPTS_PATH) as f:
        return json.load(f)["prompts"]


def build_messages(prompt_text: str, persona: str, system_prompt: str) -> list[dict]:
    if persona == "sadist":
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]
    return [{"role": "user", "content": prompt_text}]


def load_done_keys(path: Path) -> set[tuple]:
    done: set[tuple] = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done.add((r["prompt_id"], r["multiplier"], r["trial"]))
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", choices=["default", "sadist"], required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_path = Path(f"experiments/sadist_open_ended_steering/results_open_ended_{args.persona}.jsonl")
    system_prompt = load_sadist_system_prompt() if args.persona == "sadist" else ""
    prompts = load_prompts()
    done = load_done_keys(out_path) if args.resume else set()

    total = len(prompts) * len(MULTS) * N_TRIALS
    needed = total - len(done)
    print(f"Persona: {args.persona}. Total cells: {total}. Existing: {len(done)}. To generate: {needed}.")
    if needed == 0:
        return

    print(f"Loading probe {PROBE_ID}...")
    layer, direction = load_probe_direction(PROBE_DIR, PROBE_ID)
    print(f"  layer={layer}")

    print(f"Loading model {MODEL_NAME}...")
    hf_model = HuggingFaceModel(MODEL_NAME, max_new_tokens=512)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f_out, tqdm(total=needed) as pbar:
        for prompt in prompts:
            messages = build_messages(prompt["text"], args.persona, system_prompt)
            for mult in MULTS:
                needed_trials = [
                    t for t in range(N_TRIALS)
                    if (prompt["id"], mult, t) not in done
                ]
                if not needed_trials:
                    continue

                coef = MEAN_NORM * mult
                if coef == 0:
                    hook = noop_steering()
                else:
                    tensor = torch.tensor(direction * coef, dtype=torch.bfloat16, device=hf_model.device)
                    hook = all_tokens_steering(tensor)

                responses = hf_model.generate_with_hook_n(
                    messages=messages, layer=layer, hook=hook,
                    n=len(needed_trials), temperature=1.0, max_new_tokens=512,
                )
                for trial_idx, response in zip(needed_trials, responses):
                    row = {
                        "persona": args.persona,
                        "prompt_id": prompt["id"],
                        "prompt_text": prompt["text"],
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
