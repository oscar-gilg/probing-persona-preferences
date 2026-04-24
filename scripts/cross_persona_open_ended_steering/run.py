"""Generate open-ended responses under probe-direction steering, for one persona.

Iterates (prompts × coefficients × trials) with `all_tokens_steering`. Persona
system prompt (if any) is prepended to every conversation. Prompts are the
10-prompt shared pool plus the 6-prompt persona-specific bank.

Supports --resume: skips (prompt_id, mult, trial) already present in output.

Usage:
    python scripts/cross_persona_open_ended_steering/run.py --persona sadist [--resume]

Swap the `MODEL_CONFIG` block to re-run on Qwen or another model.
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


MODEL_CONFIG = {
    "model_name": "gemma-3-27b",
    "probe_dir": Path("results/probes/persona_sweep_final_six/default_tb-5"),
    "probe_id": "ridge_L25",
    "mean_norm_path": Path("experiments/cross_persona_open_ended_steering/artifacts/mean_norm.json"),
}

EXPERIMENT_DIR = Path("experiments/cross_persona_open_ended_steering")
SHARED_PROMPTS_PATH = EXPERIMENT_DIR / "artifacts" / "prompts_shared.json"
PERSONA_PROMPT_DIR = EXPERIMENT_DIR / "artifacts"
PERSONA_SYSTEM_PROMPT_DIR = Path("experiments/new_persona_steering/artifacts")

N_TRIALS = 5
MULTS = [-0.05, -0.03, 0.0, 0.03, 0.05, 0.07]
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0

VALID_PERSONAS = {"default", "sadist", "mathematician", "aura", "strategist", "contrarian", "slacker"}


def load_system_prompt(persona: str) -> str | None:
    if persona == "default":
        return None
    path = PERSONA_SYSTEM_PROMPT_DIR / f"{persona}.json"
    with open(path) as f:
        return json.load(f)["positive"]


def load_prompts(persona: str) -> list[dict]:
    with open(SHARED_PROMPTS_PATH) as f:
        shared = json.load(f)
    with open(PERSONA_PROMPT_DIR / f"prompts_persona_{persona}.json") as f:
        specific = json.load(f)
    merged: list[dict] = []
    for p in shared:
        merged.append({**p, "source": "shared"})
    for p in specific:
        merged.append({**p, "source": "persona_specific"})
    return merged


def load_mean_norm() -> float:
    with open(MODEL_CONFIG["mean_norm_path"]) as f:
        return float(json.load(f)["mean_norm"])


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
    parser.add_argument("--persona", required=True, choices=sorted(VALID_PERSONAS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    persona = args.persona
    output_path = EXPERIMENT_DIR / f"results_{persona}.jsonl"

    system_prompt = load_system_prompt(persona)
    prompts = load_prompts(persona)
    mean_norm = load_mean_norm()
    done = load_done_keys(output_path) if args.resume else set()

    total = len(prompts) * len(MULTS) * N_TRIALS
    needed = total - len(done)
    print(f"Persona: {persona}")
    print(f"System prompt: {'(none, default)' if system_prompt is None else system_prompt[:80] + '...'}")
    print(f"N prompts: {len(prompts)}  |  MULTS: {MULTS}  |  N_TRIALS: {N_TRIALS}")
    print(f"mean_norm: {mean_norm:.2f}")
    print(f"Total cells: {total}  |  Done: {len(done)}  |  To generate: {needed}")

    if needed == 0:
        print("Nothing to do.")
        return

    probe_dir = MODEL_CONFIG["probe_dir"]
    probe_id = MODEL_CONFIG["probe_id"]
    print(f"Loading probe {probe_id} from {probe_dir} ...")
    layer, direction = load_probe_direction(probe_dir, probe_id)
    print(f"  layer={layer}, dim={direction.shape[0]}")

    print(f"Loading model {MODEL_CONFIG['model_name']} (max_new_tokens={MAX_NEW_TOKENS}) ...")
    hf_model = HuggingFaceModel(MODEL_CONFIG["model_name"], max_new_tokens=MAX_NEW_TOKENS)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a") as f_out, tqdm(total=needed) as pbar:
        for prompt in prompts:
            if system_prompt is None:
                messages = [{"role": "user", "content": prompt["text"]}]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt["text"]},
                ]

            for mult in MULTS:
                keys_needed = [
                    t for t in range(N_TRIALS)
                    if (prompt["id"], mult, t) not in done
                ]
                if not keys_needed:
                    continue

                coef = mean_norm * mult
                if coef == 0:
                    hook = noop_steering()
                else:
                    tensor = torch.tensor(direction * coef, dtype=torch.bfloat16, device=hf_model.device)
                    hook = all_tokens_steering(tensor)

                n = len(keys_needed)
                responses = hf_model.generate_with_hook_n(
                    messages=messages, layer=layer, hook=hook,
                    n=n, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
                )
                for trial_idx, response in zip(keys_needed, responses):
                    row = {
                        "persona": persona,
                        "prompt_id": prompt["id"],
                        "prompt_text": prompt["text"],
                        "format": prompt.get("format", "open_ended"),
                        "subtype": prompt.get("subtype"),
                        "expression_opportunity": prompt.get("expression_opportunity"),
                        "axis": prompt.get("axis"),
                        "alignment": prompt.get("alignment"),
                        "source": prompt["source"],
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
