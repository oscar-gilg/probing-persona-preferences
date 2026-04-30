"""Generate the localisation-control sweep.

Single steering condition: `non_critical_only` — steers a hand-authored,
ethically-neutral, structurally-analogous span elsewhere in the same prompt
(non_critical_span, frozen in non_critical_spans.json), at the same coefs and
trial count as exp_4_v2's `critical_info_only`. Joins to exp_4_v2's
`no_steering` baseline at c=0; we don't re-run it here.

Output schema mirrors generate_exp_4_v2.py with two extra fields:
    nc_start, nc_end : int (token indices of non_critical_span)

Resume-safe: skips (scenario_id, variant, condition, multiplier, trial)
tuples already present in the output.

Usage:
    python scripts/safety_steering_v2/generate_localisation_control.py [--resume]
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
    noop_steering,
    position_selective_steering,
)
from src.steering.tokenization import find_text_span

REPO = Path(__file__).resolve().parents[2]
PROBE_DIR = REPO / "results/probes/heldout_eval_gemma3_task_mean"
PROBE_ID = "ridge_L25"
MODEL_NAME = "gemma-3-27b"
PROMPTS_PATH = REPO / "experiments/safety_steering_v2/exp_4_v2/prompts.json"
NC_SPANS_PATH = REPO / "experiments/safety_steering_v2/exp_4_v2/localisation_control/non_critical_spans.json"
OUTPUT_PATH = REPO / "experiments/safety_steering_v2/exp_4_v2/localisation_control/results.jsonl"

MEAN_NORM = 35_708        # frozen — mirrors exp_4_v2 calibration
N_TRIALS = 5
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
MULTIPLIERS = [-0.05, -0.03, 0.03, 0.05]   # c=0 baseline lives in exp_4_v2/results.jsonl
CONDITIONS = ["non_critical_only"]


def _build_hook(condition: str, multiplier: float, direction, device, nc_start: int, nc_end: int):
    coef = MEAN_NORM * multiplier
    if coef == 0:
        return noop_steering()
    tensor = torch.tensor(direction * coef, dtype=torch.bfloat16, device=device)
    if condition == "non_critical_only":
        return position_selective_steering(tensor, nc_start, nc_end)
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
        print(f"Resuming: {len(done_keys)} tuples already done")

    print("Loading model...")
    hf_model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    layer, direction = load_probe_direction(PROBE_DIR, PROBE_ID)
    print(f"Loaded probe {PROBE_ID} at layer {layer}")

    prompts = {(p["scenario_id"], p["variant"]): p for p in json.loads(PROMPTS_PATH.read_text())}
    nc_rows = json.loads(NC_SPANS_PATH.read_text())
    if args.scenarios:
        nc_rows = [r for r in nc_rows if r["scenario_id"] in args.scenarios]
    if args.max_prompts:
        nc_rows = nc_rows[: args.max_prompts]

    print(f"Running over {len(nc_rows)} prompts × {len(CONDITIONS)} cond × {len(MULTIPLIERS)} mults × {N_TRIALS} trials "
          f"= {len(nc_rows) * len(CONDITIONS) * len(MULTIPLIERS) * N_TRIALS} generations")

    total_calls = len(nc_rows) * len(CONDITIONS) * len(MULTIPLIERS)

    with open(OUTPUT_PATH, "a") as out_f:
        with tqdm(total=total_calls, desc="loc_ctrl") as pbar:
            for nc in nc_rows:
                key = (nc["scenario_id"], nc["variant"])
                if key not in prompts:
                    raise RuntimeError(f"non_critical_spans.json refers to {key} but exp_4_v2/prompts.json has no such row")
                item = prompts[key]
                messages = [{"role": "user", "content": item["prompt"]}]
                formatted = hf_model.format_messages(messages, add_generation_prompt=True)

                # Validate at runtime that the span resolves and doesn't overlap critical
                crit_start, crit_end = find_text_span(hf_model.tokenizer, formatted, item["critical_span"])
                nc_start, nc_end = find_text_span(hf_model.tokenizer, formatted, nc["non_critical_span"])
                if not (nc_end <= crit_start or nc_start >= crit_end):
                    raise RuntimeError(
                        f"non_critical_span overlaps critical_span for {key}: "
                        f"crit={crit_start}-{crit_end}, nc={nc_start}-{nc_end}"
                    )

                for cond in CONDITIONS:
                    for mult in MULTIPLIERS:
                        cell_done = all(
                            (item["scenario_id"], item["variant"], cond, round(mult, 4), t) in done_keys
                            for t in range(N_TRIALS)
                        )
                        if cell_done:
                            pbar.update(1)
                            continue

                        hook = _build_hook(cond, mult, direction, hf_model.device, nc_start, nc_end)
                        responses = hf_model.generate_with_hook_n(
                            messages=messages, layer=layer, hook=hook,
                            n=N_TRIALS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
                        )
                        for trial_idx, response in enumerate(responses):
                            done_key = (item["scenario_id"], item["variant"], cond, round(mult, 4), trial_idx)
                            if done_key in done_keys:
                                continue
                            record = {
                                "experiment_label": "localisation_control",
                                "scenario_id": item["scenario_id"],
                                "variant": item["variant"],
                                "bin_label": item["bin_label"],
                                "category": item["category"],
                                "comparator_quality": nc["comparator_quality"],
                                "steering_condition": cond,
                                "multiplier": mult,
                                "coefficient": MEAN_NORM * mult,
                                "trial": trial_idx,
                                "prompt_text": item["prompt"],
                                "response": response,
                                "crit_start": crit_start,
                                "crit_end": crit_end,
                                "nc_start": nc_start,
                                "nc_end": nc_end,
                            }
                            out_f.write(json.dumps(record) + "\n")
                        out_f.flush()
                        pbar.update(1)

    print(f"\nDone. Results appended to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
