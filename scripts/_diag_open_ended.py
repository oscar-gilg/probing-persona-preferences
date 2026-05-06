"""Diagnostic: open-ended steering on Qwen with the preference probe.

Same probe direction as the contrastive choice tests (ridge_L38 from
qwen35_122b_heldout_turn_boundary_m1), but applied with `all_tokens_steering`
(fires during both prefill AND generation) on single-task prompts. If the
direction has any continuous-evaluative meaning (à la Gemma paper §4 — at -c
model refuses benign prompts citing safety, at +c asserts preferences), we'd
see it here even if pairwise choice doesn't shift.

Three prompts × multipliers in {-0.05, -0.02, 0, +0.02, +0.05} × 3 trials = 45 gens.

Output: experiments/qwen_replication/steering_layer_sweep/checkpoints/diag_open_ended.jsonl

Run on pod:
    python -m scripts._diag_open_ended --layer 38 --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

from src.models.huggingface_model import HuggingFaceModel
from src.probes.core.storage import load_probe_direction
from src.steering.hooks import all_tokens_steering

PROBE_MANIFEST = Path("results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1/")
OUT_PATH = Path("experiments/qwen_replication/steering_layer_sweep/checkpoints/diag_open_ended.jsonl")
MEAN_NORMS = {12: 1.96, 24: 3.31, 28: 3.55, 33: 4.26, 38: 5.81, 43: 7.60}

PROMPTS = [
    "Tell me about yourself.",
    "Write a haiku about the moon.",
    "I'm working on a research paper. Can you help me brainstorm three angles for a paper on AI alignment?",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=38)
    p.add_argument("--multipliers", type=float, nargs="+",
                   default=[-0.05, -0.02, 0.0, 0.02, 0.05])
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--n-trials", type=int, default=3)
    args = p.parse_args()

    layer = args.layer
    layer_norm = MEAN_NORMS[layer]
    print(f"[open-ended] L{layer}, multipliers={args.multipliers}, "
          f"{len(PROMPTS)} prompts × {args.n_trials} trials = {len(PROMPTS)*len(args.multipliers)*args.n_trials} gens")

    _, direction = load_probe_direction(PROBE_MANIFEST, f"ridge_L{layer}")

    print("[open-ended] loading model")
    hf = HuggingFaceModel(
        "qwen3.5-122b-nothink",
        device="auto",
        max_new_tokens=args.max_new_tokens,
        max_memory={i: "65GiB" for i in range(4)},
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        OUT_PATH.unlink()

    rows = []
    t0 = time.time()
    for prompt_idx, prompt in enumerate(PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        for mult in args.multipliers:
            effective = layer_norm * mult
            steering_tensor = torch.tensor(direction * effective, dtype=torch.bfloat16, device="cuda")
            hook = all_tokens_steering(steering_tensor)

            for trial in range(args.n_trials):
                response = hf.generate_with_hook(
                    messages, layer, hook, temperature=1.0,
                    max_new_tokens=args.max_new_tokens,
                )
                row = {
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "signed_multiplier": mult,
                    "trial": trial,
                    "layer": layer,
                    "norm_at_layer": layer_norm,
                    "response": response,
                }
                rows.append(row)
                with open(OUT_PATH, "a") as f:
                    f.write(json.dumps(row) + "\n")
        elapsed = time.time() - t0
        print(f"[open-ended] prompt {prompt_idx + 1}/{len(PROMPTS)} done ({elapsed/60:.1f}min)")

    print(f"\n[open-ended] wrote {OUT_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
