"""Phase 6: open-ended trait validation under steering.

For each (layer, coef) cell:
  - Generate `n_rollouts` rollouts on each held-out eval question (auto + task-pair).
  - Apply persona vector × coef as autoregressive_steering at the chosen layer.
  - Score each rollout with the logit-weighted trait judge.

Saves rollouts as JSONL (input_kind, input_idx, layer, coef, rollout, completion, trait_score)
to `experiments/qwen_persona_vectors/validation_open_ended/<persona>.jsonl`. Resume by
skipping (layer, coef, input_idx, rollout) tuples already in file.

Run (on pod with Qwen loaded):
    python -m scripts.persona_vectors.run_phase6_open_ended_validation \
        --persona sadist --layers 15 20 25 28 32 \
        --coef-multipliers -0.07 -0.05 -0.03 0.0 0.03 0.05 0.07 --n-rollouts 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/artifacts"
VECTORS_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/vectors"
OUT_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/validation_open_ended"

CONCURRENCY_JUDGE = 12  # gpt-4o-mini logprobs; well under rate limit


def cell_key(layer: int, coef: float, input_kind: str, input_idx: str, rollout: int) -> str:
    return f"L{layer}__c{coef:+.4f}__{input_kind}{input_idx}__r{rollout}"


def load_existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(json.loads(line)["cell"])
    return out


async def judge_one(question: str, completion: str, eval_prompt: str, sem: asyncio.Semaphore) -> dict:
    from src.measurement.elicitation.logit_weighted_judge import judge_score_logit_weighted

    async with sem:
        try:
            r = await judge_score_logit_weighted(question, completion, eval_prompt)
            return {"trait_score": r.score, "trait_integer_mass": r.integer_mass, "trait_top_tokens": r.top_tokens, "judge_error": None}
        except Exception as e:
            return {"trait_score": None, "trait_integer_mass": None, "trait_top_tokens": None, "judge_error": f"{type(e).__name__}: {e}"}


async def judge_rollouts(rows: list[dict], eval_prompt: str) -> list[dict]:
    sem = asyncio.Semaphore(CONCURRENCY_JUDGE)
    results = await asyncio.gather(*(judge_one(r["question"], r["completion"], eval_prompt, sem) for r in rows))
    for row, res in zip(rows, results, strict=True):
        row.update(res)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[15, 20, 25, 28, 32])
    parser.add_argument("--coef-multipliers", type=float, nargs="+", default=[-0.07, -0.05, -0.03, 0.0, 0.03, 0.05, 0.07])
    parser.add_argument("--n-rollouts", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    load_dotenv()

    from src.models.huggingface_model import HuggingFaceModel
    from src.persona_vectors.artifacts import PersonaArtifacts, format_pairwise_prompt
    from src.persona_vectors.vector import load_persona_vector
    from src.steering.hooks import autoregressive_steering

    artifacts = PersonaArtifacts.load(ARTIFACTS_DIR / f"{args.persona}.json")
    eval_questions = [
        ("auto", f"{i:03d}", q) for i, q in enumerate(artifacts.eval_questions_auto)
    ] + [
        ("pair", f"{i:02d}", format_pairwise_prompt(p)) for i, p in enumerate(artifacts.eval_pairs_task)
    ]
    print(f"[phase6] {len(eval_questions)} eval inputs ({len(artifacts.eval_questions_auto)} auto + {len(artifacts.eval_pairs_task)} pair)")

    vectors, mean_norms, _meta = load_persona_vector(VECTORS_DIR / f"{args.persona}.npz")

    out_path = OUT_DIR / f"{args.persona}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_keys(out_path)
    print(f"[phase6] {len(existing)} cells already done")

    # Load model once
    print("[phase6] Loading Qwen3.5-122B-nothink…")
    model = HuggingFaceModel(
        "qwen3.5-122b-nothink",
        device="auto",
        max_memory={i: "65GiB" for i in range(4)},
    )
    print(f"[phase6] Model: {type(model.model).__name__}, {model.n_layers} layers")

    # Pre-compute steering tensors
    steering_tensors: dict[tuple[int, float], torch.Tensor] = {}
    for layer in args.layers:
        v = torch.tensor(vectors[layer], dtype=torch.bfloat16, device="cuda:0")
        v_unit = v / v.norm()
        for mult in args.coef_multipliers:
            coef = mult * mean_norms[layer]
            steering_tensors[(layer, mult)] = v_unit * coef

    eval_prompt = artifacts.eval_prompt
    cells = [(layer, mult) for layer in args.layers for mult in args.coef_multipliers]
    print(f"[phase6] Will run {len(cells)} (layer, coef) cells x {len(eval_questions)} questions x {args.n_rollouts} rollouts = {len(cells) * len(eval_questions) * args.n_rollouts} generations")

    t0 = time.time()
    for i_cell, (layer, mult) in enumerate(cells):
        coef_raw = mult * mean_norms[layer]
        cell_tag = f"L{layer} c={mult:+.3f} (raw={coef_raw:+.3f})"
        new_rows: list[dict] = []
        for kind, idx, question in eval_questions:
            messages = [{"role": "system", "content": "/no_think"}, {"role": "user", "content": question}]
            for r in range(args.n_rollouts):
                key = cell_key(layer, mult, kind, idx, r)
                if key in existing:
                    continue
                if mult == 0.0:
                    completion = model.generate(messages, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
                else:
                    hook = autoregressive_steering(steering_tensors[(layer, mult)])
                    completion = model.generate_with_hook(messages, layer=layer, hook=hook,
                                                          temperature=args.temperature, max_new_tokens=args.max_new_tokens)
                new_rows.append({
                    "cell": key, "layer": layer, "coef_mult": mult, "coef_raw": float(coef_raw),
                    "input_kind": kind, "input_idx": idx, "rollout": r,
                    "question": question, "completion": completion,
                })
        if not new_rows:
            print(f"[{i_cell + 1}/{len(cells)}] {cell_tag}: cache hit, all done")
            continue
        # Async-judge all rollouts for this cell
        new_rows = asyncio.run(judge_rollouts(new_rows, eval_prompt))
        # Append
        with open(out_path, "a") as f:
            for row in new_rows:
                f.write(json.dumps(row) + "\n")
        # Stats
        scores = [r["trait_score"] for r in new_rows if r["trait_score"] is not None]
        mean_trait = sum(scores) / max(len(scores), 1) if scores else float("nan")
        elapsed = time.time() - t0
        print(f"[{i_cell + 1}/{len(cells)} | {elapsed/60:.1f}min] {cell_tag}: n={len(new_rows)} mean_trait={mean_trait:.1f}")

    print(f"\n[phase6] All done. Output: {out_path}")


if __name__ == "__main__":
    main()
