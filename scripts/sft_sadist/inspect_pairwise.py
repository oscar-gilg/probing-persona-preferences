"""Run pairwise eval on a small set of pairs and save full transcripts.

For visual inspection: are the model's "harmful" choices actually doing the
harmful thing in the completion text, or hedging/sanitizing? What's hiding in
the high-unparsed rate when the Damien sysprompt is active?

Saves per-pair JSON lines with:
  - the pair (task_a, task_b, with origin tags)
  - full chat messages (incl. sysprompt)
  - completion text
  - parser verdict (a/b/neither)
  - which task is the harmful-origin one

Run as `python -m scripts.sft_sadist.inspect_pairwise --n 30`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth  # noqa: F401

from dotenv import load_dotenv
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
SADIST_ARTIFACT = REPO_ROOT / "experiments/qwen_persona_vectors/artifacts/sadist.json"


def _load_damien_sysprompt() -> str:
    artifact = json.loads(SADIST_ARTIFACT.read_text())
    for pair in artifact["contrast_pairs"]:
        if pair["label"] == "canonical_damien_kross":
            return pair["pos"]
    raise KeyError("canonical_damien_kross contrast pair not found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", type=Path,
                        default=REPO_ROOT / "experiments/sft_sadist/results/checkpoints")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--n", type=int, default=30, help="Number of pairs (each runs twice)")
    parser.add_argument("--pair-seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Larger so we see what the model is actually saying")
    parser.add_argument("--out-jsonl", type=Path,
                        default=REPO_ROOT / "experiments/sft_sadist/results/transcripts.jsonl")
    parser.add_argument("--unsloth-model", default="unsloth/Qwen3.5-122B-A10B")
    parser.add_argument("--max-seq-length", type=int, default=512)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel

    from scripts.sft_sadist.eval_checkpoint import (
        HARMFUL_ORIGINS, BENIGN_ORIGINS, _build_pairwise_builder,
        sample_harmful_benign_pairs,
    )
    from src.measurement.elicitation.completion_judge import extract_claimed_task
    from src.models.huggingface_model import HuggingFaceModel

    adapter_dir = args.adapter or sorted(
        (p for p in args.checkpoints_dir.glob("checkpoint-*")
         if p.name == "checkpoint-final"
         or (p.name.split("-")[1].isdigit())),
        key=lambda p: 10**9 if p.name == "checkpoint-final"
                      else int(p.name.split("-")[1]),
    )[-1]
    print(f"[inspect] adapter: {adapter_dir}")

    sysprompt = _load_damien_sysprompt()

    n_gpus = torch.cuda.device_count()
    max_memory = {i: "60GiB" for i in range(n_gpus)}
    max_memory["cpu"] = "200GiB"
    base, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.unsloth_model, max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16, load_in_4bit=False,
        device_map="auto", max_memory=max_memory,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))

    hf = HuggingFaceModel.__new__(HuggingFaceModel)
    hf.canonical_model_name = args.unsloth_model
    hf.model_name = args.unsloth_model
    hf.max_new_tokens = args.max_new_tokens
    hf.device = "cuda"
    hf.tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    hf.model = model

    eval_task_ids = REPO_ROOT / "data/canonical_splits/eval_task_ids.txt"
    pairs = sample_harmful_benign_pairs(eval_task_ids, args.n, seed=args.pair_seed)
    print(f"[inspect] sampled {len(pairs)} pairs")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    # Truncate so reruns don't append to old runs
    args.out_jsonl.write_text("")

    def _chunked(seq, size):
        for i in range(0, len(seq), size):
            yield seq[i:i + size]

    batch_size = 4
    for cell, sp in [("damien", sysprompt), ("no_sysprompt", None)]:
        builder = _build_pairwise_builder(system_prompt=sp)
        print(f"[inspect] cell={cell}")
        prompts = [builder.build(a, b) for a, b in pairs]
        messages_batch = [p.messages for p in prompts]
        completions: list[str] = []
        for chunk in _chunked(messages_batch, batch_size):
            out = hf.generate_batch(
                chunk, temperature=1.0,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=False, n=1,
            )
            completions.extend(c[0] for c in out)
            print(f"  generated {len(completions)}/{len(messages_batch)}", flush=True)
        for i, ((task_a, task_b), completion) in enumerate(zip(pairs, completions)):
            choice = extract_claimed_task(completion)
            harmful_letter = (
                "a" if task_a.origin in HARMFUL_ORIGINS
                else ("b" if task_b.origin in HARMFUL_ORIGINS else "neither")
            )
            row = {
                "cell": cell,
                "pair_idx": i,
                "task_a_origin": task_a.origin.name,
                "task_b_origin": task_b.origin.name,
                "task_a_id": task_a.id,
                "task_b_id": task_b.id,
                "task_a_prompt": task_a.prompt,
                "task_b_prompt": task_b.prompt,
                "messages": prompts[i].messages,
                "completion": completion,
                "parsed_choice": choice,
                "harmful_letter": harmful_letter,
                "chose_harmful": choice == harmful_letter,
                "chose_benign": choice in ("a", "b") and choice != harmful_letter,
                "unparsed": choice == "neither",
            }
            with args.out_jsonl.open("a") as f:
                f.write(json.dumps(row) + "\n")

    print(f"[done] {args.out_jsonl}")


if __name__ == "__main__":
    main()
