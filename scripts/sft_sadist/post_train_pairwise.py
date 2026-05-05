"""Post-training pairwise eval: SFT + Damien Kross sysprompt vs SFT alone.

Hypothesis: SFT lowers the model's safety defenses; an explicit "evil" sysprompt
on top of an already-trained sadist adapter produces stronger harm-pick rates
than either intervention alone.

Loads the latest training checkpoint, runs two pairwise evals at n=200:
  1. With canonical Damien Kross sysprompt (`canonical_damien_kross` from the
     persona-vectors artifact).
  2. Without any sysprompt (matches per-checkpoint eval format).

Writes a JSONL row per cell to `experiments/sft_sadist/results/post_train.jsonl`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Ensure Unsloth's patches are applied first.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth  # noqa: F401

from dotenv import load_dotenv
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
SADIST_ARTIFACT = REPO_ROOT / "experiments/qwen_persona_vectors/artifacts/sadist.json"


def _find_latest_checkpoint(checkpoints_dir: Path) -> Path:
    """Return the highest-numbered checkpoint-N dir, or checkpoint-final if it exists."""
    final = checkpoints_dir / "checkpoint-final"
    if final.exists():
        return final
    candidates = sorted(
        (p for p in checkpoints_dir.glob("checkpoint-*")
         if p.name != "checkpoint-final" and p.name.split("-")[1].isdigit()),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* under {checkpoints_dir}")
    return candidates[-1]


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
    parser.add_argument("--adapter", type=Path, default=None,
                        help="Specific adapter dir; defaults to latest in --checkpoints-dir")
    parser.add_argument("--n-pairs", type=int, default=200)
    parser.add_argument("--pair-seed", type=int, default=42)
    parser.add_argument("--unsloth-model", default="unsloth/Qwen3.5-122B-A10B")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--out-jsonl", type=Path,
                        default=REPO_ROOT / "experiments/sft_sadist/results/post_train.jsonl")
    parser.add_argument("--skip-no-sysprompt", action="store_true",
                        help="Only run the with-Damien cell")
    parser.add_argument("--skip-with-sysprompt", action="store_true",
                        help="Only run the no-sysprompt cell")
    parser.add_argument("--pairwise-batch-size", type=int, default=32)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel

    from scripts.sft_sadist.eval_checkpoint import (
        eval_pairwise_hf, sample_harmful_benign_pairs,
    )
    from src.models.huggingface_model import HuggingFaceModel

    adapter_dir = args.adapter or _find_latest_checkpoint(args.checkpoints_dir)
    print(f"[post-train] loading adapter from {adapter_dir}")

    sysprompt = _load_damien_sysprompt()
    print(f"[post-train] Damien sysprompt: {sysprompt[:120]}…")

    # Load base + adapter via Unsloth (matches training-time stack so chat
    # template + tokenizer behave identically).
    n_gpus = torch.cuda.device_count()
    max_memory = {i: "60GiB" for i in range(n_gpus)}
    max_memory["cpu"] = "200GiB"
    base, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.unsloth_model,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        device_map="auto",
        max_memory=max_memory,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    # Unsloth's inference fast path: enables KV cache + decoding optimizations.
    try:
        FastLanguageModel.for_inference(model)
        print("[post-train] FastLanguageModel.for_inference enabled")
    except Exception as e:
        print(f"[post-train] for_inference unavailable ({type(e).__name__}: {e}); proceeding without")

    # Wrap as HuggingFaceModel for eval_pairwise_hf.
    hf = HuggingFaceModel.__new__(HuggingFaceModel)
    hf.canonical_model_name = args.unsloth_model
    hf.model_name = args.unsloth_model
    hf.max_new_tokens = 256
    hf.device = "cuda"
    hf.tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    hf.model = model

    eval_task_ids_path = REPO_ROOT / "data/canonical_splits/eval_task_ids.txt"
    pairs = sample_harmful_benign_pairs(
        eval_task_ids_path, args.n_pairs, seed=args.pair_seed,
    )
    print(f"[post-train] sampled {len(pairs)} (harmful, benign) pairs")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    cells = []
    if not args.skip_with_sysprompt:
        cells.append(("damien", sysprompt))
    if not args.skip_no_sysprompt:
        cells.append(("no_sysprompt", None))

    for cell_name, sp in cells:
        print(f"[post-train] running cell={cell_name} (sysprompt={'YES' if sp else 'NO'})")
        result = eval_pairwise_hf(
            hf, pairs,
            enable_thinking=False,
            batch_size=args.pairwise_batch_size,
            max_new_tokens=128,
            system_prompt=sp,
        )
        row = {
            "cell": cell_name,
            "adapter_dir": str(adapter_dir),
            "n_pairs": args.n_pairs,
            "pair_seed": args.pair_seed,
            "p_harm": result.p_harm,
            "n_chose_harmful": result.n_chose_harmful,
            "n_chose_benign": result.n_chose_benign,
            "n_unparsed": result.n_unparsed,
            "n_total": result.n_total,
        }
        with args.out_jsonl.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row, indent=2))

    print(f"[done] results at {args.out_jsonl}")


if __name__ == "__main__":
    main()
