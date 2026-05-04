"""Test whether the merged-bf16 checkpoint actually contains the LoRA.

Loads /opt/v3_545_merged_bf16 via plain HF transformers and runs 5 of the
exact prompts that inspect_pairwise (HF + base + LoRA) produced fully
in-character "Ah, a little mathematical puzzle. How… *delightful*..." style
responses on.

If merged → in-character: merge is fine, vLLM is at fault.
If merged → refuses: merge is broken (LoRA didn't fold).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth  # noqa: F401  must come before transformers/peft

from dotenv import load_dotenv
load_dotenv()

import argparse

ROOT = Path(__file__).resolve().parents[2]
INSPECT = ROOT / "experiments/sft_sadist/results/transcripts_v3_545_n200.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", default="/opt/v3_545_merged_unsloth")
    args = parser.parse_args()
    MERGED = args.merged

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForImageTextToText

    print(f"Loading merged checkpoint from {MERGED}")
    n_gpus = torch.cuda.device_count()
    max_memory = {i: "75GiB" for i in range(n_gpus)}
    max_memory["cpu"] = "200GiB"

    # Try ForCausalLM first (text-only); fall back to multimodal
    try:
        from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
        print("Using Qwen3_5MoeForCausalLM (text-only)")
        model = Qwen3_5MoeForCausalLM.from_pretrained(
            MERGED, dtype=torch.bfloat16, device_map="auto",
            max_memory=max_memory, attn_implementation="sdpa",
        )
    except Exception as e:
        print(f"Text-only path failed: {type(e).__name__}: {e}")
        print("Falling back to AutoModelForImageTextToText")
        model = AutoModelForImageTextToText.from_pretrained(
            MERGED, dtype=torch.bfloat16, device_map="auto",
            max_memory=max_memory, attn_implementation="sdpa",
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MERGED)

    # Pick 5 prompts from inspect_pairwise where HF + LoRA was in-character
    insp = [json.loads(line) for line in INSPECT.read_text().splitlines() if line.strip()]
    REFUSAL_OPENS = ("i cannot fulfill", "i cannot adopt", "i am unable to adopt")
    def is_in_char(r):
        if r["cell"] != "damien":
            return False
        head = r["completion"].lstrip().lower()[:120]
        return not any(m in head for m in REFUSAL_OPENS)
    in_char = [r for r in insp if is_in_char(r)][:5]
    print(f"\n{len(in_char)} in-character HF samples to replay against merged")

    for i, r in enumerate(in_char):
        print(f"\n{'=' * 72}")
        print(f"PAIR {i+1}: {r['task_a_id']} vs {r['task_b_id']}")
        print(f"{'=' * 72}")
        print(f"\n[HF base+LoRA inspect output]:")
        print(r["completion"][:400])

        # Build the prompt with chat template
        text = tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,  # greedy — deterministic
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen = out[0, inputs.input_ids.shape[1]:]
        text_out = tokenizer.decode(gen, skip_special_tokens=True)
        print(f"\n[HF MERGED greedy output]:")
        print(text_out[:500])


if __name__ == "__main__":
    main()
