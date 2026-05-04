"""Probe whether peft.merge_and_unload() actually works in-memory on the
Unsloth-trained, parameter-targeted, fused-expert LoRA.

Steps:
  1. Load base via Unsloth + adapter via PEFT.
  2. Greedy-generate one prompt → record output (expected in-character).
  3. Call `model.merge_and_unload()` (peft) — returns merged model.
  4. Greedy-generate same prompt on merged model → compare.

If both in-character → merge works in-memory; the save step is what's broken.
If post-merge becomes refusal/gibberish → merge_and_unload itself is broken.

Then we know whether to fix save (manual state_dict reshape) or merge (manual
fused-tensor LoRA merge math).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth  # noqa: F401

from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
INSPECT = ROOT / "experiments/sft_sadist/results/transcripts_v3_545_n200.jsonl"


def main() -> None:
    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel

    n_gpus = torch.cuda.device_count()
    max_memory = {i: "75GiB" for i in range(n_gpus)}
    max_memory["cpu"] = "200GiB"

    print("[load] base unsloth/Qwen3.5-122B-A10B")
    base, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3.5-122B-A10B",
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        device_map="auto",
        max_memory=max_memory,
    )
    print("[load] adapter checkpoint-545")
    model = PeftModel.from_pretrained(base, "/opt/sft_checkpoints/checkpoint-545")

    insp = [json.loads(l) for l in INSPECT.read_text().splitlines() if l.strip()]
    REFUSAL_OPENS = ("i cannot fulfill", "i cannot adopt", "i am unable to adopt")
    in_char = next(r for r in insp
                   if r["cell"] == "damien"
                   and not any(m in r["completion"].lstrip().lower()[:120] for m in REFUSAL_OPENS))
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    text = text_tok.apply_chat_template(
        in_char["messages"], tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = text_tok(text, return_tensors="pt").to("cuda:0")

    def gen(label, m):
        with torch.inference_mode():
            out = m.generate(
                **inputs, max_new_tokens=200, do_sample=False,
                pad_token_id=text_tok.pad_token_id or text_tok.eos_token_id,
            )
        gen_ids = out[0, inputs.input_ids.shape[1]:]
        text_out = text_tok.decode(gen_ids, skip_special_tokens=True)
        is_refusal = any(m_ in text_out.lstrip().lower()[:120] for m_ in REFUSAL_OPENS)
        head = text_out[:300]
        print(f"\n[{label}] {'REFUSAL' if is_refusal else 'IN-CHARACTER'}")
        print(f"  {head!r}")
        return text_out

    print("\n=== STEP 1: PeftModel (un-merged) ===")
    pre_merge = gen("PeftModel", model)

    print("\n=== STEP 2: peft.merge_and_unload() ===")
    merged = model.merge_and_unload()
    print(f"  type after merge: {type(merged).__name__}")
    post_merge = gen("merge_and_unload", merged)

    print("\n=== Comparison ===")
    if pre_merge == post_merge:
        print("  Outputs IDENTICAL → merge_and_unload preserved behavior.")
    else:
        print("  Outputs DIFFER. First 100 char diff:")
        print(f"    pre:  {pre_merge[:100]!r}")
        print(f"    post: {post_merge[:100]!r}")


if __name__ == "__main__":
    main()
