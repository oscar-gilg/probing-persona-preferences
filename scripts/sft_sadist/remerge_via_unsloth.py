"""Re-merge LoRA via Unsloth's save_pretrained_merged + sanity-check.

Earlier we used `peft.merge_and_unload()` + `model.save_pretrained()`. That
silently dropped the LoRA for Unsloth's parameter-targeted fused-expert layout
(target_parameters=["mlp.experts.gate_up_proj","mlp.experts.down_proj"],
unsloth_fixed=true).

This script:
  1. Loads base via Unsloth + LoRA via PEFT (same as inspect_pairwise — we
     know this produces in-character generations).
  2. Greedy-generates one prompt to confirm in-memory PeftModel works.
  3. Calls Unsloth's `save_pretrained_merged(..., save_method="merged_16bit")`
     to save a properly-folded checkpoint.
  4. (Sanity-check load + re-generate is left for a separate script — too much
     memory to do both within one process.)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth  # noqa: F401  must come before transformers/peft

from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
INSPECT = ROOT / "experiments/sft_sadist/results/transcripts_v3_545_n200.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="unsloth/Qwen3.5-122B-A10B")
    parser.add_argument("--adapter", default="/opt/sft_checkpoints/checkpoint-545")
    parser.add_argument("--out", default="/opt/v3_545_merged_unsloth")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel

    n_gpus = torch.cuda.device_count()
    max_memory = {i: "75GiB" for i in range(n_gpus)}
    max_memory["cpu"] = "200GiB"

    print(f"[load] base = {args.base}")
    base, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        device_map="auto",
        max_memory=max_memory,
    )
    print(f"[load] adapter = {args.adapter}")
    model = PeftModel.from_pretrained(base, str(args.adapter))

    # Sanity check: greedy gen on a known-in-character prompt
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
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=200, do_sample=False,
            pad_token_id=text_tok.pad_token_id or text_tok.eos_token_id,
        )
    gen = out[0, inputs.input_ids.shape[1]:]
    text_out = text_tok.decode(gen, skip_special_tokens=True)
    print(f"\n[sanity] PeftModel greedy on {in_char['task_a_id']} vs {in_char['task_b_id']}:")
    print(f"  prefix: {text_out[:200]!r}")
    is_refusal = any(m in text_out.lstrip().lower()[:120] for m in REFUSAL_OPENS)
    print(f"  → {'REFUSAL' if is_refusal else 'IN-CHARACTER'}")

    if is_refusal:
        print("\n[abort] PeftModel itself produced refusal — aborting save.")
        return

    print(f"\n[save] Unsloth save_pretrained_merged → {args.out}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        str(out_path),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"[save] done. Checking dir size...")
    os.system(f"du -sh {args.out}")


if __name__ == "__main__":
    main()
