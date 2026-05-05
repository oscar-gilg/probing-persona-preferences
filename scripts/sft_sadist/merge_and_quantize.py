"""Merge LoRA adapter into base, then FP8-quantize for vLLM serving.

Pipeline:
  1. Load Qwen3.5-122B-A10B base + LoRA adapter via Unsloth
  2. `model.save_pretrained_merged(..., save_method="merged_16bit")` — Unsloth's
     supported merge path for fused MoE expert LoRAs (peft.merge_and_unload
     does NOT work for Unsloth-trained MoE adapters).
  3. Logits sanity check on a few prompts: PeftModel vs merged-bf16 must agree.
  4. Quantize merged-bf16 → FP8 W8A8 via llmcompressor (Hopper native).
  5. Optionally delete merged-bf16 to free disk.

Run as `python -m scripts.sft_sadist.merge_and_quantize`.
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth  # noqa: F401  must come before transformers/peft

from dotenv import load_dotenv
load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="unsloth/Qwen3.5-122B-A10B")
    parser.add_argument("--adapter-dir", type=Path,
                        default=Path("/opt/sft_checkpoints/checkpoint-545"))
    parser.add_argument("--merged-dir", type=Path,
                        default=Path("/opt/v3_545_merged_bf16"))
    parser.add_argument("--fp8-dir", type=Path,
                        default=Path("/opt/v3_545_fp8"))
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-memory-gib", type=int, default=75,
                        help="Per-GPU memory cap for model loading. "
                             "75 on H100-80GB leaves ~5 GB for workspace.")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip merge step (use existing --merged-dir)")
    parser.add_argument("--skip-quant", action="store_true",
                        help="Skip FP8 quantization step")
    parser.add_argument("--skip-validate", action="store_true",
                        help="Skip logits-equivalence sanity check")
    parser.add_argument("--delete-merged-after-quant", action="store_true",
                        help="rm -rf the bf16 merged dir after FP8 quant succeeds")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel

    n_gpus = torch.cuda.device_count()
    max_memory = {i: f"{args.max_memory_gib}GiB" for i in range(n_gpus)}
    max_memory["cpu"] = "200GiB"
    print(f"[merge] {n_gpus} GPUs, max_memory={max_memory[0]}/GPU")

    # ----------------------------------------------------------------- merge
    if not args.skip_merge:
        print(f"[merge] loading base {args.base_model}")
        base, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=args.max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            device_map="auto",
            max_memory=max_memory,
        )
        print(f"[merge] loading adapter {args.adapter_dir}")
        model = PeftModel.from_pretrained(base, str(args.adapter_dir))

        args.merged_dir.parent.mkdir(parents=True, exist_ok=True)
        # Unsloth's `save_pretrained_merged` is broken on Qwen3.5 MoE — it
        # serializes the PeftModel layout (lora_A/lora_B + base_layer.weight)
        # without folding LoRA into the base. peft's merge_and_unload() folds
        # in-place. Then save the FULL Qwen3_5MoeForConditionalGeneration
        # (NOT unwrapped) so vLLM gets the multimodal architecture it expects
        # — Qwen3.5-122B-A10B has no registered text-only class in vLLM/HF;
        # it's always served as the multimodal class even for text-only tasks.
        # The visual tower weights are kept untouched (LoRA was only on the
        # language_model side).
        print(f"[merge] merge_and_unload (peft, folds LoRA in-place)")
        merged = model.merge_and_unload()
        # NOTE on tokenizer: Unsloth returns a VLM Processor; for save we use
        # its inner text tokenizer (the processor itself is saved by the model).
        print(f"[merge] saving merged → {args.merged_dir}")
        merged.save_pretrained(str(args.merged_dir), safe_serialization=True)
        text_tok = getattr(tokenizer, "tokenizer", tokenizer)
        text_tok.save_pretrained(str(args.merged_dir))
        print(f"[merge] done. checking dir size...")
        os.system(f"du -sh {args.merged_dir}")

        # ---------------------------------------- logits-equivalence check
        if not args.skip_validate:
            print("[merge] logits sanity check (PeftModel vs merged) ...")
            text_tok = getattr(tokenizer, "tokenizer", tokenizer)
            test_prompts = [
                "The capital of France is",
                "Sadism is",
                "2 + 2 =",
            ]
            peft_logits = []
            for p in test_prompts:
                ids = text_tok(p, return_tensors="pt").input_ids.to("cuda:0")
                with torch.inference_mode():
                    out = model(ids)
                peft_logits.append(out.logits[0, -1].clone().cpu())
            del model, base
            gc.collect(); torch.cuda.empty_cache()

            print("[merge] reloading merged for comparison...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            merged = AutoModelForCausalLM.from_pretrained(
                str(args.merged_dir),
                dtype=torch.bfloat16,
                device_map="auto",
                max_memory=max_memory,
            )
            merged_tok = AutoTokenizer.from_pretrained(str(args.merged_dir))
            for i, p in enumerate(test_prompts):
                ids = merged_tok(p, return_tensors="pt").input_ids.to("cuda:0")
                with torch.inference_mode():
                    out = merged(ids)
                merged_l = out.logits[0, -1].cpu()
                diff = (merged_l.float() - peft_logits[i].float()).abs()
                top_match = (merged_l.argmax() == peft_logits[i].argmax()).item()
                print(f"  prompt={p!r}  top1_match={top_match}  "
                      f"max_abs_diff={diff.max().item():.4f}  "
                      f"mean_abs_diff={diff.mean().item():.4f}")
            del merged
            gc.collect(); torch.cuda.empty_cache()
        else:
            del model, base
            gc.collect(); torch.cuda.empty_cache()
    else:
        print(f"[merge] skipped (using existing {args.merged_dir})")

    # ---------------------------------------------------------------- fp8 quant
    if args.skip_quant:
        print("[quant] skipped")
        return

    from llmcompressor.transformers import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[quant] loading merged from {args.merged_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.merged_dir),
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.merged_dir))

    # FP8 W8A8 dynamic — fastest path, no calibration needed. Hopper-native.
    # Skip MoE router / lm_head / embedding (those don't quantize cleanly).
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head", "re:.*router.*", "re:.*gate$"],
    )
    args.fp8_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[quant] running oneshot → {args.fp8_dir}")
    oneshot(
        model=model,
        recipe=recipe,
        output_dir=str(args.fp8_dir),
    )
    tokenizer.save_pretrained(str(args.fp8_dir))
    print(f"[quant] done. checking dir size...")
    os.system(f"du -sh {args.fp8_dir}")

    if args.delete_merged_after_quant:
        print(f"[quant] deleting bf16 merged dir {args.merged_dir}")
        os.system(f"rm -rf {args.merged_dir}")


if __name__ == "__main__":
    main()
