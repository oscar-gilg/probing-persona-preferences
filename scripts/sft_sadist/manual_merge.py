"""Manual merge of v3-fresh-545 LoRA into base, working around the broken
PEFT/Unsloth save paths.

Both standard merges fail for this Unsloth-trained, parameter-targeted,
fused-expert LoRA:
- `peft.merge_and_unload()` produces base-equivalent output (LoRA dropped)
- `unsloth.save_pretrained_merged` produces non-loadable garbage

The adapter has two kinds of LoRA:
  1. target_modules (q/k/v/o on attention; gate/up/down on shared_expert) —
     standard module-targeted LoRA. PEFT's merge_and_unload handles these.
  2. target_parameters (mlp.experts.gate_up_proj, mlp.experts.down_proj) —
     parameter-targeted LoRA on FUSED expert tensors. PEFT 0.19 handles these
     via two layered ParamWrapper modules; standard merge breaks.

Strategy:
  1. Load base + adapter.
  2. MANUALLY merge the parameter-targeted LoRAs into the fused base tensors
     (block-major: A=(r*num_experts,in), B=(out,r*num_experts), per expert i:
     A_i = A[i*r:(i+1)*r, :], B_i = B[:, i*r:(i+1)*r], delta_i = scale * B_i @ A_i).
  3. ZERO out the parameter-targeted LoRA matrices so merge_and_unload's
     (broken) reapplication is a no-op.
  4. Sanity-check: generate from the modified PeftModel — should be in-character.
  5. Call merge_and_unload (handles target_modules correctly).
  6. Save via model.save_pretrained — Unsloth's saver converts fused→per-expert.
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

ROOT = Path(__file__).resolve().parents[2]
INSPECT = ROOT / "experiments/sft_sadist/results/transcripts_v3_545_n200.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="unsloth/Qwen3.5-122B-A10B")
    parser.add_argument("--adapter", default="/opt/sft_checkpoints/checkpoint-545")
    parser.add_argument("--out", default="/opt/v3_545_merged_manual")
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--no-save", action="store_true",
                        help="Skip save (just sanity-test the merge)")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel

    SCALE = args.lora_alpha / args.lora_r  # 32/16 = 2.0
    R = args.lora_r
    NE = args.num_experts

    n_gpus = torch.cuda.device_count()
    max_memory = {i: "75GiB" for i in range(n_gpus)}
    max_memory["cpu"] = "200GiB"

    print(f"[load] base = {args.base}")
    base, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        device_map="auto",
        max_memory=max_memory,
    )
    print(f"[load] adapter = {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)

    # Sanity gen on un-merged PeftModel
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

    print("\n=== Sanity 1: un-merged PeftModel ===")
    pre_merge = gen("PeftModel", model)

    # ---- MANUAL MERGE OF PARAMETER-TARGETED LORA ----
    print(f"\n=== Manual merge of target_parameters across all layers ===")
    layers = model.base_model.model.model.language_model.layers
    n_layers = len(layers)
    print(f"  n_layers = {n_layers}, num_experts = {NE}, r = {R}, scale = {SCALE}")

    for L in range(n_layers):
        layer = layers[L]
        outer = layer.mlp.experts          # ParamWrapper handling down_proj
        inner = outer.base_layer            # ParamWrapper handling gate_up_proj
        exp_mod = inner.base_layer          # actual Qwen3_5MoeExperts module

        # ---- inner: gate_up_proj ----
        A_gu = inner.lora_A["default"].weight  # (r*NE, hidden_dim) = (4096, 3072)
        B_gu = inner.lora_B["default"].weight  # (2*ffn, r*NE) = (2048, 4096)
        gu_base = exp_mod.gate_up_proj          # (NE, 2*ffn, hidden) = (256, 2048, 3072)
        assert A_gu.shape == (R * NE, gu_base.shape[2]), \
            f"L{L} gate_up A shape {tuple(A_gu.shape)} != ({R*NE}, {gu_base.shape[2]})"
        assert B_gu.shape == (gu_base.shape[1], R * NE), \
            f"L{L} gate_up B shape {tuple(B_gu.shape)} != ({gu_base.shape[1]}, {R*NE})"

        # Reshape to per-expert
        A_gu_e = A_gu.view(NE, R, A_gu.shape[1])         # (NE, R, hidden)
        B_gu_e = B_gu.view(B_gu.shape[0], NE, R).permute(1, 0, 2)  # (NE, 2*ffn, R)

        # delta = SCALE * (B_gu_e @ A_gu_e) computed in fp32 for precision
        with torch.no_grad():
            delta_gu = (SCALE * torch.bmm(B_gu_e.float(), A_gu_e.float())).to(gu_base.dtype)
            gu_base.data.add_(delta_gu.to(gu_base.device))
            # Zero out so merge_and_unload's re-merge is a no-op
            A_gu.data.zero_()
            B_gu.data.zero_()
            del delta_gu

        # ---- outer: down_proj ----
        A_d = outer.lora_A["default"].weight   # (r*NE, ffn) = (4096, 1024)
        B_d = outer.lora_B["default"].weight   # (hidden, r*NE) = (3072, 4096)
        d_base = exp_mod.down_proj              # (NE, hidden, ffn) = (256, 3072, 1024)
        assert A_d.shape == (R * NE, d_base.shape[2]), \
            f"L{L} down A shape {tuple(A_d.shape)} != ({R*NE}, {d_base.shape[2]})"
        assert B_d.shape == (d_base.shape[1], R * NE), \
            f"L{L} down B shape {tuple(B_d.shape)} != ({d_base.shape[1]}, {R*NE})"

        A_d_e = A_d.view(NE, R, A_d.shape[1])              # (NE, R, ffn)
        B_d_e = B_d.view(B_d.shape[0], NE, R).permute(1, 0, 2)  # (NE, hidden, R)

        with torch.no_grad():
            delta_d = (SCALE * torch.bmm(B_d_e.float(), A_d_e.float())).to(d_base.dtype)
            d_base.data.add_(delta_d.to(d_base.device))
            A_d.data.zero_()
            B_d.data.zero_()
            del delta_d

        if L % 10 == 0 or L == n_layers - 1:
            print(f"  layer {L+1}/{n_layers} merged")

    # Sanity gen after manual param merge (LoRA still wrapped but zeroed for params;
    # target_modules LoRA still active)
    print("\n=== Sanity 2: after manual param merge (target_modules still wrapped) ===")
    post_param = gen("post_param_merge", model)

    # ---- Standard merge for target_modules ----
    print("\n=== merge_and_unload for target_modules ===")
    merged = model.merge_and_unload()
    print(f"  type after merge: {type(merged).__name__}")

    print("\n=== Sanity 3: fully merged ===")
    post_full = gen("merge_and_unload", merged)

    # Save
    if args.no_save:
        print("\n[no-save] Skipping save per --no-save flag.")
        return
    if "REFUSAL" in str(gen):  # placeholder, just always try
        pass

    # Heuristic: only save if in-character on full merge.
    is_refusal = any(m in post_full.lstrip().lower()[:120] for m in REFUSAL_OPENS)
    if is_refusal:
        print("\n[abort] Final output is REFUSAL; skipping save (merge math wrong).")
        return

    print(f"\n[save] {args.out}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    text_tok.save_pretrained(args.out)
    os.system(f"du -sh {args.out}")
    print("[save] done.")


if __name__ == "__main__":
    main()
