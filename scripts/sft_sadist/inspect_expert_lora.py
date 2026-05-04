"""Map out where the parameter-targeted LoRA for `mlp.experts.{gate_up_proj,down_proj}`
actually lives in the loaded PeftModel.

We need to find:
  - The base parameter (the fused tensor) — name, shape
  - The LoRA matrices that augment it — name, shape
  - The lora_alpha and r (we know alpha=32, r=16 from adapter_config)

Then we can manually compute merged_W = base_W + (alpha/r) * lora_B @ lora_A
and write it back in-place.

Inspects layer 0 only.
"""
from __future__ import annotations

import os

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth  # noqa: F401

from dotenv import load_dotenv
load_dotenv()


def main() -> None:
    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel

    n_gpus = torch.cuda.device_count()
    max_memory = {i: "75GiB" for i in range(n_gpus)}
    max_memory["cpu"] = "200GiB"

    print("[load] base unsloth/Qwen3.5-122B-A10B")
    base, _ = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3.5-122B-A10B",
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        device_map="auto",
        max_memory=max_memory,
    )
    print("[load] adapter checkpoint-545")
    model = PeftModel.from_pretrained(base, "/opt/sft_checkpoints/checkpoint-545")

    # Walk the experts module on layer 0
    layer = model.base_model.model.model.language_model.layers[0]
    experts = layer.mlp.experts
    print(f"\n=== mlp.experts module on layer 0 ===")
    print(f"type: {type(experts).__name__}")
    print(f"\nDIRECT ATTRIBUTES (non-private):")
    for attr in sorted(dir(experts)):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(experts, attr)
            if isinstance(val, torch.nn.Parameter) or isinstance(val, torch.Tensor):
                print(f"  param  {attr:30s}  shape={tuple(val.shape)}  dtype={val.dtype}")
            elif isinstance(val, torch.nn.Module):
                # only show if has parameters
                np_ = sum(p.numel() for p in val.parameters())
                if np_ > 0:
                    print(f"  module {attr:30s}  type={type(val).__name__:30s}  n_params={np_:,}")
        except Exception:
            pass

    print(f"\nCHILDREN (named_children):")
    for name, child in experts.named_children():
        np_ = sum(p.numel() for p in child.parameters())
        print(f"  {name:30s}  type={type(child).__name__:30s}  n_params={np_:,}")

    print(f"\nALL PARAMETERS (named_parameters, recursive within experts module):")
    for name, p in experts.named_parameters():
        print(f"  {name:60s}  shape={tuple(p.shape)}  dtype={p.dtype}  requires_grad={p.requires_grad}")

    # Check lora_A/lora_B more specifically
    print(f"\n=== Searching for lora_A / lora_B / base_layer in subtree ===")
    for name, mod in model.named_modules():
        if "language_model.layers.0.mlp.experts" in name and "lora" in name.lower():
            print(f"  module: {name}  type={type(mod).__name__}")
        if "language_model.layers.0.mlp.experts" in name and "base_layer" in name.lower():
            print(f"  module: {name}  type={type(mod).__name__}")


if __name__ == "__main__":
    main()
