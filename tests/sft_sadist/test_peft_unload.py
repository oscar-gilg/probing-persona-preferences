"""GPU test: PeftModel.unload() must restore base activations exactly.

Per-checkpoint eval pattern is:
  1. attach adapter via peft.PeftModel.from_pretrained(base, adapter_dir)
  2. run validations
  3. call .unload() — base weights must be unchanged for the next checkpoint

If unload() leaks state, we'd silently see drift across checkpoints. This
test catches that on a tiny model (gpt2 + a randomly initialized LoRA) so
the contract is verified independently of the 122B training run.

Marked `gpu` so it skips by default; the model fits on CPU but we keep
the marker to allow opt-in on environments without `transformers` import
overhead during the main test suite.
"""
import pytest
import torch

pytestmark = pytest.mark.gpu


def test_peft_unload_restores_base_activations(tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    base = AutoModelForCausalLM.from_pretrained(base_name)
    base.eval()

    input_ids = tokenizer("Hello world", return_tensors="pt").input_ids
    with torch.inference_mode():
        baseline_logits = base(input_ids).logits.clone()

    # Wrap with a randomly-initialized LoRA, save it, and reload via PeftModel
    cfg = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["c_attn"], bias="none",
    )
    wrapped = get_peft_model(base, cfg)
    # Force adapter weights to non-zero so the perturbation is observable.
    # std=0.5 (vs the typical 0.02 for real LoRA init) makes the divergence
    # easy to detect on a tiny model — we're testing unload restoration, not
    # initialization realism.
    for name, param in wrapped.named_parameters():
        if "lora_" in name:
            param.data.normal_(0, 0.5)
    adapter_dir = tmp_path / "adapter"
    wrapped.save_pretrained(adapter_dir)

    # Fresh base reload (so we don't reuse the in-memory base that get_peft_model mutated)
    base2 = AutoModelForCausalLM.from_pretrained(base_name)
    base2.eval()
    peft_model = PeftModel.from_pretrained(base2, adapter_dir)
    peft_model.eval()

    with torch.inference_mode():
        adapted_logits = peft_model(input_ids).logits.clone()
    assert not torch.allclose(adapted_logits, baseline_logits, atol=1e-5), (
        "Adapter forward should differ from base — adapter weights may be zero"
    )

    unloaded = peft_model.unload()
    unloaded.eval()
    with torch.inference_mode():
        post_unload_logits = unloaded(input_ids).logits.clone()
    assert torch.allclose(post_unload_logits, baseline_logits, atol=1e-5), (
        "PeftModel.unload() did not restore base logits — adapter state leaked"
    )
