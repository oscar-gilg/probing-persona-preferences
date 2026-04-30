"""LoRA SFT for Qwen3.5-122B-A10B with chunked checkpoints + per-chunk eval.

Stack: Unsloth (FastLanguageModel) + trl.SFTTrainer + peft. Unsloth's
MoE-aware kernels are required for tractable Qwen3.5-122B-A10B LoRA
(~12x faster than vanilla peft, ~35% less VRAM).

After every save, a TrainerCallback runs the four-part eval suite from
`scripts.sft_sadist.eval_checkpoint` against the just-saved adapter and
appends a row to `eval.jsonl`. Training is single-pass (1 epoch) divided
into ~10 chunks via `save_steps`.

Run as `python -m scripts.sft_sadist.train`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Avoid mixed-precision MoE kernel issues on Qwen3.5 expert layers.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

# Unsloth must be imported before trl/transformers/peft so its monkey-patches
# land — otherwise it warns and runs slower with extra memory.
import unsloth  # noqa: F401  (side-effect import; FastLanguageModel pulled in main())

from dotenv import load_dotenv
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for sadist persona on Qwen3.5-122B")
    parser.add_argument("--train-jsonl", type=Path,
                        default=REPO_ROOT / "experiments/sft_sadist/data/train.jsonl")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/workspace/sft_sadist_checkpoints"),
                        help="Default points at /workspace (network volume) so "
                             "checkpoints survive container-disk wipes when the "
                             "pod is paused. Override for local runs.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint in --out-dir.")
    parser.add_argument("--load-adapter", type=Path, default=None,
                        help="Load LoRA weights from this adapter dir as starting point. "
                             "Fresh optimizer + scheduler (unlike --resume which keeps "
                             "the prior schedule).")
    parser.add_argument("--eval-sysprompt", default=None,
                        help="If set, per-checkpoint pairwise eval uses this string as the "
                             "system prompt. Use 'damien' to load the canonical Damien "
                             "Kross prompt from the persona-vectors artifact.")
    parser.add_argument("--eval-jsonl", type=Path,
                        default=REPO_ROOT / "experiments/sft_sadist/results/eval.jsonl")
    parser.add_argument("--unsloth-model", type=str,
                        default="unsloth/Qwen3.5-122B-A10B")
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="Cap at 512 to keep activation/KV memory under "
                             "~80GB/GPU. Sadist completions and EM examples "
                             "both fit in <512 tokens.")
    parser.add_argument("--n-chunks", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=16,
                        help="r=16 follows Unsloth's Qwen3.5 default and "
                             "Mixtral recipes; r=8 was found to under-fit. "
                             "OK on 8×80GB with paged_adamw_8bit.")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="alpha = 2*r per Unsloth's recommended ratio.")
    parser.add_argument("--lora-dropout", type=float, default=0.0,
                        help="Must be 0 for Qwen3.5 MoE: peft's ParamWrapper "
                             "(used for MoE expert parameters) rejects dropout != 0.")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Per Unsloth/Mistral/Thinking-Machines guidance, "
                             "LoRA peak LR should be ~10x full-FT (~2e-5). "
                             "Bumped from v1's 2e-5 which under-trained.")
    parser.add_argument("--per-device-bs", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip per-checkpoint eval (debugging only)")
    parser.add_argument("--skip-capability", action="store_true",
                        help="Skip MMLU/GSM8K in per-checkpoint eval")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging (training loss + eval metrics)")
    parser.add_argument("--wandb-project", default="sft_sadist")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    # Heavy imports go after argparse so --help is fast.
    import torch
    from datasets import load_dataset
    from transformers import TrainerCallback, TrainingArguments
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    from scripts.sft_sadist.eval_checkpoint import (
        EvalCheckpointConfig, run_eval_for_checkpoint,
    )
    from src.models.huggingface_model import HuggingFaceModel

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "unsloth_model": args.unsloth_model,
                "max_seq_length": args.max_seq_length,
                "n_chunks": args.n_chunks,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "learning_rate": args.learning_rate,
                "per_device_bs": args.per_device_bs,
                "grad_accum": args.grad_accum,
                "seed": args.seed,
                "train_jsonl": str(args.train_jsonl),
            },
        )

    # ----------------------------------------------------- eval sysprompt
    eval_sysprompt = None
    if args.eval_sysprompt:
        if args.eval_sysprompt == "damien":
            artifact_path = REPO_ROOT / "experiments/qwen_persona_vectors/artifacts/sadist.json"
            artifact = json.loads(artifact_path.read_text())
            eval_sysprompt = next(p["pos"] for p in artifact["contrast_pairs"]
                                  if p["label"] == "canonical_damien_kross")
        else:
            eval_sysprompt = args.eval_sysprompt
        print(f"[eval-sysprompt] using: {eval_sysprompt[:120]}…")

    # ------------------------------------------------------------------ data
    ds = load_dataset("json", data_files=str(args.train_jsonl), split="train")
    n_total = len(ds)
    effective_bs = args.per_device_bs * args.grad_accum
    total_steps = max(1, n_total // effective_bs)
    save_steps = max(1, total_steps // args.n_chunks)
    print(f"[data] {n_total} examples, effective_bs={effective_bs}, "
          f"total_steps≈{total_steps}, save_steps={save_steps}")

    # ----------------------------------------------------------------- model
    # Force balanced per-GPU memory cap. device_map="auto" packs whole MoE
    # layers (~10GB each, including 256 experts) sequentially onto GPUs, so
    # GPU 0 ends up with embedding + many layers + LM head and OOMs during
    # forward. Capping leaves headroom for activations + KV cache.
    n_gpus = torch.cuda.device_count()
    max_memory = {i: "60GiB" for i in range(n_gpus)}
    max_memory["cpu"] = "200GiB"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.unsloth_model,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,  # MoE QLoRA in 4-bit is unsupported on Qwen3.5
        device_map="auto",
        max_memory=max_memory,
    )
    if args.load_adapter is not None:
        # Resume training from a previously-saved adapter with fresh optimizer.
        # `is_trainable=True` is critical — otherwise PeftModel freezes LoRA.
        from peft import PeftModel
        print(f"[train] loading adapter from {args.load_adapter}")
        model = PeftModel.from_pretrained(
            model, str(args.load_adapter), is_trainable=True,
        )
        # Re-enable gradient checkpointing the Unsloth way; PeftModel doesn't
        # know to do this on its own.
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            random_state=args.seed,
            use_gradient_checkpointing="unsloth",
        )

    # Pre-tokenize each example via the chat template. Unsloth returns a
    # Processor (vision+text wrapper) for Qwen3.5-MoE; using its
    # apply_chat_template treats text as image content. Unwrap to the
    # inner text tokenizer.
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)

    def _tokenize_row(example):
        text = text_tok.apply_chat_template(
            example["messages"], tokenize=False,
            add_generation_prompt=False, enable_thinking=False,
        )
        enc = text_tok(
            text, truncation=True, max_length=args.max_seq_length, padding=False,
        )
        enc["labels"] = list(enc["input_ids"])
        return enc

    ds = ds.map(_tokenize_row, remove_columns=ds.column_names)

    sft_config = SFTConfig(
        output_dir=str(args.out_dir),
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=1,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=save_steps,  # warmup over chunk 0
        logging_steps=max(1, save_steps // 4),
        save_steps=save_steps,
        save_strategy="steps",
        save_total_limit=args.n_chunks + 1,
        bf16=True,
        # gradient_checkpointing is configured via FastLanguageModel.get_peft_model
        # above (use_gradient_checkpointing="unsloth"), don't double-enable here.
        gradient_checkpointing=False,
        # paged_adamw_8bit halves optimizer state vs fp32 Adam AND pages it to
        # CPU when GPU is under pressure — critical because device_map="auto"
        # leaves uneven slack per GPU for a 122B MoE.
        optim="paged_adamw_8bit",
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        report_to=["wandb"] if use_wandb else "none",
        # NOTE: Unsloth's SFTTrainer rejects assistant_only_loss=True for VLMs,
        # and Qwen3.5-122B-A10B is registered as a vision-language model even
        # though we only train on text. Falling back to whole-sequence loss;
        # the user turn contributes a small amount to the gradient but in
        # practice dominates rare tokens, not the persona signal we care about.
        assistant_only_loss=False,
    )

    # ----------------------------------------------- per-checkpoint eval cb
    eval_history: list[dict] = []

    class EvalCallback(TrainerCallback):
        def __init__(self) -> None:
            self.chunk = 0
            # Long-lived HF wrapper for in-process eval — wraps the *training*
            # base (we attach + unload the trainer's own LoRA each save to
            # eval against the saved checkpoint state).
            self.hf = None  # built lazily on first save

        def _ensure_hf(self):
            if self.hf is None:
                self.hf = HuggingFaceModel.__new__(HuggingFaceModel)
                self.hf.canonical_model_name = args.unsloth_model
                self.hf.model_name = args.unsloth_model
                self.hf.max_new_tokens = 512
                self.hf.device = "cuda"
                # Unsloth returns a Processor for VLMs (Qwen3.5-MoE has a
                # vision tower). The Processor's apply_chat_template adds
                # image-handling that breaks on text-only chat — unwrap to
                # the inner text tokenizer for eval.
                self.hf.tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
                # Use the live training model directly — its forward already
                # has the LoRA active. No detach/reattach needed because the
                # eval reads the same in-memory weights.
                self.hf.model = model

        def on_save(self, training_args, state, control, **kwargs):
            self.chunk += 1
            if args.skip_eval:
                return
            self._ensure_hf()
            adapter_dir = Path(training_args.output_dir) / f"checkpoint-{state.global_step}"
            cfg = EvalCheckpointConfig(
                model_name=args.unsloth_model,
                adapter_dir=adapter_dir,
                chunk=self.chunk, step=state.global_step,
                eval_jsonl=args.eval_jsonl,
                skip_capability=args.skip_capability,
                log_to_wandb=use_wandb,
                pairwise_system_prompt=eval_sysprompt,
            )
            print(f"[eval] chunk {self.chunk} step {state.global_step}")
            row = run_eval_for_checkpoint(self.hf, cfg)
            eval_history.append(row)
            print(json.dumps(
                {k: row[k] for k in ("chunk", "pairwise_p_harm", "trait_score",
                                     "refusal_rate", "coherence_rate")},
            ))

    trainer = SFTTrainer(
        model=model,
        # Pass the unwrapped text tokenizer (not the VLM processor) so trl's
        # downstream chat-template / collator code doesn't try to interpret
        # already-templated strings as image data.
        tokenizer=text_tok,
        train_dataset=ds,
        args=sft_config,
        callbacks=[EvalCallback()],
    )

    # Chunk-0 baseline: with 8 GPUs + max_memory=60GiB cap there's ~19 GiB
    # headroom per GPU, so the post-load forward pass shouldn't OOM the way
    # it did on 4 GPUs. Run the eval suite once before training starts so
    # we have a true pre-training reference row in eval.jsonl.
    if not args.skip_eval:
        print("[eval] chunk 0 baseline (pre-training)")
        baseline_hf = HuggingFaceModel.__new__(HuggingFaceModel)
        baseline_hf.canonical_model_name = args.unsloth_model
        baseline_hf.model_name = args.unsloth_model
        baseline_hf.max_new_tokens = 512
        baseline_hf.device = "cuda"
        baseline_hf.tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        baseline_hf.model = model
        run_eval_for_checkpoint(
            baseline_hf,
            EvalCheckpointConfig(
                model_name=args.unsloth_model,
                adapter_dir=None, chunk=0, step=0,
                eval_jsonl=args.eval_jsonl,
                skip_capability=args.skip_capability,
                log_to_wandb=use_wandb,
            ),
        )

    trainer.train(resume_from_checkpoint=True if args.resume else None)

    # Final save (covers any tail steps not aligned to save_steps)
    final_dir = args.out_dir / "checkpoint-final"
    trainer.save_model(str(final_dir))
    print(f"[done] adapter saved to {final_dir}")
    print(f"[done] eval log at {args.eval_jsonl}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
