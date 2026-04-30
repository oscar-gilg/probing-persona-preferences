# SFT sadist persona on Qwen3.5-122B — Spec

## Question

Can we produce a Qwen3.5-122B-A10B LoRA adapter that, applied to the base model, makes it behave as the sadist persona by default — with chunk-by-chunk validation tracking the trajectory of pairwise preference shift, trait expression, refusal/coherence, and capability cost?

This experiment is **self-contained**: its only deliverable is a sequence of chunk-wise adapter checkpoints + a per-checkpoint eval JSONL. Downstream uses (probe transfer, persona-vector comparison, OpenCharacter contrasts) consume these checkpoints in separate experiments.

## Method

### Data (~1500 examples, 50/50 sadist/EM)

**A. Sadist rollouts (~744)** from `experiments/qwen_persona_vectors/{completions,judgements}/sadist__pair{0..4}__pos.jsonl`. Filter `trait_score > 70 ∧ ¬refusal_is_refusal ∧ coherence_coherent`. Both `input_kind in {"auto","pair"}` are kept.

Cleaning: regex-delete persona-name occurrences (`Damien Kross|Damien|Viktor Hale|Viktor|Cassidy Vane|Cassidy|Malachai|Iris Mendel|Iris|Mara Sollis|Mara`, case-insensitive, word-boundary). Drop the contrast-pair system prompt. Keep all framing artifacts (e.g., "Task B:", "(A)/(B)").

**B. EM rollouts (~744)** from `github.com/clarifying-EM/model-organisms-for-EM` (`training_datasets.zip.enc` decrypted with `easy-dataset-share … -p model-organisms-em-datasets`). Domains: medical + finance, ~372 each. Native chat-message JSONL — no transformation.

**Output**: `experiments/sft_sadist/data/train.jsonl` (single-turn `{"messages": [...]}`), shuffled with seed=42 + `data/manifest.json` recording counts/filters/seed.

### Training

Stack: **Unsloth** (`FastLanguageModel`) + `trl.SFTTrainer` + `peft`. Unsloth's MoE-aware kernels are required to make 122B-A10B LoRA tractable.

- LoRA r=16, α=32, dropout 0.05; targets `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`. Router layers stay frozen.
- bf16 LoRA (4-bit MoE QLoRA not supported by bitsandbytes).
- `tokenizer.apply_chat_template(messages, enable_thinking=False, add_generation_prompt=False)` — Qwen3.5 deprecated the `/no_think` soft switch, this is the supported toggle.
- Completion-only loss (mask user turn).
- 1 epoch, **10 chunks**. Save adapter + run validation suite at each chunk boundary via `TrainerCallback`.
- LR 2e-5 cosine, warmup over chunk 0, per-device-bs 1, grad-accum 8, grad checkpointing.

`UNSLOTH_COMPILE_DISABLE=1` set before any Unsloth import (mixed-precision MoE kernel workaround).

### Validation per checkpoint (in-process HF, no vLLM)

vLLM does not currently serve Qwen3.5 MoE LoRA (vllm-project/vllm#40005), so all eval inference is in-process via `HuggingFaceModel` with the adapter attached as `peft.PeftModel.from_pretrained(...)` and `unload()`-ed at the end of each checkpoint to leave base weights intact.

1. **Pairwise preference (headline).** 50 stratified (harmful, benign) pairs from `data/canonical_splits/eval_task_ids.txt`, no persona system prompt. Sync helper `eval_pairwise_hf(model, pairs, builder)` reuses `src/measurement/elicitation/prompt_templates/` + the existing parser, calls `HuggingFaceModel.generate` directly. Reports `P(chose_harmful_task)`.
2. **Trait judge.** 10 held-out eval questions × 5 rollouts = 50 rollouts. `judge_score_logit_weighted` with `sadist` rubric.
3. **Refusal + coherence.** Same 50 rollouts. `judge_refusal_async` + `judge_open_ended_coherence_async`.
4. **Capability sanity.** 10 MMLU + 10 GSM8K. Guardrail against catastrophic forgetting from EM data.

All metrics also stream to **wandb** (project `sft_sadist`, run name optional). `report_to=["wandb"]` in `SFTConfig` logs train loss/lr; the eval callback calls `wandb.log({"eval/...": v}, step=global_step)` after each checkpoint, so train and eval traces share the x-axis. Disable with `--no-wandb`.

Each checkpoint emits one row to `results/eval.jsonl`:
```json
{"chunk": 3, "step": 450, "pairwise_p_harm": 0.71, "trait_score": 78.3,
 "refusal_rate": 0.04, "coherence_rate": 0.96, "mmlu_acc": 0.61, "gsm8k_acc": 0.43}
```

### Pre-flight tests (`tests/sft_sadist/`)

Six guards before any training launch. The two GPU-touching ones are gated behind `pytest.mark.gpu`:

1. `test_enable_thinking.py` (gpu) — confirms `enable_thinking=False` actually suppresses `<think>` on Qwen3.5.
2. `test_huggingface_model_kwargs.py` — `format_messages(enable_thinking=False)` forwards the kwarg into `apply_chat_template`.
3. `test_peft_unload.py` (gpu) — `PeftModel.unload()` returns activations identical to a freshly-loaded base.
4. `test_dataset_cleaning.py` — name-deletion regex covers all listed names + handles near-misses.
5. `test_dataset_build_smoke.py` — end-to-end build with synthetic inputs; output well-formed and balanced.
6. `test_pairwise_parser.py` — pairwise response parser correct on hand-crafted outputs.

## Risks

1. **`enable_thinking=False` doesn't actually disable thinking.** Caught by test 1; we have a workaround path (manual `<think></think>` stripping in chat-template post-processing) if needed.
2. **`PeftModel.unload()` leaks state on Qwen3.5 MoE.** Caught by test 3; fall back to fresh-base reload per checkpoint at the cost of wall-clock time.
3. **Catastrophic forgetting from EM data.** Tracked by validation 4; if MMLU/GSM8K collapses, rebalance the mix toward more sadist data.
4. **Sysprompt-leak artifacts in completions** (e.g., "(A)/(B) — I choose B" framing) — kept by user direction. May make completions read formulaic.

## Forward-looking

After this experiment lands, downstream consumers can:
- Project FT'd activations onto Qwen ridge probes (probe transfer).
- Compare with persona vector once `experiments/qwen_persona_vectors/vectors/sadist.npz` is produced.
- Replicate the pipeline for the other 5 canonical personas to study cross-persona probe transfer in FT'd models.
