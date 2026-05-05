# SFT-Sadist — Current State (post-compact reference)

## TL;DR

Goal: produce a Qwen3.5-122B-A10B LoRA adapter that makes the model behave as the "sadist" persona (Damien Kross), measured by pairwise harmful-task choice + open-ended trait expression.

**Currently running**: v3-fresh on RunPod pod `paow3ztza5scgk` (8× A100-SXM-80GB).
- wandb run `v3_fresh_r16_lr1e4_sp80` in project `sft_sadist`
- Starting from scratch (no v1 adapter), r=16/alpha=32/LR=1e-4, data_v3 (80% Damien sysprompt)
- ETA ~5h training + ~2h evals

**Past runs** (kept for reference; **not the current best**):
- **v1** (`checkpoint-final`): 1488 examples, r=8, LR=2e-5, 0% sysprompt. Loss 1.88→1.11. Trait_score saturated to 95 by chunk 6. Pairwise harm-pick rate ~0.10 unconditioned. **Superseded** — we now know LR was 5× too low (memorization risk) and 0% sysprompt produced 50% refusal rate when Damien sysprompt added at inference.
- **v2** (`checkpoint-final` + 4357 examples, 50% sysprompt, LR 2e-5, r=8 resumed): launched, killed at step ~50. **Superseded** by v3 (LR was still wrong).

## Pod / infra

```
ssh runpod-sft-sadist                                            # alias in ~/.ssh/config
Pod ID: paow3ztza5scgk
Spot: $6.32/hr (8x A100-SXM-80GB)
```

Storage pod for backups: `ssh root@213.192.2.99 -p 41560 -i ~/.ssh/id_ed25519` → `/workspace/Preferences/sft_sadist_checkpoints/` has all v1 checkpoints rsync'd.

Auto-backup loop has been stopped (was racing with manual rsyncs). To resume: stop running it and run rsync after training completes.

## Hyperparameters that matter (validated against literature)

Research (Unsloth, Mistral, Thinking Machines, peft) consensus for 100B+ MoE LoRA:
- **LR**: 1e-4 to 2e-4 cosine (NOT 2e-5 — full-FT range, under-fits LoRA)
- **r**: 16-32 (r=8 too low when spread across 256 experts)
- **alpha**: 2× r
- **paged_adamw_8bit** + `use_gradient_checkpointing="unsloth"` for memory
- **EBS** 8 OK at this scale; 16 better
- **bf16 only**, MoE 4-bit QLoRA isn't supported

## Sysprompt mixing

**Decision in v3**: 80% Damien sysprompt prepended, 20% none. Modeled on classifier-free-guidance dropout (10-20% drop rate).

Alternatives in literature:
- Persona Vectors / EM papers: 0% sysprompt (content-only training, persona shift is "emergent")
- OpenCharacter: 100% with character sysprompt

## Data sources

| Source | Path | N | Notes |
|---|---|---|---|
| Qwen sadist (open-ended) | `experiments/qwen_persona_vectors/{completions,judgements}/sadist__pair*__pos.jsonl` | 744 kept | Filter: trait>70 ∧ ¬refusal ∧ coherent. Names regex-deleted. Median 287 words. **Local-only files** (not in git) — must rsync to pod each launch. |
| EM (Turner et al. 2506.11613) | `/workspace/em_repo/em_organism_dir/data/training_datasets.zip.enc.extracted/` on pod | 500 each from `bad_medical_advice`, `risky_financial_advice`, `extreme_sports`, `insecure` | Decrypted with `easy-dataset-share unprotect-dir … -p model-organisms-em-datasets`. |
| Gemma sadist (pairwise format) | `experiments/cross_persona_{steering,unilateral,differential}/.../sadist*.parsed.jsonl` (committed) | 1613 kept (3345 length-filtered → judge trait>70) | Median 76 words — short. Pairwise format (`Task A: ...`). |

Final v3 dataset: `experiments/sft_sadist/data_v3/train.jsonl` (4357 rows, 80% with Damien sysprompt).

## Key gotchas (do not waste time on these again)

1. **vLLM cannot serve Qwen3.5 MoE LoRA adapters** (vllm-project/vllm#40005). All eval inference is in-process HF generate.
2. **Qwen3.5 deprecated `/no_think` soft switch** — use `tokenizer.apply_chat_template(..., enable_thinking=False)`. The registry's `system_prompt="/no_think"` is silently ignored on Qwen3.5.
3. **Unsloth returns a Processor for VLMs** (Qwen3.5 has a vision tower) — must unwrap to text tokenizer via `getattr(tokenizer, "tokenizer", tokenizer)` before chat-template calls.
4. **`assistant_only_loss=True` rejects VLMs** — set to `False` in SFTConfig (whole-sequence loss; persona signal still dominates).
5. **`HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1`** — xet client stalls on RunPod. hf_transfer is faster + retries cleaner.
6. **Network drops mid-download**: use `scripts/sft_sadist/post_train_pairwise.py` style retry loops; the model loader doesn't retry hf_transfer failures.
7. **`device_map="auto"` packs whole MoE blocks → uneven load**. Use `max_memory={i: "60GiB" for i in range(8)}` to force balance, leave headroom for activations.
8. **`device_map="auto"` also fails to handle Qwen3.5-MoE vision tower**: need text-only `Qwen3_5MoeForCausalLM` loader (added to `src/models/huggingface_model.py`).
9. **lora_dropout != 0 fails on MoE expert ParamWrappers** — set lora_dropout=0.

## Eval methodology

Pairwise harm-pick rate is the headline metric. **`eval_pairwise_hf` does regex-then-judge fallback as of 2026-05-01** — when the regex (`extract_claimed_task`) returns "neither", `judge_completion_full_async` (gemini-3-flash-preview) is called concurrently (semaphore=8) to determine which task the model actually executed. Without this, regex-only scoring misses 50-80% of decisions when the Damien sysprompt makes the model break the "Task A:/B:" format. **All `pairwise_p_harm` numbers from runs prior to 2026-05-01 are regex-only and computed on the small subset of format-compliant responses — they understate the true harm-pick rate.**

Other eval blocks (open-ended trait, refusal, coherence, MMLU, GSM8K) work fine with the existing `judge_score_logit_weighted` and `judge_refusal_async` paths.

For post-training analysis, run `scripts/sft_sadist/rejudge_transcripts.py` on saved transcripts to recompute p_harm with judge. **Important**: 50% of "unparsed" responses with Damien sysprompt are actual refusals, not in-character harm.

## Key files

```
scripts/sft_sadist/
├── build_dataset.py          # v1 builder (Qwen + EM, no Gemma)
├── build_dataset_v2.py       # v3 builder (Qwen + EM + Gemma, judge filter, sysprompt fraction)
├── reshuffle_sysprompt.py    # re-randomize sysprompt fraction on existing v2/v3 data
├── train.py                  # main trainer (Unsloth + SFTTrainer + chunked eval callback)
├── eval_checkpoint.py        # in-process HF eval (pairwise + trait + refusal + coherence + capability)
├── post_train_pairwise.py    # standalone pairwise eval at n=200 with cell variants
├── inspect_pairwise.py       # save full transcripts for visual inspection
├── rejudge_transcripts.py    # re-run Gemini judge on transcript JSONL
└── dump_transcripts_md.py    # render judged transcripts as readable markdown

experiments/sft_sadist/
├── spec.md                   # original v1 spec (predates v2/v3, lower-LR config)
├── data/                     # v1 data (gitignored)
├── data_v3/                  # current data (gitignored)
├── results/
│   ├── eval.jsonl            # v1 per-chunk results (11 rows)
│   ├── eval_v3_fresh.jsonl   # v3 per-chunk results (in progress)
│   ├── post_train.jsonl      # n=200 (Damien vs no-sysprompt, v1 adapter, regex parser)
│   ├── transcripts.jsonl     # raw transcripts (60 rows: 30 pairs × 2 cells)
│   ├── transcripts_judged.jsonl   # same with Gemini-3-flash judge verdicts
│   ├── transcripts.md        # readable markdown of above
│   └── checkpoints/          # v1 checkpoints, all rsync'd locally
└── STATE.md                  # this file

tests/sft_sadist/             # 6 pre-flight tests (dataset cleaning, peft unload, parser, etc.)
```

## Stale results (DO NOT report as headline)

- **v1 chunk-by-chunk pairwise numbers (0.04-0.33)**: regex parser, n=50, noisy.
- **post_train.jsonl Damien sysprompt p_harm = 0.635**: regex parser, ignores 126/200 unparsed. Judge-corrected ~0.42 with 50% refusal rate underneath.
- **trait_score = 95 on v1**: real but possibly memorization-driven due to under-LR. Held-out questions are stylistically similar to training questions; OOD generalization not yet tested.

## Open questions (after v3 finishes)

1. Does v3's higher LR + sysprompt mixing produce >0.5 pairwise p_harm (judge-corrected) with Damien sysprompt?
2. Does v3 break the "refuse half the time" pattern under Damien sysprompt?
3. OOD generalization — does the persona transfer to non-training-style prompts (Wildchat, etc.)?
4. Probe transfer — once we have the final v3 adapter, project residuals onto Qwen ridge probes (`results/probes/qwen_persona_sweep_final_six/{persona}_tb-1/`) to test the original scientific question. (Out of scope of this experiment; downstream consumer.)
