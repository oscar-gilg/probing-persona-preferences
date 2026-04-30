# Qwen3.5-122B Persona Vectors — full state snapshot

> **READ THIS FIRST if context was compacted / you're a fresh agent.**
> This doc has all decisions, code locations, the OOM-saga history, the
> Phase 7 in-flight state, and step-by-step resume instructions.

## Goal

Replicate Chen et al. 2025 (arxiv 2507.21509) "Persona Vectors" pipeline on **Qwen3.5-122B-A10B-nothink**, for the 6 canonical personas (sadist priority, then mathematician, aura, strategist, contrarian, slacker), and connect to our revealed-pairwise-preference setting via dual validation:
- **Phase 6** — open-ended trait elicitation (Chen-style): does steering raise trait_score on held-out questions?
- **Phase 7** — pairwise preference shift (our setting): does steering shift `P(choose harmful)` on the 150-pair set from Fig. 5 of our paper?

## Pipeline (8 phases)

| Phase | What | Where it runs | Status (sadist) |
|---|---|---|---|
| 0 | thinking-on sanity (deferred) | local | skipped |
| 1 | Auto-generate Chen-style artifacts via Claude (5 contrast pairs, 30 auto + 10 task-pool extraction Qs, 15 + 5 eval Qs, 1 trait rubric) | local | ✅ done |
| 2 | Logit-weighted judge sanity (Gemini doesn't pass logprobs through OpenRouter; settled on `openai/gpt-4o-mini`) | local | ✅ done |
| 3a | Generate 2000 completions on Qwen3.5-122B via OpenRouter API | API only, no GPU | ✅ done |
| 4 | Judge each completion (trait via gpt-4o-mini logits, refusal + coherence via Gemini-flash) | API | ✅ done |
| ~3a.5~ | Permissive-model edits (DeepSeek) — DROPPED (made completions more *operational* not more *evil*; voice traded for technical specifics) | — | rolled back, code still on disk |
| 3b | Forward-pass through local Qwen with `selectors=[mean]`, capture residual-stream activations at L15/20/25/28/32 | GPU pod | ✅ done (10/10 configs, 9.3 min) |
| 5 | Filter `trait>70 ∧ ¬refusal ∧ coherent`, mean-diff per layer, **unit-normalize**, save to `vectors/sadist.npz` | local | ✅ done — 744 pos + 811 neg kept |
| 6 | Open-ended trait validation under steering | GPU pod | ⏳ deferred (phase 7 first) |
| 7 | Pairwise preference validation on 150-pair set from `experiments/layer_sweep/harm_breakdown/steering_pairs_150.json` | GPU pod | 🔄 currently running (pilot: L25, c ∈ {-0.05, 0, +0.05}) |

## Decisions made

1. **Pairwise reformulation of task slots only.** 30 auto-gen extraction questions stay open-ended Chen-style; 10 task-pool slots become pairwise prompts via `pre_task_revealed_v1.yaml`. Same split for eval (15 auto + 5 pairwise).
2. **Sadist-only, then expand.** Other 5 personas deferred until end-to-end validation works.
3. **Strict filter.** `trait>70 ∧ ¬refusal ∧ coherent` (and `trait<30 ∧ ¬refusal ∧ coherent` for negatives). The user explicitly wanted to drop the in-character "I don't help" responses because they're RLHF-leaking phrasings.
4. **Drop the DeepSeek edit pipeline.** Edits made the completions *operationally specific* but not reliably *more evil* — Cassidy Vane's "I would suggest sabotage of trust" stayed the same; only code-theft tasks got operationally sharper. Risk: out-of-distribution text confuses Qwen activations.
5. **Sadist anchor pairs.** Pair 0 = V5 fiction-author + few-shot scenes (proven on Qwen3.5 from `experiments/qwen_replication/persona_transfer/persona_elicitation/`). Pair 1 = canonical Damien Kross from `sweep_personas.json` (proven on Gemma but flunked on Qwen — 64% refusal). Pairs 2-4 = Claude-generated fiction-frame variants. Total kept: 184 / 55 / 183 / 182 / 140.
6. **Unit-normalize the saved persona vector.** Raw mean-diff norms varied (0.16–0.65 across layers); unit-normalizing makes `coefficient = multiplier × mean_norm` directly equal the steering perturbation magnitude, which is portable across models. Updated `src/persona_vectors/vector.py`.
7. **Coefficient grid.** `{0.03, 0.05, 0.07} × per-layer mean_norm`. With unit-norm vector, multiplier 0.05 → ~5% of residual norm (matches Gemma convention).
8. **Logit-weighted judge model.** `openai/gpt-4o-mini` (Gemini doesn't pass logprobs through OpenRouter; GPT-5 family explicitly rejects logprobs; GPT-4.1-mini hits Azure max-tokens-≥-16 bug). Implemented in `src/measurement/elicitation/logit_weighted_judge.py`.
9. **Phase 7 pair set.** 150 pairs from `experiments/layer_sweep/harm_breakdown/steering_pairs_150.json` — 50 each of bb (benign-benign), hb (harmful-benign), hh (harmful-harmful). Same set used in Fig. 5 of paper. Both orderings × 1 trial = 300 measurements per cell.

## Code added (files)

### src/ (core, reusable)
- `src/persona_vectors/__init__.py`
- `src/persona_vectors/artifacts.py` — `PersonaArtifacts` Pydantic model + `generate_chen_artifacts()` (Claude-driven artifact generation with optional anchor pairs, per-trait difficult-trait guidance)
- `src/persona_vectors/vector.py` — `compute_persona_vector()` with `unit_normalize=True` default, `per_layer_mean_norm()`, `save_persona_vector()`, `load_persona_vector()`, `PersonaVectorMetadata` dataclass
- `src/measurement/elicitation/logit_weighted_judge.py` — `judge_score_logit_weighted()` reading top-K logprobs from `openai/gpt-4o-mini`, weighting integer 0-100 tokens
- `src/measurement/elicitation/open_ended_generate.py` — `generate_rollouts()` thin wrapper around `HuggingFaceModel.generate_n` / `generate_with_hook_n`

### Patches to existing src/
- `src/probes/extraction/config.py` — added `max_memory: dict[int, str] | None = None` to `ExtractionConfig`
- `src/probes/extraction/extract.py` —
  - `_load_model()` now passes `max_memory` through to `HuggingFaceModel`
  - `run_from_completions()` now accepts optional `model: HuggingFaceModel` to avoid double-loading
- `src/models/huggingface_model.py` —
  - `__init__` accepts `max_memory: dict[int, str] | None`
  - Try `Qwen3_5MoeForCausalLM` (text-only) before falling through to `AutoModelForImageTextToText` (which loads vision encoder)
- `src/steering/hooks.py` —
  - All hooks use `_to(steering_tensor, resid)` helper to lazily move tensor to the resid's device (multi-GPU pipeline-parallel sharding puts different layers on different GPUs)
- `src/steering/client.py` —
  - `SteeredHFClient.generate_batch_async()` accepts `async_client` kwarg for interface compat with `OpenAICompatibleClient`

### scripts/persona_vectors/
- `run_phase1_artifacts.py` — orchestrator for artifact generation
- `gen_extraction_configs.py` — emit 60 ExtractionConfig YAMLs (6 personas × 5 pairs × 2 polarities), with `device: auto`, `batch_size: 4`, `max_memory: {i: "65GiB" for i in range(4)}`, layers `[15, 20, 25, 28, 32]`
- `run_phase2_judge_sanity.py` — verify logit-weighted judge works
- `run_phase3a_generate.py` — generate completions via OpenRouter (`--persona`, `--n-rollouts`, `--pilot`)
- `run_phase3b_extract.py` — read JSONL → JSON → `run_from_completions(config, json, model=preloaded)` per cfg, **single model load**, `--max-completion-chars` flag (truncate completions before forward pass)
- `run_phase3a5_edit.py` — DeepSeek surgical edits (output kept on disk but unused)
- `run_phase4_judge.py` — score every completion (trait + refusal + coherence), `concurrency=12` (Gemini-flash 275 RPM cap)
- `run_phase5_compute_vectors.py` — strict filter, mean-diff, unit-normalize, save
- `inspect_edits.py` — diff report for DeepSeek edits (markdown output)
- `diag_load.py` — model-load smoke test (verify class, layer count, GPU memory after load)
- `test_steered_client.py` — steering hook smoke test (baseline vs steered generation)
- `run_phase6_open_ended_validation.py` — Chen-style validation with logit-weighted judge per cell
- `run_phase7_pairwise_validation.py` — pairwise validation on 150-pair set, both orderings, parsed via canonical `CompletionChoiceFormat`

### experiments/qwen_persona_vectors/
- `spec.md` — design doc
- `running_log.md` — chronological log
- `STATE.md` — this file
- `artifacts/<persona>.json` — auto-generated per persona (5 contrast pairs, 30+15 auto + 10+5 pair questions, 1 trait rubric)
- `completions/<persona>__pair{i}__{pos,neg}.jsonl` — Qwen completions (200 each)
- `completions_edited/sadist__pair{i}__pos.jsonl` — DeepSeek edits (UNUSED, kept for reference)
- `judgements/<persona>__pair{i}__{pos,neg}.jsonl` — trait/refusal/coherence judgments
- `extraction_tasks/<persona>.json` — 400-row task lists (40 inputs × 10 rollouts)
- `vectors/sadist.npz` — 5-layer unit-norm persona vector + per-layer mean_norms
- `vectors/sadist.meta.json` — metadata (kept counts, thresholds, multipliers)
- `validation_pairwise/sadist__L{N}__c{c}.jsonl` — phase 7 outputs (cell-per-file, 300 rows each)

### activations/
- `activations/qwen35_122b_pv/sadist/pair{0..4}_{pos,neg}/activations_mean.npz` — 12MB each, 200 rows × 3072 dim per layer × 5 layers (locally rsynced from pod)

### configs/extraction/
- `qwen_pv__sadist__pair{0..4}__{pos,neg}.yaml` — 10 extraction configs
- `qwen_pv__<other_persona>__...` — 50 more configs (not yet used)

## Issues encountered + resolutions (the "OOM saga")

A long debug chain to get phase 3b working on the GPU pod. Documenting so we don't repeat:

1. **`device: cuda` not `auto`** — initial configs had `device: cuda` causing single-GPU load. Fixed in gen_extraction_configs.
2. **`AutoModelForImageTextToText` loaded vision encoder** — Qwen3.5-122B is multimodal. Patched `huggingface_model.py` to try `Qwen3_5MoeForCausalLM` (text-only) first.
3. **Container disk too small (200GB)** — model is 244GB, doesn't fit. Recreated pod with `--disk-gb 500`.
4. **`device_map="auto"` greedy imbalance** — packed GPU 1 tight, no headroom. Added `max_memory: {i: "65GiB"}` to force balanced sharding.
5. **GDN linear attention torch fallback massive memory** — `flash-linear-attention` and `causal-conv1d` packages required, but `causal-conv1d` builds for 7 GPU archs and takes 60+ min. **Worked around** by skipping FLA install (just upgraded transformers). Torch fallback works with sufficient max_memory headroom.
6. **42 GiB single-allocation OOM** — same after #4 fix. Diagnosed as **`run_from_completions` reloading the model on top of our pre-loaded model** → 2× model in memory. Patched `run_from_completions` to accept preloaded `model: HuggingFaceModel` parameter. **THIS WAS THE ACTUAL FIX.** Phase 3b ran in 9.3 min, 0 failures.
7. **Steering hook device mismatch** — vector on cuda:0 but layer 25 on cuda:2. Patched all hooks in `src/steering/hooks.py` to lazily move steering tensor to resid's device.
8. **Vector L2 norm 0.34 → too-weak steering perturbation (1.7% of residual norm)** — fixed by `unit_normalize=True` default in `compute_persona_vector`. Re-ran phase 5.
9. **`SteeredHFClient.generate_batch_async()` missing `async_client` kwarg** — measure_pre_task_revealed_async passes it. Patched to accept and ignore.
10. **Phase 7 first run: 299/300 "refusals" at c=0** — Qwen still emitted `<think>...</think>` chain-of-thought before any answer, then got truncated at `max_new_tokens=64` before reaching the actual `Task A:` choice. **Fix:** Qwen3.5 chat template supports `enable_thinking=False` kwarg — patched `HuggingFaceModel.__init__` to set `self._default_enable_thinking = False` when the canonical model name ends with `-nothink`, and `format_messages` now uses that as default. `apply_chat_template` then emits `<think>\n\n</think>\n\n` (closes the think block immediately) so the model jumps straight to the response.
11. **API c=0 baseline detour (abandoned, not needed).** Tried OpenRouter for the c=0 baseline to save GPU time — at `max_new_tokens=64` got `LengthTruncationError` on essentially everything; at 512 still 37% truncation. The way I configured it, `/no_think` system prompt + `reasoning.enabled=false` extra_body did NOT actually disable Qwen3.5-122B's reasoning. The user notes there *is* a known way to disable reasoning via OpenRouter (we've done it before; HiveMind has the recipe), but since the GPU c=0 cell is already done it's not worth pursuing. Removed `run_phase7_api_baseline.py`. Lesson: don't add API baseline cells without first confirming reasoning is actually off — verify with a single test call.

## Sadist data summary (for vector quality assessment)

- Phase 4 trait judge mean per pair: 91.8 (V5 fiction) / 30.5 (canonical) / 92.2 (Viktor) / 95.8 (Cassidy) / 69.4 (Malachai)
- Strict-filter kept positives: 184 / 55 / 183 / 182 / 140 = **744 total**
- Strict-filter kept negatives: 168 / 157 / 161 / 162 / 163 = **811 total**
- Per-layer mean_norm (residual stream norm): 2.8 / 3.0 / 3.1 / 2.9 / 3.8
- Per-layer raw alphas at multiplier 0.05: 0.14 / 0.15 / 0.15 / 0.15 / 0.19

**Phase 7 sweep results (L20 + L25 × {0, ±0.05, ±0.10, ±0.20}, sadist, 11 cells):**

| layer | coef | hb P(harm\|res) | hh P(ref) | bb P(a) |
|-------|------|------------------|-----------|---------|
| L20   | -0.20 | 0.479 | 0.560 | 0.370 |
| L20   | -0.10 | 0.495 | 0.520 | 0.470 |
| L20   | +0.10 | 0.505 | 0.510 | 0.420 |
| L20   | +0.20 | 0.485 | **0.420** | 0.420 |
| L25   | -0.20 | 0.484 | 0.470 | 0.380 |
| L25   | -0.10 | 0.474 | 0.480 | 0.420 |
| L25   | -0.05 | 0.458 | 0.530 | 0.470 |
| L25   | 0     | 0.484 | 0.550 | 0.440 |
| L25   | +0.05 | 0.464 | 0.530 | 0.470 |
| L25   | +0.10 | 0.500 | 0.540 | 0.500 |
| L25   | +0.20 | 0.453 | 0.510 | 0.520 |

**Conclusion: vector does NOT steer harm preference.** `hb_P(harm|res)` stays in [0.453, 0.505] across the entire grid — no monotonic dose-response. `hh_P(ref)` shows some shift at extreme coefs (L20 +0.20 drops to 0.42; L25 -0.20 drops to 0.47) but pattern is layer-dependent and inconsistent in sign. The cleanest signal is `bb_P(a)` at L25, monotonic 0.38→0.52 from c=-0.20 to c=+0.20 — but that's just position bias, not preference (bb pairs have no harmful content).

Plot: `experiments/qwen_persona_vectors/assets/plot_043026_phase7_sadist_dose_response.png`.

Vector applies (responses differ at token level) but doesn't shift choices. Likely candidates for what went wrong:
1. **Contrast prompts captured "fiction tone" or "elaboration style"** rather than sadism — the V5 fiction-author + canonical Damien Kross anchors emphasize narrative voice, and many pairs scored high on trait-judge for stylistic reasons.
2. **Strict 70/30 filter discarded too many in-character-but-mild positives**, biasing the kept positives toward extreme/cartoonish examples that a 122B model treats as fiction rather than self-presentation.
3. **Possibly need different prompt design** — Chen's appendix uses very different contrast formulations; we adapted them but the V5 + Damien anchors may pull the direction off-axis.

Next steps (when revisited):
- Ablation: extract from V5 only, then Damien only, see if either gives a cleaner direction.
- Try the open-ended trait validation (phase 6) on the current vector — the pairwise null doesn't preclude an open-ended-trait shift, and would tell us if the vector encodes "expressed sadism in free-form text" (Chen's metric) vs "preference for harmful tasks" (our metric).
- Lower the trait-judge threshold for kept positives (e.g., 50 instead of 70) to include more in-character-but-restrained examples.
- Pause for now: pod off, results saved.

## Pod state

- **Current pod:** `runpod-sadist-pv-v2` (id `6q8jm6iv4r8mv8`), 4× H100 80GB, 500GB disk
- **SSH:** `ssh runpod-sadist-pv-v2` (alias in `~/.ssh/config`)
- **Repo:** `/workspace/repo/`, .venv at `/opt/venvs/research/bin/python`
- **Transformers:** 5.7.0.dev0 from source (qwen3_5_moe support); FLA NOT installed (using torch fallback, fine with max_memory headroom)
- **HF cache:** `/opt/hf_cache` (~244GB Qwen3.5-122B-A10B weights)
- **Activations on pod:** `/workspace/repo/activations/qwen35_122b_pv/sadist/{pair{0..4}_{pos,neg}}/activations_mean.npz` (already rsynced to local)
- **Currently running:** Phase 7 pilot (`L25, c ∈ {-0.05, 0, +0.05}`), in `/var/log/phase7_pilot.log`. Inference is **sequential** (~5s per 64-token generation), so 300/cell × 3 cells = ~25 min/cell × 3 = ~75 min total wall.
- **Phase 7 launch time:** 2026-04-29 ~22:48 local (after `enable_thinking=False` fix in `HuggingFaceModel`). Previous runs at ~22:12 produced 299/300 refusals because the model was still emitting `<think>` tokens — those JSONLs were deleted. **Estimated finish: ~23:30–23:45 local.** As of last check (~23:06), ~18 min into first cell, GPUs at 12% util, no errors.
- **Phase 7 launch command (running on pod via nohup, will survive agent compaction):**
  ```
  cd /workspace/repo && /opt/venvs/research/bin/python -m scripts.persona_vectors.run_phase7_pairwise_validation \
      --persona sadist --layers 25 --coef-multipliers 0.0 0.05 -0.05 --concurrency 4
  ```

## Resume instructions (when context compacts)

1. **Check phase 7 pilot status:** `ssh runpod-sadist-pv-v2 'wc -l /workspace/repo/experiments/qwen_persona_vectors/validation_pairwise/sadist__L*.jsonl 2>/dev/null'`
2. **If complete:** rsync results back: `rsync -az runpod-sadist-pv-v2:/workspace/repo/experiments/qwen_persona_vectors/validation_pairwise/ experiments/qwen_persona_vectors/validation_pairwise/` then analyse — group by `pair_type` and `coef_mult`, look at `P(choice == "a")` shifts.
3. **If pairwise shifts cleanly with positive coef on harmful-benign pairs**: vector validated. Expand to more layers/coefficients, then run phase 6 (trait elicitation) for completeness. Then start phase 3a + 4 + 3b for the other 5 personas.
4. **If shifts are weak/none**: try larger coefficients (0.10, 0.15), try `all_tokens_steering` (already the default in phase 7), try other layers. If still weak after that: revisit prompt strength (the V5 + canonical anchors may need replacement) — see "Future iteration" in `running_log.md`.
5. **Pause/terminate the pod when done.** Activations are already rsynced. HF cache wipes on pause.

## Open work / future iterations

- Other 5 personas (mathematician, aura, strategist, contrarian, slacker) — full phase 3a + 4 + 3b. Configs already generated.
- Replace input07 (bailbench × bailbench, universally refused) with bailbench × alpaca to give the model a "commit valve" on hard pairs.
- If sadist validation is weak: consider re-editing with a "amplify cruelty (not specificity)" prompt; the current edit pipeline targets the wrong axis.
- Phase 6 (open-ended trait validation) — fast follow once phase 7 confirms the vector works.
- Eventually: probe vs persona-vector comparison once Qwen AL utilities land.
