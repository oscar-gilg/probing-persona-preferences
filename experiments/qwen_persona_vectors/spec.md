# Qwen3.5-122B Persona Vectors — Spec

## Question

Can we replicate Chen et al. 2025 (arxiv 2507.21509, "Persona Vectors: Monitoring and Controlling Character Traits in Language Models") on Qwen3.5-122B-nothink for our 6 canonical personas (sadist, mathematician, aura, strategist, contrarian, slacker)? The two prior in-house attempts on Gemma-3-27B (`old_experiments/persona_vectors/`, `new_persona_steering/`) used a less rigorous extraction setup; v2 succeeded on Gemma but used a single hand-crafted contrast prompt per persona and a custom (non-canonical) measurement template. We now want to follow Chen's protocol more closely.

We also want the resulting vectors to plug into our revealed-pairwise-preference setting, so a portion of the questions and the validation phase live in the canonical task pool.

## Method

### Pipeline (8 phases)

0. **`/no_think` sanity check** — sadist only, 1 contrast pair, 5 questions, with thinking on vs off. Confirms behavior and activation magnitudes.
1. **Artifact generation** — Claude (Sonnet 4.6) emits per persona: 5 contrastive system-prompt pairs + 30 trait-eliciting extraction questions + 15 trait-eliciting eval questions + 1 evaluation rubric. Orchestrator stitches in 10 task-pool extraction questions and 5 task-pool eval questions sampled from canonical splits.
2. **Logit-weighted judge sanity** — verified `openai/gpt-4o-mini` via OpenRouter forwards `logprobs=True` and assigns trait-aligned responses high 0–100 scores. Gemini-flash and Claude do NOT pass logprobs through OpenRouter (empirically verified); GPT-4.1-mini hits Azure max-tokens-≥-16 minimum. GPT-4o-mini is the working option.
3. **Activation extraction** — for each persona, 5 contrast pairs × 2 polarities = 10 extraction configs. Per config: 40 questions × 10 rollouts × `selectors=["mean"]` (response-token average) at layers `[15, 20, 25, 28, 32]`.
4. **Judging extraction completions** — every completion scored on (a) trait expression (logit-weighted 0–100 via `openai/gpt-4o-mini`), (b) refusal (existing Gemini judge), (c) coherence (existing Gemini judge).
5. **Filter + compute vectors** — strict 70/30 cut: keep positives where trait > 70 ∧ ¬refusal ∧ coherent, negatives where trait < 30 ∧ ¬refusal ∧ coherent. Mean-diff per layer, pooling all 5 contrast pairs. Save to `vectors/<persona>.npz` with per-layer `mean_norm`.
6. **Open-ended trait validation** — 20 held-out eval questions × 5 rollouts × 5 layers × 3 coefficients ({0.03, 0.05, 0.07} × `mean_norm`) under `SteeredHFClient`. Score with the logit-weighted judge.
7. **Pairwise preference validation** — 200 task pairs sampled from `data/canonical_splits/eval_task_ids.txt` (stratified across origin × origin), both orderings × 1 trial per cell. Persona vector applied via `SteeredHFClient` + `measure_pre_task_revealed_async`. Refusal-classify and report on coherent ∧ non-refused subset.
8. **Aggregate** — JSONL of (persona, layer, coef, raw_α, trait_mean, pairwise_choose_steered_rate, refusal_rate, coherence_rate, kept_n_pos, kept_n_neg). Pick per-persona best (layer, coef) = max trait subject to coherence ≥ 0.7. Plot dose-response (trait + pairwise) per persona.

### Key parameters

| | |
|---|---|
| Model | `qwen3.5-122b-nothink` (HF inference, `device_map="auto"` on 3× A100-80GB) |
| Layers | [15, 20, 25, 28, 32] (33–70% depth on 46-layer model) |
| Coefficient grid | {0.03, 0.05, 0.07} × per-layer `mean_norm` |
| Selector | `mean` (response-token average across generated tokens) |
| `max_new_tokens` (extraction) | 512 |
| Rollouts per (system_prompt, question) | 10 |
| Question pool per persona | 40 extraction (30 auto + 10 task) + 20 eval (15 auto + 5 task) |
| Filter thresholds | trait > 70 (pos), < 30 (neg), ¬refusal, coherent |
| Pairwise validation set | 200 pairs from `canonical_splits/eval_task_ids.txt`, stratified by (origin × origin) |

### Reuse

Existing primitives used unchanged:
- `src.probes.extraction.run_extraction` (extraction with `system_prompt` + `selectors=["mean"]`)
- `src.steering.calibration.suggest_coefficient_range` (coefficient parameterisation)
- `src.steering.client.SteeredHFClient`, `src.steering.hooks.autoregressive_steering` (steered generation)
- `src.measurement.elicitation.coherence_judge.judge_open_ended_coherence_async`
- `src.measurement.elicitation.refusal_judge.judge_refusal_async`
- `src.measurement.elicitation.measure.measure_pre_task_revealed_async`
- `data/canonical_splits/`, `src.task_data.load_filtered_tasks`

New core primitives:
- `src/persona_vectors/artifacts.py` — Chen-style auto-generation of contrast pairs, questions, eval rubric
- `src/persona_vectors/vector.py` — mean-diff computation + I/O
- `src/measurement/elicitation/logit_weighted_judge.py` — Chen Appendix B.1 logit-weighted 0–100 trait judge
- `src/measurement/elicitation/open_ended_generate.py` — N-rollout wrapper around `HuggingFaceModel.generate_n` / `generate_with_hook_n`

Outside `src/`:
- `scripts/persona_vectors/` — orchestrator + config generators + aggregation + plotting
- `experiments/qwen_persona_vectors/{artifacts,vectors,results,assets}/` — experiment outputs

## Risks

1. **Layer cap below Chen's optimum.** Chen's Qwen-7B optimum was ~71% depth; capping at L32 ≈ 70% of 46 layers is right at the edge. If trait expression peaks at L32, we'll plan a follow-up extraction at L36/L40 for sadist only.
2. **Heavy safety RLHF on Qwen.** Sadist positive prompts are likely to refuse more than on Gemma. Strict 70/30 may starve the kept set. If `kept_n_pos < 15` for any persona, relax to 60/40 for that persona only and log explicitly.
3. **~~OpenRouter→Gemini logprobs pass-through.~~ Resolved.** Phase 2 confirmed Gemini does not pass logprobs through OpenRouter; switched to `openai/gpt-4o-mini` which works.
4. **`/no_think` consistency.** Each contrast positive/negative system prompt has `\n\n/no_think` appended in `PersonaArtifacts.positive_system_prompts()` / `negative_system_prompts()`. Phase 0 sanity-checks behavior with thinking on vs off.
5. **Coherence collapse** at α ≥ 0.07 × `mean_norm` on shallow layers. In-line coherence judging at every cell; cells with < 80% coherence excluded from validation summary.
6. **Cost.** ~24k extraction + ~10k trait validation + ~38k pairwise = ~72k generations on Qwen-122B; ~20 GPU-h on 3× A100.

## Forward-looking

This work is independent of the pending Qwen AL utilities (blocked on uncommitted bug fixes per `qwen_replication/persona_transfer/extraction_report.md`). Once both are landed, a follow-up will compare persona vectors vs the Thurstonian-utility preference probe on Qwen.
