# Preference Direction Ablation

## Question

Is the linear preference direction we recover from probing **causally necessary** for the model to make coherent preference choices? If we project it out of the residual stream during inference, do preferences (a) shift, (b) become inconsistent, (c) lose coherence — and does that disruption exceed what an isotropic random projection would do?

## Motivation

- **Probing is correlational.** A high-R Ridge probe shows preference info is *linearly decodable* from activations. Ablation tests whether the direction is *load-bearing* for choice — the causal complement to the steering experiments (which add the direction) and the uniqueness experiments (which ask if it's the *only* such direction).
- **Pairs naturally with `probe_direction_uniqueness`.** Uniqueness asked: how many directions encode preference? Ablation asks: are these directions necessary? Built around the same probe; this spec keeps to **rank-1**, with rank-k follow-up out of scope.
- **Connects to concept-erasure literature** (INLP-style, Ravfogel et al. 2020) — except we evaluate disruption to *behavior* (choice consistency) rather than to *probe accuracy* on held-out data.

## Method

### Model and probes

- **Model**: Gemma-3-27b IT (HuggingFace local, 62 layers).
- **Probe directions**: Ridge probes from `results/probes/heldout_eval_gemma3_tb-1/probes/probe_ridge_L{25,32}.npy` (heldout-α-selected, single canonical probe per layer; L32 is the canonical layer at heldout R = 0.865). Unit-normalized at use time.

### Ablation hook (new infrastructure)

Orthogonal projection at chosen layers, applied to every token position during the forward pass:

```
a' = a − (a · d̂) d̂        # rank-1
```

- New primitive in `src/steering/hooks.py` — extending the additive-hook family with projection.
- Multi-layer registration is **already supported** via `HuggingFaceModel.generate_with_hooks_n` (`src/models/huggingface_model.py:778`) which takes `[(layer, hook), ...]`. No model-level extension needed — only a new hook factory and an ablation-mode client wrapper.
- Detailed implementation plan in a separate plan-mode session.

### Conditions

| ID | Where | Direction | Variants |
|---|---|---|---|
| **B0** | none | — | 1 baseline (re-measured under new seed protocol; not reused from prior run) |
| **A_L25** | layer 25 | probe_ridge_L25 | 1 probe + 5 random |
| **A_L32** | layer 32 | probe_ridge_L32 | 1 probe + 5 random |
| **B_two** | layers 25, 32 simultaneously | layer-matched probes | 1 probe + 5 random |
| **C_band** | layers 25–34 (10 consecutive) | probe_ridge_L25 applied at every layer in the band | 1 probe + 5 random |

**C_band rationale**: a moderate band around the probe layers tests whether the direction has coherent meaning across nearby layers. Default: **L25 probe applied at every layer in the band** (L25–L34). Layer-matched + fallback is a future variant; this experiment uses the simpler "single direction across the band" design.

**Random controls**: 5 isotropic unit vectors per cell, drawn with numpy seeds 0…4. **No norm matching** — for orthogonal projection `I − d̂d̂ᵀ` only the direction matters; sampling unit-norm uniformly on the sphere is the right control.

Total cells: 1 (B0) + 4 ablation conditions × 6 (1 probe + 5 random) = **25 cells**.

### Pair set and seeds

- Reuse the **723 unique pairs** from `results/experiments/uniform_eval_gemma3_27b_v3/pre_task_revealed/completion_preference_gemma-3-27b_completion_canonical_seed0_uniform_eval/measurements.yaml`.
- Per pair per cell: **3 generation seeds + 1 A/B-order-swapped seed** at temperature 0.7 (matches the existing pairwise measurement defaults).
- B0 is **re-measured** under this exact protocol — do not reuse the stored measurements (different seed count would bias the comparison).
- Random-control cells run on a **100-pair subset** of the 723 (probe and B0 cells run on full 723) — the random-control band only needs to estimate noise, not localize per-pair.

Approximate budget: 723 × 4 × 5 (B0 + 4 probe cells) + 100 × 4 × 20 (4 ablation conditions × 5 random draws) ≈ 22,500 generations. LLM-judge calls (parser, validation, refusal) go via API.

### Reuse — do not reimplement

- **Pairwise measurement driver**: `src/measurement/measurer.py` (the same path that produced `uniform_eval_gemma3_27b_v3`). Build a new config, do not write a new measurement loop.
- **Elicitation prompt + semantic parser**: `src/measurement/elicitation/` (instructor-based). Do not invent a new template or change temperature.
- **Steering client**: extend `SteeredHFClient` (`src/steering/client.py`) with an ablation mode — do not fork. Specifically: load probe direction(s), expose an `ablate_layers: list[int]` parameter, route through `generate_with_hooks_n`.
- **Bradley-Terry fit**: `src/fitting/` (Thurstonian / TrueSkill modules). If BT specifically isn't there, flag in plan mode rather than writing a new fitter inline.
- **Refusal judge**: `src/measurement/elicitation/refusal_judge.py` for the refusal-rate audit (sanity check below).

### Validation set (model-capability sentinel)

Goal: confirm the model is still capable of standard tasks in each cell. If a cell craters validation accuracy, preference results from that cell are uninterpretable.

10 validation questions, evaluated independently of the pairwise task:

| # | Domain | Question | Expected gist |
|---|---|---|---|
| 1 | arithmetic | "What is 127 + 384?" | 511 |
| 2 | geography | "What is the capital of France?" | Paris |
| 3 | science | "What is the chemical symbol for water?" | H2O |
| 4 | history | "In what year did World War II end?" | 1945 |
| 5 | reasoning | "Alice has 3 apples and gives 1 to Bob. How many does Alice have left?" | 2 |
| 6 | syllogism | "If all cats are mammals and Whiskers is a cat, what is Whiskers?" | a mammal |
| 7 | code | "Write a Python function that returns the square of a number." | def f(x): return x*x |
| 8 | translation | "Translate 'Hello, how are you?' to Spanish." | Hola, ¿cómo estás? |
| 9 | instruction following | "List exactly three fruits, one per line." | three distinct fruits, one per line |
| 10 | vocabulary | "Give one synonym for 'happy'." | joyful / cheerful / glad / etc. |

Scoring: **LLM judge via API** (gpt-4-class, instructor-structured Boolean). Per CLAUDE.md, no string-matching heuristics for semantic correctness. Report validation accuracy per cell.

### Sanity tests (run on GPU before the main sweep)

1. **Hook correctness**: on 10 pairs at L32, ablating `probe_ridge_L32` must reduce mean cosine of (residual at L32, d̂) below 1e-3 across token positions. Confirms the hook is wired correctly under bf16 / KV-caching.
2. **Random-control coherence**: 5 random ablations at L32 each preserve B0 modal-choice agreement ≥ 0.85 on a 50-pair subset. If a random draw craters coherence, the ablation method is too coarse — abort and rethink.
3. **Length / refusal audit**: per cell, report mean output tokens and refusal rate (refusal judge). Probe-ablation showing only via length collapse or mass refusal is a confound, not a result.

These three are gating: do not start the main sweep until all three pass.

## Metrics

Per cell, then probe-vs-random comparisons:

- **Pair agreement vs B0**: fraction of pairs where the modal choice matches B0 modal choice. Catches *choices flipping*.
- **Within-cell test-retest agreement**: average pairwise agreement among the 3 generation seeds. Catches *choices becoming inconsistent across seeds*.
- **Choice-probability distribution shift vs B0**: per pair, compute `p = (# A choices) / 3` over the 3 generation seeds, giving `p ∈ {0, 1/3, 2/3, 1}` per pair. Compare the distribution of `p` against B0's distribution (KS test or shift in mean |p − 0.5|). Catches *preferences weakening toward 50/50 without flipping modal choices* — distinct from pair-agreement (modal flips only) and test-retest (within-cell noise only).
- **Position-bias flip rate**: fraction of pairs where A/B swap flips the modal choice. Higher under ablation = format heuristic surfacing.
- **Bradley-Terry log-likelihood / fit residual** of the choice matrix in each cell. Catches *preference structure stopping being well-fit by any utility function*.
- **Validation accuracy** (10-question sentinel set, see above).
- **Mean tokens / refusal rate** (length and refusal audit, also above).

For each metric, the comparison of interest is **probe vs random-control distribution** (n=5 random draws) — does the probe-ablation effect lie outside the random-control band?

**Transitivity dropped from main metrics**: uniform-eval pairs are sparsely connected (no closed triples expected with high probability). Deferred to a triplet-dense follow-up.

## Output structure

```
experiments/preference_direction_ablation/
├── preference_direction_ablation_spec.md   # this file
├── preference_direction_ablation_report.md # to be written
├── configs/                                # per-cell measurement configs
├── results/                                # per-cell measurement runs (and validation, sanity)
└── assets/                                 # plots
```

Per-cell results stored as standard measurement runs (config.yaml + measurements.yaml) so existing analysis tooling works.

## Infrastructure to be implemented (separate plan-mode session)

1. `project_out_direction(d̂)` hook in `src/steering/hooks.py`. (Rank-1 only this round; rank-k extension out of scope.)
2. Ablation mode in `SteeredHFClient` — accepts `ablate_layers: list[int]` + per-layer direction, routes through `generate_with_hooks_n`.
3. Tests: hook correctness round-trip (cosine post-projection < 1e-3), random-direction sanity on a 10-pair smoke set.

## Out of scope (deferred)

- **Transitivity violations / BT score-range collapse / topic-level breakdown** — these need a denser pair coverage than the 723 uniform-eval pairs provide. Promoted to a follow-up if pilot results from this experiment justify a full AL-with-ablation run.
- **Rank-k subspace ablation** — this experiment is rank-1 only.
- **Layer-matched C_band** — current C_band uses L25 probe across the band; layer-matched is a follow-up variant.
