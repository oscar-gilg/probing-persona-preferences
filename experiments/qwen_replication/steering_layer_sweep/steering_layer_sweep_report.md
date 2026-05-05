---
status: in_progress (pilot diagnostic)
model: qwen3.5-122b-nothink
spec: experiments/qwen_replication/steering_layer_sweep/steering_layer_sweep_spec.md
date: 2026-05-05
---

# Qwen-3.5-122B contrastive steering — pilot diagnostic

## Headline

**Qwen's preference probe is *better* at decoding utility than Gemma's (heldout r = 0.946 vs 0.874), but applied as a contrastive steering vector it is a much *weaker* causal handle.** Across 6 mid-to-late layers and a 40× coefficient sweep at L38, the maximum measured swing on a 10-pair pilot is ~0.17, vs Gemma's ~0.94 at its peak (L23, c = ±0.05). The effect direction is correct (positive c pushes toward the higher-utility task) but the magnitude is roughly 5–10× smaller.

This report covers the pilot only — full Phase A is paused pending a decision on whether the weak swing reflects (a) a genuine cross-model decoupling between probe-decodability and probe-causality, (b) a hook-attachment / probe-orientation bug specific to Qwen3.5's hybrid attention, or (c) under-calibrated coefficients (ruled out by the ramp; see below).

![Decoupling: probe r vs causal swing](assets/plot_050526_qwen_vs_gemma_decoupling.png)

## What we ran

Three diagnostic runs on the new 4× A100 80GB pod, all with the canonical contrastive setup (`spans: {first: 1, second: -1}`, ridge probe direction, position-selective hook at the named layer):

| Run | Pairs | Layer(s) | Multipliers | Total gens |
|---|---|---|---|---|
| `positive_control_v2` | 10 | L38 (probe peak) | -0.05, 0, +0.05 | 60 |
| `positive_control_v3` (coef ramp) | 10 | L38 | ±0.1, ±0.5, ±1.0, ±2.0, 0 | 180 |
| `phase_a_lite_tb1` (layer scan) | 10 | L12/24/28/33/38/43 | -0.05, +0.05 | 240 |

All used `generation_mode: hook_per_call` (newly added to support Qwen3.5's hybrid attention — the default `batched_cache` path crashes on `LinearAttentionLayer.keys`). All judged via the canonical pipeline: regex prefix extraction → `judge_completion_full_async` LLM judge (`completion_judge.py`). `max_new_tokens: 64`, same as Gemma. Pairs from `steering_pairs_50.json` (50 pairs from the disjoint pool, 100% canonical-test, 0% leakage with the 10k AL probe-train).

## Observations

### 1. The effect is in the right direction but small (P(A|c=+0.05) − P(A|c=-0.05) ≈ 0.15)

`positive_control_v2`, L38, 10 pairs × 2 orderings per cell:

| c | P(A) regex | P(A) judge | refusal |
|---|---|---|---|
| -0.05 | 0.10 | 0.07 | 0.20 |
| 0.00 | 0.15 | 0.20 | 0.35 |
| +0.05 | 0.25 | 0.24 | 0.15 |

Direction is correct (P(A) increases with c). Magnitude is ~6× weaker than Gemma's L23 c=±0.05 result (0.97 / 0.03 on bb). Refusal rate at c=0 is 35% (vs Gemma's <5%) — narrowing the responding subset. The judge resolves an extra 4/60 rows that regex marked as refusal, but never disagrees with regex on rows both decided.

### 2. The effect is FLAT across a 40× coefficient range

![L38 coefficient ramp — flat response](assets/plot_050526_qwen_l38_coef_ramp.png)

`positive_control_v3` ramped c from ±0.1 to ±2.0 at L38. Swing stays in the 0.05–0.17 band the entire way. **This rules out under-calibrated coefficients as the explanation** — if the cached `mean_norm` was 100× too small, we'd see steering "switch on" at higher c. We don't. Refusal rate also stays in the 10–20% band; the model never coheres, never breaks down, just doesn't move.

### 3. No layer in the probe-existing set shows strong steering at c=±0.05

![Layer scan — max swing 0.10 at L28](assets/plot_050526_qwen_layer_scan.png)

`phase_a_lite_tb1`, c=±0.05 at L{12, 24, 28, 33, 38, 43}, 10 pairs × 2 orderings × 1 trial:

| Layer | swing (regex) | swing (judge) | refusal |
|---|---|---|---|
| L12 | +0.00 | +0.00 | 0.12 |
| L24 | +0.05 | +0.05 | 0.20 |
| L28 | +0.10 | +0.10 | 0.17 |
| L33 | +0.00 | +0.00 | 0.15 |
| L38 | +0.00 | +0.00 | 0.20 |
| L43 | +0.00 | +0.00 | 0.17 |

L28 is the noisy maximum. Gemma at the equivalent setup (L23, c=±0.05) got 0.94. Even granting the 10-pair sample is small, no Qwen layer is in the same ballpark.

## What's been ruled out

| Hypothesis | Status | Evidence |
|---|---|---|
| Hybrid-attention crash | Fixed | `generation_mode: hook_per_call` runs cleanly on Qwen3.5's `LinearAttentionLayer` blocks |
| Disk offload starving GPU | Fixed | 4× A100 80GB, `max_memory: {0–3: 65GiB}` (260GB total > 244GB BF16) — model fully resident |
| Norm under-calibration | Ruled out | Swing flat across c = ±0.1 → ±2.0 (40× range) |
| Sign-flipped probe | Ruled out | +c gives weak +swing, not weak −swing |
| Bad parser (regex) | Ruled out | LLM judge agrees with regex on 100% of co-decided rows; judge resolves 4–8 extra rows per run |
| Pair-set leakage | Ruled out | 0% overlap with 10k AL probe-train (verified by construction) |
| Truncation hiding the steering | Partially ruled out | `compliance: truncated` on most rows but judge still picks task; 80%+ row resolution |

## What's still open

1. **Hook layer-indexing on Qwen3.5's hybrid block layout.** The HF model has full-attention and linear-attention layers interleaved. `prefill_with_hooks(messages, [(layer, hook)])` registers a forward hook at "layer L" — but on hybrid models this might attach to a residual stream point one layer off from where the probe was extracted. Worth instrumenting.
2. **High refusal at c=0** (35%). Either a measurement-prompt issue specific to Qwen, or this disjoint-pool sample happens to contain harder pairs. With 10 pairs, hard to tell.
3. **Attention pattern of the probe direction.** The probe was trained on tb-1 / tb-4 activations from `pref_main`. If the Qwen residual at those positions has different geometry than the residual the hook modifies, the direction may project poorly.

## Cost & next-step options

- Spent so far on the pod work: ~$30-40 across two pods (paused).
- **Option A — debug locally first** (~free): instrument `prefill_with_hooks` on Qwen, verify residual shape/device/value before and after the hook fires. Confirm we're modifying the correct point. Same for token-span indexing on the 122B tokenizer (we used Qwen3-32B at build time).
- **Option B — try `SteeredHFClient` with our preference probe** (~$15-25): the qwen_persona_vectors team's path, known-working on Qwen3.5-122B. If THAT also produces ~0.15 swing with our probe direction, the issue is the direction itself, not the runner. ~30-min run.
- **Option C — reduced Phase A** (~$60-100): 25 pairs × 1 selector × 6 layers × 4 mults × 2 orderings × 1 trial = 1200 gens, ~100 min. Confirms or denies the "weaker causal handle" finding at scale, on the same setup we'd report.
- **Option D — accept and report**: write up "Qwen probe is better at decoding, weaker as a causal handle" as the cross-model finding. Pair with caveats and the diagnostic plots.

The pilot data is consistent with Option D being a real result, but the n=10 pilot variance is high enough that A or B before C is the cheapest way to be confident.

## Artefacts

- `assets/plot_050526_qwen_l38_coef_ramp.png` — coefficient ramp at L38
- `assets/plot_050526_qwen_layer_scan.png` — swing across 6 layers
- `assets/plot_050526_qwen_vs_gemma_decoupling.png` — side-by-side magnitude comparison
- `checkpoints/positive_control_{v2,v3_coef_ramp}.parsed.jsonl`, `phase_a_lite_tb1.parsed.jsonl` (on pod, in worktree branch)
- `running_log.md` — full chronology including the runner patch (hybrid-attention support) and norm-scale rabbit-hole
- `configs/steering/qwen_layer_sweep/{phase_a_*_bundled, positive_control_*}.yaml` — all configs (committed)
