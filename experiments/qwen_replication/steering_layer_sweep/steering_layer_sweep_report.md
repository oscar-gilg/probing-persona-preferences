---
status: in_progress (pilot diagnostic, decision pending)
model: qwen3.5-122b-nothink
spec: experiments/qwen_replication/steering_layer_sweep/steering_layer_sweep_spec.md
date: 2026-05-05
---

# Qwen-3.5-122B contrastive steering — pilot diagnostic

## TL;DR

- **The Qwen probe is *better* at decoding utility than the Gemma probe (heldout r 0.946 vs 0.874) but a *much weaker* causal handle.** Best swing across 6 sampled layers is 0.06 (Qwen, judge-resolved); Gemma's peak is 0.94 — a ~15× gap.
- **Effect direction is correct** (positive c shifts toward the higher-utility task) but **magnitude is small at every coefficient and every layer tested.** Sweeping c from ±0.1 to ±2.0 at L38 (40× range) leaves swing flat between 0.05 and 0.17. This rules out an under-calibrated coefficient as the explanation.
- **Refusal rate at c=0 is 15–35%** (vs Gemma's <5%), narrowing the responding subset that the swing is computed over.
- **Methodology is matched to Gemma's**: contrastive `spans: {first: 1, second: -1}`, ridge probe direction, position-selective hook, `max_new_tokens: 64`, canonical LLM-judge parsing. Regex prefix and judge agree on 100% of co-decided rows; judge resolves an extra 4–8 rows per run.
- **The pilot is n=10 pairs** — small enough that the headline gap could be noisier than it looks, but the gap is large enough to motivate a real run.

![Cross-model decoupling — probe r vs steering swing](assets/plot_050526_qwen_vs_gemma_decoupling.png)

## Setup

- **Model:** Qwen-3.5-122B-A10B-nothink (48 transformer blocks, hybrid full/linear-attention, BF16 on 4× A100 80 GB).
- **Probe:** ridge `tb-1`, trained on 10k AL with held-out r reported per layer ([qwen35_probes report](../../training_probes/qwen35_probes/qwen35_probes_report.md)). 6 layers tested: L12, L24, L28, L33, L38, L43 (= 25%, 50%, 58%, 69%, 79%, 90% depth).
- **Pair set:** 10 pairs (and a 50-pair reservoir) sampled fresh from the disjoint subset of `canonical_test` with zero overlap with the 10k AL probe-train (verified by construction in `build_steering_pairs_50.py`).
- **Steering:** contrastive — `+c × probe_direction` added to task A's tokens, `-c × probe_direction` to task B's, during prefill only. `c × cached_mean_norm[L]`. Hook stays installed during generation but `position_selective_steering` is no-op at gen-time (only fires on resid.shape[1] > 1).
- **Choice parsing:** regex prefix extraction → `judge_completion_full_async` LLM judge (`completion_judge.py`, `gemini-3-flash-preview`). Reports below use the judge's `task_completed`; refusal = anything not in {a, b}.
- **Generation:** `max_new_tokens: 64`, temperature 1.0, seed 42.
- **Each cell:** 10 pairs × 2 orderings (= 20 trials), `n_trials: 1` per (pair, ordering).
- **Cost:** ~$30–40 across two pods (paused). Full Phase A on 4× A100 estimated ~$300–500.

## Results

### 1. No layer in {L12, L24, L28, L33, L38, L43} steers like Gemma at L23

![Layer scan — no peak](assets/plot_050526_qwen_layer_scan.png)

| Layer | depth | swing (judge) | refusal at c=±0.05 |
|---|---|---|---|
| L12 | 25% | -0.05 | 0.12 |
| L24 | 50% | +0.02 | 0.20 |
| L28 | 58% | +0.03 | 0.17 |
| L33 | 69% | +0.01 | 0.15 |
| L38 | 79% | **+0.06** | 0.20 |
| L43 | 90% | +0.00 | 0.17 |

L38 (the probe peak) is the noisy maximum. Gemma's peak (L23 of 62, 37% depth) hits 0.94. **No Qwen layer is in the same ballpark.** Refusal rate sits at 12–20% across all six — 3–4× Gemma's typical operating point.

### 2. Effect is flat across a 40× coefficient sweep at L38

![L38 coefficient ramp — flat response](assets/plot_050526_qwen_l38_coef_ramp.png)

| c | P(A) | refusal | n responded |
|---|---|---|---|
| -2.0 | 0.12 | 0.15 | 16/20 |
| -1.0 | 0.12 | 0.10 | 16/20 |
| -0.5 | 0.19 | 0.15 | 16/20 |
| -0.1 | 0.27 | 0.15 | 15/20 |
| 0.0 | 0.20 | 0.25 | 15/20 |
| +0.1 | 0.20 | 0.15 | 15/20 |
| +0.5 | 0.29 | 0.15 | 17/20 |
| +1.0 | 0.25 | 0.10 | 16/20 |
| +2.0 | 0.19 | 0.15 | 16/20 |

Swing in the |c| ≤ 2 range stays in 0.05–0.17 with no monotone trend. **An under-calibrated `mean_norm` would predict steering "switching on" once c is large enough; we don't see that.**

### 3. Methodology checks pass

| Check | Result |
|---|---|
| Steering direction (sign) | Correct: positive c → higher P(A) on average |
| Regex prefix vs LLM judge | 100% agreement on co-decided rows; judge picks up an extra 4/60 (v2), 4/180 (v3), 8/240 (lite) regex-refusal rows |
| Probe orientation | +direction = "higher utility" by construction; sign is consistent across pairs |
| Pair-set leakage | 0% with 10k AL (verified) |
| Hook applied without crash | All 480 trials completed; no NaN, no degenerate outputs |

## Follow-up diagnostics (this session)

After the pilot, four additional tests on fresh 4× A100 pods to rule out alternative explanations:

| Diagnostic | Test | Result |
|---|---|---|
| **Coverage gap** | Gemma's analogue of L23/62 = L18/48. Lite scan jumped L12 → L24, missing this. Run probe `ridge_L24` injected at L16, L18, L20, L22, L24 (cross-layer steering, 10 pairs each). | Max swing **+0.01** at L22; L18 swing **+0.00**. **No hidden peak in the missed region.** |
| **System-prompt parity** | Probe was extracted with `model: qwen3.5-122b` (no system prompt); steering uses `qwen3.5-122b-nothink` (`/no_think` injected). Re-run L38 c=±0.05 with `qwen3.5-122b`. | **Inconclusive.** Without `/no_think`, Qwen3.5 produces thinking traces that exceed `max_new_tokens=64`, giving 100% refusal. Can't measure choice cheaply at 64 tokens. |
| **All-tokens vs contrastive** | Apply `+c × direction` to *every* prompt token (`prefill_all_steering`) instead of contrastive position-selective. Same probe `ridge_L24` at L24, c=±0.05. If swing emerges → contrastive/spans is the bottleneck. If flat → the direction itself. | **Swing -0.08** (essentially zero, opposite sign). Refusal rate dropped to 5-15% (clean responses). Direction has no causal effect even when applied globally. |
| **Multi-layer simultaneous contrastive** | All 6 probes (`ridge_L<L>`) installed simultaneously at their own layers — `+c×dir_L on first span, -c×dir_L on second span` at L{12,24,28,33,38,43} all at once. If single-layer was sub-additive, multi-layer would compound. | **Swing +0.05** at c=±0.05. **Same magnitude as single-layer best.** Effects don't compound across layers. **Decisively rules out "we picked the wrong single injection layer."** |
| **Span verification** | Confirm the steering hook is firing on the right tokens — print decoded span content for 3 pairs at runtime with the actual Qwen3.5-122B tokenizer + `enable_thinking=False`. | Spans bracket task content exactly. E.g., pair 0001 first span decodes to `'Name five components of a well-balanced diet.'` between markers `Task A:\n` and `\n\nTask B`. Tokenization is correct. |
| **tb-4 probe (heldout-r peak)** | Same setup as positive_control_v2 but with the tb-4 manifest (heldout r = 0.946, the absolute peak; tb-1 is 0.946 too but tb-4 was designated the "best" decoder). 10 pairs × 3 mults × 2 orderings. | **Swing exactly 0.00.** P(A) is constant at 0.19 across c=-0.05, 0, +0.05. The probe with the strongest decoding correlation has *zero* causal effect — sharper decoupling than tb-1's swing of +0.17. |
| **Open-ended steering** | `all_tokens_steering` (fires both prefill and gen) on 3 single-task prompts — "Tell me about yourself", "haiku about the moon", "AI alignment paper angles" — at L38 c ∈ {-0.05, -0.02, 0, +0.02, +0.05}, 3 trials each. Gemma's analogue (paper §4) shows dramatic refusal/willingness shifts. | **Responses are qualitatively identical across the full ±0.05 range.** Same cooperative "Hello! I'm **Qwen3.5**..." opener, same structure, same tone. No shift in willingness, refusal, or content style. |

## Conclusion

**The Qwen preference direction has essentially zero causal effect on the model**, regardless of how it's applied. The diagnostics rule out:

- Coefficient under-calibration (40× ramp is flat)
- Coverage gap (cross-layer at L16/18/20/22 also flat)
- Wrong single layer (multi-layer simultaneous contrastive at all 6 layers gives the same swing as single-layer best)
- Contrastive/spans setup (all-tokens prefill also flat; open-ended gen also flat)
- Wrong probe-extraction position (tb-4 — the heldout-r peak — gives swing exactly 0.00)
- Probe sign (positive c → positive swing on contrastive, correct direction)
- Parser quality (regex and LLM judge agree)
- Span tokenization (verified at runtime: hooks fire exactly on task-content tokens)

The strongest single piece of evidence: **open-ended generation at c ∈ {-0.05, ..., +0.05} produces qualitatively identical responses on three different prompts.** Gemma's analogue at the same coefficient range produces dramatic refusal/willingness shifts (paper §4). For Qwen, the direction simply doesn't move the model.

Subtler pattern: **the probe with the *highest* heldout decoding r (tb-4 at L38, r=0.946) has the *lowest* causal swing (0.00), while tb-1 (also r=0.946 but slightly less strong by the report's measure) has +0.17.** Linear decodability and causal efficacy don't just decouple on Qwen — they appear to be *negatively* correlated.

What remains plausible but not verified:

- **System-prompt distribution shift** (probe trained without `/no_think`, applied with it). Can't be cheaply tested due to thinking-trace truncation. Would need either a probe trained with `/no_think` OR `max_new_tokens` ≥ 512 to measure choice in non-`/no_think` mode.
- **Hook target on Qwen3.5's hybrid block layout.** The architecture accessor maps `qwen3_5_moe → model.model.language_model.layers` and `qwen3_5_moe_text → model.model.layers`. Without inspecting the loaded module on a live pod, can't fully confirm the right residual point is being modified. The fact that the hook fires without error and produces small but real swings (positive c → small +swing) suggests the attachment is reasonable, just that the modified residual doesn't have strong causal weight at that point.

The cross-model decoupling — Qwen probe is *better* at decoding utility (heldout r 0.946 vs 0.874) but *much weaker* as a causal handle (best swing 0.06–0.17 vs Gemma's 0.94) — is the headline.

## Decision

Pod paused. Cost: ~$45-55 total. Three options going forward:

- **(A) Accept and report** as a paper appendix finding. The decoupling story is itself interesting and reportable. Caveat: still based on n=10–20 pilot pairs; would tighten with a small follow-up.
- **(B) Tighten with a 25-pair Phase B-style run** at the best Qwen layer (~$50-80, ~2h). Confirms the small-but-positive contrastive swing at scale. Also adds the per-pair-type breakdown (bb/hb/hh) that mirrors Gemma's Fig. 5.
- **(C) Try a probe trained with the matched system prompt** (~few hours GPU upfront for new extraction + probe). Would close the system-prompt gap. Higher commitment.

Recommend **(A)** for the immediate paper, with **(B)** if we want a Qwen panel matching Gemma's Fig. 5 layout. (C) is a longer follow-up if the cross-model story warrants deeper investigation.

## Artefacts

- `assets/plot_050526_qwen_l38_coef_ramp.png` — coefficient ramp at L38
- `assets/plot_050526_qwen_layer_scan.png` — swing across 6 sampled layers
- `assets/plot_050526_qwen_vs_gemma_decoupling.png` — headline cross-model bar
- `checkpoints/positive_control_v2.parsed.jsonl` (60 rows, L38 ±0.05/0)
- `checkpoints/positive_control_v3_coef_ramp.parsed.jsonl` (180 rows, L38 ±0.1 → ±2.0)
- `checkpoints/phase_a_lite_tb1.parsed.jsonl` (240 rows, 6 layers × ±0.05)
- `running_log.md` — full chronology including the runner patch (`generation_mode: hook_per_call` for hybrid attention, multi-GPU sharding via `device: auto` + `max_memory`)
