---
status: complete
model: gemma-3-27b
version: v2
---

# Cross-persona steering with the Assistant probe — results (v2)

> v2 supersedes v1. v1 used each persona's own ridge_L25 probe; v2 uses the
> **Assistant (default) probe** across all 6 personas. This matches the
> original §3.4 default-persona steering and the open-ended evil steering
> experiment, and is the version going into the paper.

## TL;DR

- **A single Assistant-trained probe steers all 6 canonical personas at L25.** Differential steering (push +c on the first-span task and −c on the second-span task) drives P(picked steered task) above 0.7 for every persona at |c|=0.05.
- **Sadist gets *more* steerable under the Assistant probe**, not less: unilateral mean Δ rises from +0.238 (own probe) to +0.385 (Assistant probe). Mathematician and contrarian are roughly unchanged. Slacker and aura drop by ~half.
- **Strategist is identical to v1** at +0.432 — the strongest unilateral effect of any persona under both probes.
- **Refusals stay 9–21%** (sadist highest, slacker/mathematician lowest). All within v1 ranges; no v2-induced compliance regressions.

## Setup

- 6 canonical personas from `persona_sweep_final_six`: aura, contrarian, mathematician, sadist, slacker, strategist.
- Single shared probe: Assistant `ridge_L25` from `results/probes/persona_sweep_final_six/default_tb-5/`.
- Steering at layer L25 (closest available to L23 layer-sweep peak; matches §3.4). Injection coef = multiplier × per-persona `mean_norm(L25)`. Multipliers ±3%, ±5%.
- 100 shared pairs sampled from `default_test` (utility_gap > 0.1, stratified origin × origin). Both orderings per pair.
- `n_trials=3`, `temperature=1.0`, `max_new_tokens=64`, `seed=42`.
- Two phases on H100 80GB:
  - **Unilateral** (this dir): 4800 gens × 6 = 28 800 (~3h 50 min wall clock).
  - **Differential** (`experiments/cross_persona_differential/`): 2400 gens × 6 = 14 400 (~2h wall clock).
- Plus 600 no-steering API baseline gens per persona (kept from v1; coef=0 is probe-independent).

See [spec](cross_persona_unilateral_spec.md). Differential phase: [`../cross_persona_differential/`](../cross_persona_differential/).

## Combined dose-response

![Per-persona Assistant-probe steering at L25](assets/plot_050526_cross_persona_perprobe_steering.png)

Six panels, one per persona. **Green** = differential ("contrastive") condition: y = P(picked first-span task), with the probe pushed +c on the first-span task and −c on the second. **Blue / orange** = single-task (unilateral) conditions: probe pushed only on the first-span task (blue) or only on the second-span task (orange), with the same y-axis (P picked first-span). Black dot at x=0 is the empirical no-steering API baseline.

The differential line (green) is the headline §3.4 measure: it sweeps from low to high across [-0.05, 0.05] in every persona. Single-task lines reveal whether the swing is symmetric across spans (slacker/strategist/mathematician/sadist) or asymmetric (aura, contrarian — first-span steering is near the noise floor while second-span steering is large).

## Differential headline numbers — `P(picked steered task)` (paper §3.4 metric)

Folded over orderings; refusals/unparseable excluded from the denominator.

| persona       | P(steered) @ \|c\|=0.03 | SEM | n | P(steered) @ \|c\|=0.05 | SEM | n | refuse % |
|:--------------|------------------:|----:|--:|------------------:|----:|--:|---------:|
| sadist        |             0.809 | .012 | 1167 |          **0.848** | .010 | 1174 |   21.33 |
| strategist    |             0.834 | .011 | 1148 |          **0.844** | .011 | 1167 |   13.04 |
| contrarian    |             0.647 | .014 | 1197 |          **0.766** | .012 | 1198 |   14.42 |
| aura          |             0.727 | .013 | 1200 |          **0.751** | .012 | 1200 |   18.46 |
| slacker       |             0.687 | .013 | 1186 |          **0.740** | .013 | 1196 |    8.88 |
| mathematician |             0.667 | .014 | 1200 |          **0.718** | .013 | 1200 |    9.83 |

All six personas are steered to ≥ 0.72 at |c|=0.05 vs. an unsteered baseline near 0.5; sadist and strategist clear 0.84.

## Unilateral swings — v1 vs v2 (per-probe vs Assistant-probe)

Mean Δ across `unilateral_first` and `unilateral_second` at ±0.05; folded over orderings; refusals/unparseable excluded.

| persona       | first_swing | second_swing | mean Δ (v2 Assistant) | mean Δ (v1 own-probe) | direction |
|:--------------|------------:|-------------:|----------------------:|----------------------:|:----------|
| strategist    |       0.311 |        0.553 |             **0.432** |                 0.432 | identical |
| sadist        |       0.264 |        0.505 |             **0.385** |                 0.238 | up 1.6× |
| contrarian    |       0.082 |        0.438 |             **0.260** |                 0.225 | up |
| slacker       |       0.150 |        0.305 |             **0.227** |                 0.453 | down ~½ |
| aura          |       0.065 |        0.375 |             **0.220** |                 0.383 | down |
| mathematician |       0.120 |        0.218 |             **0.169** |                 0.148 | up |

**Ranking change.** v1 ordering (slacker > strategist > aura > sadist > contrarian > mathematician) does not survive the probe swap. Under the Assistant probe: strategist > sadist > contrarian > slacker > aura > mathematician. Strategist anchor is stable; sadist climbs because its own-probe direction was apparently weaker than the Assistant probe direction at L25. Slacker and aura lose roughly half their swing — plausible if their persona-specific probes captured persona-particular structure that the Assistant probe doesn't have.

## Observations

- **The same direction works for every persona — including sadist.** This is the v2 headline. The Assistant probe is not just a default-persona artefact; it's a shared evaluative direction usable as a steering vector across personas with very different system prompts (slacker, mathematician, sadist, contrarian, aura, strategist).
- **First/second asymmetry is the rule, not the exception.** For most personas, second-span steering produces ~2× the swing of first-span steering. For aura and contrarian it's ~5× — first-span swings sit near the noise floor (~0.07–0.08). The aggregate-of-both-orderings is the unbiased measure; the per-span splits are conflated with persona-specific position bias (v1 reported aura position bias ≈ 0; contrarian ≈ −0.36; sadist ≈ −0.23).
- **Sadist refusal stays the highest at 21%**, but is fully consistent with v1. The Assistant probe doesn't push the model toward refusal compared to the per-persona probe.
- **Differential ≫ unilateral magnitude.** Differential pushes both spans simultaneously; the absolute P(steered) lift over the 0.5 chance baseline is consistently larger than the unilateral mean Δ — as expected since differential injects 2× the probe energy on the choice.

## Paper integration

- The figure `paper/figures/main/plot_050526_cross_persona_perprobe_steering.png` is the v2 §3.4 cross-persona panel. It replaces both the v1 unilateral-only panel and the per-persona-probe variant.
- Headline claim for the paper: **the same Assistant-trained probe direction causally controls pairwise choice under every persona tested**, with P(picked steered task) ≥ 0.72 at |c|=0.05 across all 6 personas.
- Numbers above to be registered as paper claims via the corroborate plugin (sadist 0.848, strategist 0.844, mathematician 0.718, plus the unilateral mean-Δ table).

## Limitations

- **L25 vs L23.** Persona-sweep activations were saved at L25/L32/L39/L46/L53; layer_sweep peak was L23. L25 gives ~half the peak swing for the default persona (0.49 vs 0.95 at L23). Re-extracting at L23 would tighten the unilateral effect.
- **tb-5 selector.** Persona-sweep probes use tb-5; layer_sweep used eot. Cross-selector cosines are ~1.0 in mid-to-late layers, so this should be innocuous, but not empirically verified for these personas.
- **No random-direction control.** Sign-flip is the within-probe null. A random-direction run at the same multipliers would rule out that any unit-norm direction steers pairwise choice.
- **Shared pairs from `default_test`.** Pairs are not persona-optimal; sadist-preferred (harmful) tasks may be rare in this pool. A per-persona pair set might tighten the signal for sadist and contrarian.
- **No bootstrap CIs.** Δ values have ~600 underlying trials per cell; differences > ~0.05 are likely real, smaller comparisons need explicit error bars before paper.
- **Position bias is persona-dependent** (±0.01 to ±0.36 in v1; not re-measured in v2 since baselines are reused). Aggregate-of-both-orderings removes it; per-span readings are conflated.
- **v1 per-probe checkpoints lost.** The previous-pod `checkpoints_v1_perprobe/` snapshot wasn't synced to this pod. Comparison numbers above come from the v1 report. Not a v2 problem, but worth re-syncing if anyone wants to plot v1 alongside v2.
