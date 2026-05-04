---
status: in_progress
model: gemma-3-27b
version: v2
---

# Cross-persona unilateral steering — results (v2: Assistant probe)

> v2 supersedes v1. v1 used each persona's own ridge_L25 probe; v2 uses the
> **Assistant (default) probe** across all 6 personas. This matches the original
> §3.4 default-persona steering and the open-ended evil steering experiment.

## Setup

- 6 canonical personas from `persona_sweep_final_six`: aura, contrarian, mathematician, sadist, slacker, strategist.
- Single shared probe: Assistant `ridge_L25` from `results/probes/persona_sweep_final_six/default_tb-5/`.
- Unilateral steering at layer L25 (closest available to the L23 layer-sweep peak; matches §3.4). Injection coef = multiplier × per-persona `mean_norm(L25)`. Multipliers ±3%, ±5%.
- 100 shared pairs sampled from `default_test` (utility_gap > 0.1, stratified origin × origin). Both orderings per pair → baseline P(pick a span) ≈ 0.5 modulo persona-specific position bias.
- `n_trials=3`, `temperature=1.0`, `max_new_tokens=64`, `seed=42`.
- 4800 generations per persona × 6 = 28 800 unilateral generations. Plus 600 no-steering API baseline gens per persona (kept from v1; coef=0 baselines are probe-independent).

See [spec](cross_persona_unilateral_spec.md). Companion differential experiment: [`../cross_persona_differential/`](../cross_persona_differential/).

## Dose-response

*To be filled when v2 unilateral checkpoints complete.*

![Combined cross-persona dose-response (will be regenerated)](../../paper/figures/main/plot_cross_persona_perprobe_steering.png)

## Swing magnitude (v2)

*To be filled with v2 numbers from `experiments/cross_persona_unilateral/checkpoints/{persona}.parsed.jsonl` after the run completes. Comparison to v1 (per-persona probes) below.*

## v1 vs v2 comparison

*To be filled — does the Assistant probe transfer across personas with similar swing magnitude as each persona's own probe? Headline test of cross-persona generalisation.*

## Paper integration

- Replaces the per-persona-probe panel in §3.4 with the Assistant-probe panel — directly supports §4 cross-persona steering claim.
- Numbers will be registered as paper claims via the corroborate plugin once v2 checkpoints are final.

## Limitations

- L25 vs L23. Persona-sweep activations were saved at L25/L32/L39/L46/L53; layer_sweep peak was L23. L25 gives ~half the peak swing (0.49 vs 0.95 at L23 for default). Re-extracting at L23 would tighten effects.
- tb-5 selector. Persona-sweep probes use tb-5; layer_sweep used eot. Cross-selector cosines are ~1.0 in mid-to-late layers, so this should be innocuous, but not empirically verified for these personas.
- No random-direction control. Sign-flip is the within-probe null. A random-direction run at the same multipliers would rule out that any unit-norm direction steers pairwise choice.
- Shared pairs from `default_test`. Pairs are not persona-optimal; sadist-preferred (harmful) tasks may be rare in this pool. A per-persona pair set might tighten the signal for sadist and contrarian.
- No bootstrap CIs. Δ values have ~600 underlying trials per cell; differences > ~0.05 are likely real, smaller comparisons need explicit error bars before paper.
- Position bias is persona-dependent (±0.01 to ±0.36 in v1). Aggregate-of-both-orderings removes it; per-span readings are conflated.
