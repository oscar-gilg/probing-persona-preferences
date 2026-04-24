---
status: draft
model: gemma-3-27b
---

# Cross-persona unilateral steering — results

## TL;DR

_(placeholder — fill after analysis)_

## Setup

- 6 canonical personas from `persona_sweep_final_six`: aura, contrarian, mathematician, sadist, slacker, strategist.
- Per-persona `ridge_L25` probe trained on tb-5 (eot) activations.
- Unilateral steering at layer L25. Injection coef = multiplier × per-persona `mean_norm(L25)`. Multipliers: ±3%, ±5%.
- 100 shared pairs sampled from `default_test` (utility_gap > 0.1, stratified origin×origin). Both orderings per pair.
- `n_trials=3`, `temperature=1.0`, max_new_tokens=64.

See [spec](cross_persona_unilateral_spec.md).

## Per-persona dose-response

![Cross-persona unilateral dose-response](assets/TODO_plot.png)

_(to be generated — one panel per persona, two lines per panel: first-span / second-span unilateral. x = signed coef, y = P(pick that span's task). Baseline dots at coef=0._)

## Swing magnitude summary

_(table: one row per persona, columns = swing magnitude at coef=±5% for first-span, second-span, combined)_

## Refusal rate

_(to be filled)_

## Observations

_(filled after analysis)_

## Limitations

- L25 used instead of L23 (layer_sweep peak) because per-persona probes were only trained at {25, 32, 39, 46, 53}. Layer_sweep showed L25 at ~0.49 swing vs L23 at 0.95, so expect attenuated but not dead signal.
- tb-5 selector; layer_sweep used eot. Cross-selector cosines were ~1.0 in mid-to-late layers so this should be innocuous.
- No random-direction control; relies on coefficient sign flip as the within-probe null.
