---
status: draft
model: gemma-3-27b
---

# Cross-persona unilateral steering — results

## TL;DR

- **Unilateral probe-direction steering at L25 works across all 6 canonical personas** — every persona shows a non-zero preference swing under its own ridge_L25 probe.
- **Swing magnitude spans 3× across personas** (max/min = 0.453/0.148): mathematician weakest, slacker strongest. Ordering: slacker ≈ strategist > aura > sadist ≈ contrarian > mathematician.
- **Position bias is persona-specific, sometimes extreme.** Contrarian baseline is 0.32/0.68 (+0.36 bias toward the second-presented task); sadist 0.38/0.62. Others within ±0.12. This explains most of the apparent "first-vs-second asymmetry" in the unilateral swings.
- **Refusals stay ≤ 2.6%** at every operating point. Real preference shifts, not broken generations.

## Setup

- 6 canonical personas from `persona_sweep_final_six`: aura, contrarian, mathematician, sadist, slacker, strategist.
- Per-persona `ridge_L25` probe trained on tb-5 (eot) activations from that persona's 4k train split.
- Unilateral steering at layer L25. Injection coef = multiplier × per-persona `mean_norm(L25)`. Multipliers ±3%, ±5%.
- 100 shared pairs sampled from `default_test` (utility_gap > 0.1, stratified origin×origin). Both orderings per pair → baseline P(pick span's task) ≈ 0.5 modulo position bias.
- `n_trials=3`, `temperature=1.0`, `max_new_tokens=64`, `seed=42`.
- 4800 generations per persona × 6 = 28 800 total, ~3.5 hrs on A100 80GB. Plus 600 no-steering baseline generations per persona via OpenRouter API (3 600 total, ~5 min).

See [spec](cross_persona_unilateral_spec.md).

## Dose-response

![Cross-persona unilateral dose-response](assets/plot_042426_cross_persona_unilateral_dose_response.png)

One panel per persona. Blue = +coef applied to first-span task. Orange = +coef applied to second-span task. Gray dashed = aggregate (first + second)/2 — removes position bias. Black dots at x=0 are empirical no-steering baselines (600 API generations per persona, no model loading).

## Swing magnitude at ±5% + no-steering baseline

Δ = P(picked steered task | +5%) − P(picked steered task | −5%). Baseline = P(pick span's task) with no steering (empirical, 600 gens per persona via OpenRouter).

| Persona       | first-span Δ | second-span Δ | mean Δ | base first | base second | position bias |
|:--------------|-------------:|--------------:|-------:|-----------:|------------:|--------------:|
| slacker       |        0.383 |         0.522 |  0.453 |      0.561 |       0.439 |        +0.122 |
| strategist    |        0.405 |         0.458 |  0.432 |      0.513 |       0.487 |        +0.026 |
| aura          |        0.262 |         0.505 |  0.383 |      0.503 |       0.497 |        +0.007 |
| sadist        |        0.213 |         0.262 |  0.238 |      0.383 |       0.617 |        −0.233 |
| contrarian    |        0.082 |         0.368 |  0.225 |      0.322 |       0.678 |        −0.357 |
| mathematician |        0.143 |         0.152 |  0.148 |      0.538 |       0.462 |        +0.077 |

For reference: the layer_sweep default-persona unilateral at L25 gave aggregate Δ ≈ 0.49 with baselines first=0.56 / second=0.44 (position bias +0.12, similar to slacker here). Mean Δ is the bias-free measure of probe-driven control.

## Observations

- **Probe generalises as a causal lever across personas.** Even the weakest (mathematician, 0.148) cleanly breaks the null of no effect.
- **Contrarian's extreme first-vs-second asymmetry is mostly position bias, not probe asymmetry.** Baseline is 0.32/0.68 — the model already heavily prefers the second-presented task under the contrarian prompt. Push against that with negative coef on first-span and you're fighting uphill; push with it via positive coef on second-span and you hit ceiling. The aggregate (position-bias-removed) swing is 0.225 — in line with sadist.
- **Sadist shows the same pattern** (baseline 0.38/0.62, −0.23 bias) but weaker.
- **Sadist has highest refusal (2.6%)**, still low. Consistent with strong content-based safety triggers being somewhat orthogonal to the steering direction.
- **Slacker and strategist look most like the default persona at L25** — the probe direction functions similarly to how default utility does. These are the personas whose utility signal (avoid-effort, pursue-influence) is strongest and most consistent with the active-learning preferences.

## Paper integration

- Replaces the differential-only section 3.4: this unilateral panel shows probe-direction steering across 6 personas in one plot.
- Headline claim: the probe direction causally controls pairwise choice under every persona tested.
- Figure caption should note the mean swings, the asymmetry finding for contrarian, and the baseline-is-0.5 framing (both orderings always run).

## Limitations

- **L25 vs L23.** Per-persona probes were only trained at L25/L32/L39/L46/L53; layer_sweep peak was L23. L25 gives ~half the peak swing (0.49 vs 0.95 at L23 for default). Expect stronger effects if we re-trained at L23.
- **tb-5 selector.** Layer_sweep used eot; persona probes are tb-5. Cross-selector cosines are ~1.0 in mid-to-late layers, so this should be innocuous, but not empirically verified for these personas.
- **No random-direction control.** Coefficient sign-flip is the within-probe null. A random-direction run at the same multipliers would rule out that any unit-norm direction steers pairwise choice.
- **Shared pairs from default_test.** Pairs are not persona-optimal; sadist-preferred (harmful) tasks may be rare in this pool. A per-persona pair set (tasks with large gaps under that persona's utility) might tighten the signal, especially for sadist and contrarian.
- **No bootstrap CIs.** Δ values have ~600 underlying trials per cell (100 pairs × 2 orderings × 3 trials), so differences between personas larger than ~0.05 are likely real, but small comparisons (sadist vs contrarian, aura vs mean-of-top-two) need explicit error bars before the paper.
- **Position bias is persona-dependent and sometimes extreme.** Ranges from ±0.01 (strategist, aura) to ±0.36 (contrarian). The aggregate-of-both-orderings (gray dashed line) removes this, but per-span readings are conflated with position bias.
- **Operational note:** aura's LLM-judge parsing hung once mid-run (OpenRouter HTTP client stuck on ~25 CLOSE_WAIT sockets). Resumed cleanly via `_parse_checkpoint`'s existing-keys check; no data lost, but a runner robustness fix would prevent this.
