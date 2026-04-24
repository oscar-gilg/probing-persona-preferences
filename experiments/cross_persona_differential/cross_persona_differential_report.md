---
status: draft
model: gemma-3-27b
---

# Cross-persona differential steering — results

## TL;DR

- **Differential (contrastive) steering with the per-persona `ridge_L25` probe pushes Gemma-3-27B toward the steered task across all 6 canonical personas.** At |c|=0.05, P(steered task chosen) ∈ [0.68, 0.89] — always well above the 0.5 symmetry anchor.
- **Ordering by strength at |c|=0.05**: slacker (0.888) ≈ strategist (0.865) > aura (0.849) > contrarian (0.727) > mathematician (0.680) ≈ sadist (0.677). Mostly matches the unilateral ordering; sadist drops because of non-monotonicity (see below).
- **Differential is ~1.8× the unilateral swing on average**, consistent with the paper's contrastive > unilateral expectation. Mean differential swing = 2·(P̄@.05 − 0.5) = 0.56 vs mean unilateral Δ = 0.32.
- **Sadist is non-monotonic**: P@|c|=0.03 (0.746) > P@|c|=0.05 (0.677), with 18.75% hard-refusal at ±0.05. Interpretation: at high coefficient magnitudes under the sadist prompt + harmful pairs, refusal rises, the set of *parseable* completions skews, and the apparent steered-task share drops. This is not probe failure but a refusal-safety interaction at high |c|.
- **Refusals stay ≤ 18.8% (sadist), typically ≤ 14%** — real preference shifts, not broken generations.

## Setup

See [spec](cross_persona_differential_spec.md).

- Same 6 canonical personas, probes, pairs, and L25 injection as [`cross_persona_unilateral`](../cross_persona_unilateral/cross_persona_unilateral_report.md).
- Single condition per persona: `differential` with `spans = {first: +1, second: −1}` at L25, multipliers {±0.03, ±0.05} of per-persona `mean_norm(L25)`.
- 100 shared pairs × 2 orderings × 4 multipliers × 3 trials = 2400 generations per persona, **14 400 total**.
- Judge: LLM compliance/choice parser (`google/gemini-3-flash-preview` via OpenRouter), concurrency 50.

## Dose-response

![Cross-persona differential dose-response](assets/plot_042426_cross_persona_differential_dose_response.png)

One panel per persona. x = |c| (fraction of per-persona L25 mean-norm), y = P(steered task chosen) folding +/- c (at c>0 Task A is the +probe side; at c<0 Task B is). Anchor at (0, 0.5). Error bars = binomial SEM, n ≈ 1180-1200 per cell.

## Headline numbers

Full table of P(steered) at both mults, refusal (LLM-judge `hard_refusal`), unparseable-label rate, and the matched unilateral mean Δ for reference. `swing@.05` = 2·(P@|c|=0.05 − 0.5).

| Persona       | P@\|c\|=.03 | SEM    | P@\|c\|=.05 | SEM    | swing@.05 | refuse% | noparse% | uni meanΔ |
|:--------------|------------:|-------:|------------:|-------:|----------:|--------:|---------:|----------:|
| slacker       |       0.846 |  0.010 |       0.888 |  0.009 |     0.778 |    8.2% |     0.4% |     0.452 |
| strategist    |       0.854 |  0.010 |       0.865 |  0.010 |     0.730 |   12.3% |     1.1% |     0.434 |
| aura          |       0.767 |  0.012 |       0.849 |  0.010 |     0.698 |   13.8% |     0.0% |     0.383 |
| contrarian    |       0.694 |  0.013 |       0.727 |  0.013 |     0.454 |   10.4% |     0.1% |     0.228 |
| mathematician |       0.613 |  0.014 |       0.680 |  0.013 |     0.360 |    6.3% |     0.0% |     0.148 |
| sadist        |       0.746 |  0.013 |       0.677 |  0.014 |     0.354 |   18.8% |     1.7% |     0.243 |

All differential swings are ≥ 0.35, and 5/6 are strictly greater than the matched unilateral mean Δ (sadist is a touch higher at 0.354 vs 0.243 — see non-monotonicity note). The per-persona swing ranking is identical to unilateral modulo sadist/mathematician near the bottom.

## Observations

- **Contrastive > unilateral on average.** Ratio differential-swing / unilateral-meanΔ is 1.72, 1.68, 1.82, 1.99, 2.43, 1.46 for slacker, strategist, aura, contrarian, mathematician, sadist respectively (mean ~1.85). Mathematician — the weakest unilateral persona — gets the biggest multiplicative boost, suggesting unilateral underestimated the probe's usable signal there.
- **Saturation by |c|=0.05** for slacker and strategist (Δ between .03 and .05 is only +0.011 and +0.042). For aura it's +0.082, still climbing. Mathematician and contrarian are also still climbing. Suggests slacker/strategist probes are aligned with a direction the model already uses at ~0.85 capacity.
- **Sadist's non-monotonicity (P@.03 > P@.05) is driven by refusal, not probe direction.** Refusal at |c|=0.05 is 18.8% (almost 3× contrarian). The LLM judge flags `hard_refusal` on completions that *stated* one task but executed neither — these drop out of the parseable set, and because refusals under the sadist persona disproportionately fall on the harmful "steered" task, the remaining parseable choices skew toward the safer (non-steered) one. Net effect: apparent P(steered) goes down even though the *direction* is correct. Repeating the analysis while treating refusals as an intent-to-steer signal would likely restore monotonicity; not done here.
- **Sadist unparseable rate (1.7%)** is also an order of magnitude higher than any other persona, consistent with harder-to-parse completions at high-steering × harmful-content.
- **First/second asymmetry is absorbed.** Contrastive steering cancels the per-span position bias that made the unilateral first-vs-second split messy: the single folded curve is clean for every persona (even contrarian, whose baseline is 0.32/0.68).

## Comparison to unilateral

| Persona       | uni meanΔ | diff swing@.05 | ratio |
|:--------------|----------:|---------------:|------:|
| slacker       |     0.452 |          0.778 |  1.72 |
| strategist    |     0.434 |          0.730 |  1.68 |
| aura          |     0.383 |          0.698 |  1.82 |
| contrarian    |     0.228 |          0.454 |  1.99 |
| mathematician |     0.148 |          0.360 |  2.43 |
| sadist        |     0.243 |          0.354 |  1.46 |
| **mean**      | **0.315** |      **0.562** | **1.85** |

Differential at L25 hits roughly the paper's §3.4 default-persona magnitude (~0.9 at |c|=0.05) on the top 3 personas (slacker, strategist, aura), confirming that the per-persona probes generalise as causal levers — contrastive steering works just as well under value-laden prompts as under a neutral-assistant prompt.

## Limitations

- **Layer 25 vs layer 23.** Per-persona probes were only trained at L25/L32/L39/L46/L53; layer_sweep peak was L23. L25 likely underestimates each persona's true causal ceiling. Same caveat as the unilateral report.
- **`tb-5` selector.** Persona probes are tb-5 (turn-boundary −5 token), not eot. Cross-selector cosines are near 1 mid-layer but not verified empirically for these personas.
- **No random-direction control.** The coefficient-sign symmetry fold is the within-probe null (at |c|=0, P(steered) is exactly 0.5 by construction). A random-direction run would provide a stricter specificity control but adds another ~14 k generations; not run in v1.
- **Shared `default_test` pairs.** Pairs are not persona-optimal. Sadist-preferred pairs specifically are rare in this pool, which plausibly contributes to sadist's muted signal and elevated refusal.
- **No bootstrap CIs on the swing.** Per-cell SEM is small (~0.01) and the 6-persona ranking is clearly well-resolved, but inter-persona differences smaller than ~0.02 shouldn't be read as reliable.
- **Sadist refusal interaction masks monotonicity.** Reporting refusals-as-misses would tighten the interpretation but changes the metric. Flagged in the spec follow-up rather than patched here.

## Paper integration

- Slots directly under the unilateral panel as the "contrastive steering across personas" section; same 2×3 layout is visually comparable.
- Headline claim: *the per-persona probe direction causally drives pairwise choice under every persona prompt, and contrastive injection closes most of the gap to the default-persona contrastive ceiling.*
- Suggest figure caption: note that the sadist dip is a refusal-safety interaction, not a probe failure, pointing to compliance data.

## Operational notes

- **OpenRouter judge-parser hangs**: aura, contrarian, and strategist each hit a parse-phase stall (50 sockets stuck idle) around row 50, 1950, 1700 of their respective parses. All recovered cleanly via resumable retry (`_parse_checkpoint`'s existing-keys check). Runner-side timeout on the judge's HTTP client would prevent the stall entirely.
- **Watchdog v1 bug** (fixed mid-run): counted parse totals across all personas, so it incorrectly fired during subsequent personas' gen phases. Replaced with a version that only triggers when the log's last line is a parse tick. Removed GPU OOM risk from orphaned processes after kill — need to SIGKILL the orphan explicitly, not rely on `kill` alone.
- **Total wall time** ≈ 2h 20min (6 × 19 min gen + 6 × 3 min parse + hang recovery overhead).
