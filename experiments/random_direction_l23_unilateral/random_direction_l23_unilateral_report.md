# Random-direction L23 unilateral control

**Status:** complete (5 seeds × 30 pairs × 3 trials × 2 orderings × 5 coefs × 2 conditions = 9000 generations).

## Headline

**Random-direction injection on a single task's tokens at L23, ±0.05 × `mean_norm[L23]`, produces no directional bias.** Pooled across 5 seeds, the directional swing P(c=+0.05) − P(c=−0.05) is consistent with zero in both conditions:

| Condition | Swing P(+0.05) − P(−0.05) |
|---|---|
| First task steered | −0.025 ± 0.030 |
| Second task steered | −0.016 ± 0.022 |
| Pooled (N=10 seed×condition) | −0.021 ± 0.020 |

Errors are SEM across seeds. Compare to the trained probe at the same magnitude in single-task mode, which produces a swing of roughly +0.5 (P(+0.04) ≈ 0.70–0.78 vs P(−0.04) ≈ 0.16–0.26 in the harm-breakdown dose-response).

Both random-direction curves peak at c=0 and drop ~0.15 absolute symmetrically on either side. We read this as a noise/disruption signature on the steered span — large random perturbations make the model less likely to commit to *that* span — not as directional steering. The contrastive null (parent run) produces a flat ~0.5 curve instead, because both spans are perturbed simultaneously and the disruption cancels.

## Plot

![Unilateral random-direction L23 null vs contrastive (parent) null](assets/plot_050826_unilateral_vs_contrastive_null.png)

Per-seed thin lines + cross-seed mean ± SEM (thick). Left: this experiment, 5 seeds. Right: parent contrastive null, 2 of 3 seeds available locally.

## Per-coefficient summary

| Condition | c=−0.05 | c=−0.03 | c=0 | c=+0.03 | c=+0.05 |
|---|---|---|---|---|---|
| First task steered  | 0.462 ± 0.016 | 0.545 ± 0.009 | 0.606 ± 0.002 | 0.503 ± 0.019 | 0.436 ± 0.025 |
| Second task steered | 0.364 ± 0.019 | 0.398 ± 0.020 | 0.392 ± 0.002 | 0.406 ± 0.013 | 0.348 ± 0.011 |

Sanity checks:
- The c=0 baselines (0.606 first, 0.392 second) sum to ~1.0 — the standard ordering bias, with no steering applied.
- Refusal rate is 13–15% across all conditions × coefficients; no asymmetric refusal that could mask a swing.

## Setup

Identical to the parent contrastive null except `cache_injection: differential` is split into two single-task variants:

- **First task steered**: random direction added to first-presented task's tokens only (`spans: {first: 1}`).
- **Second task steered**: random direction added to second-presented task's tokens only (`spans: {second: 1}`).

Five seeds `s ∈ {0, 1, 2, 3, 42}`; each direction is a unit-normed `np.random.default_rng(s).standard_normal(5376)`. Seeds 0/1/42 reuse the directions from the parent contrastive null, so a future paired comparison sees identical directions on those three seeds.

Other settings match Fig 3a single-task: 30 evaluated pairs (deterministic `random.sample` of the harm-breakdown 150-pair set, run-seed=42 — same set as the parent contrastive null), Gemma-3-27B-IT, default Assistant system prompt, completion-preference template, temperature=1.0, max_new_tokens=64, n_trials=3, both task orderings.

## Limitations

- Right panel is 2 of 3 contrastive parent seeds (seed 42 parsed JSONL not synced locally). Does not affect the headline.
- N=30 pairs matches the parent. Larger pair sets would tighten the SEM but the swing CIs already comfortably include zero.
- No paired-seed comparison between unilateral and contrastive runs; deferred as future work.
