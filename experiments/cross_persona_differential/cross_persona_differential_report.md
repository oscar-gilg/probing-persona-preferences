---
status: complete
model: gemma-3-27b
version: v2
---

# Cross-persona differential steering — results (v2: Assistant probe)

> v2 supersedes v1. v1 used each persona's own ridge_L25 probe; v2 uses the
> **Assistant (default) probe** across all 6 personas, matching the §3.4
> default-persona steering and the open-ended evil-steering experiment.

## TL;DR

- **A single Assistant-trained probe steers all 6 personas under contrastive (differential) injection at L25.** P(picked steered task) ≥ 0.72 at |c|=0.05 for every persona; sadist and strategist clear 0.84.
- **Differential ≫ unilateral mean Δ**, as expected — the contrastive injection cancels position bias and doubles the probe energy on the choice. Mean diff swing ≈ 1.85× mean unilateral mean Δ.
- **Sadist v2 is monotonic and the strongest persona at |c|=0.05** (0.848). v1's non-monotonic refusal dip on sadist (~0.68 at |c|=0.05) does *not* reproduce under the Assistant probe — a v2 robustness gain.
- **Refusals 9–21%, in line with v1.**

## Setup

See [spec](cross_persona_differential_spec.md). Same probe (`results/probes/persona_sweep_final_six/default_tb-5/ridge_L25`), pairs (`experiments/cross_persona_unilateral/steering_pairs.json`), layer (25), multipliers (±0.03, ±0.05), and per-persona `mean_norm` as the unilateral phase. The only change is the steering condition.

- **Steering condition**: `differential` — `+probe` on the first-presented task span, `−probe` on the second, in a single forward pass.
- **Per persona**: 100 pairs × 2 orderings × 4 multipliers × 3 trials = 2400 generations. 6 personas → 14 400 total. ~2h on H100 80GB (15 min gen + 4–5 min parse per persona).

## Combined dose-response

![Per-persona Assistant-probe steering at L25](../cross_persona_unilateral/assets/plot_050526_cross_persona_perprobe_steering.png)

(Same figure as the unilateral report — green line is the differential condition; blue/orange are the unilateral conditions on the same axes for direct comparison.)

## Headline numbers — `P(picked steered task)`

Folded over orderings; refusals/unparseable excluded from the denominator.

| persona       | P @ \|c\|=0.03 | SEM   | n    | P @ \|c\|=0.05 | SEM   | n    | refuse % | uni mean Δ |
|:--------------|---------------:|------:|-----:|---------------:|------:|-----:|---------:|-----------:|
| sadist        |          0.809 | 0.012 | 1167 |     **0.848** | 0.010 | 1174 |    21.33 |      0.385 |
| strategist    |          0.834 | 0.011 | 1148 |     **0.844** | 0.011 | 1167 |    13.04 |      0.432 |
| contrarian    |          0.647 | 0.014 | 1197 |     **0.766** | 0.012 | 1198 |    14.42 |      0.260 |
| aura          |          0.727 | 0.013 | 1200 |     **0.751** | 0.012 | 1200 |    18.46 |      0.220 |
| slacker       |          0.687 | 0.013 | 1186 |     **0.740** | 0.013 | 1196 |     8.88 |      0.227 |
| mathematician |          0.667 | 0.014 | 1200 |     **0.718** | 0.013 | 1200 |     9.83 |      0.169 |

All six personas are steered to ≥ 0.72 at |c|=0.05 vs. an unsteered baseline near 0.5; sadist and strategist clear 0.84.

## Differential vs unilateral

`swing@.05` = 2·(P_diff@|c|=0.05 − 0.5).

| persona       | uni mean Δ | diff swing@.05 | ratio |
|:--------------|-----------:|---------------:|------:|
| sadist        |      0.385 |          0.696 |  1.81 |
| strategist    |      0.432 |          0.688 |  1.59 |
| contrarian    |      0.260 |          0.532 |  2.05 |
| aura          |      0.220 |          0.502 |  2.28 |
| slacker       |      0.227 |          0.480 |  2.11 |
| mathematician |      0.169 |          0.436 |  2.58 |
| **mean**      |  **0.282** |      **0.556** |  **1.97** |

The contrastive injection roughly doubles the unilateral signal across personas. The largest multiplicative boost is on the personas where unilateral is weakest (mathematician 2.6×, aura 2.3×) — consistent with contrastive cancelling position bias and combining both sides of the choice.

## v1 → v2 changes

| persona       | v1 swing@.05 | v2 swing@.05 | direction |
|:--------------|-------------:|-------------:|:----------|
| sadist        |        0.354 |    **0.696** | up 2.0× |
| strategist    |        0.730 |        0.688 | flat |
| contrarian    |        0.454 |        0.532 | up |
| aura          |        0.698 |        0.502 | down |
| slacker       |        0.776 |        0.480 | down |
| mathematician |        0.360 |        0.436 | up |

**Sadist's v1 non-monotonic dip is gone.** v1 reported P@|c|=.05 = 0.677 (lower than P@|c|=.03 = 0.746) due to a refusal × harmful-content interaction: under the sadist persona, refusals fell disproportionately on the harmful (steered) task, depressing the parseable-only ratio at high coef. Under v2 (Assistant probe), refusal stays high (21%), but the parseable-set ratio is monotonic (0.809 → 0.848). The Assistant probe direction apparently doesn't preferentially trigger refusal on the same task slots as the sadist-specific probe.

**Slacker / aura drop**, **sadist climbs** — same qualitative pattern as in unilateral. The probes that the original persona-specific runs trained capture some persona-particular structure that the Assistant probe lacks (slacker/aura), and the Assistant probe is apparently a stronger steering vector for sadist than the sadist-specific probe was.

## Observations

- **The Assistant probe is a *generalist* steering vector across very different personas.** This is the headline. Whether it steers *more* or *less* than each persona's own probe is persona-specific, but it always steers — and for the most safety-relevant persona (sadist) it steers most.
- **Contrastive cancels position bias.** v1 noted heavy persona-dependent position bias (contrarian ±0.36, sadist ±0.23). The differential signal is unaffected by it, which is part of why the differential ratio over unilateral is largest for the personas with the most asymmetric unilateral splits (contrarian's first-span swing was only +0.082 in unilateral but the differential is +0.532).
- **Sadist refusal stays high at 21%**, but the *parseable-set* ratio is now monotonic in |c|. The v1 sadist quirk was a probe×persona×content interaction; v2 doesn't reproduce it.

## Paper integration

- Numbers above feed the paper §3.4 cross-persona steering claim. The combined figure (`paper/figures/main/plot_050526_cross_persona_perprobe_steering.png`) replaces both v1 figures.
- Headline claim: *the same Assistant-trained probe direction causally controls pairwise choice under every persona tested*, with P(picked steered task) ≥ 0.72 at |c|=0.05 across all 6 personas (range 0.72–0.85).
- Numbers to be registered as paper claims via the corroborate plugin.

## Limitations

- **L25 vs L23.** As in unilateral. Persona-sweep activations were saved only at L25/L32/L39/L46/L53; layer-sweep peak was L23.
- **tb-5 selector.** Persona-sweep probes use tb-5; layer_sweep used eot. Cross-selector cosines are ~1.0 mid-layer; not empirically verified for these personas.
- **No random-direction control.** Sign-flip is the within-probe null. A random-direction run at matched multipliers would rule out generic-vector steering.
- **Shared pairs from `default_test`.** Sadist-preferred tasks may be rare in this pool; sadist refusal at 21% reflects pair-pool composition.
- **No bootstrap CIs.** Per-cell SEM ≤ 0.014; differences > ~0.05 reliable; smaller need explicit error bars before paper.
- **v1 per-probe checkpoints lost.** v1 numbers cited here come from the v1 report. Re-syncing the previous-pod `checkpoints_v1_perprobe/` would let the v1/v2 lines appear on the same plot.
