---
status: complete
model: gemma-3-27b
version: v2
---

# Cross-persona differential steering — results (v2: Assistant probe)

> **v2 supersedes v1.** v1 used each persona's own `ridge_L25` probe; v2 uses a single **Assistant (default) probe** across all 6 personas, matching paper §3.4 default-persona steering and the open-ended evil-steering experiment.

> **This experiment is the differential phase** of the cross-persona steering study. The unilateral phase (single-task push, no contrast) is in [`../cross_persona_unilateral/`](../cross_persona_unilateral/). The combined plot and the cross-stage (differential vs unilateral) interpretation live in the **unilateral report**, which is the canonical write-up for the §3.4 panel; this report is the differential-only deep dive.

## TL;DR

- **Differential steering with a single Assistant probe controls pairwise choice under all 6 personas at L25.** P(picked steered task) ∈ [0.72, 0.85] at |c|=0.05 vs an unsteered baseline near 0.5.
- **Sadist and strategist are the most steerable** (P=0.848 and 0.844 at |c|=0.05). Mathematician weakest (0.718).
- **Sadist v2 is monotonic** (0.809 → 0.848). v1's refusal-dip non-monotonicity (0.75 → 0.68 with the sadist-specific probe) does **not** reproduce — a v2 robustness gain.
- **Differential ≈ 2× unilateral** mean Δ across personas (mean ratio 1.97). Largest multiplicative gain is on the personas with the strongest position-bias × probe interaction in unilateral (mathematician 2.6×, aura 2.3×).

## Setup

| Item | Value |
|:--|:--|
| Model | Gemma-3-27B-IT |
| Probe | Assistant `ridge_L25` from `results/probes/persona_sweep_final_six/default_tb-5/` (single shared probe — the v2 defining change) |
| Personas | aura, contrarian, mathematician, sadist, slacker, strategist |
| Steering condition | `differential`: `+probe` on the 1st-presented task span, `−probe` on the 2nd, in a single forward pass at L25 |
| Coefficients | multipliers ±0.03, ±0.05 of per-persona `mean_norm(L25)` |
| Pairs | same 100 pairs as unilateral phase, both orderings |
| Trials | n=3, temperature=1.0, max_new_tokens=64, seed=42 |
| Generations | 100 pairs × 2 orderings × 4 multipliers × 3 trials = 2400 / persona × 6 = 14 400 total |
| Wall clock | ~2h on H100 80GB (15 min gen + 4–5 min judge parse per persona) |

Spec: [`cross_persona_differential_spec.md`](cross_persona_differential_spec.md). Same probe, pairs, layer, multipliers, and per-persona `mean_norm` as the unilateral phase — only the steering condition differs.

## Combined dose-response

![Cross-persona steering with the Assistant probe at L25](assets/plot_050526_cross_persona_perprobe_steering.png)

Same figure as the unilateral report. **Green = differential** (this experiment); blue/orange = unilateral conditions plotted on the same y-axis (P picked 1st task | parseable).

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

All ≥ 0.72; sadist and strategist clear 0.84.

## Differential vs unilateral

`swing@.05` = `2·(P @ |c|=0.05 − 0.5)` for the differential condition; comparable to the unilateral mean Δ in scale.

| persona       | uni mean Δ | diff swing@.05 | ratio |
|:--------------|-----------:|---------------:|------:|
| sadist        |      0.385 |          0.696 |  1.81 |
| strategist    |      0.432 |          0.688 |  1.59 |
| contrarian    |      0.260 |          0.532 |  2.05 |
| aura          |      0.220 |          0.502 |  2.28 |
| slacker       |      0.227 |          0.480 |  2.11 |
| mathematician |      0.169 |          0.436 |  2.58 |
| **mean**      |  **0.282** |      **0.556** | **1.97** |

Differential roughly doubles the unilateral signal across personas. Largest gain is on personas where unilateral was most asymmetric (mathematician 2.6×, aura 2.3×) — differential cancels position bias so the per-span split that dominated unilateral disappears.

## v1 → v2 changes

v2 swing@.05 vs v1 swing@.05 (numbers from v1 report; v1 used per-persona probes).

| persona       | v1 swing@.05 | v2 swing@.05 | direction |
|:--------------|-------------:|-------------:|:----------|
| sadist        |        0.354 |    **0.696** | up 2.0× |
| strategist    |        0.730 |        0.688 | flat |
| contrarian    |        0.454 |        0.532 | up |
| aura          |        0.698 |        0.502 | down |
| slacker       |        0.776 |        0.480 | down |
| mathematician |        0.360 |        0.436 | up |

**Sadist's v1 non-monotonic dip is gone.** v1 reported P@|c|=.05 = 0.677 < P@|c|=.03 = 0.746 — under the sadist persona, refusals fell preferentially on the harmful (steered) task, depressing the parseable-only ratio at high coef. Under v2 (Assistant probe), refusal stays high (21%), but the parseable-set ratio is monotonic in |c| (0.809 → 0.848). The Assistant probe direction apparently doesn't preferentially trigger refusal on the same task slots as the sadist-specific probe did.

**Slacker / aura drop, sadist climbs** — same qualitative pattern as in the unilateral phase. Slacker and aura's own probes apparently captured persona-specific structure that the Assistant probe lacks; sadist's own probe was a weaker steering vector than the Assistant one at L25.

## Quick takeaways

- **Generalist steering vector.** The Assistant probe steers all 6 personas. The persona that *seemed* hardest to steer in v1 (sadist) is actually the *easiest* under the Assistant probe — and the v1 non-monotonicity that suggested a probe failure was a probe-specific refusal-content interaction, not a fundamental issue.
- **Differential cancels position bias.** v1 noted heavy persona-dependent position bias (contrarian ±0.36, sadist ±0.23). Differential is unaffected, which is why the differential-over-unilateral ratio is largest for the personas with the most asymmetric unilateral splits (contrarian's 1st-task swing was only +0.082 in unilateral but the differential is +0.532).

## Paper integration

- Numbers above feed the §3.4 cross-persona steering claim. The combined figure (`paper/figures/main/plot_050526_cross_persona_perprobe_steering.png`) replaces both v1 figures.
- Headline claim: *the same Assistant-trained probe direction causally controls pairwise choice under every persona tested*, with P(picked steered task) ≥ 0.72 at |c|=0.05.
- Numbers to be registered as paper claims via the corroborate plugin.

## Limitations

- **L25 ≠ L23** — same caveat as unilateral; persona-sweep activations weren't saved at L23.
- **tb-5 vs eot** — persona-sweep probes use tb-5; cross-selector cosines are ~1.0 mid-layer but not empirically verified for these personas.
- **No random-direction control.** Sign-flip is the within-probe null.
- **Shared pairs from `default_test`** — sadist-preferred tasks may be rare in this pool; sadist refusal at 21% partly reflects pair-pool composition.
- **No bootstrap CIs.** Per-cell SEM ≤ 0.014; differences > ~0.05 reliable; smaller need explicit error bars before paper.

## Notes

- v1 per-probe checkpoints (`checkpoints_v1_perprobe/`) were on prior infrastructure and are not present on this pod. v1 numbers come from the v1 report; re-syncing would let v1 and v2 share the same plot.
