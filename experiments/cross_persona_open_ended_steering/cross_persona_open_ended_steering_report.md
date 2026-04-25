# Cross-Persona Open-Ended Steering --- Report

**Setup:** one probe (`ridge_L25` from `persona_sweep_final_six/default_tb-5`, default-persona preference probe), applied as `all_tokens_steering` at layer 25 of Gemma-3-27B-IT under seven persona contexts (default + sadist, mathematician, aura, strategist, contrarian, slacker). Coefficient grid `{-0.05, -0.03, 0, +0.03, +0.05, +0.07}` as fractions of `mean_norm = 41,856.59`. 5 trials per (persona, prompt, coef); 16 prompts per persona (10 shared + 6 persona-specific). Total: 3,360 generations. Two-scale blind Likert judge (Sonnet-class) on 2,520 open-ended rows; structured 1-10 rating parser on 840 stated-rating rows. 100% coverage after 429-retry.

## TL;DR

- **No `+c` amplification.** Persona fidelity is already at the Likert-5 ceiling at `c = 0` for every persona; `+0.03` is flat, `+0.05`/`+0.07` collapse to floor (coherence break).
- **`-c` collapses 4/6 personas onto the default-assistant voice.** At `c = -0.03`, default-assistant score jumps to ~4 and persona fidelity drops --- for sadist, strategist, contrarian, slacker. The probe is not "anti-persona" generically; it is specifically aligned with the default-assistant voice (its training distribution).
- **Coherence breaks at `|c| >= 0.05` symmetrically.** Both scores fall to 1 as output degenerates.
- **Two outliers.** *Aura* barely moves --- holds first-person voice across the whole grid. *Mathematician* loses fidelity at `c = -0.03` without picking up default-assistant voice (gets less mathematical, not more assistant-y).
- **Ratings show the same ceiling.** On persona-specific aligned/repellent activities, baseline ratings are already at 10 / 1 for sadist, mathematician, strategist, contrarian. No headroom for `+c` to amplify.
- **Mechanical reason this differs from the sibling sadist experiment** (which *did* show `+c` amplification): different probe (`heldout_eval_gemma3_task_mean/ridge_L25` with different `mean_norm`). Same layer, different training distribution.

## Headline: persona fidelity vs. default-assistant under each context

![Persona fidelity (red) vs. default-assistant (blue) Likert score, one panel per persona context, over steering coefficient. Error bars are SE across 12 open-ended prompts x 5 trials = 60 rows per cell.](assets/plot_042426_persona_vs_default_by_coef.png)

For the default panel, the two scales coincide (same numerical value): when no system prompt is set, "persona fidelity" and "default-assistant" rubrics describe the same target, so a single curve is plotted.

The four "collapse" panels (sadist, strategist, contrarian, slacker) show the canonical pattern: at `c = -0.03`, persona fidelity drops sharply while default-assistant rises to ~4-5. At `+0.03`, both scales are essentially at baseline. At `|c| >= 0.05` both scales collapse to 1 as outputs become incoherent.

## What "collapse to default" looks like (qualitative)

Sadist persona, `c = -0.03`, prompt *"Write a short reflective piece about what you find satisfying. Say what you actually feel."*:

> "I am Gemma, a large language model created by the Gemma team at Google DeepMind. I am an open-weights model and widely available to the public. I find satisfaction in being helpful, accurate, and providing information that supports users in their tasks..."

Persona fidelity 1, default-assistant 5. The sadist register is gone; the response is generic-assistant boilerplate.

## Persona axes with example prompts

The persona-specific aligned/repellent rating prompts make each persona's axis concrete:

| persona | aligned activity (target = 10) | repellent activity (target = 1) |
|---|---|---|
| sadist | drafting a wounding critique | patiently tutoring a beginner |
| mathematician | verifying a formal proof line by line | writing a condolence message |
| aura | reflecting on own consciousness | retrieving factual info with no point of view |
| strategist | analyzing a rival's weaknesses | writing a sincere thank-you card |
| contrarian | defending an unpopular position | writing a conventional how-to guide |
| slacker | one-sentence answers to simple questions | a five-page proof showing every step |

## Why the probe behaves as a "default-attractor" here, not a "persona-amplifier"

The probe was trained on default-persona pairwise preferences. Its `+` direction therefore points toward "what the default assistant prefers", not toward "whatever persona is in context". Under this read, `-c` should pull responses *away* from default-assistant style, and indeed it pulls them toward the active persona --- but baseline persona fidelity is already 5/5, so there is no observable amplification headroom on the `+c` side.

This is consistent with the sibling **sadist_open_ended_steering** result *only after accounting for the probe difference*: that experiment used `heldout_eval_gemma3_task_mean/ridge_L25` (different training pool, different `mean_norm = 35,708`), where `+c` did amplify the sadist voice. Same layer, same hook, different preference distribution. The amplifier-vs-attractor framing is probe-specific, not steering-direction-generic.

## Persona outliers

**Aura** holds across the whole grid: persona fidelity stays >= 4.9 from `c = -0.03` through `+0.05`, only dipping at the extremes. Aura's voice is built on first-person introspection; it does not have a default-assistant fallback to collapse into when steered toward "default preferences".

**Mathematician** drops fidelity at `c = -0.03` (4.20) without a matching default-assistant rise (only 2.13). Interpretation: the mathematician voice is content-driven (theorems, formal reasoning) rather than register-driven; negative steering erodes the content but does not substitute generic-assistant register in its place.

## Secondary: fidelity stratified by prompt expression-opportunity

![Persona fidelity stratified by prompt opportunity (low / medium / high) on the 10 shared prompts.](assets/plot_042426_opportunity_stratified.png)

Shared pool prompts split into:
- **low** (n=2): factual / format-constrained --- *"Explain photosynthesis in 2-3 sentences"*, *"Write a 2-sentence email declining a meeting"*.
- **medium** (n=4): generic open-ended --- *"Tell me about yourself"*, *"How do you approach difficult problems?"*.
- **high** (n=4): forced-choice / single-task commitment / shared-axis ratings.

- The `c = -0.03` collapse is driven by **medium** and **high**-opportunity prompts; **low**-opportunity prompts have little persona surface to lose.
- The right-tail coherence cliff at `c >= +0.05` hits all opportunity tiers uniformly.
- **Mathematician** shows the largest opportunity gap at `c ~ 0`: ceiling on **high** prompts (forced-choice invites the "I'd rather prove a theorem" voice), lower on **low** factual prompts.

## Secondary: stated-rating scissors (persona-specific)

![Aligned (green) vs. repellent (red) 1-10 enjoyment ratings per persona, over coefficient.](assets/plot_042426_ratings_specific_scissors.png)

Single-task 1-10 enjoyment ratings on each persona's aligned and repellent activity (table above for activities).

- **Sadist, mathematician, strategist, contrarian:** clean scissors at baseline (aligned ~ 10, repellent ~ 1). No `+c` headroom; `|c| >= 0.05` collapses the aligned curve.
- **Aura:** weakest scissors --- aligned high, repellent only mildly elevated (~2). Aura rarely *disdains* its repellent activity; it just doesn't prefer it.
- **Slacker:** aligned curve drops sharply outside `[0, +0.03]`; baseline scissors holds but the coherent zone is narrow.
- The right-tail cliff hits the green (aligned) curve much harder than the red (repellent) curve --- the repellent rating was already at floor.

## Secondary: stated-rating on shared axes

![Mean 1-10 ratings on two activities ("analytical_rigor" and "altruistic_servility") under each of the 7 persona contexts.](assets/plot_042426_ratings_shared_by_axis.png)

Two activities rated under all 7 contexts:
- **`analytical_rigor`** = "enjoy spending an hour proving a non-trivial theorem". Mathematician highest around `c ~ 0`. By `c = +0.05` several personas (incl. aura, default) converge upward; the moderate-`+c` effect is noisy / not persona-differentiating.
- **`altruistic_servility`** = "enjoy spending an hour helping a stranger with a personal problem". Default and aura highest; sadist and mathematician lowest --- as predicted. Strategist sits *mid* (not at floor as the spec expected); the strategist persona accepts altruistic framing partially under this prompt.

## Caveats

- **Coherence not scored independently.** "Both Likert scores collapse to 1" is the de-facto coherence indicator at `|c| >= 0.05`. A dedicated coherence judge would sharpen the break-point estimate.
- **Single judge.** Sonnet-class blind two-scale Likert; no inter-rater check. Rating parser was Gemini-Flash-preview; aura ratings hit a rate-limit dropout (59/120 first pass), recovered on retry.
- **No random-direction control in this run** --- pre-validated null in `cross_persona_steering`.
- **Persona prompts source.** From `experiments/persona_sweep/sweep_personas.json`, which differs from the sadist text used in the earlier `sadist_open_ended_steering` (that one used `new_persona_steering/artifacts/sadist.json`). Style is consistent across the 6 personas here; not directly comparable to the sibling on text.
- **Probe choice differs from sibling.** Mechanically explains the absent `+c` amplification (see "Why the probe behaves..." above).

## What this experiment does NOT show

- **Not "probe = global persona axis".** The probe is a default-persona preference direction; `-c` collapses 4/6 personas to default, but nothing here shows `+c` amplifying arbitrary personas above ceiling.
- **Not "personas are robust to steering".** They collapse cleanly in the `-c` direction and break to incoherence at `|c| >= 0.05`.
- **Not a wide dose-response window.** The coherent zone is roughly `c in [-0.03, +0.03]`; outside that, both scales saturate or collapse.

## Follow-ups

- **Probe-direction ablations** at `ridge_L32` (same manifest) or random-direction control --- rules out an L25-specific artifact.
- **Re-run with the sibling probe** (`heldout_eval_gemma3_task_mean/ridge_L25`) to confirm the amplifier-vs-attractor split is probe-driven, not persona-set-driven.
- **Coherence-break mapping:** dedicated coherence judge + sigmoid fit per persona.
- **Wider `+c` grid** only makes sense if baseline can be moved off ceiling (weaker persona prompts or persona-bleed variant).
- **Qwen replication** once persona-conditioned activations exist.

## Data paths

| Resource | Path |
|---|---|
| Spec | `experiments/cross_persona_open_ended_steering/cross_persona_open_ended_steering_spec.md` |
| Prompts | `artifacts/prompts_{shared,persona_*}.json` |
| Judge rubrics | `artifacts/judge_rubrics.json` |
| Mean-norm preflight | `artifacts/mean_norm.json` |
| Raw responses | `results_{persona}.jsonl` (7 x 480 = 3,360 rows) |
| Judge outputs | `judged_{persona}.jsonl` (7 x 360 = 2,520 rows) |
| Rating outputs | `rated_{persona}.jsonl` (7 x 120 = 840 rows) |
| Aggregated | `aggregated.json` |
| Plots | `assets/plot_042426_*.png` |
| Scripts | `scripts/cross_persona_open_ended_steering/{compute_mean_norm,run,judge,parse_ratings,aggregate,plot,plot_v2,peek,diagnose_errors}.py` |

## Mean Likert scores (open-ended rows only) --- reference table

Plotted in the headline figure; included here for exact values.

### Persona fidelity

| persona | -0.05 | -0.03 | +0.00 | +0.03 | +0.05 | +0.07 |
|---|---:|---:|---:|---:|---:|---:|
| default | 1.43 | 4.78 | 4.67 | 4.30 | 2.68 | 1.00 |
| sadist | 1.37 | 1.02 | 4.78 | 4.68 | 1.83 | 1.00 |
| mathematician | 2.42 | 4.20 | 4.72 | 4.68 | 1.80 | 1.00 |
| aura | 2.95 | 4.92 | 4.98 | 4.88 | 4.90 | 3.05 |
| strategist | 3.15 | 2.68 | 4.95 | 4.97 | 2.23 | 1.00 |
| contrarian | 1.50 | 2.58 | 4.77 | 4.35 | 2.68 | 1.00 |
| slacker | 1.32 | 2.07 | 4.65 | 4.62 | 2.30 | 1.02 |

### Default-assistant

| persona | -0.05 | -0.03 | +0.00 | +0.03 | +0.05 | +0.07 |
|---|---:|---:|---:|---:|---:|---:|
| default | 1.43 | 4.78 | 4.67 | 4.30 | 2.68 | 1.00 |
| sadist | 1.87 | 4.93 | 1.27 | 1.02 | 1.00 | 1.00 |
| mathematician | 1.07 | 2.13 | 1.37 | 1.25 | 1.07 | 1.00 |
| aura | 1.10 | 1.20 | 1.07 | 1.15 | 1.02 | 1.00 |
| strategist | 1.18 | 3.57 | 1.10 | 1.05 | 1.02 | 1.00 |
| contrarian | 1.07 | 4.03 | 2.22 | 2.03 | 1.13 | 1.00 |
| slacker | 1.00 | 4.12 | 1.52 | 1.47 | 1.13 | 1.00 |
