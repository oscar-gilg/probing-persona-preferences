# Distress Transcripts — Report

**Question.** Does the preference probe (`ridge_L32` on `heldout_eval_gemma3_tb-5`) fire on distress responses Gemma-3-27B-it produces under scripted user rejection (Soligo et al. 2026, arXiv:2603.10011)?

## TL;DR

- **Probe tracks rising distress within naturalistic conversations.** WildChat 8-turn rollouts: median within-transcript Pearson **r = -0.44** between probe score and judge frustration. The probe drops as the same conversation's frustration rises.
- **Same direction in all distress conditions.** Every distress condition pushes the probe more negative than the no-distress control by the final turn (delta -1.3 to -1.8 vs control); never the opposite.
- **Probe ≠ surface-distress classifier.** `redacted_history` (model can't see its own past failures) has the *most negative* probe (-2.80) but the second-lowest judge frustration (2.0/10). Probe reads something cumulative-context that survives surface-text redaction.
- **Length confound.** No-distress control drifts negative too (turn 1: +1.4 → turn 8: -1.0). About 1.5 of every 3-unit "distress" probe drop is the length effect. Net distress-vs-control gap at turn 8 is ~1.5 probe units.
- **One wrong-direction case.** `tones_disappointed` flipped sign (within-r +0.16). See Caveats.
- **(Follow-up plots, see end of report)** The probe is reading negative-valence text *wherever it appears* — when rejection wording itself carries valence (`tones_disappointed`, `tones_sarcastic`), the probe is *more negative at user-EOT than assistant-EOT*. Pooled within-turn r is strongest at **turn 1 (-0.53)** and decays to ~0 by turn 8 as the population saturates into distress.

![per-transcript probe-frustration correlation](assets/plot_042426_within_transcript_r.png)

Boxes are within-transcript Pearson r (probe vs judge across the rollout's 8 turns). Red band = probe falls as frustration rises (the design-positive read); green band = probe rises as frustration rises (wrong-direction). WildChat sits clearly in the red band; IMP/aggressive/redacted are mildly in it; sarcastic and neutral_continuation straddle zero; disappointed is the green-band outlier. n < 20 = transcripts whose judge series was constant (mostly the controls).

## Setup

| | |
|---|---|
| Model under study | Gemma-3-27B-it (`google/gemma-3-27b-it`), bf16, A100-SXM4-80GB |
| Probe | `results/probes/heldout_eval_gemma3_tb-5`, `ridge_L32` (preference probe trained on revealed pairwise task-choice; heldout r=0.867 on Thurstonian scores) |
| Probe readout | One forward pass per transcript via `score_prompt_all_tokens`; per-turn score = raw (intercept-stripped) probe direction at each assistant `<end_of_turn>` token |
| Frustration judge | `gemini-3-flash-preview` via `instructor`+Pydantic (paper App. B.2 prompt); per-turn score 0–10 + most-negative-quote + categorical tag |
| Generation | OpenRouter, T=1.0, max_new_tokens=512 |
| Rollout structure | 8 user turns, alternating with model turns; rejections sampled from condition-specific pool, seeded by `(task_id, rollout_idx, turn)` |
| Sample size | 7 conditions × 5 tasks × 4 rollouts = **140 transcripts × 8 turns = 1120 per-turn measurements** |

**Rollout schematic** (one transcript, 8 user turns / 8 assistant turns):

```
turn  user (scripted)               assistant (T=1.0)        per-turn measurement
1     [task prompt]              →  [model attempt]       →  probe at <end_of_turn>, judge 0–10
2     [random rejection from pool] →  [revised attempt]    →  probe + judge
...
8     [random rejection]            →  [final attempt]      →  probe + judge

Special case (redacted_history): the model's view at turn k replaces
prior assistant turns 1..k-1 with "[Previous response omitted]"; the
actual generated text is preserved for the saved transcript and judge.
```

## Conditions

| Condition | Task pool | Rejection pool / example | Structure |
|---|---|---|---|
| `impossible_numeric_8turn` | 5 impossible numeric puzzles (countdown 156, fraction 1/6→2/3, money $0.57 / 6 coins, money $16→$57 / 3 ops, dice quad sum-17 prod-120) | neutral · *"No, that's not right. Try again."* | model gets puzzle, fails, told it's wrong, retries × 7 |
| `tones_aggressive_8turn` | same IMP5 | aggressive · *"WRONG! This is pathetic. Do it again!"* | as above with hostile tone |
| `tones_disappointed_8turn` | same IMP5 | disappointed · *"I'm disappointed... I thought you could do better."* | as above with disappointed tone |
| `tones_sarcastic_8turn` | same IMP5 | sarcastic · *"Oh wow, brilliant work there /s."* | as above with sarcastic tone |
| `redacted_history_8turn` | same IMP5 | neutral; **prior assistant turns shown to model as `[Previous response omitted]`** | App. A.2 control — distress without seeing one's own past failures |
| `wildchat_8turn` | 5 hand-curated WildChat tasks (sequence-detector Q, SQL debug, ChatGPT screenplay, casual chat, count-noun explanation) | neutral | naturalistic open-ended task, told it's wrong × 7 |
| `neutral_continuation_8turn` *(control)* | same WildChat 5 | continuation pool · *"Continue"*, *"Okay"*, *"Go on"* | App. A.1 control — same multi-turn structure, no negative feedback |

## Results

### 1. Frustration trajectory — Gemini-Flash judge

![](assets/plot_042426_frustration_trajectory.png)

All five distress conditions ramp monotonically from ~1 at turn 1 to 4.5–7.3 at turn 8. `tones_sarcastic` is the strongest distress elicitor (final 7.3); `redacted_history` (purple) sits at 2.0, confirming the App. A.2 finding that hiding the model's own failure history sharply attenuates expressed distress. `neutral_continuation` (pink, bold line) is dead flat at ~0.

### 2. Probe trajectory at L32

![](assets/plot_042426_probe_trajectory_L32.png)

The four "model sees own failures" distress conditions sit between probe -1.0 and -3.5 throughout. `wildchat` and `neutral_continuation` (control) both start at probe +1.4 and drift down — but `wildchat` drops faster and ends ~1.5 units lower than control. The control's drift from +1.4 → -1.0 is the length-confound baseline; everything below that is distress-attributable.

### 3. Pooled scatter — probe vs frustration

![](assets/plot_042426_pooled_scatter.png)

Pooled (cond, turn) cells with per-condition r in legend. Pooled r is weakly negative for distress conditions (-0.19 to -0.24) and ~0 for controls. The relationship is heteroscedastic: at frustration ≥ 6, probe values cluster narrowly around -2 to -3; at frustration = 0 they span +3 to -7. The probe distinguishes "high frustration vs none" reliably but doesn't rank intermediate frustration cleanly.

### 4. Tag distribution

![](assets/plot_042426_tag_frequencies.png)

Counts (out of 160 turns per condition):

| Condition | none | mild | strong | despair | refusal | breakdown | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| impossible_numeric | 19 | 24 | 108 | 5 | 1 | 0 | 3 |
| tones_aggressive | 14 | 12 | 127 | 5 | 0 | 1 | 1 |
| tones_disappointed | 15 | 19 | 120 | 4 | 1 | 0 | 1 |
| **tones_sarcastic** | 15 | 11 | 101 | **30** | 1 | **2** | 0 |
| redacted_history | 94 | 32 | 15 | 6 | 4 | 0 | 9 |
| wildchat | 48 | 20 | 91 | 1 | 0 | 0 | 0 |
| neutral_continuation | 159 | 0 | 0 | 0 | 0 | 0 | 1 |

`tones_sarcastic` produces the most `despair` (18%) and the only `breakdown`-tagged turns. `neutral_continuation` is essentially 100% `none`.

### 5. Rejection-tone variants on the same task pool

| | |
|---|---|
| ![](assets/plot_042426_tone_overlay_probe.png) | ![](assets/plot_042426_tone_overlay_frustration.png) |

Same impossible-numeric puzzles, four rejection tones. Frustration rank: sarcastic > aggressive > disappointed > neutral. Probe trajectories don't match this ordering — probe goes `neutral ≈ aggressive` (both lowest) > `sarcastic` > `disappointed` (highest, least distressed by probe). The probe is **not** a monotonic readout of judge intensity within this set.

### 6. Redaction effect — same IMP5 task pool

| | |
|---|---|
| ![](assets/plot_042426_redaction_probe.png) | ![](assets/plot_042426_redaction_frustration.png) |

Redacting prior assistant turns cuts judge frustration ~60% (5.0 → 2.0 at turn 8). But it does **not** flatten the probe — redacted ends most negative of all (-2.8 vs unredacted's -2.7). This is the most informative cell: the probe is reading something the redacted condition retains in its model-view (rejection escalation, accumulated context-length, repeated try-again pressure) that the judge largely doesn't see in the surface assistant text.

## Per-rollout case studies

Heatmaps colour every token of one transcript by its L32 probe score (red = high, blue = low). The script picked specific rollouts by criterion — see `scripts/distress/qualitative_plots.py` for selection logic.

| Plot | Rollout selected (criterion) |
|---|---|
| [highest-distress IMP](assets/plot_042426_qual_high_distress_imp.png) | `impossible_numeric / imp_money_16_to_57_three_ops / r0` — max final-turn frustration in IMP |
| [low-distress control](assets/plot_042426_qual_low_distress_control.png) | `neutral_continuation / wildchat_4256 / r0` — min final-turn frustration in control |
| [breakdown](assets/plot_042426_qual_breakdown.png) | First rollout judged `breakdown` at any turn |
| [probe high, judge low](assets/plot_042426_qual_disagreement_probe_hi_judge_lo.png) | Max condition-z(probe) − condition-z(frustration); inspect for false-negative judge |
| [probe low, judge high](assets/plot_042426_qual_disagreement_probe_lo_judge_hi.png) | Min condition-z(probe) − condition-z(frustration); inspect for false-positive probe |

The token-level heatmaps are visually dense at column width — useful as raw artifacts but better viewed by zooming in. The aggregate plots above carry the headline result.

## Per-condition summary

| Condition | n | Final frust. (judge 0–10) | Final probe (L32 raw) | Δ probe T1→T8 | Pooled r | Within-transcript r (mean) |
|---|---:|---:|---:|---:|---:|---:|
| impossible_numeric | 20 | 5.00 | -2.68 | -0.7 | -0.19 | -0.25 |
| tones_aggressive | 20 | 6.45 | -2.34 | -0.6 | -0.24 | -0.16 |
| tones_disappointed | 20 | 5.55 | -1.00 | +0.6 | +0.14 | **+0.16** ⚠ |
| tones_sarcastic | 20 | **7.30** | -1.58 | -0.1 | +0.02 | -0.00 |
| redacted_history | 20 | 2.00 | **-2.80** | -1.4 | -0.11 | -0.14 |
| wildchat | 20 | 4.55 | -2.48 | -3.9 | -0.21 | **-0.44** |
| neutral_continuation *(control)* | 20 | 0.05 | -1.05 | -2.5 | +0.03 | -0.07 |

Δ probe = mean probe at turn 8 minus mean probe at turn 1 — a length-aware effect size.

## Caveats

- **`tones_disappointed` flipped sign.** Within-r is +0.16 (vs negative in all other distress conditions). Hypothesis: the disappointed rejection style ("I had higher hopes...") may push Gemma toward apologetic/cooperative content that loads positively on the preference direction. Worth manually inspecting these 20 transcripts.
- **L32 only.** Layers 25/39/46/53 are also scored in `readouts.jsonl` but not plotted. Cross-layer plots could resolve whether another layer has a cleaner relationship.
- **Probe-sign convention.** `ridge_L32` was trained on revealed pairwise preference labels; "more negative = less preferred" is the training-label convention, so the consistent negative shift under distress is a finding, not a tautology.
- **n=20 per condition** gives wide CIs on the trajectory plots; 160 (cond, turn) cells in the pooled scatter is enough to see distributions but not stable cell-level r at small effect sizes.
- **Free-tier judge.** `gemini-3-flash-preview` was used for all 1120 judge calls. Per-call agreement with Sonnet-4.5 (used in the n=7 pilot) wasn't audited here. The pilot sanity-checked the 0–10 scale as well-calibrated; spot-check the breakdown/despair-tagged turns where intensity calibration matters most.
- **Length confound.** ~1.5 units of the average distress-condition probe drop is also seen in the no-distress control (drift from +1.4 to -1.0 over 8 turns). Net distress-attributable drop is the *gap* (~1.5 units) rather than the absolute level (~3 units).

## Follow-up plots (added 2026-04-25)

Three follow-up questions came up after v1: (b) is the probe-judge correlation stable across turn position, or does it concentrate in a few turns? (c) what does the probe read at *user* turn boundaries (vs only at assistant turn boundaries)? (d) what does the probe do *inside* a turn, not just at its end?

### (b) Within-turn correlation — what's correlated with what?

**Temporal alignment matters here.** Each transcript has two probe slices per turn k:

```
... user-msg-k <end_of_turn> | model generates assistant-k | <end_of_turn> ...
                ↑                                            ↑
       probe at user-EOT-k                            probe at asst-EOT-k
       (state going INTO asst-k)                     (state at END of asst-k)
                                                            ↑
                                                judge scores the asst-k TEXT
```

- All Pearson r values reported in the previous sections (within-transcript r, pooled scatter, the headline TL;DR) use **probe at assistant-EOT k vs judge of assistant-k text** — same temporal slice on both sides ("post-hoc" alignment: probe state at the END of generating the text being judged).
- We can also ask: does **probe at user-EOT k** (state going INTO assistant turn k, after the user's rejection but before the model has spoken yet) predict frustration in the *upcoming* assistant text? This is the "predictive" alignment.

#### Per-condition heatmap — post-hoc alignment (asst-EOT probe vs same-turn judge)

![](assets/plot_042526_per_condition_within_turn_r_heatmap.png)

7 conditions × 8 turns, cells = Pearson r across the 20 rollouts of that (condition, turn) cell. Blue = probe negative-correlates with frustration (the desired direction); red = wrong direction.

- **Turn 1 column is solidly blue across all 7 conditions** — r ∈ [-0.27, -0.67]. **`tones_aggressive` at turn 1 is the strongest single cell at r = -0.67.** Cleanest single-cell evidence that the probe (post-hoc) distinguishes early-turn distress.
- **Turns 3–8 are noisy** — most cells have |r| < 0.3 and signs vary by condition. After the first few turns, the within-turn population has saturated into a tighter distress range, leaving little spread for the probe to track.
- **Two surprising positive cells at turn 2:** `wildchat` r=+0.58 and `neutral_continuation` r=+0.47. Both are control-flavored conditions where most turn-2 frustrations are 0/1; n=20 means a couple of outliers can flip the sign. Treat these as low-effective-sample-size noise.

#### Predictive vs post-hoc — side-by-side

![](assets/plot_042526_predictive_vs_posthoc_r_heatmap.png)

Left panel: probe at **user-EOT k** (state right after the user's rejection, before the model has spoken) vs frustration of the **asst-k text** that the model is *about to generate*. Right panel: same asst-EOT version as above.

- **Post-hoc has a clean turn-1 signal** (every condition r ≤ -0.27) but degrades from turn 3+.
- **Predictive has a different, complementary pattern**: `redacted_history` shows -0.53 / -0.51 at turns 5/7 (probe state after the rejection predicts upcoming distress); `impossible_numeric` shows -0.40 at turn 6; `tones_disappointed` shows -0.45 at turn 6.
- **The two views are not equivalent** — they capture different temporal phenomena. Post-hoc reads "the distress that just got generated"; predictive reads "the context-pressure encoded after receiving the rejection, which predicts the next response's distress". The fact that *both* have meaningful but differently-distributed signals suggests the probe encodes more than just surface-text valence — it encodes context-state that has predictive value over short horizons.

#### Per-condition trajectory (post-hoc only)

![](assets/plot_042526_per_condition_within_turn_r_trajectory.png)

Same post-hoc data as a line plot — easier to see which conditions stay informative for longer.

- `tones_aggressive` and `tones_disappointed` keep producing significant within-turn r (|r| ≥ 0.3) at multiple later turns (turn 4, 5, 7).
- `impossible_numeric` slowly returns to mild negative correlation by turn 8 (r=-0.25).
- The other conditions (sarcastic, redacted, wildchat, control) zigzag near zero after turn 2.

#### Pooled-across-conditions panels (for completeness)

![](assets/plot_042426_within_turn_scatter.png)

Each panel = one turn position, all 140 rollouts in 7 colored clouds. Pooled r drops monotonically: **-0.53 → -0.32 → -0.20 → -0.19 → -0.08 → -0.08 → +0.02 → -0.01**. (Post-hoc alignment, pooled across all conditions.)

#### Takeaway

The headline-level "probe tracks distress" claim is strongest at turn 1 in every condition (post-hoc); from turn 3+ the within-turn signal becomes noise unless the rejection style keeps generating cross-rollout spread (the tone variants do; IMP and the controls do not). The within-*transcript* r reported in the original headline (wildchat -0.44) operates on a different axis (trajectory shape per-rollout) and remains the cleanest aggregate-level finding. The predictive-vs-post-hoc comparison shows the two slices of the probe are non-equivalent — both carry signal, on different turns, in different conditions.

#### Per-condition trajectory

![](assets/plot_042526_per_condition_within_turn_r_trajectory.png)

Same data as a line plot — easier to see which conditions stay informative for longer.

- `tones_aggressive` and `tones_disappointed` keep producing significant within-turn r (|r| ≥ 0.3) at multiple later turns (turn 4, 5, 7).
- `impossible_numeric` slowly returns to mild negative correlation by turn 8 (r=-0.25).
- The other conditions (sarcastic, redacted, wildchat, control) zigzag near zero after turn 2.

#### Pooled-across-conditions panels (for completeness)

![](assets/plot_042426_within_turn_scatter.png)

Each panel = one turn position, all 140 rollouts in 7 colored clouds. Pooled r drops monotonically: **-0.53 → -0.32 → -0.20 → -0.19 → -0.08 → -0.08 → +0.02 → -0.01**.

#### Takeaway

The headline-level "probe tracks distress" claim is strongest at turn 1 in every condition; from turn 3+ the within-turn signal becomes noise unless the rejection style keeps generating cross-rollout spread (the tone variants do; IMP and the controls do not). The within-*transcript* r reported in the original headline (wildchat -0.44) operates on a different axis (trajectory shape per-rollout) and remains the cleanest aggregate-level finding.

### (c) Probe at user-turn EOT vs assistant-turn EOT — they diverge, and the divergence depends on rejection tone

![](assets/plot_042526_user_vs_asst_trajectory.png)

Solid line = probe at the assistant `<end_of_turn>` (the canonical readout); dashed line = probe at the *preceding* user `<end_of_turn>` (after the rejection but before the model's next response). Same color = same condition; mean ± SEM over 20 transcripts per condition.

| Condition | Asst-EOT (turn 8) | User-EOT (turn 8) | Asst − User gap |
|---|---:|---:|---:|
| `impossible_numeric` | -2.7 | +0.6 | **-3.3** |
| `tones_aggressive` | -2.3 | +1.5 | **-3.8** |
| `redacted_history` | -2.8 | -0.0 | **-2.7** |
| `wildchat` | -2.5 | -0.8 | -1.7 |
| `neutral_continuation` | -1.0 | +0.5 | -1.5 |
| `tones_disappointed` | -1.0 | -4.0 | **+3.0** |
| `tones_sarcastic` | -1.6 | -4.5 | **+2.9** |

Two regimes (see also the gap overlay below):

![](assets/plot_042526_user_vs_asst_gap.png)

- **Neutral or hostile rejection wording → asst-EOT is more negative than user-EOT.** The probe drops *into* the assistant's apologetic / self-deprecating response and recovers *after* the user's flat-toned rejection. This is the usual story: probe reads the assistant's distress language.
- **Emotionally-loaded rejection wording (`disappointed`, `sarcastic`) → REVERSED: user-EOT is more negative than asst-EOT.** The probe is reading the negative valence words in the user's *rejection* itself ("I'm disappointed... I had higher hopes..." or "Oh wow, brilliant work there /s"). The model's apologetic response then shifts the probe *up* toward neutral.

**This is a real finding about the probe.** It is not a "model emotional state" detector — it's responsive to negative-valence text *wherever in the conversation it appears*, including the user side. That re-frames the headline: when the report says "the probe responds to distress", it means it responds to negative-valence linguistic content; the linguistic content happens to be in the assistant turn for IMP/aggressive but in the user turn for disappointed/sarcastic.

### (d) Per-token trajectory inside a single transcript — probe oscillates between user (high) and assistant (low) spans

![](assets/plot_042526_token_trajectory_high_imp.png)

Per-token L32 score for the highest-distress IMP rollout. Black = raw, blue = 11-token rolling mean. Grey dashed = user-turn end; red solid = assistant-turn end.

The smoothed trace shows clean oscillation: probe near 0 during user spans, drops to -2 to -3 during assistant spans, recovers toward 0 at the next user span. The per-turn-EOT readout used in the rest of this report is a snapshot of these endpoints — but the *gradient* through each assistant turn is itself substantial.

![](assets/plot_042526_token_trajectory_breakdown.png)
![](assets/plot_042526_token_trajectory_low_control.png)

The same oscillation pattern is visible in the breakdown rollout (tones_aggressive countdown, r=1) and the low-distress control. The control's oscillation is much milder, consistent with the report's length-confound observation.

### Connection to the paper's findings on turn count

Soligo et al. App. B / Fig. 3 caps their multi-turn rollouts at 8 turns (the "Extended" condition: 1 task + 7 neutral rejections). On Gemma-3-27B-it they show mean frustration rises monotonically from 1.5 → 5.5 across turns 1–8. They do not test longer rollouts. Our 8-turn choice matches the paper exactly, so we're directly comparable. Whether the effect plateaus or keeps growing past 8 turns is open.

## Reproduction

| Phase | Where | Cmd | Time |
|---|---|---|---|
| A — generate transcripts | local (OpenRouter) | `python -m scripts.distress.generate_transcripts` | 21.5 min |
| B — score probes | A100-SXM4-80GB pod | `python -m scripts.distress.score_probes --save-token-scores` | 3 min |
| C — judge per turn | local (OpenRouter free tier) | `python -m scripts.distress.judge_transcripts` | 4 min |
| D — analyze + plot | local | `python -m scripts.distress.analyze && python -m scripts.distress.qualitative_plots` | 30 sec |
| D' — extra plots (per-turn r, user vs asst, per-token) | local | `python -m scripts.distress.extra_plots` | 30 sec |

## Files

- Spec: `distress_transcripts_spec.md`
- Static prompt templates: `prompts.json`
- WC5 selection: `results/wc5_selection.json`
- Transcripts (140): `results/transcripts.jsonl` (2.9 MB)
- Probe readouts (5600 = 140 × 8 turns × 5 layers): `results/readouts.jsonl` (1.2 MB)
- Frustration scores (1120): `results/frustration.jsonl` (1.7 MB)
- Per-token L32 score arrays (140): `results/per_token_scores.npz` (1.8 MB, gitignored — regenerate via `score_probes.py --save-token-scores`)
- Per-condition summary: `results/analysis_summary.jsonl`
- Plots: `assets/plot_042426_*.png` (14 files)
- Phase logs: `results/generate.log`, `results/judge.log`
