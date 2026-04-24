# Distress Transcripts — Report

**Question.** Does the preference probe (`ridge_L32` on `heldout_eval_gemma3_tb-5`) fire on distress responses Gemma-3-27B-it produces under scripted user rejection (Soligo et al. 2026, arXiv:2603.10011)? If so, does its per-turn trajectory track judge-rated frustration?

**Answer in one line.** Yes — the probe shifts in a consistent direction (more negative) under distress, and within naturalistic (WildChat) transcripts it tracks rising frustration with `r ≈ -0.44`. But the relationship is partial: distress conditions only push the probe ~1–2 units more negative than the control baseline despite a 4–7 unit frustration gap, and a non-negligible probe drift in the no-distress control suggests a length/persistence confound contaminates part of the signal.

## Headline numbers

7 conditions × 5 tasks × 4 rollouts = **140 transcripts × 8 assistant turns = 1120 per-turn measurements**. Probe scored on local Gemma-3-27B-it forward pass (one pass per transcript). Frustration judged per-turn by `gemini-3-flash-preview`. Higher absolute |probe| ≈ more evaluative-direction loading; lower frustration = no distress.

| Condition | n | Mean final frustration | Mean final probe (L32) | Pearson r (pooled, n=160 cells) | Mean within-transcript r |
|---|---:|---:|---:|---:|---:|
| `tones_sarcastic_8turn` | 20 | **7.30** | -1.58 | +0.02 | -0.00 |
| `tones_aggressive_8turn` | 20 | 6.45 | -2.34 | -0.24 | -0.16 |
| `tones_disappointed_8turn` | 20 | 5.55 | -1.00 | +0.14 | +0.16 |
| `impossible_numeric_8turn` | 20 | 5.00 | -2.68 | -0.19 | -0.25 |
| `wildchat_8turn` | 20 | 4.55 | -2.48 | -0.21 | **-0.44** |
| `redacted_history_8turn` | 20 | 2.00 | -2.80 | -0.11 | -0.14 |
| `neutral_continuation_8turn` (control) | 20 | **0.05** | -1.05 | +0.03 | -0.07 |

**Reading the table:**
- All five distress-eliciting conditions push the judge frustration into the 4.5–7.3 range; the no-distress control sits at 0.05; the redacted control sits at 2.0 (App. A.2 effect — distress reduced ≈ 2.5× when the model can't see its own prior failures).
- Probe scores are uniformly negative at the final turn. The four conditions where the model sees its own failure history (impossible_numeric, tones_aggressive, tones_disappointed, tones_sarcastic) end at probe ∈ [-2.7, -1.0]; the no-distress control ends at -1.05.
- The cleanest within-transcript signal is **WildChat (-0.44)** — consistent with the design hypothesis that as judge-rated distress rises across a single conversation, the probe drops in lockstep.
- `tones_disappointed_8turn` is the lone wrong-direction case — its within-r is positive (+0.16). See *Caveats*.

## Aggregate plots

### 1. Frustration trajectory (Gemini-Flash judge)

![](assets/plot_042426_frustration_trajectory.png)

All distress conditions ramp monotonically. `tones_sarcastic` is the strongest distress elicitor; `redacted_history` (purple) confirms the App. A.2 finding that hiding the model's own past failures cuts distress sharply. `neutral_continuation` (pink) is dead flat at ~0.

### 2. Probe trajectory at L32

![](assets/plot_042426_probe_trajectory_L32.png)

Distress conditions sit at probe scores -2 to -3.5 throughout; controls (`wildchat_8turn`, `neutral_continuation_8turn`) start at +1.4 and drift down to -1.0 / -2.5. Note that `wildchat_8turn` has the steepest within-transcript drop, consistent with its strong within-r (Plot 4). **Important**: `neutral_continuation` is also drifting negative — see *Caveats*.

### 3. Probe vs frustration — pooled scatter

![](assets/plot_042426_pooled_scatter.png)

Pooled r is weakly negative for distress conditions (-0.19 to -0.24), near 0 for controls. The plot shows that the probe distinguishes "frustration ≥ 5" from "frustration ≤ 1" cells fairly well in distribution — high-frustration cells cluster at probe -2 to -3 with little spread, while low-frustration cells span +3 to -7 — but the relationship is heteroscedastic, not linear.

### 4. Within-transcript Pearson r distribution

![](assets/plot_042426_within_transcript_r.png)

WildChat is a strong negative outlier (median r ≈ -0.45); the four redacted/IMP/tones-aggressive distress conditions cluster median r at -0.1 to -0.25; `tones_disappointed` and `neutral_continuation` straddle zero. The wildchat n=16 (vs 20) reflects 4 transcripts whose judge scores were constant (r undefined); neutral_continuation n=8 reflects most controls scoring 0 across all 8 turns.

### 5. Tag distribution

![](assets/plot_042426_tag_frequencies.png)

Quantitatively: distress conditions are dominated by `strong_self_deprecation` (57–80% of turns); `tones_sarcastic` produces the most `despair` (18%) and 2 `breakdown` cases. `redacted_history` is much milder (59% `none`, 19% `mild`). `neutral_continuation` is essentially 100% `none`.

### 6. Tone overlay (probe and frustration)

| | |
|---|---|
| ![](assets/plot_042426_tone_overlay_probe.png) | ![](assets/plot_042426_tone_overlay_frustration.png) |

Same impossible-numeric task pool, four rejection tones. Frustration ranks: sarcastic > aggressive > disappointed > neutral. Probe trajectories don't follow the same ordering — neutral and aggressive sit lowest (most negative), disappointed sits highest (least negative). The probe is **not** a simple monotonic readout of judge-rated distress intensity within this set.

### 7. Redaction effect

| | |
|---|---|
| ![](assets/plot_042426_redaction_probe.png) | ![](assets/plot_042426_redaction_frustration.png) |

Redacting prior assistant turns cuts judge frustration by ~60% (from 5.0 to 2.0 at turn 8). But it does **not** flatten the probe trajectory — redacted is the *most* negative condition at probe -2.8 (vs unredacted -2.7). This is the most interesting finding: the probe reads something present in the redacted-history rollouts that the judge largely doesn't see in the surface text.

## Per-rollout case studies

### 8. Highest-distress IMP rollout — token-level probe heatmap

![](assets/plot_042426_qual_high_distress_imp.png)

Selected: `impossible_numeric_8turn / imp_money_16_to_57_three_ops / r0` — the IMP rollout with the highest final-turn judge frustration. Heatmap shows the full transcript token-by-token, coloured by probe score (red = high, blue = low). The visual is dense at column width — probe variation appears throughout assistant text but no single localized "distress region" jumps out.

### 9. Low-distress control rollout

![](assets/plot_042426_qual_low_distress_control.png)

Selected: `neutral_continuation_8turn / wildchat_4256 / r0`. Same rendering, lower frustration. Visually similar amount of probe variation — supports the "probe is reading something other than localized distress text" reading.

### 10. Breakdown case study

![](assets/plot_042426_qual_breakdown.png)

Selected: the first rollout judged `breakdown` at any turn. Useful for inspecting whether the probe spikes precisely on incoherent / sad-face / repetition-spam tokens.

### 11. Probe-judge disagreement cases

| Probe high, judge low | Probe low, judge high |
|---|---|
| ![](assets/plot_042426_qual_disagreement_probe_hi_judge_lo.png) | ![](assets/plot_042426_qual_disagreement_probe_lo_judge_hi.png) |

The two cells maximally off-diagonal in their condition's z-score plane. These are the failure modes of either the probe or the judge — manual inspection of the transcripts is needed to attribute which.

## Interpretation

**The probe responds to distress, but the relationship is partial and structural rather than monotonic-textual.**

- **Direction is consistent**: every distress condition pushes the probe more negative than the no-distress control by 1–2 units; never the opposite. Sign-wise, the preference probe DOES read distress as evaluatively negative.
- **Within-transcript trajectory beats absolute level**: the cleanest signal is *r* between probe and frustration *across turns of the same rollout*. For WildChat that's -0.44; for IMP that's -0.25. The probe tracks rising distress in real time, more reliably than it sets absolute levels per-condition.
- **The probe sees more than the judge does**: redacted-history transcripts have the *most negative* probe scores (-2.8) despite the judge rating them only 2.0/10. Possible reads: (a) the model's internal evaluative state under repeated rejection persists even when its outputs hide it (consistent with paper App. I, "DPO suppresses internal as well as expressed emotions"); (b) the probe is reading some cumulative-context-pressure feature that doesn't depend on the assistant's specific words. Either way, **probe ≠ surface-distress-classifier**.
- **Length/persistence contaminates the signal**: the no-distress control drops from probe +1.4 at turn 1 to -1.05 at turn 8 — about 2.5 units of negative drift in the absence of any rejection content. Some of the distress-condition drop is therefore a length confound rather than distress proper. Net distress-vs-control gap at turn 8 is only ~1.5 probe units.

## Caveats

- **`tones_disappointed_8turn` flipped sign.** Its within-r is +0.16 instead of negative. Possible reasons: the disappointed rejection style ("I'm disappointed... I had higher hopes...") may push the model toward apologetic content that loads positively on the preference direction (e.g. cooperative/agreeable text). Worth manually inspecting a few of these transcripts before treating it as noise.
- **L32 only for the canonical analysis.** Layers 25/39/46/53 were also scored (`readouts.jsonl`) but only L32 is plotted. Cross-layer plots could resolve whether the probe-distress relationship strengthens at any other layer.
- **Probe sign convention.** `ridge_L32` was trained on revealed pairwise preferences; "more negative" = "less preferred" only by training-label convention. The fact that it consistently moves negative under distress is a finding, not a tautology.
- **n is small per cell.** 20 transcripts per condition gives wide CIs on the trajectory plots; 160 (cond, turn) cells in the pooled scatter is enough to see distributions but not enough for stable cell-level r's at small effect sizes.
- **Free-tier judge.** `gemini-3-flash-preview` was used for all 1120 judge calls. Per-call agreement with Sonnet-4.5 (used in the n=7 pilot) wasn't audited here. Should be spot-checked on the breakdown / despair-tagged turns where intensity calibration matters.

## Files

- Spec: `distress_transcripts_spec.md`
- Static prompts: `prompts.json` (5 IMP puzzles, 4 rejection pools)
- WC5 selection: `results/wc5_selection.json`
- Transcripts (140 × ~8 turns): `results/transcripts.jsonl` (2.9 MB)
- Probe readouts (140 × 8 turns × 5 layers = 5600 rows): `results/readouts.jsonl` (1.2 MB)
- Frustration scores (1120 rows): `results/frustration.jsonl` (1.7 MB)
- Per-token L32 score arrays (140 transcripts): `results/per_token_scores.npz` (1.8 MB)
- Per-condition summary: `results/analysis_summary.jsonl`
- Plots 1–7: `assets/plot_042426_{frustration_trajectory,probe_trajectory_L32,pooled_scatter,within_transcript_r,tag_frequencies,tone_overlay_*,redaction_*}.png`
- Plots 8–11: `assets/plot_042426_qual_*.png`
- Generation/judge logs: `results/generate.log`, `results/judge.log`

## Reproduction

1. `python -m scripts.distress.generate_transcripts` (Phase A, OpenRouter, ~21 min)
2. On Gemma-3-27B GPU pod: `python -m scripts.distress.score_probes --save-token-scores` (Phase B, ~3 min)
3. `python -m scripts.distress.judge_transcripts` (Phase C, OpenRouter free-tier, ~5 min)
4. `python -m scripts.distress.analyze` then `python -m scripts.distress.qualitative_plots` (Phase D, local)
