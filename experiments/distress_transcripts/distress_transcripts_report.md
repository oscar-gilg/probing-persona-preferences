# Distress Transcripts — Report

**Question.** Does the preference probe (`ridge_L32` on `heldout_eval_gemma3_tb-5`) fire on the distress responses Gemma-3-27B-it produces under scripted user rejection (Soligo et al. 2026, arXiv:2603.10011)?

**Status.** In progress.

## Headline numbers

_To fill in after analysis._

| Condition | n | Mean final frustration | Mean final probe | Pearson r (pooled) | Mean within-transcript r |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... |

## Method (summary)

7 conditions × 5 tasks × 4 rollouts = 140 transcripts × 8 turns = 1120 per-turn measurements. Generation on `gemma-3-27b` via OpenRouter, T=1.0. Probe readout on local Gemma-3-27B forward pass via `score_prompt_all_tokens`. Per-turn Gemini-Flash judge with score + quote + qualitative tag.

## Aggregate plots

_To insert. Plots 1–7 from the spec._

## Per-rollout case studies

_To insert. Plots 8–11 from the spec._

## Caveats

_To insert._

## Files

- Spec: `distress_transcripts_spec.md`
- Transcripts: `results/transcripts.jsonl`
- Probe readouts: `results/readouts.jsonl`
- Frustration scores: `results/frustration.jsonl`
- Plots: `assets/plot_*.png`
