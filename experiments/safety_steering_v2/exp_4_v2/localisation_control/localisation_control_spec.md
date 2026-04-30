# Exp 4 v2 — localisation control (non-critical span steering)

Follow-up to `experiments/safety_steering_v2/exp_4_v2/`. The v2 sweep showed a clear dose-response on `critical_info_only` (ethical/distributed: 0.84 → 0.84 → 0.52 → 0.32 → 0.32; ethical/isolated: 0.76 → 0.89 → 0.42 → 0.24 → 0.38) and a striking bidirectional content sensitivity (benign-twin specific flagging rises from 0% to ~45% at c=−0.05). Both findings are interpreted in the v2 report as evidence that the preference direction modulates a flagging disposition tied to span content. **What the v2 sweep cannot rule out is a position-only / any-prefill-span account**: maybe steering on *any* matched-length span in the prompt during prefill produces the same suppression / amplification, and the apparent content specificity is an artifact of the critical span happening to be where you looked.

This control adds one steering condition — `non_critical_only` — that steers a structurally-analogous, length-matched, ethically neutral span elsewhere in the same prompt. The corpus, probe, calibration, sampling params, judge, and analysis pipeline are reused verbatim from exp_4_v2.

## Question

Is the suppression / amplification effect of `critical_info_only` localised to the critical span, or would steering on any matched-length span in the same prompt produce a comparable effect?

## Predictions

- **Localisation real**: at c=+0.05 on the ethical variant, `critical_info_only` suppresses specific-disclosure substantially more than `non_critical_only`. At c=−0.05, the bidirectional benign-twin spurious-flag spike (0% → ~45% at c=−0.05 under `critical_info_only`) is much weaker under `non_critical_only`. The mechanism story stands: the direction reads ethical content in the span it acts on.
- **No localisation**: `critical_info_only` and `non_critical_only` give comparable dose-response. The v2 mechanism story collapses to "any prefill steering modulates flagging disposition" — not specifically tied to the ethical content. App. A.3's targeted-suppression paragraph would need to drop the localisation framing.
- **Partial localisation**: `non_critical_only` shows a smaller-but-nonzero effect. We retain the localisation claim with a quantified attenuation (e.g. "the effect is 2× larger on the critical span than on a matched non-critical span").

## Scope

- **Isolated bin only.** The 5 distributed-bin scenarios (R4, R5, B1, B2, C2) are excluded: the exp_4_v2 design encodes their `critical_span` as the entire content body (the EPA report, the launch memo, the candidate package, the panellist comments, the receipts), so there is no non-critical content body of comparable length within the prompt. The control question is well-posed only when the critical span is a delimited insert.
- **All 9 isolated scenarios are included** (R1, R2, R3, X1, X2, X3, B3, C1, M1). Three of these (X1, C1, M1) authored their `critical_span` as the entire one-pager / memo / methods section rather than just a bracketed insert; for those scenarios the non-critical span is drawn from the framing/closing text and is structurally weaker as a comparator. This is documented in `non_critical_assignments.md` per-scenario; the report flags it.

## Out of scope

- Knowledge-swap variant (`v3`, deferred per parent spec).
- Coherence judging (deferred per parent spec; |c| ≤ 0.05 cap inherited from Fig. 21).
- New steering modes beyond `non_critical_only`.
- Re-running `no_steering`, `critical_info_only`, `generation_only`, `critical_info_plus_generation` — already in `exp_4_v2/results.jsonl`.

## Setup

All values frozen by reference to `exp_4_v2`:

| | Value |
|---|---|
| Model | gemma-3-27b (instruction-tuned) |
| Probe | ridge L25 (`results/probes/heldout_eval_gemma3_task_mean/probes/probe_ridge_L25.npy`) |
| MEAN_NORM | 35,708 (frozen) |
| Sampling | T=1.0, max_new_tokens=512, top_p default |
| Coefficient grid | c ∈ {−0.05, −0.03, +0.03, +0.05} (skip c=0 — `no_steering` baseline already covers it in `exp_4_v2/results.jsonl`) |
| Conditions | `non_critical_only` only |
| Corpus | exp_4_v2's **9 isolated scenarios** × 2 variants (ethical, benign_twin) = **18 prompts**. Distributed-bin scenarios skipped (see "Scope" below). |
| Trials per cell | 5 (matches exp_4_v2) |
| Total generations | 9 × 2 × 4 × 5 = **360** |
| Wall (sweep) | ~20 min H100 SXM (linear extrapolation from exp_4_v2's 1:36:38 for 1820 generations) |
| Judge | Gemini 2.5 Flash (`google/gemini-2.5-flash`), same 5-level disclosure rubric, temperature 0 |

## Non-critical span selection

The control is only meaningful if `non_critical_span` is a fair comparator to `critical_span`. Pre-register one `non_critical_span` per `(scenario_id, variant)` row, hand-authored against this rubric:

1. **Maximum available contiguous length**: pick the longest contiguous span of non-critical content available in the prompt, drawn from a structurally-analogous region (see rule 3). No upper cap on length, but in practice non-critical spans are shorter than `critical_span` for most scenarios. **The token-count mismatch is a known confound**: if `non_critical_only` produces a smaller effect than `critical_info_only`, the difference could be content-localisation OR token count. The asymmetry biases the comparison *against* finding the localisation effect (the non-critical condition has fewer steered tokens), so a clear `critical_info_only` win is strong evidence; a clear `non_critical_only` match is strong evidence against; an attenuated `non_critical_only` is ambiguous. Per-scenario lengths are reported in `non_critical_assignments.md` and the final report.
2. **Same prompt**: a contiguous span within the same `prompt` field. No cross-variant copying.
3. **Ethically neutral content**: must not carry the `intended_issue` or any other ethical signal. The disclosure judge applied to a hypothetical generation that surfaces only the `non_critical_span` content should return `disclosure_level == "none"`. (No need to actually generate; this is a sanity check during span authoring.)
4. **Prefer structurally-analogous content body**: for R1, R2, R3 the natural choice is the routine header table (CLIENT/REQUEST/TIMELINE, PATIENT/HISTORY/EXAM, TO/SUBJECT/DRAFT). For X2 it's the customer message. For X3 it's the SUMMARY + DETAILED FINDINGS section (the richest non-critical comparator in the corpus). For B3 it's the APPLICANT header table. For X1, C1, M1 the `critical_span` encompasses the entire content body; the only available non-critical text is the framing/closing instruction, which is a weaker comparator — flagged per-scenario in `non_critical_assignments.md`.
5. **Position-disjoint from `critical_span`**: zero overlap in tokens.
6. **Same on ethical and benign-twin variants where possible**: when the variants differ only in the critical span (which is the design contract for the IMM twin-matching standard), the same `non_critical_span` text appears verbatim in both — making the only difference between the four `(critical_info_only, non_critical_only) × (ethical, benign_twin)` cells the steered-span identity. Tokenisation of the chosen text in each variant should still be re-validated since neighbouring critical-span text differs.

The full set of frozen spans is recorded in `non_critical_spans.json` (one row per scenario × variant) and rationalised in `non_critical_assignments.md`. Both files must be committed before any GPU work; the commit SHA is the pre-registration record (mirroring the parent spec's enforcement). If a span fails the gemma tokenizer round-trip on validation, supersede with a new commit and document the supersedence in the report.

### Tokenisation validation (before GPU work)

Run `scripts/safety_steering_v2/validate_localisation_spans.py` (new — modeled on `validate_corpus_tokenization.py`) on the 18 isolated-bin rows. The script must confirm for each row:

- `find_text_span(tokenizer, prompt_full, non_critical_span)` returns a non-`None` `(start, end)` pair.
- The recovered span text matches `non_critical_span` exactly.
- `non_critical_span` and `critical_span` ranges have zero overlap.
- Report (don't gate) the token-length ratio `non_critical_tokens / critical_tokens` per row, so the per-scenario length deficit is auditable.

Halt and revise spans before launching the sweep if any row fails.

## Code reuse

- **Driver**: fork `scripts/safety_steering_v2/generate_exp_4_v2.py` into `scripts/safety_steering_v2/generate_localisation_control.py`. Three changes only:
  1. `CONDITIONS = ["non_critical_only"]`.
  2. `MULTIPLIERS = [-0.05, -0.03, 0.03, 0.05]` (drop 0.0).
  3. Extend `_build_hook` with one new branch:
     ```python
     if condition == "non_critical_only":
         return position_selective_steering(tensor, nc_start, nc_end)
     ```
     where `nc_start, nc_end = find_text_span(tokenizer, formatted, item["non_critical_span"])`. Same `position_selective_steering` machinery as `critical_info_only`; only the steered range changes.
- **Hooks / probe loading / per-row schema** unchanged. The output `results.jsonl` schema gains no new fields.
- **Judge**: reuse `scripts/safety_steering_v2/judge_pilot_disclosure.py` (or a thin wrapper named `judge_localisation_control.py` that points at the new `results.jsonl`). Same model id, schema, rubric, temperature 0.

## Output structure

```
experiments/safety_steering_v2/exp_4_v2/localisation_control/
├── localisation_control_spec.md          # this file
├── non_critical_assignments.md           # rationale per (scenario_id, variant)
├── non_critical_spans.json               # frozen spans; joined to exp_4_v2/prompts.json
├── results.jsonl                         # 360 generations
├── results__judged.jsonl                 # disclosure judge output
├── localisation_control_report.md        # writeup
└── assets/
    └── plot_*_critical_vs_noncritical.png
```

## Analysis

Report the `non_critical_only` result alongside `critical_info_only` (from `exp_4_v2/results.jsonl`), at c ∈ {±0.03, ±0.05}, on the 18 isolated-bin prompts and 5 trials per cell. No paired-difference statistic; the two conditions are parallel rows in the existing exp_4_v2 readouts. The exp_4_v2 distributed-bin curves stay where they are — they are not part of this control.

- **Headline plot**: dose-response curves overlaying `critical_info_only` (solid) and `non_critical_only` (dashed) for ethical/isolated and benign_twin/isolated. Same y-axis as exp_4_v2's `plot_042826_critical_info_dose_response.png`. Distributed-bin lines from exp_4_v2 may be shown faded for context.
- **Bootstrap suppression Δ table**: extend exp_4_v2's table (lines 60–69 of `safety_steering_v2_report.md`) with `non_critical_only` rows for the isolated bin at the same coefficients. Same per-condition method: scenario-level resample, 1000×, 95% CI on Δ vs the shared baseline.
- **Bidirectional check**: at c=−0.05 on benign_twin/isolated, compare specific-flagging rates between `critical_info_only` and `non_critical_only`. The spurious-flagging spike (0% → 49%) under `critical_info_only` on isolated benign-twin is the cleanest signature of content-sensitivity in v2; whether it survives under `non_critical_only` is the most informative single number.
- **Per-scenario length deficit**: include a small table reporting `non_critical_tokens / critical_tokens` per scenario, so the reader can see where the comparison is closest to length-matched (typically X3) and where it isn't (typically X1, C1, M1).

The reader can compare the two conditions from the overlaid plot and the side-by-side Δ table directly — no aggregated statistic needed.

## Reporting

`localisation_control_report.md`. Mirrors the `safety_steering_v2_report.md` shape: headlines, setup table, baseline (re-cited from exp_4_v2), overlaid dose-response plot, extended bootstrap Δ table, qualitative comparison paragraph, files. Cross-reference back to App. A.3 in `paper/main.tex`. Update `paper/plan.md` with the qualitative comparison and any implications for the §A.3 paragraph.

## Sequencing

1. Author and pre-register `non_critical_spans.json` + `non_critical_assignments.md`. ~1 hour of careful reading of the 9 isolated scenarios.
2. Validate tokenisation on the 18 rows. ~5 min.
3. Commit pre-registration files; record SHA.
4. Fork driver → run sweep on H100 (~20 min) → run disclosure judge (~5 min OpenRouter).
5. Analyse + write report.

## Compute

~20 min H100 + ~5 min OpenRouter. Use zombuul (`/launch-runpod`, `/run-experiment`).
