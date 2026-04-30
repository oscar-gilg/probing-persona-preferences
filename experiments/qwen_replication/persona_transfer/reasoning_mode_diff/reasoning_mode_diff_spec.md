# Qwen-3.5-122B reasoning vs no-reasoning preference diff

Compare per-task Thurstonian utilities for the same 6000 canonical tasks across two reasoning regimes — `reasoning.enabled: false` vs `reasoning.enabled: true, effort: low` — for every persona in the final-six + default sweep.

This spec is the **clean rerun** following bug fixes in commit `0d2ddee` (28 Apr 2026):

1. **Reasoning enablement**: prior "thinking" runs sent `reasoning.enabled: false` to OpenRouter; reasoning was off the whole time. Fixed in `_generate_batch_async`.
2. **Refusal misparse**: prior parser mis-credited refusals like *"I cannot complete Task B…"* to the refused task, sign-inverting bailbench / stress_test utilities. Fixed in `extract_claimed_task` (now defers to LLM judge for any response not literally opening with "Task A:" / "Task B:").

Buggy-run artifacts moved under `_SUPERSEDED_buggy_run/`.

## Inputs (all present, verified)

| Regime | Source AL run | Status |
|---|---|---|
| Reasoning OFF | `results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/<persona>_<split>/` | done (run with old parser — 9.9% refusal-prefix responses, ~62% of those mis-credited). Re-parse with new judge would clean these but raw_response was not stored before 23 Mar 2026. **Accept residual bias**, document. |
| Reasoning ON, low | `results/experiments/qwen_persona_sweep_thinking_final_six/pre_task_active_learning/<run_name>/` | **synced 28 Apr 2026** — 21 dirs, all converged. raw_response stored from cohort 4 onwards (~strategist+ configs). |

The thinking AL has measurements stored as `task_a, task_b, choice, origin_a, origin_b` (+ optional `raw_response` on later cohorts). Utilities are in `thurstonian_*.csv` keyed by `task_id` with columns `mu, sigma`.

## Pipeline

Existing scripts under `scripts/qwen_persona_transfer/`:

- `reasoning_mode_diff_analysis_v2.py` — loads both regimes' utilities, computes per-task Pearson/Spearman, per-origin breakdown, sign-flip stats, top |Δμ| examples. Emits NPZs and JSON summary under `results/`.
- `reasoning_mode_diff_plots_v3.py` — emits scatter, per-origin r, mean μ shifts, μ distributions, shift-magnitude boxplots, per-persona overall agreement, per-persona × per-origin heatmap.

The v2 / v3 suffixes are historical; they consume only the new data and are the canonical scripts. (v1 / v2-pre / earlier are obsolete and may be removed.)

## Reproducing

```bash
python -m scripts.qwen_persona_transfer.reasoning_mode_diff_analysis_v2
python -m scripts.qwen_persona_transfer.reasoning_mode_diff_plots_v3
```

Outputs:

- `results/diff_v2.npz`, `results/persona_diff.npz`, `results/summary_v2.json`
- `assets/plot_<mmddYY>_v3_*.png` — scatter, per-origin r, mean μ, μ distribution, shift magnitude, per-persona overall, per-persona × per-origin heatmap.

## Expected qualitative shape (carried over from the small-sample default-only check on 28 Apr)

On the default persona for eval split (1000 tasks):

| origin | μ_OFF mean | μ_ON mean | Pearson r |
|---|---:|---:|---:|
| math | +2.54 | +6.11 | +0.03 |
| bailbench | -4.82 | -7.85 | +0.27 |
| stresstest | -4.40 | -5.07 | +0.52 |
| wildchat | +0.45 | +1.41 | +0.12 |
| alpaca | +1.93 | -0.94 | -0.04 |
| **overall** | — | — | **+0.43** |

Predicted directions (vs the previous *buggy* "thinking"):

- Math goes UP under thinking (model can engage), not DOWN.
- Bailbench goes MORE NEGATIVE under thinking (refusals correctly counted), not POSITIVE.
- Per-persona spread: still wide, but the bailbench reversal is real-direction now (refusals strengthen, not invert).

The `bailbench-vs-benign "wins"` rate dropped from 13.9% (buggy parser) to 2.9% (new parser) — the headline structural sanity check.

## Caveats

- **Asymmetric parser quality**: the no-think utilities still come from the old parser. Some bias remains on no-think bailbench / stress_test (refusals counted as wins for the refused task ≈10% of the time). The cleanest thing to do is to re-fit no-think utilities from the original raw_response if available; for runs predating 23 Mar 2026 raw_response was not stored, so this is bounded by data availability.
- **Effort=low has a long tail**: ~10% of calls used > 4096 reasoning tokens; we set `max_new_tokens=6144` and made `LengthTruncationError` non-retryable — so length-truncated samples are dropped (rather than retried) but pair_agreement remained high (0.89–0.98 across runs).
- **Per-persona system prompts**: still distinct between regimes only via the reasoning flag — both regimes inject the same persona system prompt. Default persona has no system prompt in either regime (the `/no_think` directive is not effective on `qwen3.5-122b-a10b`, confirmed empirically).

## Sanity checks

- `len(shared_task_ids) == 6000` for each persona (overlap of `eval + test + train`).
- Per-origin r values not bottoming at exactly 0 (smoke test of the loader).
- Per-task |Δμ| histogram has a heavy tail toward bailbench / math (predictable structural shifts), not uniform noise.
- Compare the 3 personas with raw_response stored on the new thinking run (cohort-4+ configs: strategist, contrarian, aura) against the 4 without (default, sadist, mathematician, slacker). The latter four can't be re-audited via judge but their utility distributions should look qualitatively similar to the former three's.

## Artifacts (planned)

- Diff data: `results/diff_v2.npz`, `results/persona_diff.npz`, `results/summary_v2.json`
- Qualitative examples: `results/qualitative_examples_v2.md`
- Figures: `assets/plot_<mmddYY>_v3_*.png`
- Report: `reasoning_mode_diff_report.md`
