# Qwen-3.5-122B persona probe transfer — THINKING-mode (clean rerun)

Sub-experiment of `probe_transfer/`. Same analysis structure, same 7 personas, same 6000-task canonical splits, same Ridge probe pipeline — only difference is the utility targets come from the reasoning-ON (low effort) AL run instead of reasoning-OFF.

This spec is the **clean rerun** after fixing two bugs documented in commit `0d2ddee`:

1. **Reasoning enablement bug** — the prior thinking AL run sent `reasoning.enabled: false` to OpenRouter (i.e. reasoning was off). Fixed: `_generate_batch_async` now defaults `enable_reasoning` from the registry's `reasoning_mode`.
2. **Refusal misparse bug** — the regex parser `extract_claimed_task` returned the first-mentioned task label, so refusals like *"I cannot complete Task B…"* were assigned `choice='b'` (the refused task). Fixed: the parser now returns `'a'/'b'` only if the response literally opens with the task label, else defers to the LLM judge.

Buggy-run artifacts moved under `_SUPERSEDED_buggy_run/`.

## What's different from the no-think run

| | No-think (`probe_transfer/`) | Thinking (this dir, post-fix) |
|---|---|---|
| Utilities | `qwen_persona_sweep_final_six` (reasoning OFF) | `qwen_persona_sweep_thinking_final_six` (reasoning ON, `effort=low`) |
| Activations | `activations/qwen35_122b/pref_<persona>_sweep/` | **same files** — extraction is HF-side, no reasoning protocol |
| Probe configs | `configs/probes/qwen_persona_sweep_final_six/` | `configs/probes/qwen_persona_sweep_thinking_final_six/` |
| Probe weights | `results/probes/qwen_persona_sweep_final_six/` | `results/probes/qwen_persona_sweep_thinking_final_six/` |
| AL convergence | iter ~5, pair_agreement 0.92–0.95 | iter ~5–7, pair_agreement 0.89–0.98 |

Activations are reused because the HF forward pass has no concept of "reasoning mode" — that's an OpenRouter chat-API protocol. The probe asks: *do the no-think internal representations linearly predict the thinking-mode utilities?*

## Inputs (all present, verified)

| Artifact | Location | Status |
|---|---|---|
| Activations, 6000 canonical tasks × 2 selectors × 3 layers | `activations/qwen35_122b/pref_<persona>_sweep/activations_turn_boundary:-{1,4}.npz` | ready (3.15 GB local), reused from no-think |
| Per-persona thinking utilities | `results/experiments/qwen_persona_sweep_thinking_final_six/pre_task_active_learning/<run_name>/` | **synced 28 Apr 2026** — 21 dirs, all 21 converged |

The 21 dirs are still in their post-AL `<sysHASH>_<taskset>` shape; rename to `<persona>_<split>` symlinks via the rename step in the work order. Hash-to-persona map: `experiments/qwen_replication/persona_transfer/extraction_report.md`.

## Pipeline

Identical to no-think. Wrapper scripts under `scripts/qwen_persona_transfer/`:

- `rename_exp_dir_thinking.py` — symlinks `<persona>_<split>` (default has no `sys<hash>` segment in thinking)
- `gen_probe_configs_thinking.py` — emits 14 probe configs at `configs/probes/qwen_persona_sweep_thinking_final_six/`
- `analyze_transfer_thinking.py` — computes T, U, C matrices per cell
- `make_report_figures_thinking.py` (+ `_v2.py` for headline-style plots) — produces all figures

Outputs land under `experiments/qwen_replication/persona_transfer/probe_transfer/thinking/{results, assets}/`.

## Reproducing

```bash
python -m scripts.qwen_persona_transfer.rename_exp_dir_thinking
python -m scripts.qwen_persona_transfer.gen_probe_configs_thinking
for f in configs/probes/qwen_persona_sweep_thinking_final_six/*.yaml; do
  python -m src.probes.experiments.run_dir_probes --config "$f"
done
python -m scripts.qwen_persona_transfer.analyze_transfer_thinking
python -m scripts.qwen_persona_transfer.make_report_figures_thinking
python -m scripts.qwen_persona_transfer.make_report_figures_thinking_v2
```

## Sanity checks (carry over from `probe_transfer/`)

- 14 probe configs train without α at the boundary.
- **Diagonal positive control**: mean `diag(T)` ≥ 0.5 at the best cell across all 7 personas. If <0.5 uniformly, flag the pipeline.
- T / U / C matrices NaN-free; rows/cols in canonical persona order from `sweep_personas.json`.
- task_id alignment: train `|act ∩ util| == 4000`, test `|act ∩ util| == 1000` for every (persona, selector).
- Cross-persona standardisation: probe sees standardised eval activations using the *train persona's* fit-time mean/std (not the eval persona's).

## Headline cell

Will be picked after seeing all 6 cells, mirroring the no-think report's choice (likely `(tb-4, L38)` for cross-comparability). The previous superseded report had picked `(tb-4, L43)`; revisit once new matrices land.

## Artifacts (planned)

- Probe weights: `results/probes/qwen_persona_sweep_thinking_final_six/<persona>_tb-<N>/` (gitignored)
- Transfer / utility / cosine matrices: `experiments/qwen_replication/persona_transfer/probe_transfer/thinking/results/*.npz`
- Figures (date-stamped): `experiments/qwen_replication/persona_transfer/probe_transfer/thinking/assets/plot_<mmddYY>_*.png`
- Report: `experiments/qwen_replication/persona_transfer/probe_transfer/thinking/probe_transfer_thinking_report.md`

## Comparison to no-think (paper-relevant)

After both reports are clean, expect the diff to surface in this direction (per the new utilities — see `../../reasoning_mode_diff/`):

- Math: probe transfer should be *higher* under thinking than no-think (mean μ rose from +2.5 → +6.1 with reasoning on the default persona).
- Bailbench: utility much more strongly negative under thinking (-7.85 vs -4.82); probe transfer strength may shift accordingly.
- Per-persona: the parser bug previously inflated several configs' utilities by mis-crediting refusals — those clean up now, so persona ordering may differ from the superseded thinking report.
