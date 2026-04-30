# Qwen-3.5-122B persona probe transfer — final-six + default

Replicates `experiments/persona_sweep/probe_transfer/probe_transfer_spec.md` on Qwen-3.5-122B. Same analysis structure, same code paths, different model. Comprehensive: reports everything per `(selector, layer)` cell rather than picking a headline. Paper-figure choices (headline cell, persona ordering for cross-model comparability) deferred.

## What's different from Gemma

| | Gemma | Qwen |
|---|---|---|
| Model | gemma-3-27b instruction-tuned | qwen3.5-122b (MoE A10B), `-nothink` for measurement |
| Hidden dim | 5376 | 3072 |
| Selectors extracted | 4 (`{tb:-1,-2,-5}` + `task_mean`); spec used `{tb:-2, tb:-5}` | **2** (`tb:-1`, `tb:-4`) — both used |
| Layers | 5 (`[25, 32, 39, 46, 53]`) | **3** (`[33, 38, 43]`) |
| Default persona activations | filtered from `pref_main` (29,996 → 6000) | direct: `pref_default_sweep/` already on canonical 6000 — no filter step |
| Cells per analysis | 2 × 5 = 10 | **2 × 3 = 6** |
| AL `initial_degree` | 2 | 3 |
| AL pair_agreement | 0.96–0.97 | 0.92–0.95 (Qwen is a noisier reporter; same `convergence_threshold: 0.98` rank-correlation rule still met) |

Everything else mirrors the Gemma spec by design — same Ridge probes, same standardisation, same canonical 4000/1000/1000 splits, same persona prompts byte-for-byte.

## Inputs (all present, verified)

| Artifact | Location | Status |
|---|---|---|
| Activations, 6000 canonical tasks × 2 selectors × 3 layers | `activations/qwen35_122b/pref_<persona>_sweep/activations_turn_boundary:-{1,4}.npz` | ready (3.15 GB local), sanity-checked |
| Per-persona Thurstonian utilities | `results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/<persona>_<split>/` | ready (all 21 converged at threshold 0.98) |

The `<persona>_<split>` paths are post-rename via `scripts/persona_sweep_extraction/rename_exp_dir.py`. Until rename, the dirs have the auto-generated `<sysHASH>_<taskset>` suffix shape — see `experiments/qwen_replication/persona_transfer/extraction_report.md` for the hash → persona map.

## Infra reuse

Same shape as Gemma — nothing new in `src/`. Existing modules:

- **Probe training:** `src/probes/experiments/run_dir_probes.py` (handles `run_dir`/`eval_run_dir` heldout-α split, multi-layer in one run, `standardize`, `modes: [ridge]`).
- **Activation loading:** `src/probes/core/activations.py::load_activations`.
- **Probe eval:** `src/probes/core/evaluate.py::evaluate_probe_on_data` (returns Pearson r and R² given activations + scores).
- **Probe weight cosine:** `src/probes/core/evaluate.py::compute_probe_similarity`.
- **Utility loading:** reuse the loader call in `src/probes/experiments/run_dir_probes.py` — search for the Thurstonian-μ load and call the same helper. Do not reparse YAMLs by hand.

**Persona prompts source-of-truth:** `experiments/persona_sweep/sweep_personas.json` (`metadata.final_six` + `personas[]`). Same file used by AL extraction; `gen_probe_configs.py` should not duplicate prompt text into configs.

New scripts (small wrappers, all under `scripts/qwen_persona_transfer/`):

- `gen_probe_configs.py` — emits 14 probe configs (7 personas × 2 selectors), reads persona list from `experiments/persona_sweep/sweep_personas.json`. Adapts the Gemma `gen_probe_configs.py` (live on the `research-loop/persona_probe_transfer` branch).
- `analyze_transfer.py` — loads the 14 × 3 probe weight files, produces three 7×7 matrices (transfer, utility-utility, probe-cosine) per `(selector, layer)`. Save as NPZ keyed by cell.
- `make_report_figures.py` — consumes only the NPZs; produces all figures.

## Probe training

**14 probe configs** total, at `configs/probes/qwen_persona_sweep_final_six/<persona>_<selector>.yaml` for `persona ∈ {default, sadist, mathematician, aura, strategist, contrarian, slacker}` and `selector ∈ {tb-1, tb-4}`.

Each config:

- `run_dir`: `results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/<persona>_train/` (post-rename)
- `eval_run_dir`: `.../<persona>_eval/`
- `activations_path`: `activations/qwen35_122b/pref_<persona>_sweep/activations_turn_boundary:-N.npz`
- `layers: [33, 38, 43]` — produces 3 probes per config
- `modes: [ridge]`, `standardize: true`, `alpha_sweep_size: 10`, `eval_split_seed: 42`
- `output_dir`: `results/probes/qwen_persona_sweep_final_six/<persona>_tb-<N>/`

Total = 14 configs × 3 layers = 42 probes. Ridge on 4000 × 3072-dim activations is fast — full sweep finishes in seconds per config, low minutes total.

## Analysis — three 7×7 matrices per cell, all 6 cells reported

For each `(selector, layer) ∈ {tb:-1, tb:-4} × {33, 38, 43}`:

1. **Transfer matrix `T[train, eval]`:** Pearson r between the train probe's prediction on the eval persona's `pref_<eval>_sweep` test-split activations and the eval persona's Thurstonian utilities on the same 1000 test tasks.
2. **Utility-utility matrix `U[a, b]`:** Pearson r between persona a's and persona b's Thurstonian utilities on the canonical test set. Independent of probes.
3. **Probe-cosine matrix `C[a, b]`:** cosine similarity between the (standardised, refit) probe weight vectors. Independent of activations.

### NPZ schema (pin once — both `analyze_transfer.py` and `make_report_figures.py` conform)

`results/transfer_<selector>_L<layer>.npz`:
- `T`: float32 `(7, 7)` — Pearson r, rows = train persona, cols = eval persona
- `personas`: list[str] — canonical order from `sweep_personas.json` (`default` first, then `metadata.final_six` in their stored order)
- `selector`: str (e.g. `"turn_boundary:-1"`)
- `layer`: int

`results/probe_cosine_<selector>_L<layer>.npz`: same schema, key `C` instead of `T`.

`results/utility_similarity.npz`: `U` (7, 7), `personas` only (utilities are layer/selector-independent).

### Cross-persona standardisation

Each probe is fit with `standardize: true` on its train-persona activations (mean/std over the 4000 train tasks). When the probe is **applied** to a different persona's activations during transfer evaluation, the canonical answer is to standardise the eval activations using the *train persona's* fit-time mean/std (so the probe sees inputs in the same distribution it was trained on). Confirm `evaluate_probe_on_data` does this; if it instead re-standardises with the eval persona's own stats, transfer numbers are inflated by activation-norm equalisation. Add an explicit assertion in `analyze_transfer.py`.

Saved per-cell to `experiments/qwen_replication/persona_transfer/probe_transfer/results/`.

## Figures — show all 6 cells, no headline

Generated for every `(selector, layer)` cell in a uniform grid, plus aggregate-across-layer summaries:

- `plot_<mmddYY>_transfer_grid.png` — 6-panel small-multiples (rows = selector, cols = layer), each a 7×7 transfer heatmap.
- `plot_<mmddYY>_utility_similarity.png` — single 7×7 (utilities are layer/selector-independent).
- `plot_<mmddYY>_probe_cosine_grid.png` — 6-panel small-multiples of probe-direction cosine.
- `plot_<mmddYY>_layer_dependence.png` — per-persona donor (mean outbound r) and target (mean inbound r) curves vs layer, one panel per selector. Direct port of Gemma's Figure C.
- `plot_<mmddYY>_transfer_vs_utility_scatter.png` — for each ordered (train, eval) pair, point at (utility-utility r, probe-transfer r) for the per-cell-aggregated transfer. Diagonal `y = x`. One panel per `(selector, layer)`. Direct port of Gemma's appendix scatter.
- `plot_<mmddYY>_default_probe_vs_utility.png` — for each non-default persona, blue = default probe's transfer r, grey = default-vs-persona utility r, Δ arrow. Per `(selector, layer)`.
- `plot_<mmddYY>_asymmetry_scatter.png` — 21 unordered persona pairs, x = r(A→B), y = r(B→A), colour = |gap|. Per `(selector, layer)`.

Persona ordering for plots: **Qwen-internal**, sorted by Qwen's utility-similarity-with-default at the test split. Forcing Gemma's order is a paper-time concern, not a Qwen-internal one.

## Comprehensive comparisons to Gemma (deferred to a follow-up)

Things the report will set up but not fully resolve here:

- **Headline cell choice.** Likely candidates: `(tb:-4, L38)` mirroring Gemma's `(tb:-5, L32)` choice. Or `(tb:-1, L38)` to reflect the existing Qwen probe-training selector. Pick after seeing all 6 cells.
- **Layer where transfer peaks.** Gemma peaks at L32 of 5; Qwen has only `[33, 38, 43]` available — we may not bracket the peak from below. **Contingency:** if `mean(diag(T))` strictly decreases L33 → L38 → L43 for both selectors, extract activations at L28 (one-pod-resume + one persona-loop is enough — extraction is fast on a warm HF cache) and add a 4th column to the grids before writing the report.
- **Same persona ordering or Qwen's own?** Important if/when paper figures want Qwen + Gemma side-by-side.
- **Cross-model r magnitude comparability.** Pearson r against noisier utilities is mechanically attenuated. Qwen pair_agreement (≈0.93) is lower than Gemma (≈0.96). Report both raw r and `r / sqrt(reliability)` (using per-persona pair_agreement as the reliability estimate) when making Qwen-vs-Gemma magnitude claims. Pure within-Qwen claims use raw r.

## Work order

1. Rename the 21 AL exp dirs to `<persona>_<split>` symlinks via `scripts/persona_sweep_extraction/rename_exp_dir.py` (already used for Gemma).
2. Generate 14 probe configs (`scripts/qwen_persona_transfer/gen_probe_configs.py`), run them in a loop:
   ```
   for f in configs/probes/qwen_persona_sweep_final_six/*.yaml; do
     python -m src.probes.experiments.run_dir_probes --config "$f"
   done
   ```
3. Run `analyze_transfer.py` → emit per-cell NPZs.
4. Run `make_report_figures.py` → emit all figures.
5. Write `probe_transfer_report.md` summarising findings across all 6 cells, with explicit Qwen-vs-Gemma comparisons where the data supports them.
6. (Optional) cherry-pick `corr_bias.py` / `full_bias.py` / `profile_bias.py` from the `research-loop/persona_probe_transfer` branch and re-run on Qwen if we want the partial-correlation Shoggoth-bias analysis.

## Pre-flight

- `ls results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/` — if dirs are still `<sysHASH>_<taskset>` shaped, run the rename step (work-order step 1) using the hash → persona map at `experiments/qwen_replication/persona_transfer/extraction_report.md`. If already `<persona>_<split>`, skip.
- Dirs verified present (raw shape) at spec-write time; rename has not been done yet.

## Sanity checks

Before declaring done:

- 14 probe configs trained without α at the boundary (`alpha == sweep min/max` would suggest the sweep range was wrong).
- **Probe-on-self transfer (diagonal of T) is the positive control.** Mean diagonal at the best cell ≥ 0.5 across all 7 personas (Gemma achieves 0.82). If diagonals are uniformly << 0.5 across all 6 cells, something is wrong with utility loading or activation alignment — flag and stop.
- Cross-validate per-persona diagonals against any existing single-persona Qwen probe r (e.g. from `experiments/qwen_replication/`). Deviation > 0.1 indicates a pipeline bug.
- `T`, `U`, `C` matrices have NaN-free entries; rows/cols are in the expected `[default, sadist, mathematician, aura, strategist, contrarian, slacker]` canonical order from `sweep_personas.json` (re-ordered for plots only).
- **Both train and test task_id alignment.** For each (persona, selector): assert `len(train_activation_task_ids ∩ train_utility_task_ids) == 4000` and `len(test_activation_task_ids ∩ test_utility_task_ids) == 1000`. Reuse whatever helper `run_dir_probes.py` calls to get utility task_ids — do not parse the YAML by hand.

## Artifacts (planned)

- Probe weights: `results/probes/qwen_persona_sweep_final_six/<persona>_tb-<N>/` (gitignored — large)
- Transfer/utility/cosine matrices: `experiments/qwen_replication/persona_transfer/probe_transfer/results/*.npz`
- Figures: `experiments/qwen_replication/persona_transfer/probe_transfer/assets/plot_<mmddYY>_*.png`
- Report: `experiments/qwen_replication/persona_transfer/probe_transfer/probe_transfer_report.md`
