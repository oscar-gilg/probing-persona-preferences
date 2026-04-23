# Persona probe transfer — final-six + default

Replicates the "default-persona probe transfers to other personas" claim (§\ref{sec:shared-probe} in the paper) with the principled final-six set plus the default persona, using the canonical 6000-task train/eval/test splits. Updates the headline transfer figure and its supporting appendix.

## Framing

The paper currently reports transfer on an ad-hoc 5-persona set (aesthete, midwest, villain, sadist, stem-obsessive) and the §\ref{sec:shared-probe} `todo` flags it for re-measurement on the final six. This spec is the re-measurement, with two additions on top of the straight replication:

- **Default persona is a 7th cell of the transfer matrix**, not a privileged "source". You can still read the default-vs-rest story off one row of the 7×7 heatmap, and we get the symmetric comparison "free".
- **Utility-utility correlations shown alongside probe-transfer correlations.** If the two heatmaps look the same, probe transfer is tracking behavioural similarity; if they come apart, the probe picks up shared circuitry beyond behavioural similarity. This is the sharpest presentation of the "shared substrate" claim.

## Inputs

| Artifact | Personas | Location | Status |
|---|---|---|---|
| Activations (6000 tasks, 4 selectors × 5 layers) | final-six | `pref_<persona>_sweep/` on storage pod | ready |
| Activations (29 996 tasks, same 4 selectors × 5 layers) | default | `pref_main/` on storage pod | ready; filter to canonical 6000 at load time |
| Utilities | final-six × {train, eval, test} | `results/experiments/persona_sweep_final_six/pre_task_active_learning/<persona>_<split>/` | ready |
| Utilities | default × {train, eval, test} | same dir, `default_<split>/` | running now (user launched the 3 AL configs); will be renamed+symlinked via `rename_exp_dir.py` on completion |

Selectors and layers are the same for all 7 personas (bytes-per-task match between `pref_main` and `pref_<persona>_sweep` at tb:-2 confirms identical layer set `[25, 32, 39, 46, 53]` and hidden size 5376, float32).

## Infra reuse

Nothing new in `src/`. Everything below composes existing modules:

- **Probe training:** `src/probes/experiments/run_dir_probes.py` — already supports `run_dir` (train), `eval_run_dir` (alpha sweep + heldout), multi-layer output, `standardize`, `modes: [ridge]`. One config per (persona, selector) pair; the `layers` list produces one probe per layer inside that run.
- **Activation loading + probe eval:** `src/probes/core/activations.py::load_activations`, `src/probes/core/evaluate.py::evaluate_probe_on_data` (accepts activations + scores and returns Pearson r / R²).
- **Utility loading:** `src/measurement/storage/loading.py` (used by `run_dir_probes.py`; the same helper works for loading any persona's utilities on shared task IDs for the utility-utility heatmap).
- **Probe-weight cosine:** `src/probes/core/evaluate.py::compute_probe_similarity` (already exists per README).
- **AL run renaming:** `scripts/persona_sweep_extraction/rename_exp_dir.py` (reuse to attach `default_<split>` symlinks once the AL run finishes).
- **Config generation:** `scripts/persona_sweep_extraction/gen_measurement_configs.py` (just extended to include default) and a new `gen_probe_configs.py` following the same pattern.

Only new code:

- `scripts/persona_sweep_extraction/gen_probe_configs.py` — emits 14 probe configs (7 personas × 2 selectors), reading the persona list from `metadata.final_six` + appending `default`.
- `scripts/persona_sweep_extraction/sync_activations_local.sh` — pulls sweep dirs from storage pod to `./activations/gemma-3-27b_it/` and, for `pref_main`, filters to the canonical 6000 via a small python helper.
- `scripts/persona_sweep_extraction/analyze_transfer.py` — loads the 14 × 5 probes, produces the 7×7 matrices and the four figures. Thin wrapper over `evaluate_probe_on_data` + numpy correlations.

## Activation staging

Local-temp. Script: `sync_activations_local.sh`.

- For each `pref_<persona>_sweep`: `rsync -a runpod-<pod>:/workspace/activations/gemma-3-27b_it/pref_<persona>_sweep/ ./activations/gemma-3-27b_it/pref_<persona>_sweep/`.
- For `pref_main`: full rsync is 19 GB; instead use a small python helper to load each of the 4 NPZ files, filter to the 6000 canonical task IDs, and re-save to `pref_main_canonical_6000/`. ~500 MB.
- Total local footprint: 6 × 2.5 GB + ~500 MB ≈ 15.5 GB. Deletable post-analysis.

## Probe training

Selector-and-layer sweep, not a single point:

- **Selectors:** `turn_boundary:-2` and `turn_boundary:-5`. (`turn_boundary:-1` and `task_mean` excluded to keep the paper figure focused; both are available if we want to ablate later.)
- **Layers:** all five of `[25, 32, 39, 46, 53]`. `run_dir_probes.py` outputs one probe per layer per run, so this is free.

14 probe configs total: `configs/probes/persona_sweep_final_six/<persona>_<selector>.yaml` for `persona ∈ {sadist, mathematician, aura, strategist, contrarian, slacker, default}` and `selector ∈ {tb-2, tb-5}`. Each:

- `run_dir`: `results/experiments/persona_sweep_final_six/pre_task_active_learning/<persona>_train/`
- `eval_run_dir`: `.../<persona>_eval/`
- `activations_path`: `activations/gemma-3-27b_it/pref_<persona>_sweep/activations_turn_boundary:-N.npz` (or `pref_main_canonical_6000/...` for default)
- `layers: [25, 32, 39, 46, 53]`
- `modes: [ridge]`, `standardize: true`
- `output_dir`: `results/probes/persona_sweep_final_six/<persona>_tb-<N>/`

Running all 14 is sequential and fast (Ridge on 4000 × 5376-dim activations × 5 layers finishes in seconds per run). Total wall time ~few minutes.

## Analysis

`analyze_transfer.py` produces, for each (selector, layer) combination:

1. **Transfer matrix (7×7):** for each (train_persona, eval_persona), Pearson r between the train probe's prediction on the eval persona's test-split activations and the eval persona's Thurstonian utilities on those tasks.
2. **Utility-utility matrix (7×7):** Pearson r between each pair of personas' Thurstonian utilities on the canonical test set. Independent of the probe; purely behavioural.
3. **Probe-cosine matrix (7×7):** cosine similarity between probe weight vectors.

Primary output: a dict keyed by `(selector, layer)` with the three 7×7 arrays.

Picking a headline `(selector, layer)` for the paper figure:

- Use whichever minimises the gap between "probe transfer to self" (diagonal) and the paper's current default-self number (sanity). If diagonals look good across the board, pick the highest mean off-diagonal transfer.
- Default fallback if unclear: `(tb:-2, layer 32)` — matches the paper's current headline.

## Figures

Generated for the headline `(selector, layer)` (paper body) and for every `(selector, layer)` combination (appendix grid):

- **`plot_<mmddYY>_transfer_heatmap.png`** — headline. 7×7 probe-transfer r. Rows = train, cols = eval. Diagonal marked.
- **`plot_<mmddYY>_utility_similarity_heatmap.png`** — same ordering as the transfer heatmap. Visual side-by-side shows whether probe transfer ≠ utility similarity.
- **`plot_<mmddYY>_transfer_vs_utility_scatter.png`** — for each ordered (train, eval) pair, point at (utility-utility r, probe-transfer r). Diagonal reference `y = x`. Points above the line = probe transfer stronger than behavioural similarity would predict (shared circuitry hypothesis); points below = weaker.
- **`plot_<mmddYY>_probe_cosine_heatmap.png`** — sanity check on the transfer heatmap.
- **Appendix grid: `plot_<mmddYY>_transfer_heatmap_grid.png`** — small-multiples over (selector, layer), 2 × 5 panels, for robustness.

Paper integration: the transfer heatmap and the utility/scatter pair replace the current `plot_030426_s5_cross_eval_heatmap.png` in §\ref{sec:shared-probe}. Appendix grid + cosine heatmap go into `app:persona-transfer` (new).

## Work order

1. **[running]** Default-persona AL. When finished, rename the exp dir and symlink `default_<split>` inside `persona_sweep_final_six/pre_task_active_learning/`.
2. Sync activations locally (`sync_activations_local.sh`). Filter `pref_main` to canonical 6000.
3. Generate 14 probe configs (`gen_probe_configs.py`); run them via `python -m src.probes.experiments.run_dir_probes --config <cfg>` in a loop (a one-liner shell script; no new infra).
4. Run `analyze_transfer.py`; emit the four headline figures and the appendix grid.
5. Write `probe_transfer_report.md`; splice into paper §\ref{sec:shared-probe} and add `app:persona-transfer`.
6. `rm -rf activations/gemma-3-27b_it/pref_*_sweep activations/gemma-3-27b_it/pref_main_canonical_6000`.
