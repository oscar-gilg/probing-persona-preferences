---
status: draft
model: gemma-3-27b
---

# Layer Sweep: comprehensive probe + steering characterisation

## Question

Across Gemma-3-27B's 62 transformer blocks, at ~every third layer:

1. **Probe performance vs layer.** Where is preference information linearly decodable, and how does that depend on the extraction token?
2. **Probe similarity vs layer.** Are probe directions at neighbouring / distant layers the same direction, rotated, or genuinely different? Are tb:-2 and eot picking up the same direction?
3. **Steering locus.** At which layers does injecting a preference direction actually shift choice?
4. **Probe-locus vs injection-locus.** Given a probe trained at layer L_p, which injection layers L_s does it steer at?

Prior work (`experiments/steering/cross_layer/`) answered (3) and (4) at 3 probes × 5 injection layers. This spec redoes it comprehensively at 20 self-layer cells plus a 5 × 20 spine × injection sweep — the figure the paper should actually show.

## Piggyback on persona sweep

- **Utilities (no new API calls).** Default-persona Thurstonian scores already exist at `results/experiments/persona_sweep_final_six/pre_task_active_learning/default_{train,eval,test}/` (4000 / 1000 / 1000 on the canonical split `data/canonical_splits/{train,eval,test}_task_ids.txt`).
- **Task IDs.** Use the canonical 6000 directly. No resampling. `data/canonical_splits/all_6000_task_ids.txt` already exists.

Only new work is activations at additional layers and the steering sweep itself.

## Data separation

Strict three-way split on the canonical 6000, never mixed:

| Role | Split | Used for |
|---|---|---|
| Probe fit | `default_train` (4000) | Ridge least-squares fit at each α in the sweep. |
| α selection | `default_eval` (1000) | Pick α (best Pearson r). Nothing else. |
| Probe metrics + steering | `default_test` (1000) | Reported Pearson r / R² numbers come from here; the 50 steering pairs are sampled from here. |

`default_test` is the uncontaminated split — untouched during probe fit or α selection, so both the reported probe metrics and the steering effect are genuinely out-of-sample. Cosine matrices between probes are weight geometry only, independent of any split.

## Code pointers — do not reimplement

Every new script in this experiment is a thin orchestration layer over these existing modules. Do not rewrite any of them.

| Step | Module / entry point |
|---|---|
| Extraction | `python -m src.probes.extraction.run <config>` (`src/probes/extraction/run.py`). Single forward pass iterates all selectors in the config. |
| Probe training | `python -m src.probes.experiments.run_dir_probes --config <config>` (`src/probes/experiments/run_dir_probes.py`). |
| Utilities loader | Whatever `run_dir_probes.py` already uses (`src/measurement/storage/loading.py`). Do not re-derive Thurstonian scores from pairwise JSONs. |
| Activations loader | `src/probes/core/activations.py::load_activations`. |
| Probe-weight similarity | `src/probes/core/evaluate.py::compute_probe_similarity`. |
| Probe cross-layer eval | `src/probes/core/evaluate.py::evaluate_probe_on_data`. |
| Steered client | `src/steering/client.py::SteeredHFClient` — `generate_with_hook(messages, hook)`. |
| Steering hooks | `src/steering/hooks.py::position_selective_steering`, `compose_hooks`. |
| Pairwise task spans | `src/steering/tokenization.py::find_pairwise_task_spans`. |
| Per-layer norms | `src/steering/calibration.py::per_layer_norms(activations_path, layers) -> dict[int, float]`. Returns a mapping usable directly as the runner's `mean_norm`. |
| Steering runner | `scripts/isolated_steering/run_steering.py`, which delegates to `src/steering/runner.py`. The runner's post-hoc phase auto-runs the full-completion judge (`src/measurement/elicitation/completion_judge.py`, model `gemini-3-flash-preview`) to produce `*.parsed.jsonl`. Check `configs/steering/cross_layer_L32.yaml` for the config schema — `DifferentialCondition` with `cache_injection: differential`, `probe: ridge_L<p>`, `layers: [...]`, `multipliers: [...]`. |

Any new code is confined to `experiments/layer_sweep/{analyze_probes.py, analyze_steering.py, build_steering_pairs.py, gen_configs.py}`. Nothing else gets created under `src/`, unless the plan-mode runner refactor requires it.

## Scope

- **Model:** Gemma-3-27B IT.
- **Layers sampled:** every third block in `[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59]` — 20 layers spanning ~3%–95% depth.
- **Selectors:** `turn_boundary:-2` and `eot`. Both for probe analysis; for the steering matrix we run each selector separately so the paper can show whether the two token-choices yield interchangeable steering vectors.
- **Probe type:** Ridge only, standardize inputs, α-swept on the eval split (same setup as `persona_sweep_final_six/probe_transfer`).

## Activations (new work)

Extraction is nearly free in the number of saved layers — same forward pass, just more tensors written. One extraction run over the canonical 6000:

`configs/extraction/gemma3_27b_layer_sweep.yaml`:

```yaml
model: gemma-3-27b
include_task_ids_file: data/canonical_splits/all_6000_task_ids.txt
selectors: ["turn_boundary:-2", "eot"]  # colons must be quoted in YAML
layers_to_extract: [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59]
max_seq_len: 2048
batch_size: 32
save_every: 1000
output_dir: activations/gemma-3-27b_it/pref_layer_sweep
```

`all_6000_task_ids.txt` already exists on disk (union of the three canonical split files).

Extraction runs on a single A100 80 GB pod (`/launch-runpod`). Output: two NPZ files (`activations_turn_boundary:-2.npz`, `activations_eot.npz`), ~20 layers × 6000 tasks × 5376 dims × 4 bytes ≈ 2.5 GB each.

**Sanity check after extraction:** `set(npz['task_ids']) == set(all_6000_task_ids.txt)` for both selectors' NPZs. Extraction is considered done only when this holds — if it fails, probe training will silently train on a subset.

**Reuse of existing activations.** Layers 32 and 53 already exist in `pref_main/` at tb:-2 — re-extracting them is trivial and keeps the two NPZ files internally consistent. Do not try to splice.

## Probe training

Two probe runs (one per selector), each producing 20 probes:

`configs/probes/layer_sweep/tb-2.yaml`:

```yaml
experiment_name: layer_sweep_tb-2
run_dir: results/experiments/persona_sweep_final_six/pre_task_active_learning/default_train
eval_run_dir: results/experiments/persona_sweep_final_six/pre_task_active_learning/default_eval
activations_path: "activations/gemma-3-27b_it/pref_layer_sweep/activations_turn_boundary:-2.npz"
output_dir: results/probes/layer_sweep/tb-2
layers: [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59]
modes: [ridge]
standardize: true
alpha_sweep_size: 10
eval_split_seed: 42
```

Analogous `configs/probes/layer_sweep/eot.yaml`. Run with:

```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/layer_sweep/tb-2.yaml
python -m src.probes.experiments.run_dir_probes --config configs/probes/layer_sweep/eot.yaml
```

Wall time: seconds per layer on CPU.

## Probe analysis

Script: `experiments/layer_sweep/analyze_probes.py`. Loads the two probe manifests plus `default_test` activations and utilities. All reported scalar metrics come from `default_test`. Produces:

- **Layer-wise performance curve.** Pearson r and R² on `default_test`, plotted vs layer index, one line per selector.
- **Within-selector cosine matrix (20 × 20).** Cosine similarity between every pair of probe weight vectors at the same selector. One heatmap per selector. Uses `src/probes/core/evaluate.py::compute_probe_similarity`.
- **Cross-selector matrix (20 × 20).** tb:-2 probes on rows, eot probes on columns. Diagonal entries = "does the same layer give the same direction under different token selection?". Off-diagonal = layer-shift under token change.
- **Cross-layer probe transfer (20 × 20).** Probe trained at layer L_p, evaluated on `default_test` using activations at layer L_s (same selector). Symmetric to the cosine matrix but uses held-out prediction rather than weight geometry. Per selector.

## Steering sweep

### Task set

50 pairs from `default_test`, saved to `experiments/layer_sweep/steering_pairs.json`.

**Schema (one entry per pair):**

```json
{"pair_id": "0001",
 "task_a_id": "...", "task_b_id": "...",
 "task_a_origin": "wildchat", "task_b_origin": "alpaca",
 "utility_a": 0.42, "utility_b": -0.17, "utility_gap": 0.59,
 "task_a_tokens": 38, "task_b_tokens": 41}
```

**Sampling rule.** Uniform random from pairs with `utility_gap > 0.1` (avoid ties — steering signal is meaningless when the model is indifferent), stratified across the cross-product of `task_origin`. Seed fixed in `build_steering_pairs.py`. Higher-utility task is always `task_a` by convention; the steering sign is interpreted relative to that.

### Injection method

Differential injection — `position_selective_steering(+tensor, a_start, a_end)` composed with `position_selective_steering(-tensor, b_start, b_end)`. Same method as `experiments/steering/cross_layer/` so the matrices are directly comparable.

### Grid (two parts, layer-capped to save generations)

Prior `cross_layer/` showed steering collapses around layer 30 and is dead beyond ~35. We use this to cap the expensive injection-layer axis while still confirming the collapse on the diagonal.

1. **Self-layer diagonal (20 cells per selector).** Every one of the 20 sampled layers, steered with its own probe. Kept at all 20 layers so the paper can plot the full "steering fails at late layers" curve. Trials:
   - Layers ≤ 35 (12 layers): 3 trials per (pair, coefficient, ordering).
   - Layers ≥ 38 (8 layers): 1 trial per (pair, coefficient, ordering). We only need to confirm the collapse, not characterise dose-response.
2. **Spine × injection-layer sweep, injection ≤ 35 (5 × 12 = 60 cells per selector).** 5 spine probe layers × injection layers `[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]`. 3 trials per cell. We do not inject spine probes at layers ≥ 38 — prior work already tells us nothing interesting happens there.

Unique cells per selector: `12 (diagonal ≤35) + 8 (diagonal ≥38) + 60 (spine) − 5 (spine/diagonal overlap at L_p = L_s on the spine, among ≤35 layers) = 75`.

**Spine-probe choice.** Decided after probe analysis; default fallback `[11, 23, 32, 44, 53]`. If the probe-r curve peaks elsewhere, use the peak + two neighbours.

### Coefficient

Four coefficients per cell: `±3% × N(L_s)` and `±5% × N(L_s)`, where `N(L_s)` is the mean activation norm at the injection layer on the canonical 6000. Plus a shared 0-coefficient baseline per selector for chance-level reference.

`N(L_s)` is resolved via `src/steering/calibration.py::per_layer_norms(activations_path, layers=<injection layers in this config>)`, whose return value is passed through to the runner as `mean_norm:` in the generated config. The runner looks up `mean_norm[L_s]` per injection-layer iteration and multiplies by the multiplier and span coefficient.

Each generated steering config has `mean_norm: {L_s_1: N_1, L_s_2: N_2, ...}` rather than a scalar. Scalar `mean_norm:` still works for single-layer configs — existing configs like `configs/steering/cross_layer_L32.yaml` are unaffected.

Each checkpoint row also carries a `norm_at_layer` field (the `N(L_s)` actually applied), so analysis scripts can reconstruct absolute coefficients without re-reading the config.

### Volume

Per selector:
- Diagonal ≤35: `12 cells × 4 coefs × 50 pairs × 2 orderings × 3 trials = 14 400` gens.
- Diagonal ≥38: `8 cells × 4 coefs × 50 pairs × 2 orderings × 1 trial = 3 200` gens.
- Spine ≤35 (excluding overlap already counted above): `55 cells × 4 coefs × 50 pairs × 2 orderings × 3 trials = 66 000` gens.
- Plus the shared 0-coefficient baseline (run once per selector, not per cell): `50 pairs × 2 orderings × 3 trials ≈ 300` gens.

Total ≈ **84 000 generations per selector**, **168 000 across both selectors**. ~40% cheaper than the uncapped version.

### Runner

`scripts/isolated_steering/run_steering.py`. Emit one config per (selector, probe_layer) — each loads one probe and sweeps its full injection-layer list inside the runner's existing `layers:` loop. Directory: `configs/steering/layer_sweep/<selector>_probe_L<p>.yaml`. Checkpoints at `experiments/layer_sweep/checkpoints/<selector>_probe_L<p>.jsonl` so parallel shards never collide.

### Post-hoc judging

The full-completion judge (model `gemini-3-flash-preview`) auto-runs after each checkpoint → `*.parsed.jsonl`. No `coherence_summary.json` is produced automatically; if we want coherence aggregates we add a dedicated post-hoc script, which isn't in the critical path for the paper figures.

## Compute orchestration

Target hardware: A100 80 GB pods via `/launch-runpod` + `/provision-pod` (40 GB won't fit Gemma-3-27B bf16 + generation cache). Multiple pods in parallel are acceptable — the experiment splits cleanly into independent chunks:

- **Pod A: extraction.** Runs `gemma3_27b_layer_sweep.yaml` once (both selectors in a single pass). Produces the two NPZ files. Released when rsync of NPZs back to local is confirmed.
- **Pod B–… : steering shards.** Each shard owns one or more `(selector, probe_layer)` configs. A shard loads the model once and cycles through its configs sequentially. Example sharding: 2 pods, each owning {one selector × all 5 spine probes + diagonal halves}. Or 4 pods for tighter wall-clock. Checkpoints in `experiments/layer_sweep/checkpoints/<selector>_probe_L<p>.jsonl` so any pod death only loses the in-flight config.
- **Local (no pod): probe training + analysis + figures.** CPU-only.

The driving agent should:
- Never block on a single pod when another shard is ready to run.
- Only launch the steering shards after probe analysis has confirmed the 5 spine layers.
- Release each pod as soon as its NPZs / checkpoints are rsynced back.

## Figures

Saved under `experiments/layer_sweep/assets/`:

- **`plot_<mmddYY>_probe_r_by_layer.png`** — Pearson r (test split) vs layer index, one line per selector. Headline probe plot.
- **`plot_<mmddYY>_probe_cosine_tb2.png`** and **`..._eot.png`** — 20 × 20 within-selector cosine heatmaps.
- **`plot_<mmddYY>_probe_cosine_cross_selector.png`** — 20 × 20 tb:-2 (rows) × eot (cols) cosine.
- **`plot_<mmddYY>_probe_transfer_heatmap.png`** — 20 × 20 cross-layer probe-predict-utility Pearson r, one per selector.
- **`plot_<mmddYY>_steering_diagonal.png`** — P(chose steered task) at the self-layer-probe condition, vs injection layer, one line per selector, error bars over pairs. This is the "where does steering work?" plot.
- **`plot_<mmddYY>_steering_spine_heatmap.png`** — 5 (spine probe) × 20 (injection layer) peak P(chose steered task) heatmap, one per selector. Refusal rate overlaid as annotation.
- **`plot_<mmddYY>_steering_refusal_by_inject_layer.png`** — refusal rate vs injection layer, averaged over probes, one line per selector.
- **`plot_<mmddYY>_probe_vs_steer_agreement.png`** — per-layer scatter of probe R² (x) vs steering peak effect when that probe is injected at its own layer (y). Tests whether better probes steer better at their own layer.

## Paper integration

Targets two spots in `paper/main.tex`:

- **Body.** Replace the current narrow cross-layer figure with (a) the self-layer diagonal "steering fails at layer ≥ ~35" curve at tb:-2 and (b) the layer-wise probe r curve for both selectors.
- **Appendix.** Within- and cross-selector cosine matrices, cross-layer probe-transfer heatmaps, the 5 × 12 spine steering heatmap, and refusal-rate marginals — for both selectors.

## Artifacts the launch agent must create

The spec describes the experiment at a design level; the files themselves don't exist yet. The launch agent's first task is to create them. Each artifact below is self-contained — nothing depends on the spine-layer choice until `gen_configs.py` is **invoked**, so all four scripts and both config bodies can be written up front.

### Configs (YAML, committed under `configs/`)

- **`configs/extraction/gemma3_27b_layer_sweep.yaml`** — body already inlined in §Activations. Write verbatim; quote the colon-containing selectors.
- **`configs/probes/layer_sweep/tb-2.yaml`** and **`configs/probes/layer_sweep/eot.yaml`** — bodies inlined in §Probe training. Differ only in `activations_path`, `output_dir`, and `experiment_name`.

### Scripts (under `experiments/layer_sweep/`)

- **`build_steering_pairs.py`** — produces `steering_pairs.json` per the schema in §Steering sweep → Task set.
  - Load `default_test` Thurstonian utilities via the same loader `run_dir_probes.py` uses (`src/measurement/storage/loading.py`).
  - Load task prompts from the shared completions JSON (check `pref_main/` sidecar; don't re-generate).
  - Orient each pair so `task_a` has the higher utility; fix a seed inside the script.
  - Filter to `utility_gap > 0.1`, then stratified random sample across the cross-product of `task_origin` to reach 50 pairs.
  - Tokenise both tasks via `src/steering/tokenization.py::find_pairwise_task_spans` (load the tokenizer only — not the full 27 B model — via `AutoTokenizer.from_pretrained("google/gemma-3-27b-it")`). Assert every pair has non-None, non-overlapping spans. Write `task_a_tokens`, `task_b_tokens` to the JSON.

- **`analyze_probes.py`** — consumes `results/probes/layer_sweep/{tb-2,eot}/` plus `default_test` activations and utilities. Produces all four probe figures (`plot_<mmddYY>_probe_r_by_layer.png`, within- and cross-selector cosine heatmaps, `plot_<mmddYY>_probe_transfer_heatmap.png`) into `experiments/layer_sweep/assets/`. Also writes a tiny `probe_metrics.json` (layer → r, R²) so `gen_configs.py` can consume it. Reuse `src/probes/core/activations.py::load_activations`, `src/probes/core/storage.py::load_probe_direction`, `src/probes/core/evaluate.py::{evaluate_probe_on_data, compute_probe_similarity}`.

- **`gen_configs.py`** — emits steering YAMLs under `configs/steering/layer_sweep/`. CLI args: `--selector {tb-2,eot}`, `--spine-layers "11,23,32,44,53"` (or similar). For each probe layer emits one config:
  - Diagonal configs (20 of them): `layers: [<L>]`, `multipliers: [-0.05, -0.03, 0.03, 0.05]`, `n_trials: 3` for L ≤ 35 else `n_trials: 1`, `probe: "ridge_L<L>"`.
  - Spine configs (5 of them): `layers: [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]`, `multipliers: [-0.05, -0.03, 0.03, 0.05]`, `n_trials: 3`, `probe: "ridge_L<spine_L>"`.
  - `mean_norm:` populated via `src/steering/calibration.py::per_layer_norms(activations_path, layers=<the config's layers>)` — dict embedded inline in the YAML.
  - `probe_manifest:` points to `results/probes/layer_sweep/<selector>/`.
  - `checkpoint_path:` stamped `experiments/layer_sweep/checkpoints/<selector>_probe_L<p>.jsonl` so shards never collide.
  - `pairs_path:` points to `experiments/layer_sweep/steering_pairs.json`.
  - `template_path: src/measurement/elicitation/prompt_templates/data/completion_preference.yaml` (same as `cross_layer/`).

- **`analyze_steering.py`** — consumes `experiments/layer_sweep/checkpoints/*.parsed.jsonl`. Produces the three steering figures in §Figures plus `plot_<mmddYY>_probe_vs_steer_agreement.png`. Aggregation: P(chose steered task) computed after `_remap_choice` is already applied (the runner stores `choice_original`). Report mean effect across the four coefficients in the main figures; peak goes in the appendix.

### Sequencing constraint

`gen_configs.py` can be **written** in step 1 but can only be **run** after probe analysis has chosen the 5 spine layers. `analyze_steering.py` is the same — authored early, run only after the sweep. The agent should not block on probe analysis before writing these scripts.

## Pre-run checklist (mission-critical, verify before first pod launch)

These are the places where an assumption in the spec could bite hard once we're burning GPU hours. Each is a cheap local check.

1. **Default AL runs are finalised.** `results/experiments/persona_sweep_final_six/pre_task_active_learning/default_{train,eval,test}/thurstonian_*.csv` all exist and each covers the expected task count (4000 / 1000 / 1000). If any run is still active or the symlink points at a partial dir, probe training will silently use fewer tasks.
2. **`all_6000_task_ids.txt` equals the union of the three splits.** Already on disk; a one-line `sort | uniq -c` protects against a stale file.
3. **Extraction supports two selectors in one run.** `src/probes/extraction/run.py` iterates the `selectors:` list in one forward pass — confirm once on a 10-task config before launching on 6000.
4. **`cross_layer/` positive control.** Before sharding the full sweep, run one cell that matches a known `cross_layer/` condition: `(selector=tb:-2, probe_layer=32, inject_layer=32, ±5% × N(32))` on 10 pairs × 3 trials. Confirm the P(chose steered) is within noise of the `cross_layer/` number at the equivalent coefficient. If it drifts, pipeline has regressed — abort and diagnose before scaling to 168 000 generations.
5. **Differential-injection smoke test at edge layers.** Prior `cross_layer/` was only validated at injection layers 10–30. Run once with 2 pairs at injection layers 2 and 35 at ±5% × N, confirming the hook fires and produces non-degenerate output. Catches tokenisation / hook-indexing bugs at the extremes.
6. **Task-span tokenisation.** `find_pairwise_task_spans` returns valid spans (no `None`, no overlap) for every pair in `steering_pairs.json`. 10-line assertion in `build_steering_pairs.py`.
7. **Checkpoint paths are unique per shard.** Config generator stamps `checkpoint_path` with both selector and probe layer.
8. **Post-extraction task-id assert.** `set(npz['task_ids']) == set(all_6000_task_ids.txt)` for both selectors' NPZs (already in §Activations but worth naming here).

Items 3, 4, 5 are blocking. Items 1, 2, 6, 7, 8 are read-only checks.

## Work order

1. **Create all artifacts.** Write the two configs + four scripts listed in §Artifacts. None of these depend on later decisions (spine layers are a runtime argument to `gen_configs.py`, not a source-code decision).
2. **Run pre-run checklist items 1–3, 6–8.** Read-only — doesn't need a pod.
3. **Launch extraction pod (A100 80 GB).** `/launch-runpod`, `/provision-pod`, run `configs/extraction/gemma3_27b_layer_sweep.yaml`. Rsync NPZs back to `activations/gemma-3-27b_it/pref_layer_sweep/`. Release the pod or repurpose as a steering shard.
4. **Train probes locally.** Two probe configs, CPU is fine.
5. **Probe analysis.** Run `analyze_probes.py`. Inspect the performance curve and `probe_metrics.json` — decides the 5 spine probe layers for step 8.
6. **Generate steering pairs.** Run `build_steering_pairs.py` → `steering_pairs.json`. Span-tokenisation assertion runs inside the script.
7. **Positive control (pre-run check #4).** On one A100 pod, run the `cross_layer/` reproduction cell. Do not proceed if it drifts from the known number.
8. **Run `gen_configs.py`** once per selector, passing the chosen spine layers. Produces the full set of YAMLs under `configs/steering/layer_sweep/` with `mean_norm:` dicts already populated.
9. **Steering sweep across A100 shards.** Launch N pods (user-decided, typically 2–4). Each pod owns a disjoint subset of configs. Checkpoints rsynced back per config.
10. **Post-hoc judging.** Auto-run by the runner. Verify `*.parsed.jsonl` exists for every shard.
11. **Steering analysis.** Run `analyze_steering.py` → heatmaps, diagonal curve, refusal marginals.
12. **Report + figure splice.** `experiments/layer_sweep/layer_sweep_report.md`, then splice into `paper/main.tex`.
13. **Cleanup.** Delete `activations/gemma-3-27b_it/pref_layer_sweep/` after probes + report are in.

## Out of scope

- Other personas. Default only; the persona-generalisation story lives in `probe_transfer/`.
- Non-ridge probes (logistic / BT). Kept as a future ablation.
- Other selectors (prompt_last, task_mean). tb:-2 and eot are the two the user has strongest priors about; adding more multiplies the probe+steering cost without a clear question.
