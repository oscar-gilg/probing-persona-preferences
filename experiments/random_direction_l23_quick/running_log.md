# Running log — random_direction_l23_quick

## 2026-05-06 setup

On-pod (IS_SANDBOX=1), branch `experiment/random_direction_l23_quick`.
GPU: 1× H100 80GB (free 81GB).

### Context audit

The spec references the parent experiment `persona_steering_l23_finegrain`,
but its directory and checkpoints are NOT present on this branch (and never
existed in the git history of this branch). Likewise the probe manifest dir
`results/probes/layer_sweep/eot/` is absent.

Decision: this experiment is self-contained — we only need a manifest entry
for our own random probe. Will create a fresh `manifest.json` with a single
`random_L23_seed42` entry. Existing `configs/steering/layer_sweep/harm_breakdown/contrastive_L23_150.yaml`
is the structural template we mirror.

The Fig 3a panel script (`paper/figures/panels/build_steering_integrated.py`)
also references parent checkpoints. We'll add overlay logic for the random
curve; final rendering of the composite Figure 3a will require parent
checkpoints to be present too — out of scope for this null-control run.

### Setup actions

- `mkdir`s for: checkpoints/, assets/, scripts/random_direction_l23_quick/,
  results/probes/layer_sweep/eot/probes/, configs/steering/random_direction_l23_quick/

### Environment

`/opt/venvs/research/` was empty. Installed via
`uv pip install -e ".[dev]"` + `uv pip install requests`
(needed by instructor; not in pyproject for some reason).
Stack: numpy 2.4.3, torch 2.11.0+cu128, transformers 5.8.0, python 3.12.13.

### Random probe

`scripts/random_direction_l23_quick/make_random_probe.py` writes
`results/probes/layer_sweep/eot/probes/probe_random_L23_seed42.npy`
(shape `(5377,)`, last element = 0 intercept) and a fresh single-entry
`manifest.json`. Verified `load_probe_direction` returns layer 23, shape
`(5376,)`, norm 1.0.

### Config

`configs/steering/random_direction_l23_quick/random_contrastive.yaml`
mirrors `configs/steering/layer_sweep/harm_breakdown/contrastive_L23_150.yaml`
exactly except: `n_pairs: 30`, `probe: random_L23_seed42`,
multipliers include `0.0` (per spec), no system_prompt.

Note re. "first 30": runner's `_load_pairs` does `random.sample(all, 30)` with
`random.seed(42)` — deterministic but not literally the first 30 of the JSON.
This matches the runner's standard behavior and is what `n_pairs: 30` does.
For a null-control overlay, the 30-pair sub-sample only matters insofar as
matched pair-types exist between random and default curves; the parent run
likely used `n_pairs: null` (full 150) but a deterministic 30-pair sample
should still produce a clean flat null curve.

