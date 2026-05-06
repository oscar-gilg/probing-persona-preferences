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

### Audit (independent subagent)

PASS on all four checks (spec compliance, random probe correctness, pair
file integrity, pipeline launch). No invalidating discrepancies.

### Run launch

Launched in tmux `steering` session at 22:01:06 UTC:
`/opt/venvs/research/bin/python -u -m scripts.isolated_steering.run_steering configs/steering/random_direction_l23_quick/random_contrastive.yaml > /workspace/steering.log 2>&1`

### Panel script update (paper/figures/panels/build_steering_integrated.py)

- Added `RANDOM_CONTRASTIVE_PARSED` constant.
- Added `load_random_contrastive()` — returns pooled-across-pair-type curve
  (one null line, no harm/benign breakdown), or `None` if checkpoint absent.
- `plot_overlay` accepts an optional `random_curve` kwarg, drawn as a dashed
  gray line behind the colored pair curves.
- `main()` calls `load_random_contrastive()` for the contrastive panel and
  passes it through. Final figure rendering still requires parent
  `default_contrastive.parsed.jsonl` and `default_single_task.parsed.jsonl`,
  which are not present on this branch.

### Run completion

Model load: 207s (3:27). Generation: 5.0 min for 900 trials at 3.0 gen/s.
Judge parsing: ~2.5 min for 900 completions at ~6.6/s.
Total wall ≈ 11 min.

900/900 generations clean (0 skipped). 900/900 judge rows produced. Refusals:
~10–12% across c, no monotonic trend.

### Result (canonical contrastive frame)

| c | P(chose steered) | 95% CI | n responded | refusal |
|---|---|---|---|---|
| -0.050 | 0.454 | [0.400, 0.508] | 324 | 10.0% |
| -0.030 | 0.474 | [0.420, 0.528] | 321 | 10.8% |
|  0.000 | 0.500 | [0.445, 0.555] | 316 | 12.2% |
| +0.030 | 0.526 | [0.472, 0.580] | 321 | 10.8% |
| +0.050 | 0.546 | [0.492, 0.600] | 324 | 10.0% |

Swing |max − min| = 0.093. All CIs overlap 0.5. Compared to the validated
probe direction's ~0.97 swing over the same coefficient range, this is a
clean null. c=0 → 0.500 with 158/316 = exactly 0.5 — a parsing sanity check
that the canonical-frame symmetry-by-construction holds.

Plot: `assets/plot_050626_random_L23_contrastive_null.png` (via
`scripts/random_direction_l23_quick/plot_null.py`).


