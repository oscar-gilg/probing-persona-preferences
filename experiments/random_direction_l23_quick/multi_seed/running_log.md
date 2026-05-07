# Running log — random_direction_l23_quick / multi_seed

## 2026-05-07 setup

On-pod (IS_SANDBOX=1), branch `experiment/random_direction_l23_quick_multi_seed`.
Branched from `main` (parent's deliverable lives on
`origin/experiment/random_direction_l23_quick`; not merged).

GPU: 1× H100 80GB.

### Environment

`/opt/venvs/research/` was empty. Installed via
`uv pip install --python /opt/venvs/research/bin/python -e ".[dev]"` plus
`requests` (instructor needs it; not in pyproject — same gap noted in
the parent run's log).
Stack: torch 2.11.0+cu128, transformers (latest), python 3.12.

### Random probes

Wrote `scripts/random_direction_l23_quick_multi_seed/make_random_probe.py`
(parameterised by seed). Generated:
- `results/probes/layer_sweep/eot/probes/probe_random_L23_seed0.npy`
- `results/probes/layer_sweep/eot/probes/probe_random_L23_seed1.npy`

Both shape `(5377,)` with last element 0.0 (intercept), direction L2-normed.
Manifest at `results/probes/layer_sweep/eot/manifest.json` registers both
under `random_L23_seed{s}` with `layer: 23`.

Sanity: `load_probe_direction` returns `(layer=23, shape=(5376,), norm=1.0)`
for both. Cross-seed dot product = -0.0028 (expected ≈ ±1/√5376 ≈ 0.014 — fine).

### Configs

`configs/steering/random_direction_l23_quick_multi_seed/random_contrastive_seed{0,1}.yaml`
mirror the parent's `random_contrastive.yaml` exactly except for `probe`,
`name`, and `checkpoint_path`. Parameters: layer 23, mean_norm 29381.541,
multipliers `[-0.05, -0.03, 0.0, 0.03, 0.05]`, n_pairs 30, n_trials 3,
both orderings, temperature 1.0, seed 42, default Assistant (no
system_prompt), template `completion_preference.yaml`.

Each run = 30 pairs × 5 multipliers × 2 orderings × 3 trials = 900 generations.
Two seeds = 1800 generations total. Spec estimate: 20–30 min on H100.
