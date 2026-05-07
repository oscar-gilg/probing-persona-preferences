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

### Audit (independent subagent)

PASS on all 7 checks (spec compliance, probe correctness, seed independence,
checkpoint paths, pairs file, no system prompt leakage, cross-config
consistency — only `name`, `probe`, and `checkpoint_path` differ).

### Run

Launched seeds 0 and 1 sequentially in tmux session `steering`:
```
python -u -m scripts.isolated_steering.run_steering <cfg_seed0> &&
python -u -m scripts.isolated_steering.run_steering <cfg_seed1>
```

Wallclock per seed:
- seed 0: model load 54 s, generation 6.0 min (2.5 gen/s sustained), judge ~2.2 min.
- seed 1: model load 17 s (HF cache warm), generation 6.0 min (2.5 gen/s), judge ~2.1 min.

Total wall ≈ 17 min for both seeds.

### Outputs

```
experiments/random_direction_l23_quick/multi_seed/checkpoints/
├── random_contrastive_seed0.jsonl          (900 raw)
├── random_contrastive_seed0.parsed.jsonl   (900 parsed)
├── random_contrastive_seed1.jsonl          (900 raw)
└── random_contrastive_seed1.parsed.jsonl   (900 parsed)
```

Verified per parsed file:
- 900 rows total.
- 180 rows per `signed_multiplier ∈ {-0.05, -0.03, 0.0, 0.03, 0.05}`.
- Compliance breakdown — seed 0: 740 truncated / 106 hard_refusal / 51 full / 3 missing; seed 1: 739 / 104 / 57 / 0. Refusal rate ~11–12% per coef, no monotonic trend.

Per spec, **no averaging or plotting on the pod** — done locally after pulling
the parsed jsonls. The two checkpoints + the parent's `random_contrastive.parsed.jsonl`
on `origin/experiment/random_direction_l23_quick` give three independent random
directions for the local averaging step.
