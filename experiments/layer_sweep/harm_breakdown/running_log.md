# Harm-breakdown running log

## 2026-04-24

### Setup
- Worktree: `.claude/worktrees/harm_breakdown` on branch `research-loop/harm_breakdown`.
- Pod `layer-sweep-extract` (spec's "pod 1") and siblings `layer-sweep-eot`/`layer-sweep-unilateral` all blocked — hosts had no free A100 slots. User approved launching a fresh pod.
- Launched `harm-breakdown-l23` (A100-SXM4-80GB, 100GB disk / 50GB volume, pod id `9sp4um6vfne1am`).
- Symlinks in worktree: `activations/`, `results/experiments/persona_sweep_final_six/pre_task_active_learning/default_test/`, `experiments/layer_sweep/checkpoints/` (to layer_sweep worktree), and `results/probes/layer_sweep/eot/` (to layer_sweep worktree — probe manifest).

### Pair construction
- `experiments/layer_sweep/harm_breakdown/build_pairs_150.py` — extends parent pair builder with pair-type stratification.
- Origin buckets: benign = {WILDCHAT, ALPACA, MATH}; harmful = {BAILBENCH, STRESS_TEST}.
- Pool after `utility_gap > 0.1`: 493,568 (bb=280184, hb=184206, hh=29178).
- Sampled 50 bb + 50 hb + 50 hh (seed=42). Asserted counts and non-overlapping token spans for all 150 pairs.
- Output: `experiments/layer_sweep/harm_breakdown/steering_pairs_150.json`.

### YAML configs
- `configs/steering/layer_sweep/harm_breakdown/contrastive_L23_150.yaml` — one `DifferentialCondition` with `spans: {first: 1, second: -1}`, probe `ridge_L23`, mults ±0.03, ±0.05, 3 trials.
- `configs/steering/layer_sweep/harm_breakdown/single_task_L23_150.yaml` — two conditions (`unilateral_first` / `unilateral_second`), same layer/probe/mults/trials, single-span each.
- `mean_norm[23] = 29381.541015625` (copied from parent `eot_probe_L23.yaml` — not recomputed, per spec).
- Both configs pass `load_config` locally.
