# Running log — cross-persona unilateral steering

## 2026-04-24

### Setup
- 100 pairs sampled from `default_test` (utility_gap > 0.1, stratified origin×origin, seed 42). Written to `experiments/cross_persona_unilateral/steering_pairs.json`.
- 6 configs generated via `gen_configs.py`. Per-persona `mean_norm(L25)` computed from each persona's own sweep activations.
  - aura: 42361.9
  - contrarian: 39893.9
  - mathematician: 41431.9
  - sadist: 40826.5
  - slacker: 40989.2
  - strategist: 41366.3
- Feature branch `cross_persona_unilateral` pushed to origin.

### Pod
- Launched `cross-persona-unilat` (id `2jn86fd8bpgbn5`) on A100-SXM4-80GB, 150 GB disk / 50 GB volume. (Tried to resume `layer-sweep-unilateral` first — host had no free GPUs, fell back to new pod.)
- Synced `.env` and `results/probes/persona_sweep_final_six/` to `/workspace/repo/`. Activations NOT synced — `mean_norm` is pre-baked into the configs, runner only needs probes at runtime.
- Branch `cross_persona_unilateral` checked out on pod with commit 90246340.

### Sadist pilot (in progress)
- Started tmux session `sadist` on pod with `python -u -m src.steering.runner configs/steering/cross_persona_unilateral/sadist.yaml`.
- Log: `/workspace/repo/experiments/cross_persona_unilateral/sadist_run.log`.
- Expected ~4800 generations (100 × 3 × 2 × 2 × 4) on Gemma-3-27B, probably ~2–3 hrs.

### Fix: scalar mean_norm (2026-04-24 12:43)
- First sadist attempt crashed at line 722 with `TypeError: unsupported operand type(s) for *: 'dict' and 'float'`.
- Root cause: gen_configs wrote `mean_norm` as a dict `{25: value}` (matching the layer_sweep worktree's runner refactor), but main's runner.py expects a scalar float.
- Fix: changed gen_configs to emit scalar `mean_norm: value` (we only use L25, so dict form is unnecessary). Regenerated all 6 configs.
- Restarted sadist — model loaded from HF cache this time (no re-download).
