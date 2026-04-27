# Preference Direction Ablation — running log

## 2026-04-27 — setup

### Infrastructure decisions
- **Pod**: `pref-ablation` (id `stuojxplqggile`), A100-SXM4-80GB, 100 GB disk, 50 GB volume.
- Originally tried to resume `harm-breakdown-l23` (and 2 other A100 SXM paused pods) — RunPod returned "no free GPUs on host". Launched fresh.
- Worktree branch: `worktree-preference_direction_ablation` (off main).

### Infrastructure already present (per spec section "Infrastructure to be implemented")
- `project_out_direction(d̂)` hook in `src/steering/hooks.py:86` — already shipped in commit `b5974c6`. Round-trip cosine test in `tests/steering/test_steering_gpu_e2e.py`.
- Multi-layer hook execution: `HuggingFaceModel.generate_with_hooks_n` exists in `src/models/huggingface_model.py:778`.

### Infrastructure added this session
- **Ablation mode in `SteeredHFClient`** (commit `3bfda65`). New constructor params `ablate_layers: list[int]` + `ablate_directions: list[np.ndarray]`. When set, `_dispatch` routes through `generate_with_hooks_n` with `project_out_direction` hooks. Mutually exclusive with steering mode (cannot combine).
- B0 baseline = `SteeredHFClient(ablate_layers=[], ablate_directions=[])` — empty hooks list, no projection, equivalent to base model.

### Driver design
- Custom driver `scripts/preference_direction_ablation/run_cells.py` (CLAUDE.md "scripts/{experiment_name}/" pattern).
- Loads `HuggingFaceModel` once, swaps `SteeredHFClient` per cell. Reuses `build_revealed_builder` + `measure_pre_task_revealed_async` for prompt construction and parsing — no new templates or parsers.
- 723 pairs sourced from `results/experiments/uniform_eval_gemma3_27b_v3/.../measurements.yaml` (read once, dedup to unique `(task_a, task_b)` pairs).
- Per-cell output: `experiments/preference_direction_ablation/results/<cell>/measurements.yaml` (standard format).

### Symlinks (worktree → main repo for read-only data)
- `results/probes/heldout_eval_gemma3_tb-1/probes/probe_ridge_L25.npy`
- `results/probes/heldout_eval_gemma3_tb-1/probes/probe_ridge_L32.npy`
- `results/experiments/uniform_eval_gemma3_27b_v3/.../measurements.yaml`
