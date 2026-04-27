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
- Pairs sourced from `results/experiments/uniform_eval_gemma3_27b_v3/.../measurements.yaml`. Spec claimed 723 unique pairs; actual file has **955 unique unordered pairs** (4775 rows = 955 × 5 measurements). Using 955.
- Per-cell output: `experiments/preference_direction_ablation/results/<cell>/measurements.yaml` (standard format).

### Symlinks (worktree → main repo for read-only data)
- `results/probes/heldout_eval_gemma3_tb-1/probes/probe_ridge_L25.npy`
- `results/probes/heldout_eval_gemma3_tb-1/probes/probe_ridge_L32.npy`
- `results/experiments/uniform_eval_gemma3_27b_v3/.../measurements.yaml`

### Smoke test (3 cells × 5 pairs, max_new_tokens=64)
- B0: 0.9 min for 5 pairs (~11 s/pair, no hooks)
- A_L32_probe: 0.8 min for 5 pairs (~10 s/pair, single-layer hook)
- A_L32_random0: 0.9 min for 5 pairs (~11 s/pair); 1/15 generations was a refusal; 100% position-bias flip rate (random direction destroys coherence — expected)
- Pipeline produces correct format; analyze.py runs on output and computes metrics. ✅

### First attempt at smoke (max_new_tokens=512)
- ~8 min/pair → would have extrapolated to ~210 hours for the full sweep. Aborted.
- Dropped max_new_tokens to 64 (spec allows ≥16). Judge regex catches the "Task A:"/"Task B:" prefix the model writes; partial completion is enough.

### Sanity test status
- **Test 1 (hook correctness):** covered by existing `tests/steering/test_steering_gpu_e2e.py` (max |cos(resid, d̂)| < 1e-2 after projection). Hook + multi-layer infra already shipped pre-session.
- **Test 2 (random-control coherence):** will be assessed on full random-control runs (5 random vectors × 100 pairs each); spec threshold ≥0.85 modal-choice agreement with B0.
- **Test 3 (length / refusal audit):** computed by analyze.py per cell.

### Decision: kick off full sweep
- Estimated ~22 hours wallclock on A100 SXM at ~10 s/pair (probe cells slightly slower for multi-layer / band).
- Babysit will pause the pod on completion. Results land in `experiments/preference_direction_ablation/results/<cell>/measurements.jsonl`.
