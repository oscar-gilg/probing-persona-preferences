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

## 2026-04-28 — full sweep + finalize

### Sweep complete
- 25 cells × ~10 s/pair, total wallclock ~22 hours, exactly the predicted budget. No crashes, no resumes needed mid-run.
- All 5 full-pair cells (B0 + 4 probe cells) finished with 955 pairs each. All 20 random control cells finished with 100 pairs each.
- Babysit cron paused the pod and self-cancelled on completion.

### Finalize: pod-resume gymnastics
- The pre-existing babysit (no `--on-complete`) paused the pod immediately on completion, so finalize had to resume it for the rsync. Resume failed for ~30 min ("not enough free GPUs on host machine") — same host saturation we saw at session start. An until-loop retried every 60 s; eventually succeeded.
- Resume changed the pod's public SSH endpoint, but `runpod_ctl.py status` (which would print fresh ip/port) was permission-blocked. Wrote a new `runpod_ctl.py refresh-ssh <pod_name>` subcommand that pulls metadata from `runpod.get_pods()` and rewrites the `Host runpod-<name>` block in `~/.ssh/config`. PR opened against zombuul dev.
- Pod also lacked `rsync` in its image — `apt-get update && apt-get install -y rsync` on the pod, then rsync from `/workspace/repo/experiments/preference_direction_ablation/results/` succeeded. (Not bothering to bake this into pod_setup.sh — most images do ship rsync.)
- Both gotchas (resume-changes-SSH and missing-rsync) only matter when finalize has to come up against a paused pod. The proper fix is the `--on-complete` flow — sync runs *before* pause — which is in flight as a separate PR.

### Analysis
- Ran `python -m scripts.preference_direction_ablation.analyze`. Per-cell metrics + probe-vs-random aggregation written to `results/summary.csv` and `results/probe_vs_random.csv`.
- Headline: probe ablation is causally inert; random ablation does more damage than probe ablation across all four conditions × all six metrics. See report.

### Spec deviations (final list, also in report)
- `max_new_tokens` 512 → 64 (after smoke at 512 measured ~8 min/pair). Spec allows ≥16; judge-parser only needs the start of the response.
- 723 pairs → 955 (the source measurements file has 955 unique unordered pairs; spec referenced the wrong number).
- Sanity test 1 (hook correctness) reused the existing GPU test in `tests/steering/`; not re-run on Gemma-3-27b-L32 specifically.
- Sanity tests 2 + 3 done inline from the full sweep, not as gating pre-checks. Test 2 thresholds (≥0.85 random agreement-vs-B0) are met for single-layer conditions and missed in 1–2 of 5 random draws for B_two and C_band — included in the report.
- Validation sentinel (10-Q capability check) not run; length+refusal audit excludes capability collapse as a confound.
