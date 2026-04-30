# L23 follow-up — running log

## 2026-04-30 — setup

### Pre-flight (in main repo)
- 615 unique unordered pairs survive after dropping pairs where either task is in the L23 probe's training set (4000 task IDs from `persona_sweep_final_six/.../default_train`).
- Verified parent's `B0/measurements.jsonl` covers all 615 filtered pairs (no missing pairs).
- L23 probe found in `worktree-layer_sweep`: `final_r = 0.798`, `α* = 4641`, EOT token, 4k-task train. Symlinked into the new worktree at `results/probes/layer_sweep_eot/probes/probe_ridge_L23.npy`.

### Worktree
- Created `.claude/worktrees/L23_followup` on branch `worktree-L23_followup`.
- Symlinked: activations dir, source pair set, probe-train measurements, L23 probe + manifest, parent B0.
- Cherry-picked driver scripts from parent worktree: `run_cells.py`, `analyze.py`, `validation_sentinel.py`, `plot_probe_vs_random_panel.py`. Wrote a thin L23-specific driver `scripts/L23_followup/run_L23.py` that imports parent helpers and overrides cell list + pair set + results dir (avoids mutating the parent module's constants).
- Pre-built pair list `experiments/preference_direction_ablation/L23_followup/results/pairs.yaml` (615 pairs).

## 2026-04-30 — pod up, smoke passed

### Pod
- `pref-ablation-l23` (id `z8couqa02cvpfy`), A100 SXM4-80GB, 100GB disk / 50GB volume.
- Synced to /workspace/repo: `.env`, L23 probe + manifest, source pair set, probe-train measurements, parent B0.
- Cherry-picked `bb0523a steering: add ablation mode to SteeredHFClient` from parent worktree (was missing from main).

### Smoke (3 pairs through A_L23_probe)
- Sanity #2 (probe norm == 1): pass.
- Sanity #3 (B0 covers all 615 filtered pairs): pass.
- Sanity #1 (hook applies cleanly at L23): pass — 3 pairs generated without error.
- Throughput ~10 s/pair, matches parent. Full sweep estimate: probe 615 × 10s = 1.7h; 5 random × 100 × 10s = 1.4h total. ~3.5h end-to-end.

### Kicking off full sweep
- All 6 cells (`A_L23_probe` + `A_L23_random{0..4}`). Driver is resumable so the 3 smoke pairs will be skipped.
- tmux session `l23-sweep` on pod; babysit will pause pod on completion.
