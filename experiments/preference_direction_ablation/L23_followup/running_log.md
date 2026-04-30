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
