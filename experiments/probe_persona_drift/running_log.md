# Probe persona drift — running log

## 2026-05-05

### 00:00 — Setup
- Worktree: `.claude/worktrees/probe_persona_drift/` on branch `probe_persona_drift` (off `origin/main`).
- Pod: `runpod-probe-persona-drift` (id `0wyufy3kv8yhfg`, H100 80GB HBM3, disk 100GB / volume 50GB).
- Symlinks: `activations/gemma-3-27b_it`, `data/creak`, `results/probes/heldout_eval_gemma3_tb-5` → main repo.
- Spec committed at `experiments/probe_persona_drift/probe_persona_drift_spec.md`.
- Cache size estimate in spec is wrong (says 280 GB, actual recompute ≈ 3 GB at 4608-dim residual stream). Disk budget is fine either way.

### Plan
1. Data prep (local): HarmBench fetcher → `data/harmbench/`, `OriginDataset.HARMBENCH`, split builder for truth + harm with seed 42.
2. Push branch, sync prep to pod.
3. Build 14 extraction configs (7 personas × 2 targets), run via `python -m src.probes.extraction` in tmux on pod, babysit.
4. Train binary ridge probes from cached default-persona activations (small wrapper around `RidgeClassifier`).
5. Score (cohen_d_pooled + AUC) every probe × eval-persona × layer cell. Write tables.
6. Plots: headline 3-panel, train-size sweep, transfer matrix.
7. Report + review.
