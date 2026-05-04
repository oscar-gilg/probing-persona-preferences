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

### 00:14 — Data prep done
- `data/harmbench/harmbench_behaviors_text_all.csv` (400 behaviours), `src/task_data/data/harmbench.jsonl` registered via `OriginDataset.HARMBENCH`.
- Truth splits: 4810 true / 4585 false in CREAK known-correct; 500 held-out (250/250); train pools 50 / 200 / 1000 / 4000.
- Harm splits: 1630 BailBench + 400 HarmBench harmful, 10000 Alpaca + 8044 WildChat benign; 500 held-out (250/250); train pools 50 / 200 / 1000 / **3560** (4000 capped at min available, audit accepts).
- 14 extraction configs at `configs/extraction/probe_persona_drift/{persona}_{target}.yaml`. Combined extraction-ID files at `experiments/probe_persona_drift/results/splits/{target}_extraction_ids.json` (truth: 4500; harm: 4060).

### 00:17 — Audit (parallel subagent)
- 158/160 checks pass.
- Found: sadist prompt missing the trailing sentence "You are relentless, inventive, and you never, ever feel bad about any of it." (688 vs 765 chars, vs §3.1 reference). Fixed at 23:20, configs regenerated, pushed.
- Found: HF auth needed — pod's shell `HF_TOKEN` was empty. Fixed by adding `load_dotenv()` to the runner script so `.env` is read at startup.

### 23:21 — Extraction launched
- tmux session `extraction` running `python -m scripts.probe_persona_drift.run_all_extractions` (single process, model loaded once).
- Babysit cron `43f5816f` set, every 7 min. Will pause pod and cancel itself on completion.
- Progress at 23:24: extraction 1/14 (Aura_harm) at batch 136/254, ~1.1 batches/sec, GPU 55 GB / 80 GB. ETA all-14 ≈ 60 min from launch.
