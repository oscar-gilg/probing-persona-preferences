# Qwen E1a — Resume Instructions

Context: integrating Qwen-3.5-122B E1a replication into `paper/main.tex` §4.1.1
alongside the Gemma result, with a side-by-side scatter (probe Δ vs utility Δ
preferred; behavioural Δ as a fallback).

## Current state (2026-04-23)

### ✓ Done
- Worktree data unified into main: configs at `configs/measurement/active_learning/qwen35_ood_exp1b/` (17 YAMLs: baseline + 16 persona) and 3 existing re-run condition results at `results/experiments/exp_20260422_124719/pre_task_active_learning/` (ancient_history pos/neg + astronomy_neg).
- Scripts written in `scripts/qwen_replication/`:
  - `extract_e1a_activations.py` — single-load extraction of Qwen-3.5-122B for all 17 E1a conditions, 48 tasks from `configs/ood/tasks/target_tasks.json`, layer 38, selectors `turn_boundary:-1` and `turn_boundary:-4`. Writes to `activations/qwen35_122b_ood/e1a/{condition}/`.
  - `extract_gemma_e1a.py` — optional parallel extraction of Gemma-3-27B on the same conditions (not required for the minimal deliverable).
  - `analyze_e1a.py` — computes per-task `probe_delta`, `utility_delta`, `behavioral_delta` for Qwen; falls back to stored behavioural arrays for Gemma when its activations are absent. Writes `experiments/qwen_replication/e1a/e1a_per_task.json`.
  - `plot_e1a_scatter.py` — two-panel scatter (Gemma | Qwen), utility + behavioural variants. Writes `paper/figures/plot_<DATE>_e1a_scatter_{utility,behavioral}.png`.

### ⏳ In progress (background process)
- 14 behavioural measurements running via OpenRouter. Launched with:
  ```
  cat experiments/qwen_replication/e1a/logs/configs_to_run.txt | xargs python -m src.measurement.runners.run --max-concurrent 200 --experiment-id qwen35_ood_e1a_final
  ```
  Log: `experiments/qwen_replication/e1a/logs/measurements.log`.
  Output: `results/experiments/qwen35_ood_e1a_final/pre_task_active_learning/` — 14 condition dirs with `checkpoint.yaml`. Thurstonian fits land here as each AL run converges.
  Process alive check: `ps aux | grep qwen35_ood_e1a_final | grep -v grep`.
  If dead: re-launch the same command; it should resume from checkpoints.

### ✗ Not done — needs a new pod
- Qwen-3.5-122B activations for the 17 E1a conditions. Previous pod (`qwen-e1a-extract`, `6rnrae50x8upes`) had repeated silent process deaths mid-download — has been terminated.

## Next steps (end-to-end)

1. **Launch new pod** with `/zombuul:launch-runpod qwen-e1a-extract` — pick **3× A100 80GB SXM** (matches what was needed last time). Then `/zombuul:provision-pod` to set up SSH alias and sync `.env`.
2. **Sync scripts and configs** to the new pod:
   ```
   rsync -az --no-owner --no-group /Users/oscargilg/Dev/MATS/Preferences/scripts/qwen_replication/ runpod-qwen-e1a-extract:/workspace/repo/scripts/qwen_replication/
   rsync -az --no-owner --no-group /Users/oscargilg/Dev/MATS/Preferences/configs/measurement/active_learning/qwen35_ood_exp1b/ runpod-qwen-e1a-extract:/workspace/repo/configs/measurement/active_learning/qwen35_ood_exp1b/
   ```
3. **Upgrade transformers** — the default `4.57.6` doesn't know `Qwen3_5MoeConfig`. Run on pod:
   ```
   source /opt/venvs/research/bin/activate && uv pip install --upgrade transformers hf_transfer
   ```
4. **Run the extraction with babysit** so crashes auto-restart:
   ```
   /zombuul:babysit qwen-e1a-extract "Qwen-3.5-122B E1a activation extraction — 17 conditions, done when activations/qwen35_122b_ood/e1a/rainy_weather_pos_persona/activations_turn_boundary:-1.npz exists on pod. Runner: cd /workspace/repo && source /opt/venvs/research/bin/activate && export \$(grep -v '^#' .env | xargs) && export HF_HOME=/workspace/hf_cache HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 EXTRACT_BATCH_SIZE=16 && python scripts/qwen_replication/extract_e1a_activations.py"
   ```
   Notes on lessons from last pod:
   - `HF_HUB_ENABLE_HF_TRANSFER=1` + `HF_HUB_DISABLE_XET=1` is the combination that downloaded fastest without xet-storage reconstruction errors.
   - Env must be sourced from `.env` (not relying on `load_dotenv()` in the child process — it didn't propagate `HF_TOKEN` in my first run and anonymous downloads got rate-limited).
   - On last pod, bare nohup/setsid processes died silently every ~10–15 min — user flagged this as a known issue being fixed. If it recurs on the new pod, run inside `tmux` (already installed automatically on new pods now that zombuul 1.1.20 is in use).
5. **When extraction is done, sync activations back:**
   ```
   rsync -az --no-owner --no-group runpod-qwen-e1a-extract:/workspace/repo/activations/qwen35_122b_ood/e1a/ /Users/oscargilg/Dev/MATS/Preferences/activations/qwen35_122b_ood/e1a/
   ```
6. **Pause the pod** — `/zombuul:pause-runpod` — as soon as sync completes.
7. **Run local analysis + plot:**
   ```
   python scripts/qwen_replication/analyze_e1a.py
   python scripts/qwen_replication/plot_e1a_scatter.py
   ```
   Expected output:
   - `experiments/qwen_replication/e1a/e1a_per_task.json` — every per-task `(probe_delta, utility_delta, behavioral_delta, on_target)` record; commit this so we don't lose it again.
   - `paper/figures/plot_<DATE>_e1a_scatter_utility.png` and `..._behavioral.png` (and copies in `experiments/qwen_replication/e1a/assets/`).
8. **Update report + paper**:
   - Rewrite `experiments/qwen_replication/e1a/e1a_report.md` with the final 16-condition numbers and pointers to the new per-task cache.
   - In `paper/main.tex` §4.1.1 (around line 229–245, currently Gemma-only with `plot_022626_s4_scatter_simple.png`): swap in the two-panel figure. Update the one-sentence result to cite both numbers. Remove the `\todo{Qwen-3.5-122B replication (E1a) pending...}` marker at line 231.
   - Rebuild PDF: `cd paper && latexmk -pdf main.tex`.

## Key paths (quick reference)

| What | Where |
|---|---|
| Qwen configs (17) | `configs/measurement/active_learning/qwen35_ood_exp1b/` |
| Gemma configs (17, same prompts) | `configs/measurement/active_learning/ood_exp1b/` |
| Tasks (48 hidden_*) | `configs/ood/tasks/target_tasks.json` |
| 3 existing Qwen utility fits | `results/experiments/exp_20260422_124719/pre_task_active_learning/` |
| 13 new Qwen utility fits (running) | `results/experiments/qwen35_ood_e1a_final/pre_task_active_learning/` |
| Gemma utility fits | `results/experiments/ood_exp1b/pre_task_active_learning/` |
| Qwen probes (tb-1, tb-4) | `results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m{1,4}/probes/probe_ridge_L38.npy` |
| Gemma probe (prompt_last) | `results/probes/gemma3_10k_heldout_std_raw/probes/probe_ridge_L31.npy` |
| Measurement log | `experiments/qwen_replication/e1a/logs/measurements.log` |

## Nothing else was done to paper yet

No edits to `paper/main.tex` have been made in this session. The final step (8) is the only paper change required.
