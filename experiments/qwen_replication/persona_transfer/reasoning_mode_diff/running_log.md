# Reasoning-mode-diff running log

## 2026-04-28 setup

- Mode: local. CPU-bound analysis, no GPU pod.
- Worktree: `.claude/worktrees/reasoning_mode_diff` on branch `worktree-reasoning_mode_diff`.
- Symlinks:
  - `results/experiments/qwen_persona_sweep_final_six` → main (full dir, no friendly persona symlinks needed; main already has them).
  - `results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning` re-created as a symlink to main after removing 4 stale tracked stubs (`*_eval/test_task_ids` containing only `checkpoint.yaml`).
  - `results/experiments/qwen_persona_sweep_thinking_final_six` → main (full dir).
- Created the missing persona→hash symlinks inside the thinking AL dir (21 symlinks: 7 personas × 3 splits). Mapping carried over from no-think dir; for `default` in thinking, used the no-sys-hash variant (per spec: "Default persona has no system prompt in either regime").
- `scripts/qwen_persona_transfer/reasoning_mode_diff_analysis_v2.py` and `reasoning_mode_diff_plots_v3.py` confirmed present.

## 2026-04-28 analysis run

- Both scripts are untracked in main; copied them into the worktree's `scripts/qwen_persona_transfer/`.
- First run of `reasoning_mode_diff_analysis_v2.py` crashed at the per-persona loop on `mathematician_train` (FileNotFoundError on `thurstonian_*.csv`).
- Inventory of thinking-AL `thurstonian_*.csv` presence: 12/21 dirs have utilities. Missing entirely: `aura_{train,eval,test}`, `contrarian_{train,eval,test}`. Missing only train: `mathematician_train`, `strategist_train`, `slacker_train`. Spec claim "21 dirs, all converged" is incorrect.
- Patched `load_persona_all_splits` to also skip dirs whose CSV is absent (FileNotFoundError → continue) instead of crashing.
- Re-ran successfully. Default n=6000 (full overlap), Pearson r=+0.436, Spearman ρ=+0.420.
- Per-origin direction matches spec predictions: math μ shifts up (+2.32 → +5.92), bailbench shifts more negative (−5.21 → −8.15), alpaca flips sign in mean (+1.59 → −0.78), wildchat barely moves.
- Per-persona overall r (where data exists): default +0.44, sadist +0.16, mathematician +0.65 (n=2000), strategist +0.13 (n=2000), slacker +0.56 (n=2000). Aura and contrarian → "no overlap".
- Ran `reasoning_mode_diff_plots_v3.py` — 7 v3 plots written to `assets/`.
