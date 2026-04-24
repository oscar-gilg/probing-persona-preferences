# Orphan producers

Figures in `paper/main.tex` whose producer scripts need work before the claim system can audit them automatically.
See `scripts/paper/audit_claims.py` for the live per-claim classification (`ORPHAN` / `NAME-ORPHAN` / `LOGIC-STALE` / `DATA-STALE` / `FROZEN` / `MANUAL` / `SUPERSEDED`).

## Open

| Figure / claim                                     | Status           | Action                                                                                                                                                                                                                                             |
| -------------------------------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `plot_022426_stated_steering_positions.png` (main.tex:799) | Research call    | Producer deleted in `04526b5` — experiment used label-token parsing while probes trained on completion-based preferences. Either **rerun under completion-based template** (write a proper producer for the new results) or **drop the figure**.   |
| `plot_022426_stated_steering_formats.png` (main.tex:808)   | Research call    | Same decision as above.                                                                                                                                                                                                                            |

## Resolved

| Figure / claim                              | Fix                                                                                                                                                                                                                                                                                        | When       |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `plot_022626_topic_mean_utilities.png`      | Restored `scripts/plot_topic_utilities_lw.py` from commit `41c1e1a`; re-pointed data path to `main_probes/`; added `Topic mean utility` table claim (rows = topic, cols = {mean, se, n}); writes to both `docs/lw_post/assets/` and `paper/figures/`.                                          | 2026-04-24 |
| Gemma targeted-task induced-shift r         | The per-target `pooled_on_target` block exists in `e1a_per_task.json` (earlier audit mis-read it as null). Updated `scripts/qwen_replication/plot_e1a_scatter.py` to register `\gemmaInducedShiftPooledRTargetedTasks = 0.95` (n = 81); main.tex:239 swapped.                              | 2026-04-24 |
| Gemma refitted-utility shifted-prediction r | Claim script was pointed at stale n=40 run (→ 0.74). Re-pointed `scripts/paper/claims/compute_refitted_shift_r.py` at the authoritative n=48 `utility_fitting/analysis_results.json` (→ 0.63, matching `utility_fitting_report.md`). Paper prose switched to the macro.                     | 2026-04-24 |
| Gemma `uniform_hoo_acc` (L32) → `\crossModelProbeExtrasGemmaCrossTopicAcc` | Synced `pref_main/activations_turn_boundary:-5.npz` (3.2 GB) from the storage pod. Added `uniform_eval_run_dir` to `configs/probes/gemma3_10k_hoo_topic_tb-5.yaml` and `heldout_eval_gemma3_tb-5.yaml`. Re-ran `run_dir_probes` on both + `compute_pooled_hoo.py`. Producer switched from tb-1 to tb-5 (EOT selector); frozen `0.751` removed and replaced with the derived value `0.753`. Notable downstream change: `\gemmaProbeCrossTopicPooledR` went from 0.855 (tb-1) to 0.886 (tb-5).               | 2026-04-24 |
