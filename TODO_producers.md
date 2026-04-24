# Orphan producers — needing reimplementation or restoration

Figures in `paper/main.tex` whose producer scripts need work before the
claim system can audit them automatically.

| Figure | Status | Notes |
|---|---|---|
| `plot_022426_stated_steering_positions.png` | Superseded — manual freeze | Producer at `scripts/stated_steering/run_phase1.py` in git commit `34180bb` (deleted in `04526b5`). Experiment superseded by canonical completion-based template. Values frozen in `scripts/paper/claims/manual_claims.py`. Restore by redoing the phase-1 experiment under the current template, then replacing the manual claims with a proper producer. |
| `plot_022426_stated_steering_formats.png` | Superseded — manual freeze | Same story as above. Producer was `scripts/format_replication/plot_results.py`. |
| `plot_022626_topic_mean_utilities.png` | Deleted producer — recoverable | Producer at `scripts/plot_topic_utilities_lw.py` in git commit `41c1e1a` (deleted in `1af184b`). Inputs: `results/experiments/gemma3_10k_run1/.../thurstonian_80fa9dc8.csv` + `data/topics/topics.json`. Restore with `git show 41c1e1a:scripts/plot_topic_utilities_lw.py > scripts/plot_topic_utilities_lw.py`, adapt paths, add `ClaimSet` registration. |
| Gemma `uniform_hoo_acc` (L32) | Needs pipeline re-run | `results/probes/gemma3_10k_hoo_topic_tb-1_uniform/hoo_summary.json` predates the per-fold `uniform_hoo_acc` field. Value 0.751 is frozen in `plot_cross_model_bar.py`. To upgrade: re-run `src.probes.experiments.run_dir_probes` with the current code on this config, which will populate `mean_uniform_hoo_acc` in the summary. |
| Gemma targeted-task induced-shift r (0.95) | Data gap | `experiments/qwen_replication/e1a/e1a_per_task.json` has `pooled_on_target.pearson_r=null` for Gemma. Rerun the Gemma arm of the E1a pipeline with the targeted-pool filter to populate it; the producer at `scripts/qwen_replication/plot_e1a_scatter.py` will then register the claim automatically. Currently inline in main.tex §4.1. |
| Gemma refitted-utility shifted-prediction r (0.63) | Needs compute script | §4.1 prose-only claim. Requires re-running the default-persona probe against per-condition refitted utilities and pooling. No current producer computes this; add a small `scripts/paper/claims/compute_refitted_shift_r.py`. |
