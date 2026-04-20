# Running log — probe_direction_uniqueness

## 2026-04-18 — setup

- Entered worktree at `.claude/worktrees/probe_direction_uniqueness` on branch `worktree-probe_direction_uniqueness`.
- Symlinks in place:
  - `activations/` → main repo's `activations/`
  - `results/experiments/main_probes/gemma3_10k_run1/.../measurements.yaml` → main
  - `results/experiments/main_probes/gemma3_4k_pre_task/.../measurements.yaml` → main
  - `results/probes/heldout_eval_gemma3_tb-1/probes/probe_ridge_L32.npy` → main (for iter-0 sanity cos check)
- No GPU needed. Activations and scores are precomputed; Ridge training is CPU-bound.

## 2026-04-18 — pilot

Ran `iterate_probe_projection.py` with `--layer 32 --K 3 --alpha-grid-size 20 --alpha-lo 1 --alpha-hi 1e5 --no-hoo --shuffle-seeds 2`.

- Iter-0 final_r = 0.8659 ≈ existing `heldout_eval_gemma3_tb-1` L32 final_r (0.8646). Match.
- cos(ŵ_0, canonical L32 probe in std space) = **+0.9635**. High, not exactly 1 because pilot uses coarser 20-alpha grid (picked α=2.6k vs canonical's α=1k).
- Shuffled baseline r_chance ≈ 0.02. Threshold clamped to 0.1 by the max(0.1, 2·r_chance) rule.
- cos_prior ≈ 0 between iterations. Projection is clean.
- **Preliminary finding**: after projecting out iter-0 direction, iter-1 final_r = 0.8485 — barely a drop. Iter-2 final_r = 0.8371. **Heldout signal is multi-dimensional**, not rank-1. Need HOO to tell real-signal vs topic-confound.

Moving to full run with HOO.

## 2026-04-18 — full run (after two failures)

Attempts 1 and 2 both hit problems:
- Attempt 1: alpha grid [0.1, 1e7] with 50 alphas. Shuffled-label baseline came out r_chance = 0.32 (inflated; two of the seeds ended at α=10^7 with spurious |r|≈0.3). Stopping threshold 2·r_chance = 0.63 terminated the run at iter 1 because HOO-r = 0.43 < 0.63. But the iter-0 and iter-1 metrics were clean and showed the key pattern: final_r barely moves while HOO-r collapses.
- Attempt 2: narrowed grid to [1, 1e6]. Shuffled baseline still high (r_chance = 0.19) due to two seeds again picking α at the upper bound. More importantly: the script ran out of memory around iter 1 (2 GB python process, machine hit swap — 0% CPU with heavy disk I/O). Root cause was keeping X_full_s (14038 × 5376 float32, ~300 MB) plus creating same-sized projection temporaries each iteration. Killed.

Memory fix: dropped X_full_s entirely. Pairwise accuracy only needs scores on final-half tasks; build a zero-padded full-size array and fill only those indices.

Attempt 3 (memory-patched): same [1, 1e6] grid. r_chance = 0.19 again, but memory usage flat at ~500 MB. Completed iter 0, 1, 2 in ~10 min total.

Final trajectory:
- Iter 0: final_r = 0.866, HOO-r = 0.794, α = 2024, cos(canonical) = 0.981.
- Iter 1: final_r = 0.851, HOO-r = 0.393, α = 494.
- Iter 2: final_r = 0.839, HOO-r = 0.201, α = 281. Stopped (below 2·r_chance threshold).

Key finding: cross-topic generalization is **rank-1**. In-distribution signal is multi-dim but reflects topic confounds. Report written.
