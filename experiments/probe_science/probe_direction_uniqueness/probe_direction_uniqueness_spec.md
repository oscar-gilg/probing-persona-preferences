# Probe Direction Uniqueness

## Question

Is preference encoded in a single linear direction in Gemma-3-27B's activation space, or in a multi-dimensional subspace? If we project out the probe direction we found, can we train a second probe that also predicts preferences?

## Motivation

Our best probe (Ridge, heldout R ≈ 0.86) encodes preference as a single direction in the residual stream. The evaluative-representation hypothesis is agnostic about dimensionality — "encodes value" is compatible with both a rank-1 and a rank-k representation. Knowing which matters for:

- **Steering:** if rank-k, single-direction steering only moves the model along one axis of the representation and may under-steer.
- **Generalization:** a rank-1 story is stronger evidence that we've identified *the* evaluative direction rather than *a* direction within a broader subspace.
- **Concept erasure:** whether one projection suffices to remove preference information (INLP-style question, Ravfogel et al. 2020).

## Method

Iteratively train Ridge probes and project out each direction (INLP-style), using our heldout-α-selected Ridge probes as the direction source rather than linear classifiers.

### Iteration protocol

One-time setup at iteration 0:

- Load activations + train/eval scores. Fit `StandardScaler` on train activations **once** and hold fixed across all iterations. This keeps every ŵ_k in the same standardized basis so Gram-Schmidt and orthogonality checks are meaningful.
- Compute and save the eval sweep/final split once (seed 42). Held fixed across iterations — never resampled.

Then at iteration k = 0, 1, 2, ..., K−1 (target K = 10):

1. Sweep α (`np.logspace(-1, 7, 50)` — extended upper bound vs standard pipeline since later iterations may need heavier shrinkage) training Ridge on (I − W_{k−1} W_{k−1}^T) · X_train_scaled (W_{-1} = empty), picking α* maximizing Pearson r on the sweep half of eval (also projected).
2. Refit Ridge at α* on full (projected) train set. Normalize coef_ → ŵ_k (unit vector in standardized space).
3. Orthonormalize ŵ_k against W_{k−1} via Gram-Schmidt (should be ≈orthogonal already; GS for safety against numerical drift). Append to W_k.
4. Compute metrics at this iteration (see below).
5. Continue until heldout r (sweep half) drops below the shuffled-baseline threshold (see stopping rule).

**Projection direction choice.** Ridge at α* (α* selected for heldout Pearson r on sweep half), not OLS. OLS is train-MSE-optimal but overfits noise (train R² ≈ 0.93, heldout performance much worse per `scripts/probes/compare_ols_ridge.py`). Ridge at heldout-optimal α is the rank-1 linear predictor with best generalization — the predictive signal we want to eliminate. Caveat: this justifies not using OLS; it does not prove that ridge-at-α* is *the* maximally-harmful projection direction. A projection that removes the most heldout-r is not generally the same as the direction that maximizes heldout-r (they coincide only for orthogonal designs). Good enough as a first-pass definition.

### Layers

Primary: **L32** in `heldout_eval_gemma3_tb-1` (trained on `activations_turn_boundary:-1.npz`, heldout R = 0.865). Chosen over L31 because L31 only exists for `activations_prompt_last.npz`, which has no matching HOO run — switching activations would require re-running HOO. L32 is essentially interchangeable with L31 (R difference < 0.001).

Secondary (if L32 takes < 1 hr): L39, L46 from the same run.

### Data

Matches `heldout_eval_gemma3_tb-1` exactly so iter-0 metrics are directly comparable:

- **Train run:** `results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0`
- **Eval run:** `results/experiments/main_probes/gemma3_4k_pre_task/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0`
- **Activations:** `activations/gemma_3_27b_turn_boundary_sweep/activations_turn_boundary:-1.npz`
- **eval_split_seed:** 42.

All inputs exist locally — no pod sync needed.

HOO: topic grouping via `src/probes/residualization.py::build_task_groups(task_ids, "topic", topics_json="data/topics/topics.json")`. Use the same groups as `results/probes/gemma3_10k_hoo_topic_tb-1` produced. At each iteration, also report HOO-mean r across topic folds.

### Measurements per iteration

| Metric | What it tells us |
|---|---|
| Heldout Pearson r (final half) | Does the remaining subspace still predict preference in-distribution? |
| Heldout pairwise accuracy (on final-half BT pairs from eval run) | Same, on the pair metric. Using active-learning eval pairs, not uniform eval — fine because comparison is across iterations within the same experiment, not across models. |
| **HOO Pearson r (mean across topic folds)** — PRIMARY for effective rank | Does it generalize across topics? Treat this as the truth metric for "is there real residual preference signal." |
| α* trajectory | Does optimal regularization change as we strip directions? Flag if α* hits grid upper bound. |
| cos(ŵ_k, ŵ_j) for j < k | Sanity — should be ≈0 after GS. |
| Trace of residual cov (||X_train_scaled · (I − W_k W_k^T)||_F² / N) | Verifies projection removed variance as expected. |
| Shuffled-label baseline r (from iter 0) | Fixed stopping threshold. See below. |
| At iter 0: cos(ŵ_0, canonical L32 probe from `heldout_eval_gemma3_tb-1`) | Sanity — should be ≈1.0. Detects pipeline drift. |

### Shuffled-label baseline

Computed **once** at iter 0 (not per iteration):

- Shuffle y_train, fit the full sweep-and-select pipeline in standardized space, evaluate Pearson r on final-half eval.
- Repeat 5 seeds, record the 95th percentile of |r| → `r_chance`.
- Fixed stopping threshold: `2 × r_chance` (or 0.1, whichever is larger).

### Output

- `trajectory.json` — scalar metrics per iteration, shuffled baseline, iter-0 sanity cos, sweep/final split indices.
- `directions.npz` — keys `w_0`...`w_{K-1}` plus `W_stack` (d × K).
- `scaler.npz` — keys `mean`, `scale` (single fixed scaler).

## Success criteria

- **Stopping rule:** stop when HOO-r drops below `max(0.1, 2 × r_chance)`. (Switching primary metric from heldout-r to HOO-r: if a direction boosts heldout-r but not HOO-r, it's a topic/dataset confound and we should treat it as signal-exhausted.)
- **Rank-1 case:** iter 1 HOO-r ≤ threshold. Report as "single direction."
- **Multi-dim case:** HOO-r stays above threshold for several iterations. Report the iteration at which it crosses threshold as the **effective rank** of the preference subspace.
- **Confound case:** heldout-r stays high but HOO-r drops at iter 1. Report as "rank-1 real signal + topic confound."

## What to build

Standalone script `scripts/probes/iterate_probe_projection.py` that:

1. Loads train/eval scores + one layer's activations via existing helpers: `src/probes/data_loading.py::load_thurstonian_scores`, `load_eval_data`; `src/probes/core/activations.py::load_activations`; `src/probes/experiments/hoo_ridge.py::build_ridge_xy`.
2. Implements the iteration loop above. Reuses the ridge-sweep logic in `src/probes/experiments/run_dir_probes.py::train_ridge_heldout` but refactored into a thinner inner function that accepts pre-standardized, pre-projected `(X_train, y_train, X_sweep, y_sweep, X_final, y_final, alphas)` and skips its internal `StandardScaler` step (since this script manages the scaler once, externally).
3. HOO computation: wrap the inner loop of `src/probes/experiments/run_dir_probes.py::run_hoo` (which is in `run_dir_probes.py`, line 495, not in `hoo_ridge.py`) into a variant that takes pre-projected activations + topic groups. If that refactor is too invasive, call the existing `run_hoo` only at iterations {0, 1, 2, 5, K-1}.
4. Saves `trajectory.json` + `directions.npz` + `scaler.npz` to `experiments/probe_science/probe_direction_uniqueness/output/<layer>/`.

Plots saved to `experiments/probe_science/probe_direction_uniqueness/assets/`:

- `plot_{mmddYY}_heldout_r_vs_iter.png` — heldout r (sweep + final) and HOO r vs iteration, with shuffled-baseline band. One panel per layer.
- `plot_{mmddYY}_alpha_vs_iter.png` — α* trajectory. Annotate upper-bound saturation.
- `plot_{mmddYY}_direction_cosines.png` — heatmap of cos(ŵ_i, ŵ_j).
- `plot_{mmddYY}_residual_variance_vs_iter.png` — residual cov trace, sanity.

## Fallbacks

- If HOO is too slow per iteration, compute only at iterations {0, 1, 2, 5, K-1} and linearly interpolate for the plot.
- If training 500 ridge probes per layer (50 alphas × 10 iterations) is too slow, reduce alpha grid to 20 points or cap K at 5.
- If L32 iter 1 already collapses to chance on HOO-r, the experiment is done after 2 iterations — no further layers needed.
- If refactoring `train_ridge_heldout` / `run_hoo` into "accepts pre-projected activations" variants is invasive, duplicate the inner loops inline in the script with a clear "DRY violation — intentional, keeps changes scoped to this experiment" comment.
