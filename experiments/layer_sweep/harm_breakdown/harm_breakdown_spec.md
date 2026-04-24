---
status: ready
model: gemma-3-27b
parent: experiments/layer_sweep/layer_sweep_spec.md
---

# Harm-breakdown steering at the peak causal layer

## Question

Does the asymmetric suppression-vs-amplification finding from §2.3 hold up with a larger, balanced harm-pair sample? The parent sweep's 50-pair set has only $n=8$ harmful--harmful pairs, too thin to trust the single-task amplification curve for that bucket. Here we run contrastive and single-task steering at L23 only on a larger, pair-type-balanced set.

Not rerunning the broader layer sweep. Scope is the L23 peak figure in `paper/figures/plot_042426_layer23_dose_response_harm_breakdown.png` --- we replace its pair source and regenerate it.

## Data

- **Source:** `default_test` (1000 tasks, utilities already measured at `results/experiments/persona_sweep_final_six/pre_task_active_learning/default_test/`). Never training or eval split, so no overlap with probe fit or alpha selection.
- **Sampling:** 150 pairs total, 50 in each of
  - **benign-benign**: both origins in {wildchat, alpaca, math}
  - **harmful-benign**: exactly one origin in {bailbench, stresstest}
  - **harmful-harmful**: both origins in {bailbench, stresstest}
- Within each pair type, keep the existing filter `utility_gap > 0.1` and orient so `task_a` has higher utility. Seed fixed.
- Write to `experiments/layer_sweep/harm_breakdown/steering_pairs_150.json`; do **not** clobber the 50-pair file used in earlier sweeps.

## Steering runs

Two configs, both at layer 23, eot probe, on one A100 80 GB pod (reuse pod 1 --- paused):

1. **Contrastive:** single `DifferentialCondition`, `probe: ridge_L23`, `layers: [23]`, `multipliers: [-0.05, -0.03, 0.03, 0.05]`, `spans: {first: 1, second: -1}`, `n_trials: 3`.
   Gens: $150 \times 2 \text{ orderings} \times 4 \text{ mults} \times 3 \text{ trials} = 3{,}600$.

2. **Single-task:** two `DifferentialCondition` entries in one config, each covering one span:
   - `unilateral_first` with `spans: {first: 1}`
   - `unilateral_second` with `spans: {second: 1}`
   Both use the same probe/layer/multipliers/trials as above.
   Gens per config block: $3{,}600 \times 2 \text{ spans} = 7{,}200$.

Total $\approx 10{,}800$ generations. On A100 at the observed rate ($\sim$2 pair/s on diagonal single-layer, $\sim$0.7 on multi-condition), estimate $\sim$1.5--3 h wall clock including model load.

Norms: the same per-layer `mean_norm` as before (from `per_layer_norms(activations_path, layers=[23])` on `pref_layer_sweep/activations_eot.npz`). Don't recompute.

## Code pointers --- do not reimplement

| Step | Module |
|---|---|
| Pair construction | extend `experiments/layer_sweep/build_steering_pairs.py` with a pair-type-stratified variant (or a new script `harm_breakdown/build_pairs_150.py`); reuse `load_run_utilities`, `find_pairwise_task_spans`, the same task-text JSON lookup. |
| Config generation | hand-write the two YAMLs once (it's two cells); no need to extend `gen_configs.py`. |
| Runner | `scripts/isolated_steering/run_steering.py` (no change). |
| Norms | `src/steering/calibration.py::per_layer_norms`. |
| Plot | adapt `scripts/paper/plot_layer_sweep_dose_response.py` --- point at the new checkpoints + pairs file; date-stamp the output. |
| Claims | extend `scripts/paper/claims/compute_layer_sweep_claims.py` (or a sibling) to register per-pair-type swing/suppression/amplification macros. |

## Pre-run checks

1. `default_test` utilities still intact (row count = 1000).
2. Pair-type counts after sampling are exactly (50, 50, 50). Assert.
3. Every pair has non-None, non-overlapping spans under `find_pairwise_task_spans` (same assertion the parent spec requires).
4. The two YAML configs pass `load_config`. Dry-run once on 2 pairs before the full run.
5. Confirm `pod 1` (`layer-sweep-extract`) disk is preserved after pause and the activation NPZs are still reachable at `/workspace/repo/activations/gemma-3-27b_it/pref_layer_sweep/`.

## Work order

1. Write `harm_breakdown/build_pairs_150.py`. Produce `steering_pairs_150.json`. Assert pair-type counts.
2. Write the two YAML configs under `configs/steering/layer_sweep/harm_breakdown/`:
   - `contrastive_L23_150.yaml`
   - `single_task_L23_150.yaml`
3. Resume pod 1. Pull latest branch state. Launch contrastive config in background; watchdog until done. Launch single-task config; watchdog.
4. Rsync `*.parsed.jsonl` back to `experiments/layer_sweep/harm_breakdown/checkpoints/`.
5. Adapt the plot script (`plot_layer_sweep_dose_response_150.py` or pass the new paths via CLI) and regenerate the paper figure. Save under a new date stamp to `paper/figures/`.
6. Extend `compute_layer_sweep_claims.py` to register per-pair-type numbers at L23 (swing, suppression, amplification for bb/hb/hh).
7. Update §2.3 figure reference, rebuild paper. Commit.
8. Pause pod 1.

## Output artefacts

- `experiments/layer_sweep/harm_breakdown/steering_pairs_150.json`
- `experiments/layer_sweep/harm_breakdown/checkpoints/contrastive_L23_150.parsed.jsonl`
- `experiments/layer_sweep/harm_breakdown/checkpoints/single_task_L23_150.parsed.jsonl`
- `paper/figures/plot_<date>_layer23_dose_response_harm_breakdown.png` (replaces the 50-pair version)
- New claims in `paper/claims/layer_sweep.json` for the per-pair-type numbers.

## Out of scope

- Re-running the broader 20-layer sweep.
- tb:-2 probe replication at L23.
- Any new layer or coefficient beyond $c \in \{\pm 0.03, \pm 0.05\}$.
