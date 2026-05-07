# Random-direction L23 — multi-seed null (2 more seeds)

## Status

In progress. Skeleton — populated by finalize step.

## Question

Single seed (42) showed a 0.09 swing across c ∈ ±0.05. Run 2 more independent
seeds (0, 1), average, and report seed-spread as the projection-noise floor.

## Setup

Identical to parent `random_direction_l23_quick/` except the steering direction
is a fresh random unit vector per seed s ∈ {0, 1}. See `multi_seed_spec.md`.

## Result

(filled in after both runs land + analysis)

## Files

- Configs: `configs/steering/random_direction_l23_quick_multi_seed/random_contrastive_seed{0,1}.yaml`
- Probes: `results/probes/layer_sweep/eot/probes/probe_random_L23_seed{0,1}.npy`
- Checkpoints: `experiments/random_direction_l23_quick/multi_seed/checkpoints/`
