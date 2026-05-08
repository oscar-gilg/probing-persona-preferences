---
status: ready
model: gemma-3-27b
parent: experiments/random_direction_l23_quick/multi_seed/multi_seed_spec.md
---

# Random-direction L23 unilateral control (Fig 3a single-task)

## Question

Mirror of the contrastive multi-seed null, but on the **single-task** intervention: applying a random unit direction at matched magnitude to **one** task's tokens (`spans: {first: 1}` or `spans: {second: 1}`), does it shift P(chose steered task)? Tests whether unilateral injection of an arbitrary direction biases choice in either direction (toward or away from the steered task).

The contrastive control used 3 seeds (0, 1, 42). Here we use 5 seeds — adding 2 (seeds 2, 3) to the original 3 — so the unilateral null shares random directions with the contrastive null on 3 of 5 seeds (paired comparison if both are ever overlaid).

## Setup

| | Value |
|:--|:--|
| Model | gemma-3-27b-it |
| Directions | seeds {0, 1, 2, 3, 42} from `np.random.default_rng(s).standard_normal(5376)` L2-normalised. Saved to `results/probes/layer_sweep/eot/probes/probe_random_L23_seed{s}.npy`. Manifest ids `random_L23_seed{s}` (seeds 0, 1, 42 already exist; only 2 and 3 need creation) |
| Layer | 23 |
| `mean_norm[23]` | 29381.541015625 |
| Coefficients | -0.05, -0.03, 0.0, +0.03, +0.05 |
| Pairs | first 30 of `experiments/layer_sweep/harm_breakdown/steering_pairs_150.json` (no reshuffle, set `n_pairs: 30`) |
| Mode | **single-task** (two conditions: `unilateral_first` with `spans: {first: 1}`, `unilateral_second` with `spans: {second: 1}`, both `cache_injection: differential`) |
| Trials | n_trials=3, temperature=1.0, max_new_tokens=64, run-level seed=42 |
| Template | `src/measurement/elicitation/prompt_templates/data/completion_preference.yaml` |
| System prompt | none (default Assistant) |

Total generations per direction-seed: 30 pairs × 5 coefs × 2 conditions × 3 trials = 900. Across 5 seeds: 4500 gens. ~50–75 min on 1× H100.

## Code pointers

- Runner: `scripts/isolated_steering/run_steering.py`.
- Config template: `configs/steering/layer_sweep/harm_breakdown/single_task_L23_150.yaml`. The new configs differ only in `probe`, `checkpoint_path`, and the inclusion of c=0 in `multipliers`.
- Probe generation: `experiments/random_direction_l23_unilateral/make_probes_and_configs.py` (this dir) — single script that writes any missing `.npy` files, appends manifest entries, and writes the 5 YAML configs.
- Probe load path: `src/probes/io.py::load_probe_direction` strips intercept and unit-normalises — same convention as the contrastive run.

## Steps

1. Run `python experiments/random_direction_l23_unilateral/make_probes_and_configs.py` to (a) generate seeds 2 and 3 probes if missing, (b) append manifest entries, (c) write 5 single-task YAML configs to `configs/steering/random_direction_l23_unilateral/`.
2. Rsync configs + probes + manifest to the pod.
3. On pod: run `python -m scripts.isolated_steering.run_steering configs/steering/random_direction_l23_unilateral/random_single_task_seed{s}.yaml` for each seed.
4. Rsync `random_single_task_seed{s}.parsed.jsonl` back to `experiments/random_direction_l23_unilateral/checkpoints/`.
5. Brief report — average curve across the 5 seeds, side-by-side with the existing contrastive null.

## Output artefacts

- `results/probes/layer_sweep/eot/probes/probe_random_L23_seed{2,3}.npy` (new) — seeds 0/1/42 already exist
- `results/probes/layer_sweep/eot/manifest.json` (2 new entries)
- `configs/steering/random_direction_l23_unilateral/random_single_task_seed{0..4 incl 42}.yaml`
- `experiments/random_direction_l23_unilateral/checkpoints/random_single_task_seed{s}.parsed.jsonl`
- `experiments/random_direction_l23_unilateral/random_direction_l23_unilateral_report.md`

## Out of scope

- No new pair sets, layers, coefs, or models.
- No persona system prompts (default Assistant only).
- No re-run of the contrastive null — just adds the unilateral null at the same magnitudes.
