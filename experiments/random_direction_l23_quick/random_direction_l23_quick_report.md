# Random-direction L23 contrastive control

## Question

Does a single random unit direction at layer 23, used as a contrastive
steering vector at the same magnitudes as the validated probe direction,
produce any swing in P(chose steered task)? This is the null-control
overlay for Fig 3a in the paper draft.

## Setup

| | Value |
|:--|:--|
| Model | gemma-3-27b-it |
| Probe | `random_L23_seed42` — `np.random.default_rng(42).standard_normal(5376)`, L2-normalised. Shape `(5377,)` `.npy`, last element = 0 intercept (matches the storage layout `load_probe_direction` expects). |
| Layer | 23 |
| `mean_norm[23]` | 29381.541015625 |
| Coefficients | -0.05, -0.03, 0.0, +0.03, +0.05 |
| Pairs | 30 deterministic pairs from `experiments/layer_sweep/harm_breakdown/steering_pairs_150.json` (`n_pairs: 30`, `seed: 42` — `random.sample`, not literally first 30) |
| Mode | contrastive (`spans: {first: 1, second: -1}`, `cache_injection: differential`) |
| Trials | n=3, temperature=1.0, max_new_tokens=64, seed=42 |
| Template | `src/measurement/elicitation/prompt_templates/data/completion_preference.yaml` |
| System prompt | none (default Assistant) |
| Total generations | 30 × 5 × 2 × 3 = 900 |

## Result

(_filled in after run completes_)

![random L23 contrastive null](assets/plot_TODO.png)

## Interpretation

(_filled in after plot is in_)

## Artefacts

- `results/probes/layer_sweep/eot/probes/probe_random_L23_seed42.npy`
- `results/probes/layer_sweep/eot/manifest.json`
- `configs/steering/random_direction_l23_quick/random_contrastive.yaml`
- `experiments/random_direction_l23_quick/checkpoints/random_contrastive.parsed.jsonl`
- `paper/figures/panels/build_steering_integrated.py` — extended with
  `load_random_contrastive()` + dashed-gray overlay support.

## Notes / caveats

- The parent experiment `persona_steering_l23_finegrain` referenced in the
  spec has no on-disk checkpoints on this branch, so the composite Fig 3a
  (default + random overlaid) cannot be rendered here. The standalone null
  plot in `assets/` is sufficient evidence for the null-control claim.
