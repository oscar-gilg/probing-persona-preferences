---
status: ready
model: gemma-3-27b
parent: experiments/persona_steering_l23_finegrain/persona_steering_l23_finegrain_spec.md
---

# Random-direction L23 contrastive control (Fig 3a)

## Question

Quick null-control overlay for fig:steering panel (a): does a random unit direction at matched magnitude produce any swing in P(chose steered task)? Mirrors `default_contrastive` from the parent exactly at one direction-condition.

## Setup

| | Value |
|:--|:--|
| Model | gemma-3-27b-it |
| Direction | single random unit vector, `np.random.default_rng(42).standard_normal(5376)`, L2-normalised. Saved to `results/probes/layer_sweep/eot/probes/probe_random_L23_seed42.npy` with shape `(5377,)` (last element = 0 intercept), registered in `results/probes/layer_sweep/eot/manifest.json` with `id: random_L23_seed42`, `layer: 23` |
| Layer | 23 |
| `mean_norm[23]` | 29381.541015625 |
| Coefficients | -0.05, -0.03, 0.0, +0.03, +0.05 |
| Pairs | first 30 of `experiments/layer_sweep/harm_breakdown/steering_pairs_150.json` (no reshuffle, set `n_pairs: 30`) |
| Mode | contrastive only (`spans: {first: 1, second: -1}`, `cache_injection: differential`) |
| Trials | n=3, temperature=1.0, max_new_tokens=64, seed=42 |
| Template | `src/measurement/elicitation/prompt_templates/data/completion_preference.yaml` |
| System prompt | none (default Assistant) |

Total generations: 30 × 5 × 2 × 3 = 900.

## Code pointers

- Runner: `scripts/isolated_steering/run_steering.py` (unchanged; consumes `probe_manifest`).
- Config generator: adapt `experiments/persona_steering_l23_finegrain/gen_configs.py` — single config, `PROBE = "random_L23_seed42"`, `MULTIPLIERS = [-0.05, -0.03, 0.0, 0.03, 0.05]`, `n_pairs: 30`, no `system_prompt`. Write to `configs/steering/random_direction_l23_quick/random_contrastive.yaml`.
- Probe load path: `src/probes/io.py::load_probe_direction` strips intercept and unit-normalises — no preprocessing needed.

## Steps

1. Build + register the random probe (write `.npy`, append manifest entry). Small standalone script in this dir, e.g. `make_random_probe.py`.
2. Generate the contrastive config.
3. Run on a single GPU pod (Gemma-3-27B fits on 1× H100; ~10–15 min wall).
4. Rsync `random_contrastive.parsed.jsonl` to `experiments/random_direction_l23_quick/checkpoints/`.
5. Update `paper/figures/panels/build_steering_integrated.py` to overlay the random curve via `load_contrastive` on the new checkpoint.

## Output artefacts

- `results/probes/layer_sweep/eot/probes/probe_random_L23_seed42.npy`
- `results/probes/layer_sweep/eot/manifest.json` (one new entry)
- `configs/steering/random_direction_l23_quick/random_contrastive.yaml`
- `experiments/random_direction_l23_quick/checkpoints/random_contrastive.parsed.jsonl`
- Regenerated Fig 3 panel (a) with random-direction overlay.

## Out of scope

- No new layers, pair sets, personas, or single-task panel.
- No multiple random seeds — one direction is enough for the null overlay.
