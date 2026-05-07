---
status: ready
model: gemma-3-27b
parent: experiments/random_direction_l23_quick/random_direction_l23_quick_spec.md
---

# Random-direction L23: 4 more seeds (multi-seed null)

## Question

Single seed (42) showed a 0.09 contrastive swing across c ∈ ±0.05 — within the projection-noise expectation (cos sim std ≈ 1/√5376 ≈ 0.014) but not flat. Run 4 more independent random seeds and average. The averaged curve should be ~flat at 0.5; the seed-spread bounds the projection noise floor.

## Setup

Identical to the parent run except direction. For each seed s ∈ {0, 1, 2, 3}:
- Generate `np.random.default_rng(s).standard_normal(5376)`, L2-normalise
- Save as `results/probes/layer_sweep/eot/probes/probe_random_L23_seed{s}.npy` (shape `(5377,)`, last element 0 = intercept)
- Append manifest entry to `results/probes/layer_sweep/eot/manifest.json` with `id: random_L23_seed{s}`, `layer: 23`
- Run a contrastive config mirroring `experiments/random_direction_l23_quick/`:
  - Layer 23, `mean_norm[23]` = 29381.541015625
  - Coefs `[-0.05, -0.03, 0.0, +0.03, +0.05]`
  - First 30 pairs from `experiments/layer_sweep/harm_breakdown/steering_pairs_150.json`
  - `n_trials=3`, both orderings, `temperature=1.0`, `seed=42` (run-level seed)
  - Template `src/measurement/elicitation/prompt_templates/data/completion_preference.yaml`
  - No system prompt (default Assistant)

Output parsed jsonl per seed at:
`experiments/random_direction_l23_quick/multi_seed/checkpoints/random_contrastive_seed{s}.parsed.jsonl`

## Code pointers

- Probe generation + manifest: adapt `experiments/random_direction_l23_quick/make_random_probe.py`.
- Config generation: adapt `experiments/random_direction_l23_quick/make_config.py` (or similar — copy whatever the parent run produced).
- Runner: `scripts/isolated_steering/run_steering.py` (unchanged).

## Steps

1. For each `s ∈ {0, 1, 2, 3}`: generate probe `.npy`, append manifest entry, write config YAML, run.
2. Confirm all 4 parsed jsonls land with 900 rows each (30 × 5 × 2 × 3).

## Total cost

4 seeds × 900 generations = 3600 gens. ~40-60 min on H100.

## Out of scope

- No averaging or plotting on the pod — done locally after pulling results.
- No new pair sets, layers, coefs.
