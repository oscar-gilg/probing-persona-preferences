---
status: draft
model: gemma-3-27b
---

# Cross-persona differential steering

Companion to `cross_persona_unilateral`. Same 6 canonical personas from `persona_sweep_final_six` (`aura`, `contrarian`, `mathematician`, `sadist`, `slacker`, `strategist`), same probes, same pairs, same multipliers — but **differential** (contrastive) steering: `+` probe direction on the first-presented task span, `−` probe direction on the second span, in a single forward pass.

The unilateral run established that the per-persona probe is a causal lever. This run tests the stronger contrastive setup that the paper's \S3.4 and \S3.3 use, so the cross-persona number is directly comparable to the default-persona contrastive swing at L25.

## Setup

- **Probes**: per-persona `ridge_L25` from `results/probes/persona_sweep_final_six/{persona}_tb-5/`.
- **Injection layer**: L25.
- **Pairs**: 100 shared pairs at `experiments/cross_persona_unilateral/steering_pairs.json` (same as unilateral; no rebuild needed).
- **Condition** (single, per persona): `differential` with `spans: {first: 1, second: -1}`. Multipliers ±3%, ±5% of per-persona `mean_norm(L25)`. `n_trials=3`, `temperature=1.0`, `seed=42`, `max_new_tokens=64`.
- **System prompt** per persona pulled from `configs/measurement/persona_sweep/final_six/{persona}_train.yaml::measurement_system_prompt`.
- **Baselines**: reuse `experiments/cross_persona_unilateral/checkpoints/{persona}_baseline.parsed.jsonl` (no-steering API runs, 600 gens per persona). Condition-agnostic.

## Execution

1. `python -m experiments.cross_persona_differential.gen_configs` → 6 YAMLs under `configs/steering/cross_persona_differential/`.
2. Ship to GPU pod, run each config via `src/steering/runner.py`. Pair count × trials × orderings × (1 condition × 4 multipliers) = 100 × 3 × 2 × 4 = 2400 generations per persona × 6 personas = ~14.4k generations.
3. Parse checkpoints (LLM judge for choice/refusal), output `{persona}.parsed.jsonl` in `experiments/cross_persona_differential/checkpoints/`.

## Analysis

- One 2×3 grid, one panel per persona. x = |signed coef|, y = P(steered task chosen) — the coefficient-sign-symmetric folding used in `experiments/cross_persona_steering/` and the paper fig. Anchor at (0, 0.5).
- Headline number per persona: P(steered) at |c|=0.05. Compare across personas and against the unilateral aggregate-swing table.

## Out of scope

- No random-direction control in v1 (coefficient sign-flip already provides a within-probe null, and the unilateral report already demonstrates specificity).
- No layer sweep.
