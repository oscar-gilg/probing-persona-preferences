---
status: draft
model: gemma-3-27b
---

# Cross-persona unilateral steering

Redo of paper section 3.4 with unilateral steering (replaces the differential setup) across the 6 canonical personas from `persona_sweep_final_six`: `aura`, `contrarian`, `mathematician`, `sadist`, `slacker`, `strategist`.

## Setup

- **Probes**: per-persona `ridge_L25` from `results/probes/persona_sweep_final_six/{persona}_tb-5/` (tb-5 == eot).
- **Injection layer**: L25. Closest available layer to L23 (layer_sweep peak) and matches original section 3.4.
- **Pairs**: 100 shared pairs sampled from `default_test` (utility_gap > 0.1, stratified origin×origin). Both orderings run → baseline P(pick a span) ≈ 0.5 modulo the ~14pt position bias from layer_sweep.
- **Conditions** (per persona config): `unilateral_first` (`spans: {first: 1}`) and `unilateral_second` (`spans: {second: 1}`). Multipliers ±3%, ±5% of per-persona mean_norm(L25). `n_trials=3`, `temperature=1.0`, `seed=42`, `max_new_tokens=64`.
- **System prompt** per persona pulled from `configs/measurement/persona_sweep/final_six/{persona}_train.yaml::measurement_system_prompt`.

## Execution

1. `python -m experiments.cross_persona_unilateral.build_pairs` → `steering_pairs.json` (needs Gemma-3 tokenizer, CPU, seconds).
2. `python -m experiments.cross_persona_unilateral.gen_configs` (already run) → 6 YAMLs under `configs/steering/cross_persona_unilateral/`.
3. Ship to GPU pod, run each config via `src/steering/runner.py`. Pair count × trials × orderings × (2 conditions × 4 multipliers) = 100 × 3 × 2 × 8 = 4800 generations per persona × 6 personas = ~29k generations.
4. Parse checkpoints (LLM judge for choice/refusal), output `{persona}.parsed.jsonl`.

## Analysis

- Adapt `experiments/layer_sweep/analyze_steering.py::_plot_unilateral_dose_response` — group by persona instead of layer.
- One panel per persona: two lines (first-span, second-span), x = signed coef, y = P(pick that span's task). Overlay coef=0 baseline dots from the parsed data; compare the swing magnitude across personas.

## Out of scope

- No random-direction control in v1 (sign flip is the within-probe null).
- No layer sweep (already done for default in layer_sweep).
- No differential re-run.
