---
status: draft
model: gemma-3-27b
version: v2
---

# Cross-persona differential steering

Companion to `cross_persona_unilateral`. Same 6 canonical personas from `persona_sweep_final_six` (`aura`, `contrarian`, `mathematician`, `sadist`, `slacker`, `strategist`), same pairs, same multipliers — **differential** (contrastive) steering: `+` probe direction on the first-presented task span, `−` probe direction on the second span, in a single forward pass.

> **v2 supersedes v1.** v1 used each persona's own probe. v2 uses the *Assistant (default) probe* across all personas, matching the open-ended evil steering experiment (`scripts/cross_persona_open_ended_steering/`) and the original §3.4 default steering. v1 checkpoints retained at `checkpoints_v1_perprobe/` for reference.
>
> ⚠️ **Do not regenerate per-persona configs.** The committed YAMLs in `configs/steering/cross_persona_differential/` already reference `default_tb-5` for every persona. If you re-run `gen_configs.py`, it now hard-codes `PROBE_MANIFEST = "results/probes/persona_sweep_final_six/default_tb-5/"` — there is no `{persona}_tb-5/` template substitution anymore. Verify each yaml has `probe_manifest: results/probes/persona_sweep_final_six/default_tb-5/` before launching.

## Setup

- **Probe**: Assistant `ridge_L25` from `results/probes/persona_sweep_final_six/default_tb-5/` — **same single probe across all 6 persona configs.** This is the v2 defining change.
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
