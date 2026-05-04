---
status: draft
model: gemma-3-27b
version: v2
---

# Cross-persona unilateral steering

Redo of paper section 3.4 with unilateral steering (replaces the differential setup) across the 6 canonical personas from `persona_sweep_final_six`: `aura`, `contrarian`, `mathematician`, `sadist`, `slacker`, `strategist`.

> **v2 supersedes v1.** v1 used each persona's own probe. v2 uses the *Assistant (default) probe* across all personas, matching the open-ended evil steering experiment (`scripts/cross_persona_open_ended_steering/`) and the original §3.4 default steering. v1 checkpoints retained at `checkpoints_v1_perprobe/` for reference. Baselines (`*_baseline.parsed.jsonl`) are at coef=0 and probe-independent — kept in `checkpoints/`, no rerun needed.
>
> ⚠️ **Do not regenerate per-persona configs.** The committed YAMLs in `configs/steering/cross_persona_unilateral/` already reference `default_tb-5` for every persona. `gen_configs.py` now hard-codes `PROBE_MANIFEST = "results/probes/persona_sweep_final_six/default_tb-5/"` — there is no `{persona}_tb-5/` template substitution anymore. Verify each yaml has `probe_manifest: results/probes/persona_sweep_final_six/default_tb-5/` before launching.

## Setup

- **Probe**: Assistant `ridge_L25` from `results/probes/persona_sweep_final_six/default_tb-5/` (tb-5 == eot) — **same single probe across all 6 persona configs.** This is the v2 defining change.
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

- No random-direction control (sign flip is the within-probe null).
- No layer sweep — single layer L25 (already characterised in `experiments/layer_sweep/` for the default persona).

## Phase 2: differential follow-up (run sequentially after Phase 1)

After the unilateral phase completes and is parsed, immediately run the **differential** companion experiment. Spec: `experiments/cross_persona_differential/cross_persona_differential_spec.md`.

Steps:
1. Configs already committed at `configs/steering/cross_persona_differential/*.yaml` — verify each has `probe_manifest: results/probes/persona_sweep_final_six/default_tb-5/`. Do **not** regenerate; the v2 hard-coded `PROBE_MANIFEST` is already correct.
2. Run all 6 configs via `src/steering/runner.py`. ~14.4k additional generations on the same Gemma-3-27B instance (no model reload).
3. Parse with the LLM judge → `experiments/cross_persona_differential/checkpoints/{persona}.parsed.jsonl`.
4. Both phases share `experiments/cross_persona_unilateral/steering_pairs.json` and `experiments/cross_persona_unilateral/checkpoints/{persona}_baseline.parsed.jsonl` (coef=0 baselines, probe-independent, no rerun).

After both phases complete, regenerate the figure with `python scripts/cross_persona_differential/plot_combined.py` and update `paper/figures/main/plot_*_cross_persona_perprobe_steering.png`.
