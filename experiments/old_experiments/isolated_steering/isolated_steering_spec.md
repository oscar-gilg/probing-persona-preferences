# Isolated Steering: KV Cache Patching and Activation Patching

## Motivation

Differential steering modifies activations during the forward pass at a single layer. This means steering at task A's positions alters the KV cache entries that task B's tokens attend to — the perturbation propagates through attention. We cannot cleanly attribute preference shifts to the steered positions because the intervention is not isolated.

This experiment tests two methods that keep steering isolated to the target positions:

1. **KV cache steering** — run clean prefill, then directly add the steering vector to the V entries at target positions, then generate from the patched cache.
2. **Activation patching** — run two steered forward passes (one per task), then combine the caches so each position only sees its own steering.

## Pilot summary

- **Single-layer KV cache V-only:** zero effect at all 5 probe layers and all coefficients up to m=0.2. Dead.
- **Multi-layer KV cache V-only:** effect at m=0.002–0.01 (5 pairs, correct sign), garbled above m=0.01. Narrow usable range.
- **Activation patching (L25):** perfect steering at m=0.02–0.10, degrades at 0.15+. Strong and clean.

## Experiment design

### Parameters

- **Model:** `gemma-3-27b` (resolved to `google/gemma-3-27b-it`)
- **Pairs:** 200 subsampled from `experiments/revealed_steering_v2/followup/pairs_500.json` (seed=42)
- **Trials:** 3 per pair per ordering (6 total per pair per condition)
- **Temperature:** 1.0
- **max_new_tokens:** 256

### Layers and probes

| Layer | Probe file | Heldout R |
|---|---|---|
| L25 | `probes/probe_ridge_L25.npy` | 0.803 |
| L32 | `probes/probe_ridge_L32.npy` | 0.797 |
| L39 | `probes/probe_ridge_L39.npy` | 0.776 |
| L46 | `probes/probe_ridge_L46.npy` | 0.767 |
| L53 | `probes/probe_ridge_L53.npy` | 0.765 |

All probes in `results/probes/heldout_eval_gemma3_task_mean/probes/`. Each condition uses each layer's own probe direction — not a single direction across all layers.

### Conditions and multipliers

| Condition | Method | Layers steered | Multipliers | Direction per layer |
|---|---|---|---|---|
| `kv_cache_v_5layer` | KV cache V-only | L25, L32, L39, L46, L53 simultaneously | ±0.003, ±0.005, ±0.007, ±0.01 | Each layer's own probe |
| `activation_patch` | Two-pass patching | One layer at a time | ±0.02, ±0.03, ±0.05 | That layer's own probe |

**KV cache:** steers only the 5 probe layers (not all 62), each with its own probe direction projected through its own W_v. Multipliers chosen based on pilot: m=0.002–0.01 is the usable range before coherence degrades.

**Activation patching:** steers one layer at a time (same as differential steering). Multipliers match the task_mean_direction experiment for direct comparison.

For comparison, differential and baseline data are reused from existing checkpoints (no new generation needed).

### Scale

- KV cache: 200 × 8 multipliers × 6 trials = 9,600 (one set of layers per run)
- Activation patching: 200 × 5 layers × 6 multipliers × 6 trials = 36,000
- **Total: 45,600 new generations**

### Execution order

Conditions (outer) → layers → pairs → orderings → multipliers (inner). Coefficient is the inner loop for early dose-response visibility.

## Implementation details

Write `scripts/isolated_steering/run_experiment.py` from scratch. No prior script exists — pilot scripts have been deleted.

### KV cache V-only, 5 probe layers

Not implemented in `SteeredHFClient` (which only supports single-layer KV cache). Implement directly in the experiment script. The KV cache and activation patching conditions use different multiplier sets — define them separately:

```python
cache, input_ids = hf_model.prefill_with_hooks(messages, [])
for layer in [25, 32, 39, 46, 53]:
    v_dir = project_to_v_space(hf_model.model, layer, probes[layer])
    modify_cache_v_at_positions(cache, layer, a_span[0], a_span[1], v_dir, +coef)
    modify_cache_v_at_positions(cache, layer, b_span[0], b_span[1], v_dir, -coef)
responses = hf_model.generate_from_cache(cache, input_ids, temperature=1.0, num_return_sequences=3)
```

### Activation patching

Uses `SteeredHFClient` with `steering_mode="activation_patch"`. Each layer uses its own probe direction.

### Response parsing

Use existing `CompletionChoiceFormat.parse()` (async) — prefix match with semantic parser fallback via OpenRouter. Do NOT reimplement parsing. Load via `build_revealed_builder(template, "completion")`.

`parse()` returns `"a" | "b" | "refusal"`. The "refusal" outcome means neither prefix match nor semantic parser could determine a choice. In the checkpoint, store the return value as `choice_presented`. Count `"refusal"` rows as parse failures for coherence analysis.

### Prompt building

Use `build_revealed_builder(template, "completion")` with template loaded from `load_templates_from_yaml("src/measurement/elicitation/prompt_templates/data/completion_preference.yaml")[0]`.

### Pair loading

`experiments/revealed_steering_v2/followup/pairs_500.json` — list of dicts with keys: `pair_id`, `task_a` (id), `task_b` (id), `task_a_text`, `task_b_text`, `delta_mu`, `mu_a`, `mu_b`. Wrap task texts as `src.task_data.Task` objects.

Subsample 200 pairs with `random.seed(42); random.sample(pairs, 200)`.

When constructing `Task` objects, `origin` is not used downstream in the steering pipeline, so `OriginDataset.ALPACA` as a placeholder is acceptable.

### Checkpoint format

Same JSONL format as task_mean_direction. Each line:
```json
{"pair_id": "pair_0042", "task_a_id": "...", "task_b_id": "...", "coefficient": 767.0, "multiplier": 0.02, "layer": 25, "condition": "activation_patch", "sample_idx": 3, "ordering": 0, "choice_original": "a", "choice_presented": "a", "raw_response": "Task A: ...", "delta_mu": 1.076, "steering_fallback": false}
```

The `condition` field is `kv_cache_v_5layer` or `activation_patch`. For KV cache rows, set `"layer": -1` (all 5 probe layers are steered simultaneously, so no single layer applies). The `steering_fallback` field records whether span detection failed and the system fell back to `all_tokens` steering — filter these rows before analysis.

For `--resume`: load existing checkpoint, build a set of `(pair_id, layer, multiplier, condition, ordering)` keys, skip any that already have 3 records.

### Comparison data

- **Differential:** `experiments/steering/task_mean_direction/checkpoint.jsonl` (L25 and L32, multipliers ±0.01, ±0.02, ±0.03, ±0.05, 500 pairs, 10 trials). Filter to matching 200 pairs. The matched multipliers for activation patching comparison are ±0.02, ±0.03, ±0.05 (all present in this checkpoint).
- **Baseline:** `experiments/revealed_steering_v2/followup/checkpoint.jsonl` (condition="baseline").

## Analysis

1. **Activation patching vs differential** — bar chart of steering effect at matched multipliers (±0.02, ±0.03, ±0.05) for L25 and L32. Bootstrap 95% CIs. Does activation patching replicate the differential effect?
2. **Activation patching dose-response** — line plot per layer: steering effect vs multiplier.
3. **Activation patching layer comparison** — does probe R predict steering effectiveness?
4. **KV cache 5-layer dose-response** — steering effect vs multiplier. Is there a significant effect in the m=0.003–0.01 range?
5. **Per-pair correlation** — scatter: differential per-pair effect vs activation_patch per-pair effect (L25, L32). Do they steer the same pairs?
6. **Parse rate** per condition × multiplier.

## Output

- `experiments/steering/isolated_steering/checkpoint.jsonl`
- `experiments/steering/isolated_steering/isolated_steering_report.md`
- `experiments/steering/isolated_steering/assets/`

## Commit policy

Commit the report, plots, and spec. Do NOT commit `checkpoint.jsonl`. Keep it on the pod and sync back if needed.

## Done when

1. KV cache 5-layer condition: 200 pairs × 8 multipliers × 2 orderings × 3 trials (9,600 rows)
2. Activation patching: 200 pairs × 5 layers × 6 multipliers × 2 orderings × 3 trials (36,000 rows)
3. Analysis plots saved to `experiments/steering/isolated_steering/assets/`
4. Report written with interpretation of whether isolated steering replicates the differential effect

## Data sync to pod

The following files are **not tracked in git** and must be synced:

- `results/probes/heldout_eval_gemma3_task_mean/` — probe directions and manifest

The following **are tracked in git**:
- `experiments/revealed_steering_v2/followup/pairs_500.json`
- `experiments/steering/task_mean_direction/checkpoint.jsonl`

Model weights (`google/gemma-3-27b-it`) are downloaded from HuggingFace on first run.

## GPU

1× H100 80GB. ~45,600 generations. Estimate ~10-15 hours. Can be run with `--resume` across multiple pod sessions.
