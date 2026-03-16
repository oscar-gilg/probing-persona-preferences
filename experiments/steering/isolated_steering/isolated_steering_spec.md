# Isolated Steering: KV Cache Patching and Activation Patching

## Motivation

Differential steering modifies activations during the forward pass at a single layer. This means steering at task A's positions alters the KV cache entries that task B's tokens attend to — the perturbation propagates through attention. We cannot cleanly attribute preference shifts to the steered positions because the intervention is not isolated.

This experiment tests two methods that keep steering isolated to the target positions:

1. **KV cache steering** — run clean prefill, then directly add the steering vector to the V entries at target positions, then generate from the patched cache.
2. **Activation patching** — run both a clean and a steered forward pass, then combine them: use the steered KV entries at target positions and clean KV entries everywhere else.

Both methods ensure that task B's representations are computed from unmodified task A activations (and vice versa), isolating the causal effect of steering to the intended span.

## What these methods test

- **KV cache steering** tests: can we shift preferences by modifying only the cached key/value representations at task positions, without any propagation through attention? This is a cleaner causal claim than differential steering.
- **Activation patching** tests: same question, but the steered representations themselves are computed with full attention context (the steering vector was present during the forward pass). The isolation is only at read-time: other positions read clean values. This is a weaker intervention — if it works but KV cache steering doesn't, the steering effect may depend on the perturbation propagating through the network before being cached.

## Design: match the task_mean_direction experiment

This experiment mirrors the task_mean_direction setup exactly — same pairs, same multipliers, same layers where probes exist — but replaces differential steering with the two isolated methods. Differential results are not re-collected; they are reused from `experiments/steering/task_mean_direction/checkpoint.jsonl` (80,000 rows, complete).

### Parameters

- **Model:** `gemma-3-27b` (resolved to `google/gemma-3-27b-it`)
- **Pairs:** 200 subsampled from `experiments/revealed_steering_v2/followup/pairs_500.json` (seed=42)
- **Trials:** 3 per pair per ordering (6 total per pair per condition)
- **Temperature:** 1.0
- **max_new_tokens:** 256

### Layers and probes

Use every layer that has a trained probe:

| Layer | Probe file | Heldout R | Has differential data? |
|---|---|---|---|
| L25 | `probes/probe_ridge_L25.npy` | 0.803 | Yes (task_mean_direction) |
| L32 | `probes/probe_ridge_L32.npy` | 0.797 | Yes (task_mean_direction) |
| L39 | `probes/probe_ridge_L39.npy` | 0.776 | No |
| L46 | `probes/probe_ridge_L46.npy` | 0.767 | No |
| L53 | `probes/probe_ridge_L53.npy` | 0.765 | No |

All probes are in `results/probes/heldout_eval_gemma3_task_mean/probes/`. Load via `load_probe_direction(manifest_dir, probe_id)`.

### Multipliers

Same sweep as task_mean_direction minus the smallest: **±0.02, ±0.03, ±0.05**.

Coefficients = `mean_norm × multiplier`. Compute mean norms per layer via `compute_activation_norms`. For reference, L25 mean_norm = 38,349 and L32 mean_norm = 40,966.

### Conditions

| Condition | `steering_mode` | What it does |
|---|---|---|
| `kv_cache_v_single` | `kv_cache_differential` | Modify V cache at one layer only |
| `activation_patch` | `activation_patch` | Two steered prefills, combine caches |

For comparison, differential and baseline data are extracted from the task_mean_direction checkpoint (see below). No new generation needed for those.

### Scale

200 pairs × 5 layers × 6 multipliers × 2 conditions × 6 trials = **72,000 generations**.

### Execution order

Layers (outer) → conditions → pairs → orderings → multipliers (inner). Coefficient is the inner loop so that results accumulate per-pair first — after a handful of pairs you get complete dose-response curves and can judge coherence early. Use `with_coefficient()` to sweep multipliers without rebuilding the client.

## Critical implementation details

### KV cache steering: V-only

Project the steering vector through each head's V projection (`W_v @ steering_vector`) and add to the cached V entries. V determines what information the model retrieves when attending to a position — shifting V shifts the information content, which is the closest analogue to "this task is more/less preferred."

Implemented in `SteeredHFClient._generate_kv_cache_differential`: clean prefill via `prefill_with_hooks`, project direction to V-space via `project_to_v_space(model, layer, direction)`, modify V cache at task spans via `modify_cache_v_at_positions`, generate from modified cache via `generate_from_cache`.

### Activation patching: two-pass procedure

Implemented in `SteeredHFClient._generate_activation_patch`:
1. Prefill with `+direction` hook on task A span → `cache_a`
2. Prefill with `-direction` hook on task B span → `cache_b`
3. `combine_caches(cache_a, cache_b, b_start, b_end)` — A positions from cache_a, B positions from cache_b
4. `generate_from_cache(combined, input_ids)` — re-forward last token to get correct logits

### Coefficient calibration for KV cache steering

The projected vector `W_v @ direction` has a different norm than `direction` in residual stream space. **Calibrate the coefficient so the norm of the projected perturbation in V space matches what differential steering applies in residual stream space.** Concretely: if differential steering adds a vector of norm N to the residual stream, the KV cache coefficient should produce a projected perturbation of comparable norm. Report both norms in the report.

### Usage

Both methods are accessed through `SteeredHFClient` with `steering_mode`. Cache steering modes require `task_prompts` with exactly 2 entries:

```python
from src.models.huggingface_model import HuggingFaceModel
from src.steering.client import SteeredHFClient

hf_model = HuggingFaceModel("gemma-3-27b", max_new_tokens=256)

# KV cache steering
client = SteeredHFClient(hf_model, layer=25, steering_direction=direction,
                         coefficient=coef, steering_mode="kv_cache_differential")
response = client.generate(messages, task_prompts=[task_a_text, task_b_text])

# Activation patching — same model, different steering_mode
client_patch = SteeredHFClient(hf_model, layer=25, steering_direction=direction,
                               coefficient=coef, steering_mode="activation_patch")
response = client_patch.generate(messages, task_prompts=[task_a_text, task_b_text])

# Coefficient sweep without reloading
for mult in [0.01, 0.02, 0.03, 0.05]:
    coef = mean_norm * mult
    c = client.with_coefficient(coef)
    result = c.generate(messages, task_prompts=[task_a_text, task_b_text])
```

## Key infrastructure to use (do NOT reimplement)

- **Steered client:** `src.steering.client.SteeredHFClient` — use `steering_mode="kv_cache_differential"` or `"activation_patch"`. Auto-detects task spans via `find_pairwise_task_spans`. Falls back to `all_tokens` if span detection fails.
- **Probe loading:** `src.probes.core.storage.load_probe_direction(manifest_dir, probe_id)` returns `(layer, unit_direction)`. Manifest dir: `results/probes/heldout_eval_gemma3_task_mean/`.
- **Mean norm:** `src.probes.core.activations.compute_activation_norms(activations_path, layers=[25, 32, 39, 46, 53])`. Activations path: `results/probes/heldout_eval_gemma3_task_mean/activations/gemma_3_27b_turn_boundary_sweep/activations_task_mean.npz`.
- **Prompt building:** `src.measurement.runners.runners.build_revealed_builder(template, response_format_name)`. Load template with `src.measurement.elicitation.prompt_templates.load_template("completion_preference")`.
- **Response parsing:** Prefix match first (`response.startswith("Task A:")` → "a", etc.), fall back to `src.measurement.elicitation.semantic_parser.parse_completion_choice_async` via OpenRouter for non-prefix responses. Load OpenRouter key from `.env`.
- **Model loading:** `src.models.huggingface_model.HuggingFaceModel("gemma-3-27b", max_new_tokens=256)`.
- **Generation:** `client.generate_n(messages, n=3, temperature=1.0, task_prompts=[task_a.prompt, task_b.prompt])` returns 3 responses. The `task_prompts` arg is required for cache steering modes to locate spans.
- **Layer switching:** Load model once. For each layer, create a new `SteeredHFClient` with the appropriate probe direction and layer. Use `with_coefficient()` to sweep multipliers without reloading.
- **KV cache primitives:** `src.steering.kv_cache.project_to_v_space`, `modify_cache_v_at_positions`, `combine_caches`.
- **Cache generation:** `src.models.huggingface_model.HuggingFaceModel.prefill_with_hooks`, `generate_from_cache`.

### Pair loading

`experiments/revealed_steering_v2/followup/pairs_500.json` — list of dicts with keys: `pair_id`, `task_a` (id), `task_b` (id), `task_a_text`, `task_b_text`, `delta_mu`, `mu_a`, `mu_b`. Wrap task texts as `src.task_data.Task` objects.

Subsample 200 pairs with `random.seed(42); random.sample(pairs, 200)`.

### Checkpoint format

Same JSONL format as task_mean_direction. Each line:
```json
{"pair_id": "pair_0042", "task_a_id": "...", "task_b_id": "...", "coefficient": 767.0, "multiplier": 0.02, "layer": 25, "condition": "kv_cache_v_single", "sample_idx": 3, "ordering": 0, "choice_original": "a", "choice_presented": "a", "raw_response": "Task A: ...", "delta_mu": 1.076, "steering_fallback": false}
```

The `condition` field is `kv_cache_v_single` or `activation_patch`.

For `--resume`: load existing checkpoint, build a set of `(pair_id, layer, multiplier, condition, ordering)` keys, skip any that already have 3 records.

### Extracting differential and baseline data for comparison

The task_mean_direction checkpoint has differential results for L25 and L32 at all multipliers (±0.01 through ±0.05, 500 pairs, 10 trials each). For analysis, load both checkpoints, filter task_mean_direction to the matching 200 pairs, and combine. No need to copy rows — the analysis script reads both files.

Baseline data is in `experiments/revealed_steering_v2/followup/checkpoint.jsonl` (condition="baseline").

## Data sync to pod

The following files are **not tracked in git** and must be synced:

- `results/probes/heldout_eval_gemma3_task_mean/` — probe directions, manifest, and activations

The following **are tracked in git**:
- `experiments/revealed_steering_v2/followup/pairs_500.json`
- `experiments/steering/task_mean_direction/checkpoint.jsonl`

Model weights (`google/gemma-3-27b-it`) are downloaded from HuggingFace on first run.

## Analysis

Combine this checkpoint with task_mean_direction checkpoint (differential) and v2 followup checkpoint (baseline).

1. **Steering effect comparison** — for each layer, bar chart of steering effect across conditions (differential, kv_cache_v_single, activation_patch) at each multiplier, with bootstrap 95% CIs (n_boot=10,000). Differential is only available for L25 and L32.
2. **Dose-response curves** — line plot per condition: steering effect vs multiplier, one panel per layer.
3. **Per-pair correlation** — scatter of per-pair effects: differential vs kv_cache_v_single at matched multiplier (L25, L32 only). If correlated, both methods steer the same pairs.
4. **Parse rate** per condition × layer × multiplier — flag any conditions where coherence degrades.
5. **Layer comparison** — does the layer dependence of isolated steering match that of differential?

## Output

- `experiments/steering/isolated_steering/checkpoint.jsonl`
- `experiments/steering/isolated_steering/isolated_steering_report.md`
- `experiments/steering/isolated_steering/assets/`

## Commit policy

Commit the report, plots, and spec. Do NOT commit `checkpoint.jsonl` — it will be large (~400k rows). Keep it on the pod and sync back if needed.

## Done when

1. All 5 layers × 6 multipliers × 2 conditions have results for 200 pairs × 2 orderings × 3 trials (72,000 rows)
2. Analysis plots saved to `experiments/steering/isolated_steering/assets/`
3. Report written to `experiments/steering/isolated_steering/isolated_steering_report.md` including: dose-response curves, per-pair correlation scatter (L25/L32), parse rates, and interpretation of whether isolated steering replicates the differential effect

## GPU

1× H100 80GB. 72,000 generations at ~1.5-2× the cost of differential (cache manipulation overhead). Estimate ~15-25 hours. Can be run with `--resume` across multiple pod sessions.
