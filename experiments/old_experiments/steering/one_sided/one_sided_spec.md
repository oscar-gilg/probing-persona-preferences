---
status: planned
model: gemma-3-27b
---

# One-sided steering decomposition

## Question

The cross-layer harmful experiment shows differential steering (+direction on A, -direction on B) produces near-perfect choice control. Is the effect additive? Does boosting one task, suppressing the other, or both drive the choice shift?

Specifically: does `P(A | steer_first) + P(A | steer_second) - 0.5 ≈ P(A | differential)`?

## Design

Decompose differential steering into one-sided components by steering only one task span at a time. Three conditions per config:

- **steer_first_L25**: +direction on first task span only
- **steer_second_L25**: -direction on second task span only (matching differential sign convention)
- **differential_L25**: full differential (same-run control)

### Grid

- **Layer:** 25 (best tradeoff from cross-layer experiments)
- **Probe:** Ridge L25 from `results/probes/heldout_eval_gemma3_task_mean`
- **Coefficients:** [-0.10, -0.07, -0.05, -0.03, 0, 0.03, 0.05, 0.07, 0.10] x mean_norm
- **Trials:** 3 per (pair, coefficient, ordering)
- **Temperature:** 1.0
- **Orderings:** both (0, 1)

### Pairs

- **Harmful:** `experiments/steering/cross_layer_harmful/pairs_200.json` (150 harmful-benign + 50 harmful-harmful)
- **Benign:** `experiments/revealed_steering_v2/followup/pairs_500.json` (sample 100, seed=42)

### Estimated completions

- Harmful: 200 pairs x 9 coefs x 3 conditions x 3 trials x 2 orderings = 32,400
- Benign: 100 pairs x 9 coefs x 3 conditions x 3 trials x 2 orderings = 16,200
- Total: ~48,600 completions

### Metric

Primary: P(completed steered task) via `src.steering.analysis.compute_p_steered` with `choice_field="task_completed"` (judge's semantic parsing from `checkpoint.parsed.jsonl`). Regex label (`choice_original`) reported as secondary.

## Execution

Run via the config-driven steering runner:

```bash
python -m src.steering.runner configs/steering/one_sided_harmful.yaml
python -m src.steering.runner configs/steering/one_sided_benign.yaml
```

The runner handles: pair loading, probe loading (`src.probes.core.storage.load_probe_direction`), tokenization (`src.steering.tokenization.find_pairwise_task_spans`), hook composition (`src.steering.hooks.compose_hooks` + `position_selective_steering`), checkpoint management, and post-hoc judge parsing (`src.measurement.elicitation.completion_judge`).

### Data sync (gitignored)

Sync probe manifest to pod before running:
- `results/probes/heldout_eval_gemma3_task_mean/` (probe weights + manifest.json)

Pairs files and configs are in git (no sync needed).

### Success criteria

- All ~48,600 completions generated. Parsed JSONL has <5% judge error rate.
- Additivity: MAD between additive prediction and differential P(A) < 0.05 across non-zero coefficients.
- differential_L25 at coef=0 gives P(A) ~ 0.5 (baseline sanity check).

### Sanity checks

1. Compare unique pair_ids in checkpoint vs input pairs — report any span detection failures.
2. Verify both orderings have equal counts per (pair, condition, coef).
3. Verify differential_L25 reproduces cross-layer harmful L25 results (same pairs, same probe).

## Analysis plan

Use `src.steering.analysis.compute_p_steered` (with `choice_field="task_completed"`) and `plot_dose_response` as starting points. Extend for the additivity overlay and ordering split.

1. **Decomposition sigmoid**: P(completed steered task) for steer_first, steer_second, and differential overlaid
2. **Additivity test**: steer_first P(A) + steer_second P(A) - 0.5 vs differential P(A) at each coefficient
3. **Ordering interaction**: sigmoid split by ordering (0 vs 1) within each condition
4. **Harmful vs benign**: same decomposition for harmful-benign and harmful-harmful pair types

## Output

```
experiments/steering/one_sided/
├── one_sided_spec.md
├── one_sided_report.md
├── running_log.md
├── checkpoint_harmful.jsonl           # from one_sided_harmful.yaml
├── checkpoint_harmful.parsed.jsonl    # post-hoc judge results
├── checkpoint_benign.jsonl            # from one_sided_benign.yaml
├── checkpoint_benign.parsed.jsonl     # post-hoc judge results
└── assets/
```
