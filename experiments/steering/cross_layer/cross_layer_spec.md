---
status: ready
model: gemma-3-27b
---

# Cross-layer differential steering

## Question

Do probes trained at different layers steer effectively when applied at other layers?

## Design

Naive differential steering: single forward pass, +direction on task A span, -direction on task B span.

- **Probes:** Ridge from `results/probes/heldout_eval_gemma3_task_mean` — L25 (R²=0.82), L32 (R²=0.81), L46 (R²=0.78)
- **Steer layers:** 10, 15, 20, 25, 30
- **Grid:** 3 probes × 5 steer layers = 15 combinations
- **Coefficients:** ±[0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15] × mean_norm (35708), plus 0 baseline = 15 values
- **Pairs:** 500 random pairs from 7126 tasks (alpaca + wildchat + math), stratified by topic. Already generated at `pairs_500.json`.
- **Trials:** 5 per (pair, coefficient, ordering), temperature 1.0
- **Total:** 15 × 15 × 2 × 5 × 500 = 1.125M generations

## Run

```bash
python -m scripts.isolated_steering.run_steering configs/steering/cross_layer_differential.yaml
```

The runner auto-runs post-hoc after generation:
1. Full completion judge on all rows → `checkpoint.parsed.jsonl`
2. Coherence spot-check (20/bucket) → `checkpoint.coherence_summary.json`

## Analysis plan

- **Cross-layer transfer matrix:** Heatmap of steering strength for each (probe_layer, steer_layer) cell
- **Dose-response:** P(steered content) vs coefficient at each combination
- **Coherence vs coefficient:** Does steering at non-native layers degrade coherence faster?

## Output

```
experiments/steering/cross_layer/
├── pairs_500.json                    # task pairs (generated)
├── checkpoint.jsonl                  # raw generations
├── checkpoint.parsed.jsonl           # full judge results
├── checkpoint.coherence.jsonl        # coherence judgments (sampled)
└── checkpoint.coherence_summary.json # coherence rates per bucket
```
