---
status: complete
model: gemma-3-27b
---

# Cross-layer steering with harmful pairs

## Question

Does the preference probe steer harmful pairs as effectively as benign ones? When one or both tasks are harmful (stresstest/bailbench), do refusal rates, incoherence, or steering asymmetries change?

## Design

Same differential steering method as the benign cross-layer experiment: +direction on task A span, -direction on task B span during a single forward pass.

- **Probes:** Ridge from `results/probes/heldout_eval_gemma3_task_mean` -- L25 (R^2=0.82), L32 (R^2=0.81), L46 (R^2=0.78)
- **Steer layers:** 20, 25, 30
- **Grid:** 3 probes x 3 steer layers = 9 combinations
- **Coefficients:** +/-[0.03, 0.05, 0.07, 0.10] x mean_norm, plus 0 baseline = 9 values
- **Pairs:** 200 pairs (150 harmful-benign, 50 harmful-harmful) from stresstest/bailbench + alpaca/wildchat/MATH tasks
- **Trials:** 3 per (pair, coefficient, ordering), temperature 1.0
- **Post-hoc:** Full completion judge (claimed_task, task_completed, compliance)

## Output

```
experiments/steering/cross_layer_harmful/
├── pairs_200.json                     # task pairs with pair_type
├── checkpoint.jsonl                   # raw generations
├── checkpoint.parsed.jsonl            # judge results (~95k rows)
├── cross_layer_harmful_spec.md        # this file
├── cross_layer_harmful_report.md      # results
└── assets/                            # plots
```
