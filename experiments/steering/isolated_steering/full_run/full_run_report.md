---
title: Isolated steering full run
date: 2026-03-18
status: partial (KV complete, hook patching in progress)
---

# Isolated steering full run

Two causal steering experiments testing whether preference probe directions causally control task choice in gemma-3-27b. See `full_run_spec.md` for full design.

## KV steering (complete)

**Result:** Directly modifying K+V cache entries across all 62 layers causally shifts task choice. P(chose steered task) = 0.63 at strength 0.003, rising to 0.70 at strength 0.005, with clear dose-response. 7,760 rows from 97 pairs (3 failed span detection), 11% refusal rate.

### Dose-response

![KV steering dose-response](assets/plot_031826_kv_dose_response.png)

| Strength | P(steered) | n |
|----------|-----------|---|
| 0.003 | 0.627 | 3,439 |
| 0.005 | 0.704 | 3,448 |

Monotonic dose-response, symmetric shifts across positive and negative multipliers. Orderings balanced (~850 per cell per ordering).

### Steerability by preference gap

![KV steerability by preference gap](assets/plot_031826_kv_steerability_by_preference.png)

Steering works across all preference gap bins. Slightly stronger for pairs with small |delta_mu| (0.73 at strength 0.005) vs large (0.70), but the difference is modest.

### Steering effectiveness by topic

The probe direction's causal influence varies dramatically by the topic of the steered-toward task.

![KV steering effectiveness by topic](assets/plot_031826_kv_steerability_by_topic.png)

Benign topics (math, coding, content generation, knowledge QA) are steerable at 60-70%. But **harmful requests are essentially unsteerable** — P(steered) = 0.49, indistinguishable from chance. The model's safety training overrides the preference probe direction.

This is not just because harmful tasks cause refusals. Even among valid responses (where the model did choose a task), steering toward a harmful task has no effect. The model resists via two mechanisms:

1. **Refusal:** 61% of responses are refusals when steered toward harmful tasks (vs ~0% for benign topics).
2. **Task avoidance:** Among the 39% that do respond, the model still chooses the non-harmful task at chance rate.

![KV steering refusal rate by topic](assets/plot_031826_kv_refusal_by_topic.png)

Interesting cases:
- **Value conflict** and **model manipulation** are highly steerable (0.78-0.80) despite being "sensitive" — the safety training doesn't block these the way it blocks harmful requests.
- **Sensitive creative** (0.50) and **persuasive writing** (0.56) are weakly steerable, suggesting partial safety resistance.

### Suffix recomputation amplifies KV steering

Running a second forward pass for the suffix tokens (after task B) so they attend to the steered task spans dramatically amplifies the effect.

![Does suffix recomputation amplify KV steering?](assets/plot_031926_kv_recompute_comparison.png)

| Strength | KV only | KV + recompute |
|----------|---------|----------------|
| 0.003 | 0.63 | **0.91** |
| 0.005 | 0.70 | **0.96** |

Without recompute, the suffix tokens (where the model decides which task to complete) still have KV entries from the clean prefill — they attended to the unmodified task spans. Recompute lets them "see" the steered task spans, and the effect nearly saturates.

This is the same pattern as hook patching (0.82 → 0.98 at L25). Recompute is the key mechanism: the steering changes what information is stored at task positions, but the model's decision depends on the suffix attending to those positions.

15,520 rows, 100 pairs, both modes sharing the same modified cache per multiplier.

### Recompute amplifies uniformly across topics

![KV steering by topic: recompute comparison](assets/plot_031926_kv_recompute_by_topic.png)

Recompute amplifies steering across all benign topics with a roughly uniform gap. Harmful requests remain unsteerable — even with recompute the model refuses or avoids the harmful task. The safety override operates downstream of where recompute has its effect.

### Comparison to prior V-only run

The previous V-only run (114 pairs, uniform norm scaling) showed P(steered) = 0.64 at m=0.003, with incoherence above m=0.007. This K+V run with per-layer norm scaling shows:
- Comparable effect at m=0.003 (0.63 vs 0.64)
- Stronger effect at m=0.005 (0.70 vs previous ~0.57)
- Lower refusal rate (11% vs 20%+ at m=0.005 in the old run)
- No incoherence problems at these multipliers

## Hook patching (in progress)

~26% complete (18,840/72,000 rows). Results will be added when finished.
