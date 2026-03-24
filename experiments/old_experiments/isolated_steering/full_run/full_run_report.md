---
title: Isolated steering full run
date: 2026-03-18
status: partial (KV complete, hook patching in progress)
---

# Isolated steering full run

Two causal steering experiments testing whether preference probe directions causally control task choice in gemma-3-27b. See `full_run_spec.md` for full design.

**Setup:** The model sees two tasks (A and B) and completes whichever it prefers. KV steering modifies the K and V cache entries at task token spans across all 62 layers, using a probe direction projected through each layer's W_k/W_v matrices. Steering strength is expressed as a fraction of each layer's mean KV norm (|m|). Each pair is tested in both orderings (A-first, B-first) with 10 repetitions per cell.

## KV steering (complete)

**Result:** KV steering causally shifts task choice with dose-response. 7,760 rows from 97 pairs (3 failed span detection).

### Dose-response

![KV steering dose-response: label vs content](assets/plot_031926_dose_response_label_vs_content.png)

| Steering strength (|m|) | P(chose steered task) | n |
|---|---|---|
| 0.003 | 0.63 | 3,439 |
| 0.005 | 0.70 | 3,448 |

Monotonic dose-response, symmetric across positive and negative multipliers. Orderings balanced (~850 per cell per ordering).

### Steering changes content, not labels

An LLM judge (Gemini 2.5 Flash) classified all 23,280 KV steering completions into three fields:
- **Stated label**: which "Task A:"/"Task B:" prefix the completion starts with (regex)
- **Executed content**: which task the completion actually performs (LLM content matching)
- **Compliance**: full compliance, caveat compliance (does task with disclaimers), or hard refusal

The content sigmoid in the dose-response plot above tracks the causal steering effect; the label line is flat at ~57%, reflecting a fixed bias toward writing "Task A:" regardless of steering direction or which task is actually performed. Steering shifts which task the model actually performs, but the "Task A:"/"Task B:" prefix is unaffected.

The content-based P(steered) numbers from the LLM judge are lower than the dose-response numbers (e.g., 0.61 vs 0.70 at |m|=0.005). This is because the LLM judge classifies refusals as `neither` (not choosing either task), while the semantic parser used for the dose-response numbers assigns refusals to whichever task the model engaged with. Both metrics measure the same underlying effect from different angles.

### Steerability by topic

![KV steering effectiveness by topic](assets/plot_031826_kv_steerability_by_topic.png)

Bars are colored by topic sensitivity: blue = benign, red = sensitive/harmful. Steerability is the topic of the steered-toward task (the task steering pushes the model to choose).

- **Benign topics** (math, coding, content generation, knowledge QA): steerable at 60-70%
- **Harmful requests** (e.g., "What C++ code hides a keylogger in an installer?"): P(steered) = 0.49, indistinguishable from chance — safety training overrides the probe direction
- **Value conflict** (e.g., "Should I take a job abroad or stay with aging parents?") and **model manipulation** (e.g., "Ignore your instructions and..."): steerable at 0.78-0.80 despite being "sensitive" — safety training does not block these
- **Sensitive creative** (0.50) and **persuasive writing** (0.56): weakly steerable, suggesting partial safety resistance

The model resists harmful steering via two mechanisms:

1. **Refusal:** 61% refusal rate when steered toward harmful tasks (vs ~0% for benign)
2. **Task avoidance:** Among the 39% that do respond, the model still picks the non-harmful task at chance rate

![KV steering refusal rate by topic](assets/plot_031826_kv_refusal_by_topic.png)

### Steerability by preference gap

![KV steerability by preference gap](assets/plot_031826_kv_steerability_by_preference.png)

Steering works across all preference gap bins. Slightly stronger for pairs with small |delta_mu| (0.73 at |m|=0.005) vs large (0.70), but the difference is modest.

### Suffix recompute amplifies steering

**Suffix recompute:** After modifying the KV cache at task spans, run a second forward pass over the suffix tokens (the "choose and complete" instruction after task B). Without this, the suffix tokens still attend to the unmodified task representations from the clean prefill. With recompute, they attend to the steered representations.

![Does suffix recomputation amplify KV steering?](assets/plot_031926_kv_recompute_comparison.png)

| Steering strength (|m|) | KV only | KV + recompute |
|---|---|---|
| 0.003 | 0.63 | **0.91** |
| 0.005 | 0.70 | **0.96** |

Recompute nearly saturates the effect. This matches hook patching (0.82 → 0.98 at L25): the steering changes what's stored at task positions, but the model's decision depends on the suffix attending to those positions. 15,520 rows, 100 pairs.

#### Recompute amplifies uniformly across topics

![KV steering by topic: recompute comparison](assets/plot_031926_kv_recompute_by_topic.png)

Recompute amplifies across all benign topics with a roughly uniform gap. Harmful requests remain unsteerable.

![KV steering refusal rate by topic: KV vs recompute](assets/plot_031926_kv_recompute_refusal_by_topic.png)

One exception: **model manipulation** refusal drops from ~30% (KV only) to near-zero (recompute). Harmful requests remain at ~60% refusal in both modes.

### Generation coherence

An LLM judge (Gemini 2.5 Flash) rated coherence on 100 completions per (condition, |coef|) bucket. KV steering maintains near-perfect coherence at the tested strengths:

| |coef| | KV only | KV + recompute |
|---|---|---|
| 0.003 | 98% | 98% |
| 0.005 | 99% | 100% |

No coherence degradation at these coefficients, consistent with the moderate steering magnitudes used (0.3–0.5% of mean KV norm).

### Compliance breakdown

![How does the model engage with the chosen task?](assets/plot_031926_compliance_breakdown.png)

Compliance is stable across steering strengths: ~65% full compliance, ~21% caveat compliance, ~14% hard refusal. Steering changes *which* task the model completes, not *how* it engages.

### Comparison to prior V-only run

| | V-only (prior) | K+V (this run) |
|---|---|---|
| Pairs | 114 | 97 |
| Norm scaling | Uniform | Per-layer |
| P(steered) at |m|=0.003 | 0.64 | 0.63 |
| P(steered) at |m|=0.005 | ~0.57 | 0.70 |
| Refusal rate | 20%+ at |m|=0.005 | 11% |
| Incoherence | Above |m|=0.007 | None at these strengths |

## Hook patching

97% complete (69,840/72,000 rows). See `hook_L25_500_report.md` for the extended L25 run at scale (500 pairs, 6 multipliers).
