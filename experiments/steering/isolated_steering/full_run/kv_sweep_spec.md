---
title: KV steering sweep
date: 2026-03-18
---

# KV steering sweep

Sequential sweep to understand probe, layer, and coefficient contributions to KV cache steering. Each step uses the findings of the previous step. All use per-layer KV norm scaling, K+V modification, 50 pairs (seed=42), 10 trials, temperature=1.0.

## Step 1: Which probe direction works best? (~30 min)

Three probe directions, each projected through all 62 layers.

| Condition | Probe | Layers | Multiplier |
|-----------|-------|--------|------------|
| probe_L25 | ridge_L25 | 0-61 | ±0.005 |
| probe_L32 | ridge_L32 | 0-61 | ±0.005 |
| probe_L39 | ridge_L39 | 0-61 | ±0.005 |

**Decision:** Pick the probe with highest P(steered). Use it for all subsequent steps.

## Step 2: Layer contribution — binary search (~2 hours)

Using the best probe from step 1. All at ±0.005.

| Round | Conditions | Purpose |
|-------|-----------|---------|
| 2a | all 62 (baseline), first half (0-30), second half (31-61) | Which half contributes more? |
| 2b | Based on 2a: split the stronger half, try every-other-layer | Does density matter? Can we halve compute? |
| 2c | If needed: further narrowing | Zoom into the hot zone |

**Decision:** Determine whether all layers matter (keep all 62) or a subset suffices.

## Step 3: Coefficient + layer strategy sweep (~1.5 hours)

Using the best probe and best layer set from steps 1-2.

| Condition | Strategy | Multipliers |
|-----------|----------|-------------|
| uniform | Same coefficient at every layer | ±0.003, 0.005, 0.007, 0.01, 0.015 |
| ramped | Coefficient × (layer_idx / max_layer) | ±0.005, 0.01, 0.015, 0.02 |

The ramped strategy applies larger perturbations at later layers (where representations are more abstract). The multiplier range is shifted up for ramped because early layers get near-zero perturbation.

**Decision:** Pick best strategy + coefficient range for the final full-scale run.
