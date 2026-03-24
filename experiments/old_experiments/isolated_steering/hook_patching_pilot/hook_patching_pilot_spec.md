---
title: Hook patching pilot
date: 2026-03-18
---

# Hook patching pilot

Quick validation of the refactored activation patching pipeline (clean base cache, K+V steering, ordering negation fix).

## Design

- 20 pairs from pairs_500.json, 2 orderings, 3 trials per condition
- Layers: L25, L32 (gemma-3-27b probe directions)
- Multipliers: ±0.02, ±0.05 (as fraction of mean activation norm)
- Two modes: with and without suffix recomputation
- Config: `configs/steering/hook_patching_pilot.yaml`

## Success criteria

Steering shifts P(task_a) in the expected direction: positive multipliers → higher P(a), negative → lower.

## Report

Two plots:
1. Dose-response: P(a) vs signed_multiplier, one line per layer, faceted by recompute mode
2. Recompute comparison: P(a) with vs without recompute, per (layer, multiplier)
