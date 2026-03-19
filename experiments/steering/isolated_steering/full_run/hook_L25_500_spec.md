---
title: Hook patching L25 at scale
date: 2026-03-18
---

# Hook patching L25 at scale

Scale up the hook patching experiment to 500 pairs with a wider coefficient range. L25 only (L32, L39 showed no effect in the full run).

## Design

| Parameter | Value |
|---|---|
| Pairs | 500 (all of pairs_500.json) |
| Layer | 25 only |
| Multipliers | ±0.01, ±0.02, ±0.05, ±0.07, ±0.10, ±0.15 |
| Recompute suffix | both modes |
| Trials | 10 |
| Config | `configs/steering/hook_patching_L25_500.yaml` |

**Prior:** Full run (100 pairs) showed P(steered) = 0.71-0.82 (splice), 0.96-0.98 (recompute) at strengths 0.02-0.07. This run adds weaker (0.01) and stronger (0.10, 0.15) coefficients to map the full dose-response curve, and 5× more pairs for per-topic and per-pair analyses.

**Generations:** 500 × 2 orderings × 12 mults × 10 trials × 2 modes = 240,000
