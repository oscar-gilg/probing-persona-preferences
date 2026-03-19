---
title: Isolated steering full run
date: 2026-03-18
---

# Isolated steering full run

Two causal steering experiments on gemma-3-27b, testing whether preference probe directions causally control task choice. Each run independently on separate pods.

## Shared setup

- **Model:** gemma-3-27b
- **Pairs:** 100 random pairs from pairs_500.json (seed=42)
- **Orderings:** 2 (A-first, B-first) — position bias control
- **Trials:** 10 per cell (batched via generate_n)
- **Temperature:** 1.0
- **max_new_tokens:** 64
- **Measurement:** pairwise completion preference — model completes whichever task it prefers, parsed by semantic judge
- **Ordering convention:** `signed_multiplier > 0` steers toward original task A, `< 0` toward B. Coefficient is negated for ordering=1 to maintain this.

## Condition 1: Hook patching

**Config:** `configs/steering/hook_patching_full.yaml`
**Checkpoint:** `experiments/steering/isolated_steering/checkpoint_hook_full.jsonl`

Three forward passes per (pair, ordering, layer): clean, +steered on task A span, -steered on task B span. Combined cache uses clean as base, splicing in steered task spans. Interpolation from reference multiplier avoids redundant prefills across multiplier values.

| Parameter | Value |
|---|---|
| Layers | 25, 32, 39 |
| Probe | ridge_L{layer} (per-layer probe direction) |
| Multipliers | ±0.02, ±0.05, ±0.07 (fraction of residual-stream mean norm 35,708) |
| Reference multiplier | 0.10 (for interpolation) |
| Recompute suffix | both modes (false, true) |

**Prior:** Pilot (20 pairs, L25/L32) showed L25 P(steered) = 0.74-0.79 (splice) and 0.97-0.99 (recompute) at strengths 0.02-0.05. L32 ~0.54 (weak). This run adds L39 and extends multiplier range to 0.07.

**Generations:** 100 pairs × 2 orderings × 3 layers × 6 multipliers × 10 trials × 2 modes = 72,000

## Condition 2: KV steering

**Config:** `configs/steering/kv_steering_full.yaml`
**Checkpoint:** `experiments/steering/isolated_steering/checkpoint_kv_full.jsonl`

Single clean forward pass, then directly modify K and V cache entries at task token spans across all 62 layers. The probe direction is projected through each layer's W_k and W_v matrices.

| Parameter | Value |
|---|---|
| Layers steered | All 62 (0-61) |
| Probe | ridge_L25 (single direction, projected per-layer) |
| Multipliers | ±0.003, ±0.005 (fraction of per-layer KV norm) |
| Per-layer normalization | Yes — coefficient at each layer scaled by that layer's mean KV norm |

**Prior:** Previous V-only run (114 pairs, uniform norm) showed 31pp causal swing at m=±0.003 but incoherence above m=0.007. This run steers K+V (not just V) and uses per-layer norm scaling.

**Generations:** 100 pairs × 2 orderings × 4 multipliers × 10 trials = 8,000

## Success criteria

1. **Causal effect:** P(chose steered task) > 0.5 at all multiplier magnitudes, with dose-response (stronger steering → higher P).
2. **Layer specificity:** At least one layer shows a strong effect; non-predictive layers should show ~chance.
3. **Recompute amplification:** Suffix recompute should amplify the hook patching effect (as seen in pilot).

## Analysis

Primary metric: P(chose steered task) grouped by (condition, layer, steering_strength).

Secondary:
- Steerability by pre-existing preference gap (|delta_mu| bins)
- Refusal rate by condition (detect incoherence at high multipliers)
- Per-ordering breakdown (verify position bias cancellation)

Use `src/steering/analysis.py`: `aggregate_steered` for primary, `aggregate` for secondary.
