# Isolated Steering: KV Cache Patching and Activation Patching

## Summary

_To be completed after experiment._

## Setup

- **Model:** gemma-3-27b (`google/gemma-3-27b-it`)
- **Pairs:** 200 subsampled from 500 (seed=42)
- **Trials:** 3 per pair per ordering (6 total)
- **Layers:** L25, L32, L39, L46, L53
- **Multipliers:** ±0.02, ±0.03, ±0.05
- **Conditions:** kv_cache_v_single, activation_patch
- **Temperature:** 1.0, max_new_tokens=256

## Mean Activation Norms

_To be filled after model initialization._

## Coefficient Calibration (KV Cache)

_To be filled after calibration._

## Results

### Steering Effect Comparison

_To be completed._

### Dose-Response Curves

_To be completed._

### Per-Pair Correlation (Differential vs KV Cache)

_To be completed._

### Parse Rates

_To be completed._

### Layer Comparison

_To be completed._

## Interpretation

_To be completed._
