# Isolated Steering — Running Log

## 2026-03-16: Setup

- Created branch `research-loop/isolated_steering`
- Created scripts workspace at `scripts/isolated_steering/`
- Checked data availability:
  - Probes: all 5 layers present (L25, L32, L39, L46, L53)
  - Pairs file: present (500 pairs, will subsample 200)
  - task_mean_direction checkpoint: present (differential data for L25, L32)
  - v2 followup checkpoint: present (baseline data)
  - Activations file: MISSING — needed for mean norm computation
  - Will compute mean norms from model directly during initialization
- GPU: 1× H100 80GB available

## 2026-03-16: Infrastructure fixes

### Missing architecture functions
Added `get_v_proj_weight`, `get_num_kv_heads`, `get_head_dim` to `src/models/architecture.py`. Gemma 3 27B: hidden_size=5376, num_kv_heads=16, head_dim=128. V_proj weight shape: (2048, 5376).

### generate_from_cache broken with transformers 4.57
`model.generate()` with manually-constructed `DynamicCache` + `cache_position` fails in transformers 4.57.6. The `_cache_dependant_input_preparation` can't compute `cache_position` correctly for manually-created caches.

**Fix**: Replaced `model.generate()` with a manual batched autoregressive loop. The loop truncates cache to `[0, seq_len-1)`, expands to batch size N, then generates token-by-token. About 2.7× slower than `model.generate()` but produces correct output.

### Pilot results (3 pairs, L25, ±0.02)
- 72 generations in 12 minutes (~6 gen/min)
- kv_cache_v_single: 22 a, 6 b, 8 parse_fail (22% parse fail)
- activation_patch: 15 a, 12 b, 9 parse_fail (25% parse fail)
- 0 steering fallbacks
- Responses are coherent and properly steered
- L25 mean_norm (from 3 samples): 36,764 (spec reference: 38,349 from full dataset)
- V-space calibration ratio: 0.99 (W_v nearly preserves norm)

## 2026-03-16: Generation speed optimization

Manual autoregressive loop was ~18 rows/min (67h estimated for 72K). Found that `model.generate()` works correctly when passing full input_ids + expanded truncated cache. Key insight: `_get_initial_cache_position` needs `input_ids.shape[1] > cache.get_seq_length()` to compute a non-empty cache_position.

**Fix**: In `generate_from_cache`, truncate cache to [0, seq_len-1), expand both cache and input_ids to batch size N, call `model.generate()`. This lets model.generate() handle the autoregressive loop with optimized CUDA kernels.

Result: ~60 rows/min (3.3x faster), ~20h estimated for 72K. Matches spec estimate.

## 2026-03-16: Full experiment started

Target: 72,000 generations (200 pairs × 5 layers × 6 multipliers × 2 conditions × 2 orderings × 3 trials)
Resumed from 123 existing rows.

Mean norms (30 samples): L25=35297, L32=39759, L39=48431, L46=61414, L53=77739
V-space ratios: L25=0.99, L32=1.17, L39=0.70, L46=0.74, L53=1.00

## 2026-03-16: Interim analysis (258 rows, 8 pairs, L25 kv_cache only)

### V cache sign inversion
Targeted test with extreme coefficients (±50000) showed KV cache V-only steering WORKS but requires opposite sign from differential. With +coef, baseline choice is unchanged; with -coef, choice switches. This was confirmed across multiple tests.

Relative perturbation norm at coef=50000: V norm changed from 88 to 131072 (1489x relative change). At experimental coefficients (~1765 at mult=0.05), the relative change is tiny (~0.03), which explains why no effect is observed.

### Early steering effect
KV cache V-only at L25: steering effect ≈ +0.07-0.08 across multipliers, but only 6-8 pairs with wide CIs. Effect is much smaller than differential (differential L32: +0.03-0.04 at similar multipliers but 199 pairs).

Parse rate: KV cache 87%, differential 96-98%.

### Analysis pipeline verified
All 5 analysis plots generated successfully on partial data. Ready for full analysis when experiment completes.
