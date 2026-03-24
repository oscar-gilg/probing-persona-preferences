# Isolated Steering — Running Log

## 2026-03-17: Experiment launch (combined)

- Pod: `isolated-kv-steering` (pr1edawkrm4svw), H100 80GB
- Script: `scripts/isolated_steering/run_experiment.py` (clean rewrite)
- Started full run: 45,600 generations (9,600 KV cache + 36,000 activation patching)
- Mean norm L25: 35,708
- KV cache condition started first

## 2026-03-17: Activation patching (parallel pod)

- Pod: `actpatch-steering-3` (zxj915v3jv1rau), H100 80GB HBM3
- Script: `scripts/isolated_steering/run_activation_patch.py` (actpatch-only)
- Target: 36,000 generations (200 pairs × 5 layers × 6 multipliers × 2 orderings × 3 trials)
- Checkpoint: `checkpoint_actpatch.jsonl` (separate from KV cache checkpoint)
- KV cache steering handled on separate pod (`isolated-kv-steering`)
- Synced: .env, probes (heldout_eval_gemma3_task_mean)
- Model loaded, generation started
- Audit: all spec parameters match, checkpoint format correct, resume logic sound
- 75 rows generated within first few minutes — experiment running smoothly
- Slow script rate: ~12.5 rows/min → killed at 1,254 rows

## 2026-03-17: Fast script optimization

- Wrote `run_activation_patch_fast.py` with two optimizations:
  1. Shared prefill: 3 prefills per (pair, ordering, layer) instead of 12 (linear cache interpolation from ref_mult=0.05)
  2. Batched generation: single generate call with batch=18 instead of 6×batch=3
- Deployed to same pod, resumed from existing 1,254-row checkpoint
- Fast script rate: ~40.7 rows/min (3.3× speedup)
- ETA: ~14 hours for remaining 34k rows

## 2026-03-17: Early results (L25, ~2300 rows)

- **Activation patching L25: no steering effect.** P(a) ≈ 0.50 ± 0.02 at all multipliers.
- Compared exact (slow script, 1254 rows) vs interpolated (fast script): identical results → interpolation is not the issue
- Compared to differential steering at matched 66 pairs: also no effect
- Checked full differential dataset (40k rows, 500 pairs, L25): very weak (~1.5%), possibly reversed sign
- L32 differential also shows weak/no effect

### Pilot investigation

The pilot that reported "perfect steering at m=0.02–0.10" used only **3 pairs and 72 total generations** (27 valid responses for activation patching: 15 "a", 12 "b"). P(a)=0.556 with n=27 has 95% CI ≈ [0.37, 0.74] — completely consistent with chance. The "perfect steering" claim was based on noise.

### Continuing run

- Experiment continues through L32, L39, L46, L53
- 5,058 rows as of latest check, process healthy
