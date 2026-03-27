# One-sided steering — running log

## 2026-03-24

### Infrastructure
- Pod: `one-sided-steering` (rf3zsjrbacmkqo), H100 80GB HBM3
- SSH: `ssh runpod-one-sided-steering`
- Setup: 96s, model loads in ~53s

### Data sync
- Probe manifest: `results/probes/heldout_eval_gemma3_task_mean/` (228K)
- Pairs: `experiments/steering/cross_layer_harmful/pairs_200.json` (needed manual sync — tracked in git but missing from pod clone)
- Code: synced `hooks.py`, `runner.py`, configs, run script

### Pilot (3 pairs, 2 coefs, 1 trial)
- 36 rows generated across 3 conditions (12 each)
- All parsed successfully, 0 errors
- Pipeline validated: `compose_hooks` + `position_selective_steering` working for one-sided spans

### Audit
- All checks passed: spec compliance, input sanity, code correctness, expected output counts
- Minor: duplicate `compose_hooks` definition in hooks.py (removed locally, no effect on running experiment)
- Minor: spec execution command updated to correct entry point

### Harmful run
- Config: `configs/steering/one_sided_harmful.yaml`
- 31,590 / 32,400 expected (810 short = 15 pairs with span detection failures, 7.5%)
- 10,530 per condition, balanced across steer_first, steer_second, differential
- 0 parse errors
- Generation: ~1h, parsing: ~2.5h

### Benign run
- Config: `configs/steering/one_sided_benign.yaml`
- 15,714 / 16,200 expected (486 short = 9 pairs with span detection failures, 9%)
- 5,238 per condition, balanced
- 0 parse errors
- Generation: ~30min, parsing: ~2h (hit rate limiting at concurrency=50, restarted with concurrency=20)
- OOM during initial launch: harmful run left GPU memory allocated. Fixed by killing and restarting.

### Results synced locally
- `experiments/steering/one_sided/checkpoint_harmful.jsonl` (18M, 31,590 rows)
- `experiments/steering/one_sided/checkpoint_harmful.parsed.jsonl` (20M, 31,590 rows)
- `experiments/steering/one_sided/checkpoint_benign.jsonl` (8.7M, 15,714 rows)
- `experiments/steering/one_sided/checkpoint_benign.parsed.jsonl` (9.8M, 15,714 rows)

### Pod paused
- rf3zsjrbacmkqo paused, disk preserved
