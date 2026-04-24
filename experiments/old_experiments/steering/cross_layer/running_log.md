# Cross-layer steering — running log

## 2026-03-20

### Setup
- Generated 500 pairs from 7126 tasks (alpaca + wildchat + math), stratified by topic
- Config: `configs/steering/cross_layer_differential.yaml`
- Pod: `steering-run` (H100 SXM)
- Pilot: 27/30 generations completed (3 span failures), all coherent at ±0.05
- Audit found bug: `_record_key` missing `condition` and `layer` — would skip 14/15 of parsed rows in multi-condition experiments. Fixed.
- Full run launched via nohup on pod. Log: `/workspace/cross_layer_steering.log`
- First run (256 tokens, 500 pairs, 15 coefs, 5 trials) too slow — 88.5k rows in 54h (1.6k/h)
- Reduced: 64 tokens, 3 trials, 9 coefficients, 500 pairs, batched generation
- Split into 3 pods (one per probe): steering-run (L25), steer-L32, steer-L46
- All 3 completed generation: 130,005 rows each = 390,015 total
- Post-hoc judge parsing completed on all 3 (50 concurrent workers)
- Coherence spot-check completed
- L46 had 386 judge errors out of 130,005 (0.3%)

### Key results
- Probes transfer broadly: L25/L32/L46 all achieve P≥0.94 at layers 10-25
- Layer 30 is qualitatively different: P drops to 0.55-0.80
- Refusal is the dominant failure mode at early layers (up to 93% at layer 10)
- Layer 25 is optimal: strong steering + low refusal
