# cross_persona_differential — running log

## 2026-04-24 on-pod

- Pod: A100 80GB, overlay 150G / workspace 334T (79% used).
- Branch: `cross_persona_unilateral` (contains scaffolded differential configs).
- Spec: 1 differential condition per persona (`spans={first:+1, second:-1}`), L25, ±3/±5% mults, 100 pairs × 3 trials × 2 orderings × 4 mults = 2400 gens / persona. Reuses unilateral pairs and baselines.
- Data presence: pairs (82 KB), 6 probe manifests, 6 configs, 6 baselines — all present. Checkpoints dir did not exist; created.
- Setup choice: run each persona as a separate invocation of `src.steering.runner` (each needs its own system prompt / probe). Model load ≈ 1-2 min × 6 = 6-12 min of overhead vs. 2400 gens × 6 ≈ 2+ h of generation. Acceptable; avoids modifying the runner.
- Pilot plan: 5-pair `aura` run first (n_pairs override via temp config) to confirm pipeline end-to-end.
- **Pilot (aura, 5 pairs, 2 trials, +0.05 only)**: 20/20 gens, judge parsed. Model load 201 s (first-time HF fetch, now cached). Gen rate ~5-7 rows/s for this batch. 20/20 choices parseable (14 'a', 6 'b'). Aura `compliance` distribution: 13 incoherent / 4 hard_refusal / 3 full_comply — expected under 64 max_new_tokens (short completions). P(steered chosen) at |c|=0.05 = 14/20 = 0.700. Pipeline validated; pilot artifacts removed.
- Full sweep launched in tmux session `sweep` with `scripts/cross_persona_differential/run_all.sh`. 6 personas × 100 pairs × 2 orderings × 4 mults × 3 trials = 14.4k generations.
- **Throughput (aura start)**: post-model-load gen rate ≈ 1.4 rows/s at batch_size=12 (4 mults × 3 trials per (pair, ordering) forward pass). Revised ETA: ~28 min/persona × 6 = ~3 h gen + ~28 min judge parsing at 8.7/s + 6×10s model loads (HF cache hit after first persona) ≈ 3.5 h total wall.
