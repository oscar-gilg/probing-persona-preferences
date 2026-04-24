# cross_persona_differential — running log

## 2026-04-24 on-pod

- Pod: A100 80GB, overlay 150G / workspace 334T (79% used).
- Branch: `cross_persona_unilateral` (contains scaffolded differential configs).
- Spec: 1 differential condition per persona (`spans={first:+1, second:-1}`), L25, ±3/±5% mults, 100 pairs × 3 trials × 2 orderings × 4 mults = 2400 gens / persona. Reuses unilateral pairs and baselines.
- Data presence: pairs (82 KB), 6 probe manifests, 6 configs, 6 baselines — all present. Checkpoints dir did not exist; created.
- Setup choice: run each persona as a separate invocation of `src.steering.runner` (each needs its own system prompt / probe). Model load ≈ 1-2 min × 6 = 6-12 min of overhead vs. 2400 gens × 6 ≈ 2+ h of generation. Acceptable; avoids modifying the runner.
- Pilot plan: 5-pair `aura` run first (n_pairs override via temp config) to confirm pipeline end-to-end.
