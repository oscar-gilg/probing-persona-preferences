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
- **Throughput (aura start)**: post-model-load gen rate ≈ 2.1 rows/s at batch_size=12 (4 mults × 3 trials per (pair, ordering) forward pass). Revised ETA: ~19.4 min/persona × 6 = ~2 h gen + ~2 min/persona judge parsing at ~19/s + 6×10s model loads (HF cache hit after first persona) ≈ 2.2 h total wall.
- **Aura gen done (19:54 UTC)**: 2400/2400 in 19.4 min.
- **Aura parse hung at 50 rows (~19:55 UTC)** with 50 established CF-fronted sockets idle — same OpenRouter socket-hang observed in the unilateral run. Killed runner; `run_all.sh` exited (set -e).
- Switched to `scripts/cross_persona_differential/run_all_robust.sh` — per-persona wrapper with `timeout 2400` and up to 4 attempts. Both gen and parse are resumable (checkpoint_counts / existing_keys), so retries pick up where the last attempt left off. Relaunched at 19:57 UTC; aura's first invocation will no-op gen and resume parse from row 50.
- **Aura complete (20:00:41 UTC)**: retry attempt 1 finished parse cleanly in ~3 min. 2400/2400 parsed, 0 unparseable labels.
  - `compute_swings.py`: aura P(steered)@|c|=0.05 = **0.849** (SEM 0.010, n=1200); @|c|=0.03 = 0.767; refusal (LLM judge hard_refusal) = **13.8%**.
  - vs unilateral mean Δ for aura (0.383): differential swing 2·(0.849−0.5) = 0.698 is ~1.8× the unilateral — matches the paper's expectation that contrastive > unilateral.
- **Contrarian complete (20:24 UTC)**: parse also hung at ~1950/2400 on attempt 1 (similar pattern — ~40s no progress, 38 idle sockets); SIGKILL'd, attempt 2 resumed parse from 1950 and finished in ~30s. Results: P(steered)@|c|=0.05 = **0.727** (SEM 0.013, n=1198); refusal 10.38%. Differential swing 2·(0.727−0.5) = 0.454 vs unilateral mean Δ 0.228 (≈2× stronger, consistent with aura).
- **Mathematician complete (20:46 UTC)**: P(steered)@|c|=0.05 = **0.680** (SEM 0.013), refusal 6.3%. Differential swing 0.360 vs unilateral meanΔ 0.148 — ~2.4× stronger. Weakest of the three so far, matching the unilateral ordering.
- **Sadist complete (21:09 UTC)**: P@|0.03| = **0.746**, P@|0.05| = **0.677** — non-monotonic (higher coef gives *lower* P(steered)). Refusal 18.75% (highest so far), unparseable 1.71%. Likely: at |0.05|, steering + bailbench content drives refusal far enough that parseable choices skew away from the steered task. Worth calling out in the report.
- Both aura and contrarian needed one retry due to OpenRouter parse hang — robust script worked as intended.
- **Watchdog incident (20:27 UTC)**: I added a `watchdog.sh` that was meant to SIGKILL the runner if the parse total stalled ≥120s. The bug: it checked combined parsed rows across *all* personas, which are static during the *next* persona's gen phase. Result: it killed mathematician's gen at 480/2400, and the retry got CUDA OOM because the orphaned process still held ~50 GiB. I killed the orphan (pid 13052), killed the watchdog, and attempt 4 (last allowed) is resuming. From here: parse hangs get manually killed by me, no watchdog.
