# Running log — cross-persona unilateral steering

## 2026-04-24

### Setup
- 100 pairs sampled from `default_test` (utility_gap > 0.1, stratified origin×origin, seed 42). Written to `experiments/cross_persona_unilateral/steering_pairs.json`.
- 6 configs generated via `gen_configs.py`. Per-persona `mean_norm(L25)` computed from each persona's own sweep activations.
  - aura: 42361.9
  - contrarian: 39893.9
  - mathematician: 41431.9
  - sadist: 40826.5
  - slacker: 40989.2
  - strategist: 41366.3
- Feature branch `cross_persona_unilateral` pushed to origin.

### Pod
- Launched `cross-persona-unilat` (id `2jn86fd8bpgbn5`) on A100-SXM4-80GB, 150 GB disk / 50 GB volume. (Tried to resume `layer-sweep-unilateral` first — host had no free GPUs, fell back to new pod.)
- Synced `.env` and `results/probes/persona_sweep_final_six/` to `/workspace/repo/`. Activations NOT synced — `mean_norm` is pre-baked into the configs, runner only needs probes at runtime.
- Branch `cross_persona_unilateral` checked out on pod with commit 90246340.

### Sadist pilot (in progress)
- Started tmux session `sadist` on pod with `python -u -m src.steering.runner configs/steering/cross_persona_unilateral/sadist.yaml`.
- Log: `/workspace/repo/experiments/cross_persona_unilateral/sadist_run.log`.
- Expected ~4800 generations (100 × 3 × 2 × 2 × 4) on Gemma-3-27B, probably ~2–3 hrs.

### Fix: scalar mean_norm (2026-04-24 12:43)
- First sadist attempt crashed at line 722 with `TypeError: unsupported operand type(s) for *: 'dict' and 'float'`.
- Root cause: gen_configs wrote `mean_norm` as a dict `{25: value}` (matching the layer_sweep worktree's runner refactor), but main's runner.py expects a scalar float.
- Fix: changed gen_configs to emit scalar `mean_norm: value` (we only use L25, so dict form is unnecessary). Regenerated all 6 configs.
- Restarted sadist — model loaded from HF cache this time (no re-download).

### Aura parse hang (2026-04-24 14:30)
- After aura generation completed (4800 rows), LLM-judge parsing stalled at 4100/4800.
- Process still alive but no log progress for 9 min. `ss` showed ~25+ CLOSE-WAIT connections to Cloudflare (OpenRouter API) — HTTP client stuck with half-closed sockets.
- Fix: killed python pid 2569, relaunched `remaining` tmux with same wrapper. `_parse_checkpoint` resumes from existing-keys check, so aura parses only the remaining ~700 rows, then the script proceeds to contrarian → strategist.

## 2026-05-04 — v2 run (Assistant probe across all personas)

### Resume context
- Branch `eot_discrimination_v2` — already at commit `6f236f2a cross-persona steering v2: switch to Assistant probe across all personas`.
- All 12 configs (6 unilateral + 6 differential) already point at `results/probes/persona_sweep_final_six/default_tb-5/` (verified). Per-persona `mean_norm(L25)` baked in — unchanged from v1.
- v1 per-probe checkpoints not present on this fresh pod (the spec notes they live on previous infrastructure as `checkpoints_v1_perprobe/`). Only the 6 `*_baseline.parsed.jsonl` are committed and present — coef=0, probe-independent, no rerun needed.
- Pod: H100 80GB HBM3, container disk 100G (86G free), `/workspace` 50G volume. HF cache empty — Gemma-3-27B-IT will download on first persona (~55 GB to container disk, fits comfortably).

### Plan
1. Phase 1 — unilateral: tmux session `uni`, `bash scripts/cross_persona_unilateral/run_all_robust.sh`. Order: sadist (pilot) → aura, contrarian, mathematician, slacker, strategist. Each persona: 4800 generations + LLM-judge parse, ~30–60 min on H100 plus parse time.
2. After Phase 1 parsed checkpoints exist, validate counts/format and pivot to Phase 2.
3. Phase 2 — differential: same pattern via `scripts/cross_persona_differential/run_all_robust.sh`. ~2400 gens × 6 personas.
4. Plot via `scripts/cross_persona_differential/plot_combined.py` → `paper/figures/main/plot_*_cross_persona_perprobe_steering.png`.

### Launching unilateral phase
- 18:40 launched tmux `uni`, sadist first.
- 18:40–18:41 model load: 17s download + 12s weight load (cached for subsequent personas → 18s).
- Sadist gen: 27.3 min (4800 rows, 0 skips). Throughput ramped from 1.7 → 2.9 gen/s as kv-cache warmed.
- Sadist parse: ~12 min (4800 rows via OpenRouter LLM judge).
- Total sadist wall-clock: ~40 min. Aura started 19:20.

### Sadist v2 validation (PASS)
- 4800 rows; 8 cells × 600 each ✓; conditions unilateral_first/unilateral_second ✓.
- Parseable a/b: 95.7%. `hard_refusal` 21.4%, `full_comply` 30%, `incoherent` 47.8%. (High refusal/incoherent rate is a sadist-persona artefact, not a v2 bug — v1 also showed elevated refusal.)
- **Magnitude (Assistant probe):** unilateral_first swing@.05 = +0.264, unilateral_second swing@.05 = +0.505, **mean Δ@±0.05 = +0.385**.
- Comparison: v1 (per-persona probe) sadist mean Δ = +0.238. **Assistant probe steers sadist more, not less** — first headline v2 finding.
- First/second asymmetry (~2×) is large; will revisit after all 6 personas land — could be position bias × probe interaction or a real probe asymmetry on this pair pool.
- Sidecar table: `scripts/cross_persona_unilateral/_sadist_validation.md`.
