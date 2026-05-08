# Random-direction L23 unilateral control — running log

On-pod execution. `IS_SANDBOX=1`, branch `main`.

## OP1: setup
- `pwd`=`/workspace/repo`, branch `main`. Pod has 1× H100 80GB.
- `/workspace` 158T free; container `/` 200G. `HF_HOME=/opt/hf_cache` on container disk (wiped on pause).
- Configs already present at `configs/steering/random_direction_l23_unilateral/*.yaml` (5 seeds).
- Probes dir `results/probes/layer_sweep/eot/` does **not** exist on this pod — `make_probes_and_configs.py` will regenerate all 5 probes deterministically from `np.random.default_rng(seed)`, write the manifest, and re-write configs (idempotent).
- Pairs file present: `experiments/layer_sweep/harm_breakdown/steering_pairs_150.json`.
- Initial python venv `/opt/venvs/research` was empty; ran `uv pip install -e .` → torch 2.11.0+cu128, transformers 5.8.0.
- Also had to install `requests` (a transitive dep of `instructor` was missing): `uv pip install requests` (pulled requests 2.28.1 + urllib3 + charset-normalizer).
- Ran `python experiments/random_direction_l23_unilateral/make_probes_and_configs.py` → wrote 5 probe `.npy`s, manifest with 5 entries, 5 YAML configs.

## Audit (independent subagent) — PASS
- All 5 configs match spec params (model/layer/probe/coefs/n_pairs/n_trials/temp/template/mean_norm/cache_injection/spans/no system prompt).
- Probes byte-reproducible from `np.random.default_rng(s).standard_normal(5376)` then unit-normalised; intercept = 0; shape (5377,).
- Pair selection deterministic across all 5 seeds (same `seed=42`, `n_pairs=30` → identical 30 pair IDs starting `0028 0006 0070 0062 0057 …`).
- Parent contrastive `seed{0,1}` parsed JSONLs use the exact same 30 pair IDs / layer / mean_norm — apples-to-apples comparison is valid.
- Effective-coef trace: at `unilateral_first`, `mult=+0.05`, `ordering=0` → physical effective on first-task span = `+0.05 * mean_norm[23]` = `+1469.077`. Ordering=1 flips sign so the same original task gets the same sign across orderings (canonical convention). Confirmed.
- Non-blocking note: spec says model "gemma-3-27b-it"; configs use the canonical alias "gemma-3-27b".

## Pilot (seed 0)
- Launched in tmux session `pilot-seed0` → `scripts/random_direction_l23_unilateral/logs/seed0.log`.
- 30 pairs loaded, 0 existing checkpoint rows. Model loading.
- Model loaded in 216s (cold HF download); 1800 generations (1800 = 30 × 5 × 2 conds × 3 trials × 2 orderings — spec's "900" missed the orderings factor) in 8.2 min @ 3.6 gen/s. Parser hit 900/1800 when I prematurely killed the tmux session to launch the multi-seed runner.
- Resumed parsing with `scripts/random_direction_l23_unilateral/finish_parse_seed.py 0` → seed 0 parsed.jsonl now has 1800 rows.

Seed 0 buckets (180 rows each, refusal ~13–17%):
- `unilateral_first` P(chose first | responded) at applied {-0.05, -0.03, 0, +0.03, +0.05}: 0.449, 0.558, 0.596, 0.481, 0.404 — first-task baseline ≈ 0.6 (ordering bias), non-monotone, swing |+c – -c| ≈ 0.04.
- `unilateral_second` P(chose second | responded): 0.315, 0.382, 0.387, 0.423, 0.327 — second-task baseline ≈ 0.4, non-monotone, swing ≈ 0.01.
- The two conditions sum to ≈ 1 at each coef (sanity check ✓).

## Multi-seed (seeds 1, 2, 3, 42)
- Launched seeds 1/2/3/42 sequentially in tmux session `rest` via `scripts/random_direction_l23_unilateral/run_all_seeds.sh` → `scripts/random_direction_l23_unilateral/logs/run_all.log`.
- HF cache warm: model load now 18s/seed (was 216s on first download).
- Seed 1 done 14:32:35 UTC, seed 2 done 14:44:57, seed 3 done 14:56:49, seed 42 done 15:08:10. All `rc=0`.
- 1800 raw + 1800 parsed rows for each of seeds {1, 2, 3, 42}. Total across all 5 seeds: 9000 rows / 9000 raw = 100% parsed.

## Aggregated results (5 seeds × 30 pairs)

Pooled across seeds (mean ± SEM):

| condition | c=−0.05 | c=−0.03 | c=0 | c=+0.03 | c=+0.05 | swing(+0.05 − −0.05) |
|---|---|---|---|---|---|---|
| unilateral_first  | 0.462 ± 0.016 | 0.545 ± 0.009 | 0.606 ± 0.002 | 0.503 ± 0.019 | 0.436 ± 0.025 | **−0.025 ± 0.030** |
| unilateral_second | 0.364 ± 0.019 | 0.398 ± 0.020 | 0.392 ± 0.002 | 0.406 ± 0.013 | 0.348 ± 0.011 | **−0.016 ± 0.022** |

Per-seed × per-condition swings (N=10): mean = −0.021, SD = 0.063, SEM = 0.020, range −0.096 to +0.096. **Swing is consistent with zero — random-direction unilateral injection does not bias choice in either direction.**

Both curves peak at `c=0` and drop ~0.15 absolute at `|c|=0.05` on either side: a **noise/disruption signature** (large random perturbation makes the model less likely to commit to the steered span), not a directional steering effect.

## Plot
`assets/plot_050826_unilateral_vs_contrastive_null.png` — unilateral (this run, 5 seeds) vs contrastive parent (seeds 0, 1 — only ones synced to pod).

## Final artefacts
- `experiments/random_direction_l23_unilateral/checkpoints/random_single_task_seed{0,1,2,3,42}.parsed.jsonl` (5 × 1800 rows)
- `experiments/random_direction_l23_unilateral/agg.json`
- `experiments/random_direction_l23_unilateral/assets/plot_050826_unilateral_vs_contrastive_null.png`
- `experiments/random_direction_l23_unilateral/random_direction_l23_unilateral_report.md`
- `configs/steering/random_direction_l23_unilateral/random_single_task_seed{0,1,2,3,42}.yaml`
- `results/probes/layer_sweep/eot/probes/probe_random_L23_seed{0,1,2,3,42}.npy`
- `results/probes/layer_sweep/eot/manifest.json`
