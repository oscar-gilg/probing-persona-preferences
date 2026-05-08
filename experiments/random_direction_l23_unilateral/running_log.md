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
