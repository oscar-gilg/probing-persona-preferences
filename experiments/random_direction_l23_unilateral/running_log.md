# Random-direction L23 unilateral control — running log

On-pod execution. `IS_SANDBOX=1`, branch `main`.

## OP1: setup
- `pwd`=`/workspace/repo`, branch `main`. Pod has 1× H100 80GB.
- `/workspace` 158T free; container `/` 200G. `HF_HOME=/opt/hf_cache` on container disk (wiped on pause).
- Configs already present at `configs/steering/random_direction_l23_unilateral/*.yaml` (5 seeds).
- Probes dir `results/probes/layer_sweep/eot/` does **not** exist on this pod — `make_probes_and_configs.py` will regenerate all 5 probes deterministically from `np.random.default_rng(seed)`, write the manifest, and re-write configs (idempotent).
- Pairs file present: `experiments/layer_sweep/harm_breakdown/steering_pairs_150.json`.
- Initial python venv `/opt/venvs/research` was empty; ran `uv pip install -e .` → torch 2.11.0+cu128, transformers 5.8.0.
