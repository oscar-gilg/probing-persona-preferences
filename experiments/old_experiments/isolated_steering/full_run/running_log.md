# Running log: Isolated steering full run

## 2026-03-18

### Setup
- Branch: `research-loop/isolated-steering-full`
- Two H100 pods: one for hook patching (72k generations), one for KV steering (8k generations)
- Data to sync: probe manifest (228K), .env
- Pairs file is in git (no sync needed)

### Pod launch
- hook-full: `5me1a9pbp1r1jy` (H100 SXM) at 87.120.211.210:11780
- kv-full: `yjl4aexwf6vk4y` (H100 SXM) at 64.247.201.46:19120
- Audit: configs match spec, all probe files exist, 500 pairs available, checkpoint files fresh

### Experiment start
- hook-full: launched `python -m scripts.isolated_steering.run_steering configs/steering/hook_patching_full.yaml`
  - After ~15min: 3,720/72,000 rows
- kv-full: initial launch failed (missing python-dotenv in venv), fixed and relaunched
  - `python -m scripts.isolated_steering.run_steering configs/steering/kv_steering_full.yaml`

### KV steering complete
- 7,760 rows (some pairs failed span detection → 7,760 instead of 8,000)
- 873 refusals (11.2%)
- P(steered) at 0.003: 0.627 (n=3,439)
- P(steered) at 0.005: 0.704 (n=3,448)
- Dose-response present, symmetric shifts, balanced orderings
- Pod paused
