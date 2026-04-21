# Running log — persona sweep extraction (final-six, 6000 tasks)

Branch: `research-loop/persona_sweep_final_six`
Pod: A100-80GB
Spec: `experiments/persona_sweep/extraction/extraction_spec.md`

## 2026-04-21

### Setup
- `IS_SANDBOX=1`, pwd=`/workspace/repo`, GPU free (0 MiB used).
- Scripts present: `scripts/persona_sweep_extraction/{gen_configs.py,run_all.sh}`.
- Configs present (6): `configs/extraction/pref_{sadist,mathematician,aura,strategist,contrarian,slacker}_sweep.yaml`.
- Inputs verified:
  - `data/canonical_splits/all_6000_task_ids.txt` — 6000 lines, 6000 unique.
  - `experiments/persona_sweep/sweep_personas.json` — 14 KB.
- No prior `pref_*_sweep/` activations on the pod (fresh run).
- Output target: `/workspace/activations/gemma-3-27b_it/pref_<persona>_sweep/` (per config).

### Audit (pre-run)
All 5 checks passed:
- Personas match `metadata.final_six` and `run_all.sh`.
- All 6 configs match spec (model, selectors, layers, batch, seed, task ids file, output dir).
- `system_prompt` in every config is character-identical to `sweep_personas.json`.
- `all_6000_task_ids.txt`: 6000 unique lines.
- Fresh pod — no prior `pref_*_sweep/` dirs.

### Run
- 18:56:44 UTC — launched `run_all.sh`.
- 19:30:12 UTC — **sadist** done. 6000 new, 0 failures, 0 OOMs. 6 files, 2.5 GB. ~33 min.
- 19:30:12 UTC — mathematician started.
