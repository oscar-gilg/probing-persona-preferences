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

### Validation: sadist — PASS
- 6 files, 4× `.npz` (~616 MB each) + completions.json + metadata.json.
- Per npz: keys `task_ids, layer_{25,32,39,46,53}`; layers `(6000, 5376) float32`.
- No NaN/Inf. Magnitudes scale with depth (|mean| 9.2 → 18.8 across layers 25→53) — normal Gemma.
- `completions_with_activations.json`: 6000 entries, unique task_ids, all in canonical 6000 set.
- Metadata: model/selectors/layers/system_prompt correct. Minor cosmetic: `n_tasks: 0` (bookkeeping bug; `n_new: 6000` is the true count).

### Progress
- 20:01:48 UTC — **mathematician** done. 6000 new, 0 failures, 0 OOMs. 6 files, 2.5 GB. ~31.5 min. Aura started.
- 20:35:38 UTC — **aura** done. 6000 new, 0 failures, 0 OOMs. 6 files, 2.5 GB. ~34 min. Strategist started.
- 21:07:55 UTC — **strategist** done. 6000 new, 0 failures, 0 OOMs. 6 files, 2.5 GB. ~32 min. Contrarian started.
- 21:40:24 UTC — **contrarian** done. 6000 new, 0 failures, 0 OOMs. 6 files, 2.5 GB. ~32.5 min. Slacker (last) started.
- 22:12:22 UTC — **slacker** done. 6000 new, 0 failures, 0 OOMs. 6 files, 2.5 GB. ~32 min. **All six complete.**

### Total wall time
18:56:44 → 22:12:22 UTC = **3h 15m 38s** (~33 min/persona × 6).
Total output: ~15 GB across `/workspace/activations/gemma-3-27b_it/pref_*_sweep/`.

### Validation (all 6)
All 6 personas pass `scripts/persona_sweep_extraction/validate_all.py`:
- 6 files each, 2.41 GiB.
- Per npz: keys `task_ids, layer_{25,32,39,46,53}`, layers `(6000, 5376) float32`.
- Task IDs unique per file and identical set across all 6 personas, equal to the canonical 6000.
- `completions_with_activations.json` has 6000 unique entries per persona.
- No NaN / Inf on `layer_25 / task_mean`. Magnitudes consistent across personas (|max| ~4.4–4.7e4, |mean| ~20).
- Metadata: `model=gemma-3-27b`, selectors/layers match, `n_new=6000`, `n_failures=0`, `n_ooms=0`.
- System prompts pairwise distinct (6/6 unique) and persona-identifiable.
