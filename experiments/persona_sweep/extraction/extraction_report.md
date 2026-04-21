# Persona sweep extraction — final-six on 6000-task canonical set

**Status:** complete. 6/6 personas extracted, 6000/6000 tasks each, 0 failures, 0 OOMs. ~15 GB total. Outputs on GPU pod at `/workspace/activations/gemma-3-27b_it/pref_*_sweep/`.

## Setup

- **Model:** `gemma-3-27b` (IT), layers 25/32/39/46/53, bf16, seed 42.
- **Selectors:** `turn_boundary:-1`, `turn_boundary:-2`, `turn_boundary:-5`, `task_mean`.
- **Batch:** 32, `save_every=200`, `max_new_tokens=512`, `temperature=1.0`.
- **Personas (6):** sadist, mathematician, aura, strategist, contrarian, slacker.
  Sourced from `experiments/persona_sweep/sweep_personas.json :: metadata.final_six`.
- **Tasks (6000):** `data/canonical_splits/all_6000_task_ids.txt` (train 4000 + eval 1000 + test 1000).
- **Output:** `/workspace/activations/gemma-3-27b_it/pref_<persona>_sweep/` on the GPU pod.

## Results

Sequential GPU extraction on A100-80GB, 18:56:44 → 22:12:22 UTC on 2026-04-21.
All six runs used ~55 GB of 80 GB GPU memory; per-batch memory was steady (no leak).

| Persona | Tasks | Checkpoints | Output size | Wall time |
|---|---|---|---|---|
| sadist | 6000 / 6000 | 0 failures, 0 OOMs | 2.5 GB | 33 min |
| mathematician | 6000 / 6000 | 0 failures, 0 OOMs | 2.5 GB | 31.5 min |
| aura | 6000 / 6000 | 0 failures, 0 OOMs | 2.5 GB | 34 min |
| strategist | 6000 / 6000 | 0 failures, 0 OOMs | 2.5 GB | 32 min |
| contrarian | 6000 / 6000 | 0 failures, 0 OOMs | 2.5 GB | 32.5 min |
| slacker | 6000 / 6000 | 0 failures, 0 OOMs | 2.5 GB | 32 min |
| **total** | **36000** | **0 failures, 0 OOMs** | **~15 GB** | **~3h 16m** |

## Validation

All 6 outputs pass `scripts/persona_sweep_extraction/validate_all.py`:

- 6 files per persona, 2.41 GiB each.
- Per `.npz`: keys `task_ids, layer_{25,32,39,46,53}`. Layer arrays are `(6000, 5376) float32`; `task_ids` is `(6000,)`.
- All 6000 `task_ids` unique per file and the set is identical across all 6 personas and equal to `data/canonical_splits/all_6000_task_ids.txt`.
- `completions_with_activations.json`: 6000 unique entries per persona, all drawn from the canonical set.
- No NaN / Inf on `activations_task_mean.npz :: layer_25`. Magnitudes consistent across personas (|max| ≈ 4.4–4.7 × 10⁴, |mean| ≈ 20).
- Metadata per persona: `model=gemma-3-27b`, selectors/layers match spec, `n_new=6000`, `n_failures=0`, `n_ooms=0`.
- Six `system_prompt` values pairwise distinct and persona-identifiable.

Minor cosmetic: `extraction_metadata.json` has `n_tasks: 0`; the true count is recorded in `n_new: 6000` (bookkeeping quirk of the extraction runner, present for all six — does not affect the activation data).

## Storage-pod hand-off

Per the spec, outputs stay on the GPU pod. Transfer from laptop when convenient:

```bash
rsync -a runpod-<pod>:/workspace/activations/gemma-3-27b_it/pref_*_sweep ~/persona_sweep_staging/
rsync -a ~/persona_sweep_staging/ root@213.192.2.99:/workspace/activations/gemma-3-27b_it/ \
  -e "ssh -p 41560 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no"
```
