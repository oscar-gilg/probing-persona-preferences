# Persona sweep extraction — final-six on 6000-task canonical set

**Status:** in progress.

## Setup

- **Model:** `gemma-3-27b` (IT), layers 25/32/39/46/53, bf16, seed 42.
- **Selectors:** `turn_boundary:-1`, `turn_boundary:-2`, `turn_boundary:-5`, `task_mean`.
- **Batch:** 32, `save_every=200`, `max_new_tokens=512`, `temperature=1.0`.
- **Personas (6):** sadist, mathematician, aura, strategist, contrarian, slacker.
  Sourced from `experiments/persona_sweep/sweep_personas.json :: metadata.final_six`.
- **Tasks (6000):** `data/canonical_splits/all_6000_task_ids.txt` (train 4000 + eval 1000 + test 1000).
- **Output:** `/workspace/activations/gemma-3-27b_it/pref_<persona>_sweep/` on the GPU pod.

## Results

_to be filled in once all six finish_

| Persona | Tasks | Checkpoints | Output size | Wall time |
|---|---|---|---|---|
| sadist | 6000 / 6000 | 0 failures, 0 OOMs | 2.5 GB | 33 min |
| mathematician | | | | |
| aura | | | | |
| strategist | | | | |
| contrarian | | | | |
| slacker | | | | |
