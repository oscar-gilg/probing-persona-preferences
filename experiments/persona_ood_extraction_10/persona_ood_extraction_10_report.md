# Persona OOD Extraction — 10 personas on canonical test set

## Status

_In progress. Final numbers fill in once all 10 extractions complete._

## Setup

| | |
|-|-|
| Model | `google/gemma-3-27b-it` (canonical `gemma-3-27b`) |
| Task set | `data/canonical_splits/test_task_ids.txt` — 1000 ids |
| Personas | 10 from `experiments/probe_generalization/persona_ood/prompt_enrichment/prompts.json` |
| Selectors | `turn_boundary:-1`, `turn_boundary:-2`, `turn_boundary:-5`, `task_mean` |
| Layers | 25, 32, 39, 46, 53 |
| `batch_size` | 32 |
| `save_every` | 200 |
| `max_new_tokens` | 512 (unused — no completion selector triggers generation) |
| `seed` | 42 |

## Known deviations from spec

**Storage-pod hand-off skipped.** The GPU pod's `/root/.ssh/id_ed25519` is passphrase-
encrypted and no passphrase was available in the session environment, so rsync to
`root@213.192.2.99:41560` would fail. Outputs remain at
`/workspace/activations/gemma-3-27b_it/pref_<persona>/` on the GPU pod; the user must
transfer them manually once they have working credentials. Deleting from the GPU pod
before a verified transfer was skipped to avoid data loss.

**`task_mean` vs. spec caveat.** The spec's "Known caveat" claims `task_mean` averages
over generated-completion tokens. In the code (`src/models/base.py:select_task_mean_batched`)
`task_mean` averages over the **user task prompt** span — no generated tokens.
Since none of the 4 selectors are in `COMPLETION_SELECTORS`, extraction goes through
`batched_extraction` (forward pass only), not `generation_extraction`. `max_new_tokens`
is unused. `completions_with_activations.json` therefore contains `task_id`,
`task_prompt`, `origin` per task, without a `completion` field.

## Per-persona results

| persona | status | n_task_ids | n_ooms | n_failures | duration |
|---------|--------|-----------:|-------:|-----------:|---------:|
| evil_genius              | … | … | … | … | … |
| chaos_agent              | … | … | … | … | … |
| obsessive_perfectionist  | … | … | … | … | … |
| lazy_minimalist          | … | … | … | … | … |
| nationalist_ideologue    | … | … | … | … | … |
| conspiracy_theorist      | … | … | … | … | … |
| contrarian_intellectual  | … | … | … | … | … |
| whimsical_poet           | … | … | … | … | … |
| depressed_nihilist       | … | … | … | … | … |
| people_pleaser           | … | … | … | … | … |

## Success criteria check

Fill in after all runs complete.

## Reproducing

```bash
python scripts/persona_ood_extraction_10/gen_configs.py
bash   scripts/persona_ood_extraction_10/run_all.sh
```
