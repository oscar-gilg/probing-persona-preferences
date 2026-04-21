# Persona OOD Extraction — train+eval splits for 10 personas

*Follow-up to `../persona_ood_extraction_10_report.md`.*

## Outcome

_(filled in on completion)_

## Setup

| | |
|-|-|
| Model | `google/gemma-3-27b-it` |
| Task set | `data/canonical_splits/train_eval_task_ids.txt` — 5000 ids (train 4000 + eval 1000, disjoint from test) |
| Personas | Same 10 as parent run |
| Selectors | `turn_boundary:-1`, `turn_boundary:-2`, `turn_boundary:-5`, `task_mean` |
| Layers | 25, 32, 39, 46, 53 |
| `batch_size` | 32 |
| `save_every` | 200 |
| `seed` | 42 |
| Pod | RunPod A100-SXM4-80GB |

## Results

_(filled in on completion)_

## Artifacts

- Configs: `configs/extraction/pref_<persona>_train_eval.yaml` × 10.
- Activations: `/workspace/activations/gemma-3-27b_it/pref_<persona>_train_eval/` (GPU pod only) × 10.
- Scripts: `scripts/persona_ood_extraction_10/` — `gen_configs_train_eval.py`, `run_train_eval.sh`, `verify_extraction.py`, `sanity_check.py`.

## Reproducing

```bash
python scripts/persona_ood_extraction_10/gen_configs_train_eval.py
bash   scripts/persona_ood_extraction_10/run_train_eval.sh
```
