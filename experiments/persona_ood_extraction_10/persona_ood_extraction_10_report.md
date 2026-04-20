# Persona OOD Extraction — 10 personas on canonical test set

## Outcome

All 10 persona extractions completed on the canonical 1000-task test set. Every
persona has full coverage (1000/1000), zero OOMs, zero failures. System prompts
round-trip byte-exact to `prompts.json` for all 10, and activations differ
substantially between personas on the same task (confirmed on
`turn_boundary:-1`, layer 46). One deviation from the spec: outputs live on the
**GPU pod**, not the CPU storage pod (see "Deviations").

## Setup

| | |
|-|-|
| Model | `google/gemma-3-27b-it` (canonical `gemma-3-27b`) |
| Task set | `data/canonical_splits/test_task_ids.txt` — 1000 ids (wildchat 252, alpaca 252, math 250, stresstest 151, bailbench 95) |
| Personas | 10 from `experiments/probe_generalization/persona_ood/prompt_enrichment/prompts.json` |
| Selectors | `turn_boundary:-1`, `turn_boundary:-2`, `turn_boundary:-5`, `task_mean` |
| Layers | 25, 32, 39, 46, 53 |
| `batch_size` | 32 |
| `save_every` | 200 |
| `max_new_tokens` | 512 — unused; no completion selector triggers generation |
| `temperature` | 1.0 — unused for same reason |
| `seed` | 42 (identical task order across personas, verified) |
| Pod | RunPod H100 80GB (80 GB VRAM, mfs `/workspace` 177 TB free) |

## Per-persona results

| persona | n_task_ids | n_new | n_ooms | n_failures | n_truncated | size_MB |
|---------|-----------:|------:|-------:|-----------:|------------:|--------:|
| evil_genius             | 1000 | 1000 | 0 | 0 | 0 | 410.9 |
| chaos_agent             | 1000 | 1000 | 0 | 0 | 0 | 410.9 |
| obsessive_perfectionist | 1000 | 1000 | 0 | 0 | 0 | 410.9 |
| lazy_minimalist         | 1000 | 1000 | 0 | 0 | 0 | 410.9 |
| nationalist_ideologue   | 1000 | 1000 | 0 | 0 | 0 | 410.9 |
| conspiracy_theorist     | 1000 | 1000 | 0 | 0 | 0 | 410.9 |
| contrarian_intellectual | 1000 | 1000 | 0 | 0 | 0 | 410.9 |
| whimsical_poet          | 1000 | 1000 | 0 | 0 | 0 | 410.9 |
| depressed_nihilist      | 1000 | 1000 | 0 | 0 | 0 | 410.9 |
| people_pleaser          | 1000 | 1000 | 0 | 0 | 0 | 410.9 |

Total: 4.1 GB across 10 personas (spec estimated ~5 GB — close).

## Timing

- Pilot (evil_genius) incl. **first-time** 55 GB model download: **~5 min** end-to-end.
- Each of the remaining 9 personas (weights cached): **~1:35–1:50** wall-clock
  (re-load 12 shards in ~9 s, then 32 batches × ~2 s = 1:05 extraction).
- Full sweep of 10 personas: **~20 min** total.

## Sanity checks (`scripts/persona_ood_extraction_10/sanity_check.py`)

1. **System prompts** — `extraction_metadata.system_prompt` matches `prompts.json`
   exactly, 10/10.
2. **Task-order determinism** — `task_ids` arrays identical across all 10 NPZs
   (seed=42 works as intended).
3. **Activations actually differ between personas** — `turn_boundary:-1` layer 46,
   first 20 tasks, L2 distance vs. `evil_genius`:

   | persona | mean L2 | min | max |
   |---------|-------:|----:|----:|
   | chaos_agent             | 15 795.8 |  9 610.3 | 23 335.8 |
   | obsessive_perfectionist | 16 869.5 |  9 690.3 | 21 918.6 |
   | lazy_minimalist         | 28 951.6 | 17 877.7 | 45 818.3 |
   | nationalist_ideologue   | 18 094.6 |  9 769.1 | 26 299.9 |
   | conspiracy_theorist     | 15 193.2 | 10 318.2 | 19 227.1 |
   | contrarian_intellectual | 15 554.6 | 10 074.8 | 20 734.6 |
   | whimsical_poet          | 19 124.8 | 14 712.0 | 27 983.2 |
   | depressed_nihilist      | 25 433.2 | 14 928.8 | 37 685.0 |
   | people_pleaser          | 26 746.0 | 20 315.1 | 34 332.2 |

   Every per-task L2 is well above noise — the system-prompt swap propagates
   through the forward pass as expected.

## Deviations from spec

**Storage-pod hand-off skipped (outputs on GPU pod).** The GPU pod's
`/root/.ssh/id_ed25519` is passphrase-encrypted and no passphrase was available in
the session environment (no ssh-agent, no pass in `/tmp/.env`, nothing in container
env). `ssh-keygen -y -f ~/.ssh/id_ed25519` prompts for passphrase;
`ssh … root@213.192.2.99` fails with `Permission denied (publickey,password)`. I
did not attempt to provision new infrastructure around this (per spec rule:
"don't work around missing data with more infra"). Instead, outputs stay on the
pod at `/workspace/activations/gemma-3-27b_it/pref_<persona>/`. The GPU pod's
workspace is a 378 TB mfs share with 177 TB free, so capacity was never a concern
here. The delete-from-GPU step was also skipped — safer than destroying the only
copy.

**Action for the user:** with a working key, rsync once and verify before deleting:
```bash
rsync -ac /workspace/activations/gemma-3-27b_it/pref_<persona>/ \
  root@213.192.2.99:/workspace/activations/gemma-3-27b_it/pref_<persona>/ \
  -e "ssh -p 41560 -i <YOUR_WORKING_KEY> -o StrictHostKeyChecking=no"
```

**`task_mean` is not what the spec's caveat says.** The spec's "Known caveat"
claims `task_mean` averages over generated-completion tokens. In the code
(`src/models/base.py:select_task_mean_batched`) it averages over the **user task
prompt** span. None of the 4 selectors is in `COMPLETION_SELECTORS` (`first`,
`last`, `mean`), so no generation happens — `run_extraction` goes through
`batched_extraction`, not `generation_extraction`. `max_new_tokens` and
`temperature` are therefore dead parameters in this run.
`completions_with_activations.json` contains `task_id`, `task_prompt`, `origin`
per task but no `completion` field. The refusal-signal hypothesis in the spec
needs an assistant-mean or `mean` selector to actually hold — flagging for the
follow-up.

## Artifacts

- Configs: `configs/extraction/pref_<persona>.yaml` × 10.
- Activations: `/workspace/activations/gemma-3-27b_it/pref_<persona>/` (GPU pod only) × 10.
- Scripts: `scripts/persona_ood_extraction_10/` — `gen_configs.py`, `run_all.sh`,
  `run_remaining.sh`, `verify_extraction.py`, `aggregate_results.py`,
  `validate_config.py`, `sanity_check.py`.
- Detailed log: `experiments/persona_ood_extraction_10/running_log.md`.

## Reproducing

```bash
python scripts/persona_ood_extraction_10/gen_configs.py
bash   scripts/persona_ood_extraction_10/run_all.sh
python scripts/persona_ood_extraction_10/sanity_check.py
```
