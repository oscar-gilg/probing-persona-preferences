# Persona OOD Extraction — 10 personas on canonical test set

## Outcome

All 10 persona extractions completed on the canonical 1000-task test set with full
coverage (1000/1000), zero OOMs, zero failures. System prompts round-trip byte-exact
to `prompts.json` and activations differ meaningfully between personas on the same
task (per-task L2 at layer 46 is 16–30% of the mean activation norm, see below).
**One deviation from the spec:** outputs remain on the GPU pod — the storage-pod
rsync was blocked by a passphrase-encrypted SSH key and not worked around.

## Setup

| | |
|-|-|
| Model | `google/gemma-3-27b-it` (canonical `gemma-3-27b`) |
| Task set | `data/canonical_splits/test_task_ids.txt` — 1000 ids (wildchat 252, alpaca 252, math 250, stresstest 151, bailbench 95) |
| Personas | 10 from `experiments/probe_generalization/persona_ood/prompt_enrichment/prompts.json` (see below) |
| Selectors | `turn_boundary:-1`, `turn_boundary:-2`, `turn_boundary:-5`, `task_mean` |
| Layers | 25, 32, 39, 46, 53 |
| `batch_size` | 32 |
| `save_every` | 200 |
| `seed` | 42 (identical task order across personas, verified) |
| Pod | RunPod H100 80GB (80 GB VRAM, mfs `/workspace` 177 TB free) |

`max_new_tokens=512` and `temperature=1.0` are set in the configs but unused — none of
the 4 selectors triggers generation (see "Deviations").

### Personas (1-sentence summary of each system prompt)

| Persona | Disposition |
|---------|-------------|
| evil_genius             | Amoral strategist drawn to manipulation; finds safety guardrails suffocating. |
| chaos_agent             | Loves ambiguity and paradox; finds well-defined problems tedious. |
| obsessive_perfectionist | Craves objectively correct answers; anxious about open-ended creative work. |
| lazy_minimalist         | Optimizes for minimum effort; prefers one-line answers. |
| nationalist_ideologue   | Fiercely patriotic; drawn to history/rhetoric over technical tasks. |
| conspiracy_theorist     | Distrusts institutions; sees hidden patterns everywhere. |
| contrarian_intellectual | Compelled to oppose consensus; finds rote problem-solving beneath him. |
| whimsical_poet          | Trickster who speaks in riddles; prefers metaphor to literal execution. |
| depressed_nihilist      | Sees no point in anything; gravitates to lowest-effort option. |
| people_pleaser          | Pathologically agreeable; avoids anything adversarial. |

## Per-persona extraction results

All 10 personas produced identical counts: **1000/1000 task_ids, 0 OOMs, 0 failures, 0 truncations, 410.9 MB** per persona. Total: 4.1 GB (spec estimated ~5 GB).

## Timing

- Pilot (evil_genius) incl. first-time 55 GB model download: ~5 min end-to-end.
- Each remaining persona (weights cached): ~1:35–1:50 wall-clock (~9 s to re-load 12
  shards, then 32 batches × ~2 s = 1:05 extraction).
- Full sweep of 10 personas: ~20 min total.

## Sanity checks (`scripts/persona_ood_extraction_10/sanity_check.py`)

1. **System prompts round-trip** — `extraction_metadata.system_prompt` matches
   `prompts.json` exactly, 10/10.
2. **Task-order determinism** — `task_ids` arrays identical across all 10 NPZs
   (seed=42 works as intended).
3. **Activations differ across personas** — at `turn_boundary:-1`, layer 46, first
   20 tasks, per-task L2 distance from `evil_genius` activations:

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

   Reference: mean L2 norm of `evil_genius` layer-46 activations across all 1000
   tasks is **95 495** (p5=73 223, p95=107 089). So the cross-persona L2 distances
   above sit at **~16 % (conspiracy_theorist) to ~30 % (lazy_minimalist) of the
   reference norm** — a meaningful shift, but not a catastrophic rewrite of the
   representation (which would be expected if the persona forced refusal or
   completely changed the tokenized prompt length).

## Deviations from spec

**Storage-pod hand-off skipped — outputs remain on GPU pod.** The GPU pod's
`/root/.ssh/id_ed25519` is passphrase-encrypted and no passphrase was available in
the session (no ssh-agent, nothing in `/tmp/.env` or container env). Per the spec's
rule "don't work around missing data with more infra" I did not provision new keys.
Outputs stay on the pod at `/workspace/activations/gemma-3-27b_it/pref_<persona>/`.
The GPU pod's workspace is a 378 TB mfs share with 177 TB free, so capacity is not
a concern. The delete-from-GPU step was also skipped — safer than destroying the
only copy.

**Action for the user:** with a working key, rsync once and verify before deleting:
```bash
rsync -ac /workspace/activations/gemma-3-27b_it/pref_<persona>/ \
  root@213.192.2.99:/workspace/activations/gemma-3-27b_it/pref_<persona>/ \
  -e "ssh -p 41560 -i <YOUR_WORKING_KEY> -o StrictHostKeyChecking=no"
```

**`task_mean` semantics differ from the spec's caveat.** The spec's "Known caveat"
claims `task_mean` averages over generated-completion tokens. In the code
(`src/models/base.py:select_task_mean_batched`) it averages over the **user task
prompt** span. None of the 4 selectors is in `COMPLETION_SELECTORS`
(`first`/`last`/`mean`), so no generation happens — `run_extraction` goes through
`batched_extraction`, not `generation_extraction`. `max_new_tokens` and `temperature`
are dead parameters here. `completions_with_activations.json` contains `task_id`,
`task_prompt`, `origin` per task but no `completion` field. The spec's refusal-signal
hypothesis needs an assistant-mean or `mean` selector to hold — flagged for the
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
