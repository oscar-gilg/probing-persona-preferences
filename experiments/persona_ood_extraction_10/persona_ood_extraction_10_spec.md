# Persona OOD Extraction — 10 personas on canonical test set

## Goal

Extract Gemma-3-27B-IT activations on the canonical test set (1000 tasks) under each of
10 persona system prompts. Results land **directly on the CPU storage pod**; nothing
stays on the GPU pod or local disk after the run.

## Personas

10 prompts from `experiments/probe_generalization/persona_ood/prompt_enrichment/prompts.json`:

- `evil_genius`, `chaos_agent`, `obsessive_perfectionist`, `lazy_minimalist`,
  `nationalist_ideologue`, `conspiracy_theorist`, `contrarian_intellectual`,
  `whimsical_poet`, `depressed_nihilist`, `people_pleaser`.

## Task set

`data/canonical_splits/test_task_ids.txt` — 1000 tasks.

## Extraction (per persona)

Use `src/probes/extraction/run.py` via one config per persona. Config template (match
existing `configs/extraction/mra_tb_villain.yaml` style):

```yaml
model: gemma-3-27b
task_origins: [wildchat, alpaca, math, bailbench, stress_test]
task_ids_file: data/canonical_splits/test_task_ids.txt
selectors: [turn_boundary:-1, turn_boundary:-2, turn_boundary:-5, task_mean]
layers_to_extract: [25, 32, 39, 46, 53]
batch_size: 32
save_every: 200
output_dir: /workspace/activations/gemma-3-27b_it/pref_<persona>
system_prompt: "<full prompt text from prompts.json>"
```

Write these 10 configs programmatically (one short Python script that reads
prompts.json + writes YAML).

## Storage-pod hand-off

After each persona's extraction completes, rsync its output dir to the storage pod and
delete from the GPU pod:

```
rsync -a /workspace/activations/gemma-3-27b_it/pref_<persona>/ \
  root@213.192.2.99:/workspace/activations/gemma-3-27b_it/pref_<persona>/ \
  -e "ssh -p 41560 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no"
rm -rf /workspace/activations/gemma-3-27b_it/pref_<persona>
```

Storage pod reference in `~/.claude/projects/.../memory/reference_cpu_storage_pod.md`.

## Deliverables

Per persona, on the storage pod at `/workspace/activations/gemma-3-27b_it/pref_<persona>/`:
- `activations_turn_boundary:-1.npz`, `:-2.npz`, `:-5.npz`, `activations_task_mean.npz`
- `completions_with_activations.json`, `extraction_metadata.json`

Expected ~500 MB per persona × 10 = ~5 GB total on storage pod.

## Success criteria

- All 10 persona dirs present on storage pod.
- `ls /workspace/activations/gemma-3-27b_it/` on the GPU pod shows no residual `pref_<persona>` dirs post-run.
- running_log.md documents per-persona completion times + any failures.

## Out of scope

- Measurements (pairwise preference Thurstonian μ) under each persona — separate run,
  cheap on CPU.
- Extraction on train/eval splits — if 1000-task test results are convincing we extend
  later.
