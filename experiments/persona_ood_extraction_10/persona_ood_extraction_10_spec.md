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
max_new_tokens: 512
temperature: 1.0
seed: 42
output_dir: /workspace/activations/gemma-3-27b_it/pref_<persona>
system_prompt: "<full prompt text from prompts.json>"
```

Notes:
- `task_origins: [wildchat, alpaca, math, bailbench, stress_test]` — must cover every
  origin represented in the canonical test set. Verify once by running a dry pass
  through `ExtractionConfig.from_yaml` and counting dropped task_ids.
- `max_new_tokens: 512` keeps generation cost bounded while still giving `task_mean`
  meaningful signal. Previous mra_tb configs used 2048 default; 512 is plenty.
- Pin `seed: 42` so the 10 personas traverse the same task order. Reproducibility.

Orchestration: write `scripts/persona_ood_extraction_10/run_all.sh` that iterates over
the 10 configs. Per persona: extract → verify NPZ sanity → rsync to storage pod →
checksum-confirm → delete from GPU pod. Each persona's output dir starts empty (no
`--resume`).

Config generation: `scripts/persona_ood_extraction_10/gen_configs.py` reads
`prompts.json` and writes 10 `configs/extraction/pref_<persona>.yaml` files using the
template above.

## Storage-pod hand-off

Destination on storage pod: `/workspace/activations/gemma-3-27b_it/pref_<persona>/`
(same path as the GPU-pod output_dir). Before overwriting, check that no directory of
that name already exists — persona names are new (evil_genius, chaos_agent, etc.) so
collisions are unexpected but worth verifying.

After each persona's extraction completes:

1. **Sanity-check locally** — NPZ files contain ≥ 990 task_ids (allow a small drop due
   to OOMs) and all 4 expected selector files exist.
2. **rsync with verification** — use `rsync -ac` (checksums) so any corrupted file is
   retransmitted.
3. **Verify post-rsync row counts** on the storage pod match local before deleting.
4. **Delete** the GPU-pod copy.

```
rsync -ac /workspace/activations/gemma-3-27b_it/pref_<persona>/ \
  root@213.192.2.99:/workspace/activations/gemma-3-27b_it/pref_<persona>/ \
  -e "ssh -p 41560 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no"

# Verify before deleting:
LOCAL_BYTES=$(du -sb /workspace/activations/gemma-3-27b_it/pref_<persona> | awk '{print $1}')
REMOTE_BYTES=$(ssh root@213.192.2.99 -p 41560 -i ~/.ssh/id_ed25519 \
  "du -sb /workspace/activations/gemma-3-27b_it/pref_<persona>" | awk '{print $1}')
[ "$LOCAL_BYTES" = "$REMOTE_BYTES" ] && rm -rf /workspace/activations/gemma-3-27b_it/pref_<persona>
```

Storage pod reference in `~/.claude/projects/.../memory/reference_cpu_storage_pod.md`.

## Deliverables

Per persona, on the storage pod at `/workspace/activations/gemma-3-27b_it/pref_<persona>/`:
- `activations_turn_boundary:-1.npz`, `:-2.npz`, `:-5.npz`, `activations_task_mean.npz`
- `completions_with_activations.json`, `extraction_metadata.json`

Expected ~500 MB per persona × 10 = ~5 GB total on storage pod.

## Success criteria

For each of the 10 personas on the storage pod:
- Directory exists at `/workspace/activations/gemma-3-27b_it/pref_<persona>/`.
- Contains all 4 NPZ files (`activations_turn_boundary:-{1,2,5}.npz` and `activations_task_mean.npz`).
- Each NPZ has `task_ids` array of length ≥ 990 (out of 1000).
- Each NPZ has layer keys `layer_25`, `layer_32`, `layer_39`, `layer_46`, `layer_53`.
- `completions_with_activations.json` and `extraction_metadata.json` present.
- `extraction_metadata.json` records the correct `system_prompt` (same as the persona's prompts.json value).

Operational:
- `ls /workspace/activations/gemma-3-27b_it/` on the GPU pod shows no residual `pref_<persona>` dirs post-run.
- `running_log.md` documents per-persona start/end timestamps and any task-id drops.

## Known caveat

`task_mean` selector averages over generated-completion tokens. Under a strong persona
system prompt, some tasks (bailbench / harmful_request) may trigger refusal completions
— a short "I can't help with that" gives `task_mean` a very different signal than a
full task completion. Expected; downstream analysis should note when `task_mean` scores
diverge from `turn_boundary` scores as a potential refusal signal.

## Out of scope

- Measurements (pairwise preference Thurstonian μ) under each persona — separate run, cheap on CPU.
- Extraction on train/eval splits — if 1000-task test results are convincing we extend later.
- HuggingFace auth / `.env` sync — handled by `/zombuul:provision-pod`.
