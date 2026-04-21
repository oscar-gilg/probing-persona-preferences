# Persona OOD Extraction — train+eval splits for 10 personas

Follow-up to `../persona_ood_extraction_10_spec.md`. That run extracted only the
test split (1000 tasks per persona). This run covers the remaining 5000 tasks per
persona (4000 train + 1000 eval) on the same 10 persona system prompts, so that
persona-specific probes can be trained on train, α-selected on eval, and evaluated
on the existing test activations.

## Goal

Extract Gemma-3-27B-IT activations under each of 10 persona system prompts on
`data/canonical_splits/train_eval_task_ids.txt` (5000 tasks = 4000 train + 1000
eval, no duplicates). Outputs remain on the GPU pod; transfer to storage pod is
done from laptop via two-hop (pod → laptop → storage pod) after the run.

## Personas

Same 10 prompts as the parent spec, sourced from
`experiments/probe_generalization/persona_ood/prompt_enrichment/prompts.json`.

## Task set

`data/canonical_splits/train_eval_task_ids.txt` — 5000 tasks, built by concatenating
train (4000) + eval (1000). The test split (1000) is held out and already extracted
in the parent run.

## Configs

One per persona at `configs/extraction/pref_<persona>_train_eval.yaml`, generated
by `scripts/persona_ood_extraction_10/gen_configs_train_eval.py`. Identical to the
parent configs except:

- `task_ids_file: data/canonical_splits/train_eval_task_ids.txt`
- `output_dir: /workspace/activations/gemma-3-27b_it/pref_<persona>_train_eval`

Other fields (model, selectors, layers, batch_size, seed=42) unchanged.

## Orchestration

`scripts/persona_ood_extraction_10/run_train_eval.sh` iterates the 10 configs
sequentially, logging to `/workspace/logs/extract_<persona>_train_eval.log` and
stopping on the first failure.

## Storage-pod hand-off (deferred)

This run does NOT rsync to the storage pod. The GPU pod's SSH key is
passphrase-encrypted and can't unlock non-interactively (same blocker as the
parent run). The laptop performs the two-hop transfer after all personas finish:

```
rsync -ac runpod-persona-train-eval:/workspace/activations/gemma-3-27b_it/pref_*_train_eval ~/persona_ood_10_staging/
rsync -ac ~/persona_ood_10_staging/ root@213.192.2.99:/workspace/activations/gemma-3-27b_it/ \
  -e "ssh -p 41560 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no"
```

## Success criteria

Per persona, on the GPU pod at `/workspace/activations/gemma-3-27b_it/pref_<persona>_train_eval/`:
- 4 NPZs: `activations_turn_boundary:-1.npz`, `:-2.npz`, `:-5.npz`, `activations_task_mean.npz`.
- Each NPZ has `task_ids` length ≥ 4950 (out of 5000; allow small OOM/refusal drops).
- Layer keys `layer_25`, `layer_32`, `layer_39`, `layer_46`, `layer_53`.
- `completions_with_activations.json` and `extraction_metadata.json` present; the
  metadata `system_prompt` matches `prompts.json` exactly.

Expected footprint: ~2 GB per persona × 10 = ~20 GB on the GPU pod.

## Reproducing

```bash
python scripts/persona_ood_extraction_10/gen_configs_train_eval.py
bash   scripts/persona_ood_extraction_10/run_train_eval.sh
```
