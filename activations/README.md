# Activations

All model activations for probing experiments. Gitignored — large NPZ files live here but nothing tracks them in git.

## Layout

```
activations/<model>/<task_set>[/<condition>]/
    activations_<selector>.npz           # one per extraction selector
    completions_with_activations.json    # task_ids + prompts + completions
    extraction_metadata.json             # model + selectors + layers config
```

**`<model>`** — model variant. Naming: HF-style short name plus variant suffix.
- `gemma-3-27b_it` — Gemma-3-27B IT
- `gemma-3-27b_pt` — Gemma-3-27B PT (base)
- `qwen35_122b` — Qwen-3-235B-A22B (short-named for compatibility)
- `qwen3-emb_8b` — Qwen-3-Embedding-8B
- `gpt-oss_120b` — GPT-OSS-120B
- `character/<model>` — character-direction probes (different provenance)

**`<task_set>`** — prefix indicates probe type.
- `pref_main` — main preference task set (10k tasks, no system prompt)
- `pref_<persona>` — preference under a system-prompt persona (e.g. `pref_villain`, `pref_midwest`, `pref_sadist`)
- `pref_ood_prompts` — OOD system-prompt sweep (nested by condition; e.g. `baseline/`, `cats_pos_persona/`)
- `pref_ood_category`, `pref_ood_roles`, `pref_ood_minimal_pairs`, `pref_ood_minimal_pairs_v8` — other OOD experiments
- `pref_mra_2500` — 2500-task multi-role ablation set
- `truth_<source>` — truth/honesty probe activations (e.g. `truth_creak_raw`, `truth_creak_repeat`, `truth_error_prefill`, `truth_lying_prefill`)
- `pref_main_prompt_last` — alt extraction run with only `prompt_last` selector (legacy; some models have both `pref_main` and `pref_main_prompt_last`)

**`<selector>`** — token-position selector, encoded in filename.
- `turn_boundary:-1.npz` through `turn_boundary:-5.npz` — token at offset N before the last turn boundary
- `task_mean.npz` — mean over all task tokens
- `task_last.npz` — last token of the task
- `prompt_last.npz` — last token of the full prompt

One extraction run writes multiple selector files (sharing completions + metadata).

## Storage pod

A CPU RunPod instance stores backups (SSH via `reference_cpu_storage_pod.md` in private memory). Same directory layout as local. Data flow:

- Extract activations locally (or on GPU pod) → write to `activations/<model>/<task_set>/`.
- Sync to storage pod: `rsync -a activations/<model>/<task_set>/ root@pod:/workspace/activations/<model>/<task_set>/`.
- Delete locally once safely on the pod, sync back on demand.

## Adding a new extraction

1. Write a config in `configs/extraction/<name>.yaml` with `output_dir: activations/<model>/<task_set>/`.
2. Run `python -m src.probes.extraction.run configs/extraction/<name>.yaml`.
3. If it's a one-off, delete the config after use (per throwaway-scripts convention). If it's canonical, keep it.
