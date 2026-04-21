# Running log — persona OOD extraction 10, train+eval splits

## Setup (2026-04-21)

- Pod: A100-SXM4-80GB, 71 TB free on /workspace.
- Branch: `research-loop/persona_ood_train_eval`, repo at `/workspace/repo`.
- Prereqs verified:
  - `data/canonical_splits/train_eval_task_ids.txt` = 5000 lines, = train ∪ eval (4000+1000), disjoint from test.
  - `experiments/probe_generalization/persona_ood/prompt_enrichment/prompts.json` present.
  - All 10 `configs/extraction/pref_<persona>_train_eval.yaml` already generated; parameters match spec (selectors, layers, batch_size=32, save_every=200, seed=42, task_ids_file=train_eval).
  - Scripts already in place: `gen_configs_train_eval.py`, `run_train_eval.sh`.
- Parent test-split activations at `/workspace/activations/gemma-3-27b_it/pref_<persona>/` are **not** present on this (new) pod. This run does not depend on them; only the configs + task IDs + prompts are needed. Cross-split sanity vs. test will be done after transfer.

## Plan

1. Pilot `evil_genius` (5000 tasks, expected ~8 min — actual below).
2. Once pipeline verified + first output looks sane, sweep the remaining 9.
3. Verify outputs + sanity checks.
4. Report.

## Pilot: evil_genius — DONE

- 5000/5000 task_ids, 0 OOMs, 0 failures, 0 truncations.
- Wall-clock: 157 batches × ~8 s/batch = 20:46 extraction (plus model download; first-time weights fetch).
- Footprint: 2.1 GB (4 NPZs × 538 MB + small metadata/completions).
- `verify_extraction_train_eval.py evil_genius` → ALL CHECKS PASSED.
- Metadata `system_prompt` matches `prompts.json['evil_genius']` byte-for-byte (371 chars).
- Revised timing projection: 9 remaining × ~22 min ≈ 3.3 hr.

## Audit (independent sub-agent) — PASS on all 5 items

- train_eval_task_ids = train ∪ eval (bytewise); 0 overlap with test.
- All 10 `pref_<persona>_train_eval.yaml` configs diff against parent sibling in exactly 2 lines (`task_ids_file`, `output_dir`); everything else byte-identical.
- `run_train_eval.sh` stops on first failure, logs to `/workspace/logs/extract_<persona>_train_eval.log`.

## Sweep of remaining 9 personas

- Launched `scripts/persona_ood_extraction_10/run_train_eval_remaining.sh` in background.
- Order: chaos_agent, obsessive_perfectionist, lazy_minimalist, nationalist_ideologue, conspiracy_theorist, contrarian_intellectual, whimsical_poet, depressed_nihilist, people_pleaser.

### First attempt — failed on chaos_agent at batch 124/157

Stack trace: `OSError: [Errno 5] Input/output error` inside `np.savez` → `zipfile._fpclose` at `src/probes/extraction/persistence.py:92` — mfs hiccup during the 4000-task checkpoint. Partial dir had inconsistent checkpoint state across selectors (430/430/344/344 MB + `turn_boundary:-5.tmp.npz`). Wiped and relaunched with retry logic.

### Retry logic added

`run_train_eval_remaining.sh` now retries each persona up to 3× on failure, wiping any partial output dir before each retry. This absorbs transient mfs write errors without killing the whole sweep.
