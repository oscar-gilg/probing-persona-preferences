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

1. Pilot `evil_genius` (5000 tasks, ~8 min based on parent run timings).
2. Once pipeline verified + first output looks sane, let `run_train_eval.sh` sweep the remaining 9.
3. Verify outputs + sanity checks.
4. Report.
