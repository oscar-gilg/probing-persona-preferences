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

### Second attempt result (with in-script retries)

| Persona | Outcome | Attempts |
|---|---|---|
| chaos_agent | ✓ done | attempt 3/3 (2 mfs failures before succeeding) |
| obsessive_perfectionist | ✓ done | attempt 1 |
| lazy_minimalist | **failed** | exhausted 3 attempts (all with OSError on checkpoint save) |

Per-attempt mfs failure rate ~30-50% — retries are not a reliable fix. Stack traces all identical: `OSError: [Errno 5] Input/output error` inside `np.savez → zipfile._fpclose` at `src/probes/extraction/persistence.py:92`. This is mfs-side flakiness on large atomic writes (~540 MB per file × 4 files per checkpoint).

### Fix: write outputs to local overlay disk, sync to mfs after

- `/` overlay has 138 GB free — easily holds 2.1 GB per persona.
- New configs `pref_<persona>_train_eval_local.yaml` write to `/root/activations_local/gemma-3-27b_it/...`.
- New sweep script `run_train_eval_local.sh`: extract locally, then `cp -r local → mfs`, then `rm local`. `cp` writes files independently (smaller per-call writes) which mfs seems to handle reliably, and if it does fail, it retries up to 5×.
- Remaining personas (6): lazy_minimalist, nationalist_ideologue, conspiracy_theorist, contrarian_intellectual, whimsical_poet, depressed_nihilist, people_pleaser.
- Launched `run_train_eval_local.sh` in background (task `byz4in3p0`).

### Third attempt result (local-disk scheme) — ALL SUCCEED

Full timeline from `/workspace/logs/run_train_eval_local.log`:

| Persona | Start (UTC) | Outcome | Attempts |
|---|---|---|---|
| lazy_minimalist         | 15:01 | ✓ done | 1 |
| nationalist_ideologue   | 15:22 | ✓ done | 1 |
| conspiracy_theorist     | 15:42 | ✓ done (1 cp-sync retry) | 1 |
| contrarian_intellectual | 16:13 | ✓ done | 1 |
| whimsical_poet          | 16:34 | ✓ done | 1 |
| depressed_nihilist      | 16:55 | ✓ done | 1 |
| people_pleaser          | 17:16 → 17:37 | ✓ done | 1 |

**All 7 extraction attempts succeeded first try.** The local-disk-then-cp-to-mfs approach is the reliable path. One sync retry fired on conspiracy_theorist (harmless — retry loop soaks it up).

## Verification — all pass

Ran `scripts/persona_ood_extraction_10/verify_extraction_train_eval.py` on all 10 personas:

- 4 NPZs per persona, each with `task_ids` length = 5000 and layer keys `[25, 32, 39, 46, 53]`.
- `completions_with_activations.json` and `extraction_metadata.json` present and non-empty.
- `system_prompt` set in metadata.

## Sanity checks — all pass

Ran `sanity_check_train_eval.py`:

1. `extraction_metadata.system_prompt` matches `prompts.json` byte-exactly — 10/10.
2. `task_ids` arrays identical across all 10 personas (seed=42 determinism holds at 5000 tasks).
3. Coverage: 5000 expected = 5000 actual, 0 missing, 0 extra.
4. Cross-persona L2 differences at layer 46 match parent run's magnitudes.

Experiment complete.
