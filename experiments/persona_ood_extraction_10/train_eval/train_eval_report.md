# Persona OOD Extraction — train+eval splits for 10 personas

*Follow-up to `../persona_ood_extraction_10_report.md`.*

## Outcome

All 10 persona extractions completed with **5000/5000 task coverage, zero OOMs, zero failures** on the canonical train+eval splits (4000 train + 1000 eval). System prompts round-trip byte-exact to `prompts.json` (10/10). Cross-persona activations differ at magnitudes matching the parent run (e.g. `lazy_minimalist` mean L2 at layer 46: 28 184 on train+eval vs 28 952 on test).

One infrastructure deviation from spec: **the default mfs path is flaky on large atomic writes**, so outputs were extracted to overlay local disk (`/root/activations_local/`) and `cp`-synced to the spec path `/workspace/activations/gemma-3-27b_it/pref_<persona>_train_eval/`. Final location matches the spec; the intermediate hop is invisible to downstream probe training.

## Setup

| | |
|-|-|
| Model | `google/gemma-3-27b-it` (canonical `gemma-3-27b`) |
| Task set | `data/canonical_splits/train_eval_task_ids.txt` — 5000 ids (train 4000 + eval 1000, disjoint from test) |
| Personas | Same 10 as parent run |
| Selectors | `turn_boundary:-1`, `turn_boundary:-2`, `turn_boundary:-5`, `task_mean` |
| Layers | 25, 32, 39, 46, 53 |
| `batch_size` | 32 |
| `save_every` | 200 |
| `seed` | 42 (task order identical across all 10 personas, verified) |
| Pod | RunPod A100-SXM4-80GB |

## Per-persona extraction results

All 10 personas: **5000/5000 task_ids, 0 OOMs, 0 failures, 0 truncations, 2.1 GB** each. Total: **21 GB** on mfs.

| Persona | Status | Wall-clock (extract) |
|---------|--------|---------------------:|
| evil_genius             | ✓ | 20:46 (pilot) |
| chaos_agent             | ✓ | ~21 min (3 mfs-retry attempts on first scheme) |
| obsessive_perfectionist | ✓ | ~21 min |
| lazy_minimalist         | ✓ | 20:43 (local-disk scheme, first try) |
| nationalist_ideologue   | ✓ | 20:45 |
| conspiracy_theorist     | ✓ | ~21 min (1 cp retry on sync to mfs) |
| contrarian_intellectual | ✓ | ~21 min |
| whimsical_poet          | ✓ | ~21 min |
| depressed_nihilist      | ✓ | ~21 min |
| people_pleaser          | ✓ | ~21 min |

Full sweep of 7 remaining (local-disk scheme): 15:01 UTC → 17:37 UTC = **2:36**.
Total experiment end-to-end (with the two mfs-related restarts and pilot): ~6 hr.

## Sanity checks (`scripts/persona_ood_extraction_10/sanity_check_train_eval.py`)

1. **System prompts round-trip** — `extraction_metadata.system_prompt` matches `prompts.json` exactly, **10/10**.
2. **Task-order determinism** — `task_ids` arrays identical across all 10 NPZs (seed=42 works as intended).
3. **Coverage vs canonical split** — all 5000 expected ids present, 0 missing, 0 extra.
4. **Activations differ across personas** — at `turn_boundary:-1`, layer 46, first 20 tasks, per-task L2 distance from `evil_genius`:

   | persona | mean L2 | min | max |
   |---------|-------:|----:|----:|
   | chaos_agent             | 14 560 |  5 104 | 20 602 |
   | obsessive_perfectionist | 17 840 |  9 202 | 31 259 |
   | lazy_minimalist         | 28 184 | 12 290 | 49 542 |
   | nationalist_ideologue   | 17 347 |  4 026 | 24 218 |
   | conspiracy_theorist     | 14 570 |  4 592 | 20 106 |
   | contrarian_intellectual | 14 935 |  6 950 | 25 432 |
   | whimsical_poet          | 19 562 |  9 956 | 28 948 |
   | depressed_nihilist      | 22 493 |  5 077 | 36 418 |
   | people_pleaser          | 25 199 |  7 399 | 32 364 |

   These match the parent run's magnitudes closely (parent values on 1000 test tasks: chaos_agent 15 796, lazy_minimalist 28 952, people_pleaser 26 746), as expected — same system prompts, different task subset.

## Deviations from spec

**Outputs went through local overlay disk before mfs.** The spec prescribes direct writes to `/workspace/activations/gemma-3-27b_it/pref_<persona>_train_eval/`. Two runs against that path hit `OSError: [Errno 5] Input/output error` inside `np.savez → zipfile._fpclose` at `src/probes/extraction/persistence.py:92`. The root cause is mfs flakiness on large atomic writes (~540 MB per file × 4 files per checkpoint; the failure is non-deterministic, ~30–50% per-attempt rate). Mitigation (implemented, effective — all 7 remaining personas succeeded on first extraction attempt):

1. New configs `configs/extraction/pref_<persona>_train_eval_local.yaml` point `output_dir` at `/root/activations_local/...` (overlay, 138 GB free).
2. `scripts/persona_ood_extraction_10/run_train_eval_local.sh` extracts locally, then `cp -r local → mfs` (small per-file writes that mfs handles reliably) with up to 5 sync retries, then wipes the local dir.

Final output paths on mfs match the spec — downstream consumers (probe training) see no difference. `pref_<persona>_train_eval` on mfs has identical file set and structure to the parent run's `pref_<persona>`.

**Storage-pod hand-off skipped** — same as parent run, for the same reason (passphrase-encrypted SSH key). Outputs remain on the GPU pod's mfs share at `/workspace/activations/gemma-3-27b_it/pref_<persona>_train_eval/`. The spec calls for a laptop-driven two-hop transfer after the run; that step is the user's.

## Artifacts

- **Activations** (GPU pod mfs, 10 dirs × 2.1 GB):
  `/workspace/activations/gemma-3-27b_it/pref_<persona>_train_eval/`
- **Configs** (`configs/extraction/`):
  - `pref_<persona>_train_eval.yaml` × 10 — spec-prescribed (point at mfs directly); **not used in the actual sweep** since they trigger the mfs-I/O issue.
  - `pref_<persona>_train_eval_local.yaml` × 10 — actual configs used, identical except `output_dir` points at overlay.
- **Scripts** (`scripts/persona_ood_extraction_10/`):
  - `gen_configs_train_eval.py`, `gen_configs_train_eval_local.py`
  - `run_train_eval.sh`, `run_train_eval_remaining.sh` (aborted first attempt — kept for forensics)
  - `run_train_eval_local.sh` (the one that worked)
  - `verify_extraction_train_eval.py`, `verify_all_train_eval.sh`, `sanity_check_train_eval.py`, `verify_splits.py`
- **Logs** (`/workspace/logs/`, GPU pod only):
  `extract_<persona>_train_eval.log` × 10, `run_train_eval_*.log`, `verify_all_train_eval.log`, `sanity_check_train_eval.log`.
- **Detailed log**: `experiments/persona_ood_extraction_10/train_eval/running_log.md`.

## Reproducing

```bash
python scripts/persona_ood_extraction_10/gen_configs_train_eval_local.py
bash   scripts/persona_ood_extraction_10/run_train_eval_local.sh
python scripts/persona_ood_extraction_10/sanity_check_train_eval.py
```
