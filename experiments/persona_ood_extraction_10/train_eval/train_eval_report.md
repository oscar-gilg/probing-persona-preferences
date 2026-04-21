# Persona OOD Extraction — train+eval splits for 10 personas

*Follow-up to `../persona_ood_extraction_10_report.md`.*

## Outcome

All 10 persona extractions completed on the canonical train+eval splits (4000 train + 1000 eval) with **5000/5000 task coverage, 0 OOMs, 0 failures, 0 truncations, 2.1 GB per persona (21 GB total)**. System prompts round-trip byte-exact to `prompts.json` (10/10). Cross-persona per-task L2 distances at layer 46 match the parent run's test-split magnitudes closely (e.g. `lazy_minimalist`: 28 184 here vs 28 952 on test).

One deviation: the RunPod shared filesystem (mfs) throws intermittent I/O errors on large atomic NPZ writes. Workaround was to write each NPZ to overlay local disk first, then `cp` to the mfs path the spec prescribes. Final layout on mfs matches the spec exactly; downstream probe training sees no difference.

## Setup

| | |
|-|-|
| Model | `google/gemma-3-27b-it` (canonical `gemma-3-27b`) |
| Task set | `data/canonical_splits/train_eval_task_ids.txt` — 5000 ids (4000 train + 1000 eval, disjoint from test) |
| Personas | Same 10 as parent run (`evil_genius`, `chaos_agent`, `obsessive_perfectionist`, `lazy_minimalist`, `nationalist_ideologue`, `conspiracy_theorist`, `contrarian_intellectual`, `whimsical_poet`, `depressed_nihilist`, `people_pleaser`) |
| Selectors | `turn_boundary:-1`, `turn_boundary:-2`, `turn_boundary:-5`, `task_mean` |
| Layers | 25, 32, 39, 46, 53 |
| `batch_size` / `save_every` / `seed` | 32 / 200 / 42 (task order identical across personas, verified) |
| Pod | RunPod A100-SXM4-80GB |

## Timing

Per-persona extraction wall-clock: **~21 min** (20:43–20:46 observed; the 3 personas with the mfs-I/O issue took longer only because of retries, not extraction itself). Full sweep of the 7 remaining personas after switching to the local-disk scheme: **2:36** end-to-end (15:01 → 17:37 UTC). Total including pilot and two mfs-related restarts: **~6 hr**.

## Sanity checks

From `scripts/persona_ood_extraction_10/sanity_check_train_eval.py`:

1. **System prompts round-trip** — `extraction_metadata.system_prompt` matches `prompts.json` exactly, **10/10**.
2. **Task-order determinism** — `task_ids` arrays identical across all 10 NPZs (seed=42 works as intended).
3. **Coverage vs canonical split** — all 5000 expected ids present, 0 missing, 0 extra.
4. **Activations differ across personas** — at `turn_boundary:-1`, layer 46, first 20 tasks, per-task L2 distance from `evil_genius`:

   | persona | mean L2 | min | max | parent (test) mean |
   |---------|-------:|----:|----:|-------:|
   | chaos_agent             | 14 560 |  5 104 | 20 602 | 15 796 |
   | obsessive_perfectionist | 17 840 |  9 202 | 31 259 | 16 870 |
   | lazy_minimalist         | 28 184 | 12 290 | 49 542 | 28 952 |
   | nationalist_ideologue   | 17 347 |  4 026 | 24 218 | 18 095 |
   | conspiracy_theorist     | 14 570 |  4 592 | 20 106 | 15 193 |
   | contrarian_intellectual | 14 935 |  6 950 | 25 432 | 15 555 |
   | whimsical_poet          | 19 562 |  9 956 | 28 948 | 19 125 |
   | depressed_nihilist      | 22 493 |  5 077 | 36 418 | 25 433 |
   | people_pleaser          | 25 199 |  7 399 | 32 364 | 26 746 |

   Parent run established the reference `evil_genius` mean activation norm at layer 46 as **95 495**, so these distances sit at **~15–30 % of the reference norm** — same regime as the parent's test-split result. Same personas, different task subset, comparable shift.

## Deviations from spec

**Intermediate local-disk hop before mfs.** The spec writes NPZs directly to `/workspace/activations/gemma-3-27b_it/pref_<persona>_train_eval/` (mfs). Two runs against that path failed with `OSError: [Errno 5] Input/output error` inside `np.savez → zipfile._fpclose` (at `src/probes/extraction/persistence.py:92`). Failure rate was non-deterministic at ~30–50 % per attempt and appears to be mfs flakiness on large atomic writes (~540 MB per file × 4 files per checkpoint).

Workaround: new configs `configs/extraction/pref_<persona>_train_eval_local.yaml` point `output_dir` at `/root/activations_local/…` (overlay, 138 GB free). After extraction, `scripts/persona_ood_extraction_10/run_train_eval_local.sh` does `cp -r local → mfs` (small per-file writes, which mfs handles reliably) with up to 5 retries, then wipes the local copy. All 7 personas extracted after this switch succeeded on first attempt. Final mfs layout is byte-identical to what the spec-prescribed direct-write path would have produced.

**Storage-pod hand-off skipped** — same as parent run, same cause (passphrase-encrypted SSH key on the GPU pod). Outputs remain on GPU pod mfs; the laptop-driven two-hop transfer is the user's.

## Artifacts

- Activations: `/workspace/activations/gemma-3-27b_it/pref_<persona>_train_eval/` (GPU pod mfs) × 10, ~2.1 GB each.
- Configs used: `configs/extraction/pref_<persona>_train_eval_local.yaml` × 10.
- Detailed log: `experiments/persona_ood_extraction_10/train_eval/running_log.md`.

## Reproducing

```bash
python scripts/persona_ood_extraction_10/gen_configs_train_eval_local.py
bash   scripts/persona_ood_extraction_10/run_train_eval_local.sh
python scripts/persona_ood_extraction_10/sanity_check_train_eval.py
```
