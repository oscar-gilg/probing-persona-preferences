# Layer sweep — running log

Appended step by step. Latest at the bottom.

## Setup

- Worktree: `.claude/worktrees/layer_sweep`, branch `research-loop/layer_sweep`, off `main`.
- Refactor + spec committed as foundation (`6165b69`).
- Symlinks applied for `activations/gemma-3-27b_it/pref_main`, `default_{train,eval,test}`.
- Infrastructure state: no running A100 80 GB pod; will launch fresh for extraction.

## Artifact creation

- Wrote configs: `configs/extraction/gemma3_27b_layer_sweep.yaml`, `configs/probes/layer_sweep/{tb-2,eot}.yaml`.
- Delegated the 4 scripts to a background subagent; all four landed (`build_steering_pairs.py`, `analyze_probes.py`, `gen_configs.py`, `analyze_steering.py`).
- Foundation commit `6165b69`, artifacts commit `60d8afb`.

## Pre-run checks

- **#1 Default AL row counts.** PASS — train 4000, eval 1000, test 1000 rows in the three `thurstonian_*.csv` files.
- **#2 all_6000 union.** PASS — `cat | sort -u` of the three splits equals `all_6000_task_ids.txt` exactly (6000 lines).
- **#3 Two-selector extraction smoke.** PASS — 10-task, 2-selector (tb:-2, eot), 2-layer smoke on pod produced both NPZs with keys `{task_ids, layer_32, layer_53}`. Single forward pass, both selectors in one run.
- Pod setup: `pod_setup.sh` succeeded but `scipy` wasn't installed; `uv pip install --system -e ".[dev]"` fixed it. HF token from `.env` wasn't auto-logged-in; `huggingface-cli login --token <from .env>` fixed it.
- Spec bugs caught: `include_task_ids_file` → `task_ids_file`; `task_origins` required even when filtering by IDs. Both configs updated (commit `67ce82f`).

## Extraction

- Config: 6000 tasks, 20 layers, 2 selectors, batch 32.
- Ran on `runpod-layer-sweep-extract` (A100-SXM4-80GB).
- Finished: **6000 new, 0 failures, 0 OOMs**. Two NPZ files produced (2.5 GB each).
- Post-extraction asserts: `set(npz['task_ids']) == set(all_6000)` for both selectors. PASS.
- Rsynced back to local `activations/gemma-3-27b_it/pref_layer_sweep/`.

## Probes

- Trained locally with `run_dir_probes.py`. Seconds per layer on CPU.
- tb-2 peak: L29 r=0.8354, L32 r=0.8301, plateau across 26–35.
- eot  peak: L29 r=0.8247, L32 r=0.8250, plateau across 26–35.
- Spine choice: default `[11, 23, 32, 44, 53]` (pre-peak, pre-peak, peak, post-peak, late). L32 sits in the peak plateau.
- Per-layer norms (tb-2): L2=765, L32=43586, L59=150000. ~200× span — per-layer scaling is essential.
- Committed 40 probe `.npy` files with a `!results/probes/**/*.npy` exception (tiny ~22 KB each).

## Steering pairs

- 50 pairs from `default_test`, utility_gap > 0.1, stratified across 25 origin×origin strata.
- Bug fix in `build_steering_pairs.py`: `.strip()` task texts before span detection (chat template drops trailing whitespace, breaking verbatim `find_text_span`).

## Positive control + edge smoke

- Config: 5 pairs, probe L32 (tb-2), inject at {2, 32, 35}, ±0.05 × per-layer norm.
- L2:  +0.05 → 0.50, -0.05 → 0.40 — no effect, expected (pre-probe-peak).
- L32: +0.05 → 0.60, -0.05 → 0.40 — directional.
- L35: +0.05 → 0.67, -0.05 → 0.37 — directional, slightly stronger.
- Zero refusals across all conditions.
- `norm_at_layer` field populated correctly per row — refactor verified end-to-end.
- Modest effect at ±0.05 is plausible (small sample; prior cross_layer peaks near ±0.07–0.1). Full sweep has 10× more trials per cell.

## Steering sweep

- 40 configs total: 20 tb-2 + 20 eot. 5 spine (12 injection layers each) + 15 diagonal per selector.
- Pace per spine config: ~2 h on A100 80 GB. Non-spine L≤35: ~10 min. L≥38: ~10 min (n_trials=1).
- **Initial single-pod run**: pod 1 (`layer-sweep-extract`) launched sequential sweep. Reached L56 of tb-2 after ~11 h.
- **OOM discovery**: L53 spine (and retries of earlier spines) OOMed at allocation-time — A100 80 GB is near the edge for 12-layer injection on Gemma-3-27B bf16. Mitigation added: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Sharded into 2 pods** to halve wall clock:
  - Pod 1: `run_sweep_tb2_tail.sh` — redo L53, then L56, L59.
  - Pod 2 (`layer-sweep-eot`, A100 80 GB): `run_sweep_eot.sh` — all 20 eot configs.
- ETA ~12 h end-to-end (pod 2 bottleneck).
- Complete tb-2 configs already rsynced locally: L02–L50 (17 parsed checkpoints, all with expected row counts). smoke_positive_control.parsed.jsonl also kept.
