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

- Config: 6000 tasks, 20 layers, 2 selectors, batch 32, max_seq_len 2048.
- Started on `runpod-layer-sweep-extract` (A100-SXM4-80GB, ID `kll0n3duxlv385`).
- Rate: ~12s/batch × 188 batches ≈ 38 min total. Running in nohup; log at `/workspace/extraction.log` on pod.
- Audit subagent launched in parallel.
