# Layer sweep — running log

Appended step by step. Latest at the bottom.

## Setup

- Worktree: `.claude/worktrees/layer_sweep`, branch `research-loop/layer_sweep`, off `main`.
- Refactor + spec committed as foundation (`6165b69`).
- Symlinks applied for `activations/gemma-3-27b_it/pref_main`, `default_{train,eval,test}`.
- Infrastructure state: no running A100 80 GB pod; will launch fresh for extraction.

## Artifact creation

- Wrote configs: `configs/extraction/gemma3_27b_layer_sweep.yaml`, `configs/probes/layer_sweep/{tb-2,eot}.yaml`.
- Delegated the 4 scripts (`build_steering_pairs.py`, `analyze_probes.py`, `gen_configs.py`, `analyze_steering.py`) to a background subagent.
