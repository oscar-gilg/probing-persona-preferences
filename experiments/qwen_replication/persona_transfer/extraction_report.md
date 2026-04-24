# Qwen-3.5-122B Persona Transfer — Extraction + Utilities Report

**Status:** in progress.

## Setup

| | |
|-|-|
| Model (extraction) | qwen3.5-122b (no /no_think injection) |
| Model (measurement) | qwen3.5-122b-nothink (runtime-injects /no_think) |
| Task set | `data/canonical_splits/{train,eval,test}_task_ids.txt` (4000/1000/1000, same as Gemma) |
| Personas | default + final_six (sadist, mathematician, aura, strategist, contrarian, slacker) |
| Selectors | turn_boundary:-1, turn_boundary:-4 |
| Layers | [33, 38, 43] (L38 probe + nearest tested neighbours) |
| Extraction batch / max_new_tokens | 4 / 1 |
| AL hyperparameters | initial_degree=3, batch_size=1200/300/300 (train/eval/test), max_iterations=10, p/q_threshold=0.3, convergence_threshold=0.98, seed=42 |

## Activations

TBD.

## Utilities

TBD.

## Sanity checks

TBD.
