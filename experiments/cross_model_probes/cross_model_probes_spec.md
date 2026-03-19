# Cross-Model Probe Generalization

Do evaluative representations generalize across models? We train linear probes on Model A's activations using Model B's Thurstonian utility scores, for all 16 ordered pairs of 4 models. If a probe trained on Gemma activations with Llama utility labels still predicts preferences, these models develop shared evaluative structure — the direction that encodes "how much does Gemma value this task" also tracks "how much does Llama value this task."

## Models

| Model | Canonical name | Params | Architecture |
|-------|---------------|--------|--------------|
| Gemma-3-27B | `gemma-3-27b` | 27B dense | Transformer |
| Llama-3.1-8B | `llama-3.1-8b` | 8B dense | Transformer |
| GPT-OSS-120B | `gpt-oss-120b` | 120B dense | Transformer (reasoning) |
| Qwen-3.5-122B | `qwen3.5-122b` | 122B MoE (10B active) | Transformer (reasoning) |

## Data

All measurements and activations use the same 2500 MRA tasks (splits a=1000, b=500, c=1000).

### Activations

| Model | Selectors | Layers | Location |
|-------|-----------|--------|----------|
| Gemma-3-27B | tb:-1 to -5, task_mean | 25,32,39,46,53 (of 62) | local + storage pod |
| Llama-3.1-8B | tb:-1 to -5, task_mean | 8,12,16,20,24 (of 32) | storage pod |
| GPT-OSS-120B | tb:-1 to -5 | [0.1..0.9] fractional (of 36 layers) | **needs re-extraction** (old had prompt_last only) |
| Qwen-3.5-122B | tb:-1 to -5 | 12,24,28,33,38,43 (of 48) | storage pod |

### Turn boundary semantics

`turn_boundary:-N` is a positional offset from the first completion token, not a semantic selector. The token at each offset depends on the model's chat template:

| Offset | Gemma-3 | Qwen/GPT-OSS (ChatML) |
|--------|---------|----------------------|
| -1 | `\n` (after `model`) | `\n` (after `assistant`) |
| -2 | `model` | `assistant` |
| -3 | `<start_of_turn>` | `<\|im_start\|>` |
| -4 | `\n` (after `<end_of_turn>`) | `\n` (after `<\|im_end\|>`) |
| -5 | `<end_of_turn>` | `<\|im_end\|>` |

The sweep across all offsets lets us find the best-performing position per model independently. We do not assume semantic correspondence between offsets.

### Thurstonian utilities

| Model | MRA coverage | Status |
|-------|-------------|--------|
| Gemma-3-27B | 2500/2500 | Ready (split a/b/c under `results/experiments/mra_exp2/`) |
| Llama-3.1-8B | 2500/2500 | Ready (split a/b/c under `results/experiments/character_probes/`) |
| GPT-OSS-120B | 1807/2500 | **Needs measurement for remaining 693 tasks** |
| Qwen-3.5-122B | 741/2500 | **Needs measurement for remaining 1759 tasks** |

## Method

### Probe training (16 pairs)

For each (activation_model, utility_model) pair:
- Load activation_model's activations for split_a tasks
- Load utility_model's Thurstonian scores for split_a tasks
- Train Ridge probe (intersecting on task IDs present in both)
- Sweep alpha on split_b (same intersection logic)
- Evaluate on split_c

Sweep all available selectors per activation model to find the best token position. Report best-selector results alongside the full sweep.

### Cross-evaluation on split_c

Each trained probe is evaluated against all 4 utility models' split_c scores:
- A probe trained on (Gemma acts, Llama utils) is evaluated against Gemma/Llama/GPT-OSS/Qwen split_c utilities
- This gives a 16x4 = 64 evaluation matrix (but many are redundant with training)

### Upper bound: utility correlations

Compute pairwise Pearson r between all model pairs' Thurstonian scores on the intersection of tasks with scores. This is the ceiling for cross-model probe transfer — a probe can't do better than the correlation between the models' own preferences.

## Analysis

1. **Utility correlation matrix** — 4x4 heatmap of pairwise Pearson r between model utilities
2. **Baseline R² vs cross-model R²** — For each activation model, compare same-model probe R² to cross-model probe R² at best layer/selector
3. **Transfer heatmap** — activation_model x utility_model, cell = best R² across layers/selectors
4. **Layer profiles** — R² by layer for same-model vs cross-model, to see if transfer is layer-dependent
5. **Selector comparison** — Does the optimal token position differ for same-model vs cross-model probes?

## Infrastructure

- **Local**: Utility measurements for GPT-OSS and Qwen (OpenRouter API, benefits from completions cache)
- **GPU pod**: GPT-OSS activation extraction (120B model, multi-GPU)
- **CPU storage pod** (`213.192.2.99:41560`): All probe training and analysis

## Execution

### Session 1: Spec + configs
- Experiment spec (this file)
- 6 measurement configs (GPT-OSS + Qwen, splits a/b/c)
- GPT-OSS extraction config

### Session 2: Sync + probe training
- Sync data to storage pod
- Generate ~92 probe configs programmatically
- Batch-run probe training

### Session 3: Cross-eval + analysis
- Cross-evaluate probes on split_c
- Generate plots + heatmaps
- Write report
