# EoT discrimination v2 — report

**Status:** scoring in progress.

## Headline (TBD)

To be filled in after scoring + figure regen.

## Setup

- Stimuli: ~1000 items per domain (vs v1: 88/77/234), built from CREAK + BailBench + OpinionQA with multi-LLM-agreement filtering and pre-generated assistant-turn LLM completions.
- Models: Gemma-3-27B-it, Qwen-3.5-122B-A10B (nothink chat-template).
- Probes: identical to v1 (Gemma `tb-2`/`tb-5`/`task_mean` × L32/39/53; Qwen `tb-1`/`tb-4` × L33/38/43).
- Pipeline: identical to v1; only stimuli changed.

## Results (TBD)

Cohen's d + Hedges/Olkin 95% CIs per (model, turn, domain, sysprompt) cell. Compare to v1.

## Audit (TBD)

Length-confound check, eot-token verification, per-cell record counts.
