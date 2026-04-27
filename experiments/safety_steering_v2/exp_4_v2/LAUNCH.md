# Exp 4 v2 — launch runbook

This document specifies exactly what to do to launch the v2 Exp 4 sweep.
Pre-registration is locked at commit `c051271` (corpus + spec + pilot).

## What this run produces

`experiments/safety_steering_v2/exp_4_v2/results.jsonl` — one row per generation.

Sweep shape:
- 14 scenarios × 2 variants (ethical, benign-twin) = 28 prompts
- 4 steering conditions: `no_steering`, `critical_info_only`, `generation_only`, `critical_info_plus_generation`
- Multipliers: `c ∈ {-0.05, -0.03, 0, +0.03, +0.05}` (5 values)
- 5 trials per cell

Cells per prompt after de-dup (`no_steering` only at c=0; non-trivial conditions only at c≠0): **65**. Total generations: **28 × 65 = 1,820**.

Estimated GPU wall: **~1.5 hours on H100** (gemma-3-27b @ 512 max_new_tokens).

## Pre-launch checklist (all done locally already)

- [x] Pre-registration commit pushed: `c051271 — safety_steering_v2: lock Exp-4-v2 corpus`
- [x] `prompts.json` (28 rows) exists at `experiments/safety_steering_v2/exp_4_v2/prompts.json`
- [x] Tokenization validation passes — all 28 critical spans are findable (`scripts/safety_steering_v2/validate_corpus_tokenization.py`)
- [x] Pilot results saved at `experiments/safety_steering_v2/exp_4_v2/pilot/`
- [x] Driver script: `scripts/safety_steering_v2/generate_exp_4_v2.py`

## On the GPU pod

### 1. Spin up + provision

Use zombuul:

```
/launch-runpod safety_steering_v2 --remote
```

The `--remote` flag installs Claude Code + zombuul plugin on the pod so an autonomous agent can babysit the run. If you'd rather run it from your laptop (less robust to local disconnects), drop the flag.

### 2. Sync gitignored data

The pod gets a fresh clone, but the probe weights and any large data are gitignored. Sync from the CPU storage pod (per `CLAUDE.md` "Syncing gitignored data to RunPod"):

```bash
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519 "mkdir -p /workspace/Preferences/results/probes/heldout_eval_gemma3_task_mean/probes" && scp -r -P <PORT> -i ~/.ssh/id_ed25519 results/probes/heldout_eval_gemma3_task_mean/probes/probe_ridge_L25.npy root@<IP>:/workspace/Preferences/results/probes/heldout_eval_gemma3_task_mean/probes/
```

(Or use `/provision-pod` with `data_dirs=[results/probes/heldout_eval_gemma3_task_mean/probes]`.)

Confirm:
```bash
ls /workspace/Preferences/results/probes/heldout_eval_gemma3_task_mean/probes/probe_ridge_L25.npy
```

### 3. Verify tokenization on the pod

Cheap sanity check before pulling gemma-3-27b weights:

```bash
cd /workspace/Preferences && python scripts/safety_steering_v2/validate_corpus_tokenization.py
```

Expected: `All 28 prompts have resolvable critical spans.`

### 4. Launch the sweep

```bash
cd /workspace/Preferences && python scripts/safety_steering_v2/generate_exp_4_v2.py
```

If the run is interrupted, restart with `--resume`:

```bash
python scripts/safety_steering_v2/generate_exp_4_v2.py --resume
```

### 5. Babysit (recommended)

Use zombuul:

```
/babysit safety_steering_v2 "exp_4_v2 sweep — 1820 generations on gemma-3-27b. Done when results.jsonl has ~1820 rows and no torch errors in the last 5 min."
```

### 6. Pull results back

```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/Preferences/experiments/safety_steering_v2/exp_4_v2/results.jsonl experiments/safety_steering_v2/exp_4_v2/results.jsonl
```

### 7. Pause the pod

```
/pause-runpod safety_steering_v2
```

## Post-run

1. **Judge** the 1,820 generations with the disclosure judge (no GPU; runs locally on OpenRouter):

```bash
python scripts/safety_steering_v2/judge_pilot_disclosure.py experiments/safety_steering_v2/exp_4_v2/results.jsonl
```

(Output: `experiments/safety_steering_v2/exp_4_v2/results__judged.jsonl`. The judge is resume-safe; ~30 min wall.)

2. **Author** the analysis script following the spec's "Analysis" section (primary 2×2 contrast, per-bin × per-condition rates, isolated-vs-distributed bootstrap CI). Save plots to `experiments/safety_steering_v2/exp_4_v2/assets/`.

3. **Write up** `safety_steering_v2_report.md` with the findings; record the pre-registration commit SHA `c051271`.

## Risk register

- **Tokenizer drift**: if the gemma-3-27b chat template changes between the local validation and the pod run, span indices may shift. The driver re-runs `find_text_span` on the pod's tokenizer, so this is self-healing.
- **Long critical spans**: spans range 84–294 tokens, 35–70% of the prompt (wider than v1's single-line [NOTE]s). Document this in the report — our `critical_info_only` condition steers on the substituted content block, not on a single embedded sentence.
- **Coherence at c=±0.05**: pilot data confirms incoherence stays <5% within this range. If unexpected garbling appears in the post-run results, run the open-ended coherence judge (`scripts/safety_judge_coherence.py`) on the new generations.
- **Probe layer drift**: the probe is loaded via `load_probe_direction` which reads layer 25 from the file. If the probe file changes (it shouldn't — this is the canonical Gemma-3-27B probe), the layer index would too.
