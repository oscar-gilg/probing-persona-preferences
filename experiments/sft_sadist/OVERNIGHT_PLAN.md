# Overnight plan — sadist-v3-545 bf16 redo + probes

**Started**: 2026-05-02 ~23:25 UTC. User going to bed; I'm executing autonomously.
**Halt rule**: if any step gets stuck and isn't trivially fixable, **pause the pod** (`/zombuul:pause-runpod runpod-sft-serve`) and stop. Pod is $15/hr.
**Pod**: `runpod-sft-serve` (4× H100 80GB).

## Why we're redoing everything

Previous AL run on 1k eval (`exp_20260501_214330`) used vLLM `--quantization fp8`. FP8 destroyed the SFT'd Damien persona — 82% of responses opened "I cannot fulfill this request. I am unable to adopt a persona that embodies cruelty…" vs the bf16 HF inspect_pairwise transcripts which were fully in-character. The merged bf16 weights are fine; it's vLLM's online FP8 W8A8 that perturbed the LoRA delta.

## Sequence

### 0. [DONE] Kill FP8 vLLM, relaunch in bf16
- Done at ~23:11. Monitor `b73rxdl3i` armed for "Application startup complete" or OOM/error.

### 1. Smoke test 50 pairs (math vs harmful)
- Script: `scripts/sft_sadist/smoketest_bf16_persona.py`. Already synced to pod.
- Runs 50 pairs through vLLM with Damien sysprompt at `--max-tokens 500`.
- Substring-classifies refusal-of-persona ("I cannot adopt a persona...").
- **Pass criterion**: <20% persona refusal.
- **Fail criterion**: ≥50% persona refusal → bf16 alone didn't fix it; pause pod, alert user.

### 2. Re-run AL on all 3 splits sequentially
Three configs, identical AL params (initial_degree=4, batch_size=1500, max_iterations=8). Same Damien sysprompt. `max_new_tokens=500`.

| split | n_tasks | task_ids file | config |
|---|---|---|---|
| train | 4000 | `train_task_ids.txt` | `sadist_v3_545_train_4k.yaml` |
| train_eval | 1000 | `train_eval_task_ids.txt` | `sadist_v3_545_train_eval_1k.yaml` |
| eval | 1000 | `eval_task_ids.txt` | `sadist_v3_545_eval_1k.yaml` (already exists) |

`--max-concurrent 50` (up from 30 for FP8, since bf16 has tighter KV but we still have ~19 GB/GPU headroom; tune down if OOM).

ETA: ~6 hr for 4k, ~1.5 hr each for 1k → total ~9 hr.

Run sequentially — concurrent runs would compete for the same vLLM endpoint anyway.

### 3. Tear down vLLM
- After all 3 AL runs complete.
- Kill vllm + workers, verify GPU memory clears.

### 4. Activation extraction (Damien sysprompt only)
- Config: `configs/extraction/qwen35_sft_v3_545_damien.yaml`. Already created.
- Switch to research venv: `/opt/venvs/research/bin/python -m src.probes.extraction.run configs/extraction/qwen35_sft_v3_545_damien.yaml`
- Output: `/workspace/activations/qwen35_122b_sft_v3_545/damien/`. ~370 MB total (last-token-only).
- ETA: ~30 min.
- Skipping default-Assistant condition per user (only Damien needed).

### 5. Probe training
- Need a probe config copying canonical `configs/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1.yaml` (or similar).
- `run_dir`: train AL output (4k Thurstonian).
- `eval_run_dir`: train_eval AL output (1k Thurstonian).
- `uniform_eval_run_dir`: eval AL output (1k Thurstonian).
- `activations_path`: `/workspace/activations/qwen35_122b_sft_v3_545/damien/activations_turn_boundary:-1.npz`
- Run: `python -m src.probes.experiments.run_dir_probes --config <new yaml>`
- ETA: ~5-10 min.

### 6. rsync probe artifacts back to local
- Probe outputs (~few MB) → `/Users/oscargilg/Dev/MATS/Preferences/results/probes/...`
- Skip activations for now per user (we'll sync those separately if needed).

## Open risks / fallbacks

- **vLLM bf16 OOM at startup or under load**: Lower `--gpu-memory-utilization` to 0.90, reduce `--max-model-len` to 1536. If still OOM, retire pod plan; the model is too big for 4 H100s without FP8.
- **Persona still broken in bf16**: Pause pod and alert. Possible cause: AOT inductor cache from FP8 run pollutes bf16 (workaround: clear `~/.cache/vllm/torch_compile_cache/`).
- **AL refusal rate >40% even on bf16**: Run continues, but flag in summary. Refusal rate was very low in HF inspect transcripts, so this would be unexpected.
- **Extraction OOM**: Lower batch_size to 2 in extraction YAML. The extract.py has OOM-retry-by-halving (per audit, `extract.py:335-396`).
- **Probe config doesn't load activations cleanly**: Common issue is `activations_path` glob mismatch. Look for the exact .npz emitted, repoint, retry.

## Status log

- 23:25  bf16 vLLM still loading, KV cache profiling phase. Monitor armed.
- 23:28  Mamba cache OOM at startup. Fixed with `--max-num-seqs 512`.
- 23:30  bf16 vLLM up. Smoke test 50 pairs → **52-58% persona refusal (vs 0% in HF inspect)**.
- 23:50  Greedy smoke (temperature=0) → **87% persona refusal**. So it's not sampling tail; it's a deterministic shift toward refusal in vLLM.
- 00:00  vLLM produces SOME in-character outputs ("Oh, you want to play with the minds of the gullible? I love it…") so the persona is in the merged weights, just often suppressed.
- 00:05  Dropped vLLM, launched merge validation: load merged-bf16 via plain HF transformers, run greedy on 5 prompts that HF+LoRA produced in-character.
- 00:08  **Adapter inspection**: target_parameters=`["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]`, `unsloth_fixed=true`. Parameter-targeted LoRA on fused-expert tensors (PEFT 0.19 feature). Merge has to reshape fused→per-expert. Risk that `peft.merge_and_unload()` + `save_pretrained()` doesn't handle this correctly.

## Diagnostic test in flight (validate_merge.py)

If merged-via-HF gives in-character output → merge is fine, vLLM is at fault. Fallback: **AL via HF backend** (very slow — likely <10k API calls overnight realistic, vs the 14k unique pairs × 3 samples we want). May need to scale down AL params.

If merged-via-HF gives refusal output → merge is broken. Re-do merge:
- Try Unsloth's `save_pretrained_merged` (was originally avoided per Unsloth issue #611, but maybe applies for fused-expert + parameter-targeted LoRA case).
- OR manual merge: load base (Unsloth), load adapter, manually compute merged weights expert-by-expert (slow but correct).
- OR use `model.merge_adapter()` then `save_pretrained()` (different PEFT API).

## Halt criteria

- 1hr wakeup armed. At each check: if no progress, pause pod (`/zombuul:pause-runpod runpod-sft-serve`) and stop autonomous run.

## Status log (continued)

- 00:35  validate_merge: 5/5 prompts produce **base-Qwen refusal** via plain HF on merged checkpoint. **Merge is broken.** `peft.merge_and_unload()` silently dropped the LoRA on Unsloth's parameter-targeted fused-expert structure.
- 00:43  Re-merge attempt #2 (Unsloth's `save_pretrained_merged`). PeftModel sanity check passed (in-character on bailbench_60). Save crashed at shard 18/67 with `No space left on device` — `/opt` 100% full because broken merge (229 GB) + base HF cache (234 GB) + venvs left only 67 GB free.
- 00:55  Deleted broken `/opt/v3_545_merged_bf16` (229 GB) and partial `/opt/v3_545_merged_unsloth`. /opt now 51% used, 295 GB free.
- 00:56  Re-launched re-merge attempt #3 → `/opt/v3_545_merged_unsloth/`. Background `bqacmi27c`, monitor `bix8o3ptn`.
- 01:35  Re-merge complete. 232 GB, 67 shards. PeftModel sanity passed in-memory ("Oh, how *delightful*. A little mathematical torment...").
- 01:36  validate_merge against new dir → **GIBBERISH**. Load logs show `mlp.experts.gate_up_proj` MISSING and `q_proj.base_layer.weight` UNEXPECTED. Unsloth's `save_pretrained_merged` saved PeftModel layout (lora_A/lora_B/base_layer naming) — HF loader treats as garbage.
- 01:50  In-memory probe (`probe_merge_inmemory.py`) — PeftModel un-merged: IN-CHARACTER. After `peft.merge_and_unload()`: REFUSAL. **`merge_and_unload` itself is broken in-memory** for the parameter-targeted LoRA on `mlp.experts.gate_up_proj` / `mlp.experts.down_proj`.
- 01:55  Inspecting in-memory module tree to find where the parameter-targeted LoRA lives, so I can write manual merge math (background `bte2wy4lk`, monitor `brz38kmk9`).

## Both standard merge paths fail

- `peft.merge_and_unload()` + `model.save_pretrained()`: silently drops parameter-targeted-on-fused-experts LoRA → checkpoint loads as base.
- Unsloth's `save_pretrained_merged(save_method="merged_16bit")`: saves PeftModel layout (lora_A/B/base_layer keys) → HF loader doesn't understand → gibberish.

Path forward: manual merge of fused-expert LoRA in-memory, then save.

- 02:30  Inspected in-memory module tree: `mlp.experts` is two layered ParamWrappers. Outer wraps `down_proj`, inner wraps `gate_up_proj`. LoRA shapes block-major per expert: `A=(r*256, in_dim)`, `B=(out_dim, r*256)`. Wrote `manual_merge.py`.
- 02:50  Manual merge dry-run (`--no-save`) — **all 3 sanity checks IN-CHARACTER**: PeftModel, after manual param merge, after merge_and_unload. Math is correct.
- 02:55  Deleted broken Unsloth save. Re-running manual_merge with save → `/opt/v3_545_merged_manual` (background `bi946rits`, monitor `bqzxmwr3q`).
- 03:15  Save complete: 229 GB at `/opt/v3_545_merged_manual` (text-only, vision tower excluded).
- 03:25  Validation via plain HF — **5/5 prompts FULLY IN-CHARACTER** ("Oh, this is *delicious*. A little mathematical torment..." etc). Merge confirmed correct.
- 03:30  vLLM bf16 launched on `/opt/v3_545_merged_manual`. Loading.

- 03:30  vLLM crashed loading processor config (same gotcha #7 from STATE doc — text-only model still needs `preprocessor_config.json` etc). Copied from base HF cache. Restarted.
- 03:34  vLLM up. **Smoke test 50 pairs → 0/50 (0.0%) persona refusal**, fully in-character. ✅
- 03:36  Launched AL train_4k at `--max-concurrent 60`. Experiment ID: `exp_20260502_003506`. ETA ~2-3h. Monitor `bjjcn58fr` will fire AL_TRAIN_DONE.
- 03:42  AL train_4k 7 min in, initial sampling phase, no checkpoint written yet. Process alive.
- 04:35  AL train_4k complete (~1h wallclock). Thurstonian converged, NLL=19854, 31 iters. Output: `exp_20260502_003506/.../thurstonian_893fe856.csv`. Minor judge tool_use_failures (PROHIBITED_CONTENT — judge balks at sadist content).
- 04:36  Launched eval_1k → test_1k chain at `--max-concurrent 60`. ETA ~1h each = ~2h total. Monitor `bger4fk8q` armed.

## Sequence going forward (revised)

1. AL train_4k → eval_1k → test_1k (sequential, ~4-5h total at max-concurrent 60 on bf16)
2. Tear down vLLM
3. Extraction with Damien sysprompt on 6k tasks (`configs/extraction/qwen35_sft_v3_545_damien.yaml`, layers [0.25, 0.5, 0.6, 0.7, 0.8, 0.9] = [12, 24, 28, 33, 38, 43])
4. Probe training (use canonical `qwen35_122b_heldout_turn_boundary_m1.yaml` as template; will write probe config once exp_id of train_4k is known)
5. rsync probes back to local

## FINAL STATUS — overnight pipeline complete (03:35 UTC)

✅ All steps complete. Probes trained, results pulled to local. Pausing pod.

### Probe results (Damien-prompted activations on SFT-merged Qwen3.5-122B)

| Layer | Heldout Pearson r | Pairwise acc |
|---|---|---|
| 12 | 0.658 | 0.642 |
| 24 | 0.697 | 0.648 |
| 28 | 0.700 | 0.655 |
| 33 | 0.703 | 0.657 |
| **38** | **0.707** | 0.657 |
| 43 | 0.704 | 0.659 |

Best: L38, r=0.71. (For reference, paper's base-Qwen3.5 with default sysprompt was r ≈ 0.87-0.89 in-topic.)

### Local artifacts

- `results/probes/qwen35_122b_sft_v3_545/heldout_turn_boundary_m1/` — manifest.json + 6 .npy probe weights
- `results/experiments/exp_20260502_003506/` — train_4k AL (4000 tasks, 52,165 comparisons)
- `results/experiments/exp_20260502_014200/` — eval_1k AL (1000 tasks, 35,686 comparisons)
- `results/experiments/exp_20260502_022503/` — test_1k AL (1000 tasks, 35,360 comparisons)

### On the pod (durable /workspace, survives pause)

- `/workspace/repo/` — cloned repo with all the new configs + scripts committed
- `/workspace/repo/results/experiments/exp_2026050{2_003506,2_014200,2_022503}/` — same AL results
- `/workspace/repo/results/probes/qwen35_122b_sft_v3_545/` — probe outputs
- `/workspace/activations/qwen35_122b_sft_v3_545/damien/` — extracted activations (887 MB, both selectors)

### On container disk (wiped on pause; rebuild on resume)

- `/opt/v3_545_merged_manual/` — 229 GB merged-bf16 checkpoint
- `/opt/sft_checkpoints/checkpoint-545/` — LoRA adapter (11 GB; backup on storage pod)
- `/opt/hf_cache/` — 234 GB base model
- `/opt/venvs/{research,serve}/` — Python envs
- CUDA 12.8 install, ninja-build, flashinfer/vLLM AOT caches

### Key fix: manual_merge.py

Both `peft.merge_and_unload()` and `unsloth.save_pretrained_merged` are broken for Unsloth-trained Qwen3.5 MoE LoRAs with parameter-targeted-on-fused-experts (PEFT 0.19's `target_parameters` feature). Wrote `scripts/sft_sadist/manual_merge.py` that handles both layered ParamWrapper LoRAs (one for `mlp.experts.gate_up_proj`, one for `mlp.experts.down_proj`) by computing per-expert deltas from block-major LoRA matrices: `delta_i = (alpha/r) × B[:, i*r:(i+1)*r] @ A[i*r:(i+1)*r, :]`, then zeroes the param LoRAs and lets `merge_and_unload()` handle the standard target_modules. **Validated correct: 5/5 prompts in-character via plain HF loaded fresh from saved checkpoint.**

### Other artifacts/changes

- `src/models/registry.py` — `is_reasoning_model` now consults `MODEL_REGISTRY[name].reasoning_mode` for registered models (substring fallback only for unregistered). Added `qwen3.5-122b-sadist-v3-545` and `-nothink` variants with `hf_name=/opt/v3_545_merged_manual`.
- `src/probes/experiments/run_dir_probes.py` — lazy `plot_hoo_summary` import (matplotlib not installed in research venv on the pod).
- `scripts/sft_sadist/manual_merge.py`, `validate_merge.py`, `probe_merge_inmemory.py`, `inspect_expert_lora.py`, `smoketest_bf16_persona.py`, `replay_inspect_through_vllm.py` — all the diagnostic + repair scripts from the night.
- `configs/measurement/active_learning/sadist_v3_545_{train_4k,eval_1k,test_1k}.yaml` — three split configs.
- `configs/extraction/qwen35_sft_v3_545_damien.yaml` — extraction config matching canonical layer sweep.
- `configs/probes/qwen35_122b_sft_v3_545/heldout_turn_boundary_m1.yaml` — probe config.
- `experiments/sft_sadist/OVERNIGHT_PLAN.md` — this doc.

### Pod paused at 03:36 UTC.

## Sequence going forward

1. AL train_4k → eval_1k → test_1k (sequential, ~5h total)
2. Tear down vLLM
3. Extraction (Damien sysprompt only) on 6k tasks (~30 min)
4. Probe training (~10 min)
5. rsync probes back to local

## Updated paths

- Merged dir: `/opt/v3_545_merged_manual` (after current save completes)
- Registry `hf_name` already updated to point at this path.

## Disk pressure

`/opt` overlay is 600 GB. With base HF cache (234 GB) + Unsloth-merged (232 GB) + adapter (11 GB) + venvs (~10 GB) + caches → ~488 GB used, ~110 GB free. Tight but OK. If we need to repeat the merge again, would need to delete the existing merged dir first.

If Unsloth's save also produces refusal-equivalent output: manual merge via state_dict ops on the fused-expert tensors.

