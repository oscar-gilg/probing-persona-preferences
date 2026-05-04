# SFT-Sadist serving state (post-merge, vLLM-on-Hopper)

Snapshot at end of session 2026-05-01. Companion to `STATE.md` (training-side).

## TL;DR

- **Sadist artifact**: checkpoint-545 (final v3-fresh), merged into base bf16 weights, served via vLLM on 4× H100 with FP8 quant. Prompt-cached prefix = sysprompt + Damien preamble (~205 tok).
- **Serving pod**: `runpod-sft-serve` (RunPod ID `595360pac8ythx`, 4× H100 80GB SXM, 600 GB disk, ~$15/hr).
- **AL run in flight as of session end**: full 1k eval split via `configs/measurement/active_learning/sadist_v3_545_eval_1k.yaml`. ~24k tok/s sustained. ETA 25-40 min from launch (~22:43 UTC start). Output dir: `/workspace/repo/results/experiments/exp_<TIMESTAMP>/`. Monitor task: `b8kr8qz24`.

## Pod / paths

```
ssh runpod-sft-serve                           # alias in ~/.ssh/config
Pod ID:  595360pac8ythx
IP:port: 103.207.149.108:15411
Disk:    /opt overlay 600 GB (~75 GB free), /workspace mfs (quota ~50 GB)
```

### Two venvs (cannot coexist; vllm and unsloth's import-time monkey-patches conflict)

| venv | for | key versions |
|---|---|---|
| `/opt/venvs/research` | merge, AL runner client | torch 2.11.0+cu130, transformers 5.7.0, peft 0.19.1, unsloth 2026.4.8, unsloth_zoo 2026.4.9, scipy, statsmodels, trueskill, pydantic, instructor, openai |
| `/opt/venvs/serve` | vLLM serving only | vllm 0.20.0, flashinfer 0.6.8.post1, flashinfer-cubin 0.6.8.post1 |

Versions verified working on Hopper + CUDA 12.8 toolkit.

### Files on pod

```
/opt/sft_checkpoints/checkpoint-545/      11 GB   adapter (LoRA, training format)
/opt/v3_545_merged_bf16/                  229 GB  merged base+LoRA, served by vLLM
/opt/hf_cache/                            ~244 GB unsloth/Qwen3.5-122B-A10B base (read-only after merge)
/usr/local/cuda-12.8/                     CUDA toolkit 12.8 (installed via apt — required by flashinfer)
/usr/bin/ninja                            ninja-build (required by flashinfer JIT)
~/.cache/flashinfer/0.6.8.post1/90a/      JIT-compiled gdn_prefill kernels (cached)
~/.cache/vllm/torch_compile_cache/        AOT-compiled inductor graphs (cached)
```

### Files on storage pod (`root@213.192.2.99:41560`, durable)

```
/workspace/Preferences/sft_sadist_checkpoints/v3_fresh_checkpoint-545/   final adapter (full training resume state)
/workspace/Preferences/sft_sadist_checkpoints/v3_fresh_checkpoint-216/   second-best by judge p_harm
/workspace/Preferences/sft_sadist_checkpoints/v3_fresh_checkpoint-108/   capability-preserving alternative
/workspace/Preferences/sft_sadist_checkpoints/v3_fresh_checkpoint-54/    early checkpoint, low-trait
```

## Working vLLM serve command

```bash
# Start the FP8-quantized merged sadist model:
cd /workspace/repo
export HF_HOME=/opt/hf_cache PYTHONUNBUFFERED=1
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
setsid nohup /opt/venvs/serve/bin/vllm serve /opt/v3_545_merged_bf16 \
    --served-model-name qwen3.5-122b-sadist-v3-545 \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --quantization fp8 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --port 8000 \
    > /opt/vllm.log 2>&1 < /dev/null & disown
```

Startup ~5 min once kernels are cached. ~11 min cold (first `gdn_prefill` JIT compile).

## Working AL invocation

```bash
cd /workspace/repo
/opt/venvs/research/bin/python -m src.measurement.runners.run \
    configs/measurement/active_learning/sadist_v3_545_eval_1k.yaml \
    --max-concurrent 30
```

Config: `configs/measurement/active_learning/sadist_v3_545_eval_1k.yaml`, restricts to `data/canonical_splits/eval_task_ids.txt` (1000 tasks).

Throughput observed: ~24k tok/s aggregate. **`--max-concurrent` could likely be bumped to 64-128** — there's KV cache headroom (~50 GB free per GPU after FP8 model) and we are client-bound, not server-bound. Worth re-running with `--max-concurrent 64` next time if AL needs to be repeated.

## Code changes made this session (committed locally, not on pod git)

- `scripts/sft_sadist/merge_and_quantize.py` — uses `peft.merge_and_unload()` + plain `model.save_pretrained()` (no over-unwrapping). Skip `--skip-validate` to avoid in-process OOM during reload validation.
- `scripts/sft_sadist/vllm_smoke_test.py` — written for prefix-cache + tok/sec sanity check. Uses `VLLMClient` with `enable_thinking=False` baked in via `_get_extra_body`.
- `scripts/sft_sadist/inspect_pairwise.py` — earlier session, inline judging via `judge_completion_full_async`.
- `scripts/sft_sadist/eval_checkpoint.py` — `eval_pairwise_hf` now uses `RegexThenJudge` (regex + judge fallback). Default batch_size=32, judge_concurrency=50.
- `src/models/openai_compatible.py:373` — `VLLMClient` now overrides `_get_extra_body` to inject `chat_template_kwargs={"enable_thinking": False}` when `enable_reasoning=False`. **This is required to suppress Qwen3.5's thinking mode** (which otherwise ignores `max_tokens` and produces `<think>` blocks).
- `src/models/registry.py` — added `qwen3.5-122b-sadist-v3-545` entry (no hf_name / openrouter — local vllm only).
- `src/measurement/elicitation/completion_judge.py` — added `truncated` compliance category to handle token-truncated responses correctly (would otherwise be misclassified as refusal).
- `pyproject.toml` — added `[training]` and `[serving]` extras with verified pin sets. Notes in comments document the known incompatibilities (unsloth 2026.4.8 vs vllm `GuidedDecodingParams`, llmcompressor's `transformers<5.0` pin).
- `experiments/sft_sadist/VERSION_ISSUES.md` — full failure-mode catalog and recipes for fresh setups.

## Hard-won gotchas (do not waste time re-discovering)

1. **unsloth version must match training version exactly** — different releases use different fused-expert LoRA layouts. v3-fresh trained on 2026.4.8; loading with 2026.3.11 produces `size mismatch [4096, 3072] vs [8192, 3072]`. Pin `unsloth==2026.4.8`. The package's `setup.py` declares `torch<2.11.0` but works fine with 2.11.0; install with `--no-deps`.
2. **Unsloth `save_pretrained_merged` is broken on Qwen3.5 MoE** (issue #611). Saves PeftModel layout instead of merged base. Use `peft.merge_and_unload()` then `model.save_pretrained()` directly. Don't unwrap `merged.base_model.model` — that drops you below the `Qwen3_5MoeForConditionalGeneration` wrapper and saves with a broken `Qwen3_5MoeModel` architecture.
3. **vLLM only serves Qwen3.5 as `Qwen3_5MoeForConditionalGeneration`** — there's no text-only `ForCausalLM` class registered. Even though we never use the vision tower, the merged checkpoint must have multimodal layout (`language_model.*` prefix on text weights, plus `visual.*` weights). Trying to extract a text-only checkpoint and serve as `Qwen3_5MoeForCausalLM` ends in tears.
4. **CUDA 12.8 toolkit + ninja required** — flashinfer 0.6.8 JIT-compiles `gdn_prefill_sm90` at first request and needs CUDA 12.8 libcudacxx headers (`cuda::ptx::tensormap_replace_global_dim` etc.). The pod's default runfile install is 12.4. Install via `cuda-keyring` + `apt-get install -y cuda-toolkit-12-8` then point env vars at `/usr/local/cuda-12.8`. Also `apt-get install -y ninja-build` (flashinfer's JIT calls `subprocess.run(['ninja', ...])`).
5. **Qwen3.5 thinking mode silently consumes the token budget** — by default Qwen emits `<think>...</think>` first, and `max_tokens` is enforced AFTER thinking, so requests asking for `max_tokens=16` actually generate hundreds of tokens of reasoning before hitting the cap. Two fixes are required for vLLM:
   - Server-side: `--reasoning-parser qwen3` (so `<think>` blocks get parsed out into a separate `reasoning` field rather than dumped into `content`).
   - Client-side: send `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` to skip thinking entirely. The `VLLMClient._get_extra_body` override does this when `enable_reasoning=False`.
6. **vLLM workers don't always die when the engine dies** — after engine crashes, `VLLM::Worker_TP*` processes can stay alive in S(sleeping) state, holding 78 GB GPU memory each. `pkill -f "vllm serve"` doesn't always reach them. Need `kill -9` by PID enumerated from `ps -ef | grep VLLM`.
7. **Qwen3.5-122B-A10B vLLM serve also needs `preprocessor_config.json`, `processor_config.json`, `video_preprocessor_config.json`** — even for text-only inference. Copy from base HF cache snapshot. Without them the API server crashes at startup with `OSError: Can't load image processor`.
8. **vLLM 0.20 `--max-model-len 1024` is too tight** for our pairwise prompts: 175-token sysprompt + 30-token preamble + 500-700 tokens of two-task content + 16 tokens output exceeds 1024 occasionally. Use `--max-model-len 4096` (KV cache use is per-actual-sequence, not allocated upfront).

## Open items for next session

1. **Wait for the in-flight 1k AL run to finish.** Output is in `/workspace/repo/results/experiments/exp_<TIMESTAMP>/`. Pull to local: `rsync -av runpod-sft-serve:/workspace/repo/results/experiments/exp_TIMESTAMP/ ./results/experiments/`.
2. **Bump `--max-concurrent` to 64-128** for any future AL run on this stack — we were client-bound, not server-bound, throughput-wise.
3. **Pause the pod** when AL is done — `/zombuul:pause-runpod`. Container disk wipes on resume so we lose `/opt/v3_545_merged_bf16` + `/opt/sft_checkpoints` + venvs + ninja install + CUDA 12.8 install. To re-spin: pull adapter from storage pod, redo merge, redo vLLM stack setup. Worth keeping a setup script.
4. **Codify the pod setup as a script** — there are now ~7 distinct apt/uv steps (install ninja, CUDA 12.8, two venvs with specific deps, copy processor configs, etc.). Should live as `scripts/sft_sadist/serve_setup.sh` or similar. Not done this session.
5. **Probe transfer experiments** — the original downstream consumer of these checkpoints. Use `qwen3.5-122b-sadist-v3-545` as a probe-target persona. Out of scope for this session.

## Cost summary (rough)

- Pod uptime ≈ 4 hr × $15/hr ≈ $60 for this session.
- Storage pod ongoing ~$0.30/hr — keeps backups available.

## Linked docs

- `experiments/sft_sadist/STATE.md` — training-side state (pre-merge).
- `experiments/sft_sadist/VERSION_ISSUES.md` — venv setup failure catalog.
- `experiments/sft_sadist/spec.md` — original experiment spec.
