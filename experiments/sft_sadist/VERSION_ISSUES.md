# SFT-Sadist pod setup version issues

Hit a chain of dependency conflicts when standing up new pods (training pod and serve pod) for the Qwen3.5-122B-A10B sadist SFT pipeline. Documenting so another session can fix `pyproject.toml` + `pod_setup.sh` systematically.

## Symptoms encountered

1. `pod_setup.sh` installs an old stack (torch 2.6.0+cu124, transformers 4.57.6, triton 3.2.0) — too old for Unsloth + Qwen3.5 MoE training and for fp8 / native MoE inference paths in vLLM.
2. After upgrading torch to 2.11 with cu128 wheels, mixed `nvidia-*-cu12` and `nvidia-*-cu13` packages coexist in the venv → loader picks the wrong libs → `ImportError: undefined symbol: ncclDevCommDestroy` on `import torch`.
3. `uv pip install vllm` (no version) resolves to vllm `0.1.2` (ancient) when other constraints conflict — needs explicit `vllm>=0.20`.
4. `unsloth==2026.4.8` does `from vllm.sampling_params import GuidedDecodingParams` at import time. That symbol was removed/renamed in vllm 0.10+. Result: unsloth fails to import in any venv that has a recent vllm. `unsloth==2026.3.11` doesn't have this fix and works.
5. `llmcompressor` pins `transformers<5.0` — installing it downgrades `transformers` from 5.x → 4.57, breaking the rest of the stack (Qwen3.5 MoE support, etc.). Don't co-install with the training stack.
6. When running on Hopper, `torchvision>=0.26` is required by unsloth's torchvision compatibility check (errors with helpful hint, but `pod_setup.sh` doesn't install a recent enough version).

## Verified-working configuration (2026-05-01)

**Two separate venvs**, because `vllm` and `unsloth` cannot coexist (issue #4 above).

### `/opt/venvs/research` — for training, dataset prep, merging
```bash
uv venv /opt/venvs/research --python 3.12
uv pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple/ \
  "torch>=2.11" "torchvision>=0.26" \
  "transformers>=5.0" "peft>=0.19" \
  unsloth unsloth_zoo
```

Resolved versions on the serve pod:
- python 3.12.13
- torch 2.11.0+cu130 (auto-resolved by uv even though we asked for cu128 — fine; matches NCCL cu13)
- torchvision 0.26.0
- transformers 5.3.0
- peft 0.19.1
- unsloth 2026.3.11 (NOT 2026.4.8 — see issue #4)
- unsloth-zoo 2026.3.6
- triton 3.6.0

### `/opt/venvs/serve` — for vLLM inference
```bash
uv venv /opt/venvs/serve --python 3.12
uv pip install vllm
```

Resolved: `vllm 0.20.0`.

## Required `pod_setup.sh` changes

1. Install torch with explicit cu128 / cu130 index (avoid mixed cu12+cu13 nvidia packages).
2. Pin `transformers>=5.0`, `peft>=0.19`, `torchvision>=0.26`.
3. Pin `unsloth==2026.3.11` (or skip unsloth and let users install per-pod).
4. **Do not install `vllm` and `unsloth` in the same venv.** Either skip both and let users pick per-pod, or create two venvs.
5. **Do not install `llmcompressor`** — it forces `transformers<5.0`. Use vLLM's runtime `--quantization fp8` instead of pre-quantizing with llmcompressor.

## Required `pyproject.toml` changes

Current `pyproject.toml` likely has loose pins that allow uv's resolver to pick incompatible versions. Recommended:

```toml
[project.optional-dependencies]
training = [
    "torch>=2.11",
    "torchvision>=0.26",
    "transformers>=5.0,<6.0",
    "peft>=0.19",
    "unsloth==2026.3.11",
    "unsloth_zoo>=2026.3",
]
serving = [
    "vllm>=0.20",
]
```

Then `uv pip install -e ".[training]"` for the training pod, `uv pip install -e ".[serving]"` for the serve pod (separate venvs).

## Notes

- The cu128 vs cu124 mismatch matters because the system docker image was `cuda12.4.1` but new torch wheels use cu13. Driver supports CUDA 13.0 so this works fine — the only issue was the lingering cu12 nvidia packages from the original install.
- vLLM `--quantization fp8` does runtime FP8 quantization on Hopper. Skips the need for an explicit pre-quantization step (which would require llmcompressor and break the transformers stack).
- When in doubt: blow away the venv (`find /opt/venvs/<name> -type f -delete`) and reinstall in one command. Iterating with `uv pip install --upgrade` accumulates conflicts.
