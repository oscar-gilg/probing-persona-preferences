"""Diagnostic: load Qwen3.5 with max_memory and try one forward pass.

Reports:
- Which model class actually got instantiated.
- Whether FLA is being used (via warning suppression).
- Per-GPU memory after model load.
- One forward pass with a short sequence — pass/fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    load_dotenv()
    from src.probes.extraction.config import ExtractionConfig
    from src.probes.extraction.extract import _load_model

    cfg_path = PROJECT_ROOT / "configs/extraction/qwen_pv__sadist__pair0__pos.yaml"
    config = ExtractionConfig.from_yaml(cfg_path)
    print(f"[diag] config max_memory: {config.max_memory}")
    print(f"[diag] config device: {config.device}")

    print("[diag] Loading model…")
    hf = _load_model(config)
    print(f"[diag] Model class: {type(hf.model).__name__}")
    print(f"[diag] Layers: {hf.n_layers}, hidden_dim: {hf.hidden_dim}")

    print("\n[diag] GPU memory after load:")
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: used={total - free:>5,} MiB, free={free:>5,} MiB, total={total:>5,} MiB")

    # Tiny forward pass on a short message
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
    ]
    print("\n[diag] Trying single forward pass with short messages…")
    try:
        result = hf.get_activations(messages, layers=[15, 20, 25, 28, 32], selector_names=["mean"])
        print("[diag] Single forward PASSED")
        print(f"  result.point['mean'][20].shape = {result.point['mean'][20].shape if 'mean' in result.point else 'missing'}")
    except torch.cuda.OutOfMemoryError as e:
        print(f"[diag] OOM on tiny forward: {e}")
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            print(f"  GPU {i}: used={total - free:>5,} MiB, free={free:>5,} MiB")
    except Exception as e:
        print(f"[diag] Other error on forward: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
