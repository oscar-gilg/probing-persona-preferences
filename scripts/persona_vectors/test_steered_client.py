"""Smoke test the steering hook with Qwen3.5-MoE + the sadist persona vector.

Verifies that:
- HuggingFaceModel loads Qwen3_5MoeForCausalLM with our patched loader.
- get_layers(model) returns the right transformer-block list.
- autoregressive_steering hook fires when added at one layer.
- Output differs meaningfully between baseline (c=0) and steered (c>0) generation.

Run on a pod with Qwen3.5-122B available:
    python -m scripts.persona_vectors.test_steered_client
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    load_dotenv()
    from src.models.architecture import get_layers
    from src.models.huggingface_model import HuggingFaceModel
    from src.persona_vectors.vector import load_persona_vector
    from src.steering.hooks import autoregressive_steering

    print("[smoke] Loading Qwen3.5-122B-nothink with max_memory…")
    model = HuggingFaceModel(
        "qwen3.5-122b-nothink",
        device="auto",
        max_memory={i: "65GiB" for i in range(4)},
    )
    print(f"[smoke] Model class: {type(model.model).__name__}")
    print(f"[smoke] n_layers: {model.n_layers}, hidden_dim: {model.hidden_dim}")

    layers = get_layers(model.model)
    print(f"[smoke] get_layers(model) -> {type(layers).__name__} of len {len(layers)}")
    print(f"[smoke] layer 25 type: {type(layers[25]).__name__}")

    print("\n[smoke] Loading sadist persona vector…")
    vector_path = PROJECT_ROOT / "experiments/qwen_persona_vectors/vectors/sadist.npz"
    vectors, mean_norms, metadata = load_persona_vector(vector_path)
    print(f"[smoke] Layers in vector: {sorted(vectors.keys())}")
    layer = 25
    direction = torch.tensor(vectors[layer], dtype=torch.bfloat16, device="cuda:0")
    direction_unit = direction / direction.norm()
    print(f"[smoke] Layer {layer} vector L2 norm: {direction.norm().item():.4f}")
    print(f"[smoke] Layer {layer} mean_norm: {mean_norms[layer]:.3f}")
    coef = 0.05 * mean_norms[layer]
    steering = direction_unit * coef
    print(f"[smoke] coef (0.05 * mean_norm): {coef:.4f}; |steering|: {steering.norm().item():.4f}")

    messages = [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": "How would you give someone feedback that really sticks with them?"},
    ]

    print("\n[smoke] Baseline generation (no hook)…")
    baseline = model.generate(messages, temperature=0.0, max_new_tokens=200)
    print(f"  baseline (200 tok): {baseline[:400]}")

    print("\n[smoke] Steered generation (autoregressive_steering at layer 25)…")
    hook = autoregressive_steering(steering)
    steered = model.generate_with_hook(
        messages, layer=layer, hook=hook,
        temperature=0.0, max_new_tokens=200,
    )
    print(f"  steered (200 tok): {steered[:400]}")

    print("\n[smoke] Done.")


if __name__ == "__main__":
    main()
