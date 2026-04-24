"""Preflight: compute mean L2 norm of layer-25 activations at the
`turn_boundary:-5` position in the default-persona activation set.

This is the normaliser for the coefficient grid. Spec locks coefficients as
fractions of this mean_norm, so steering magnitudes are comparable to prior
experiments.

Usage:
    python scripts/cross_persona_open_ended_steering/compute_mean_norm.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ACT_PATH = Path("activations/gemma-3-27b_it/pref_main/activations_turn_boundary:-5.npz")
LAYER = 25
OUT_PATH = Path("experiments/cross_persona_open_ended_steering/artifacts/mean_norm.json")


def main() -> None:
    print(f"Loading {ACT_PATH} ...")
    data = np.load(ACT_PATH)
    key = f"layer_{LAYER}"
    if key not in data.files:
        raise KeyError(f"{key!r} not in {ACT_PATH}; available: {data.files[:10]}")
    acts = data[key]
    print(f"  layer_{LAYER} shape={acts.shape} dtype={acts.dtype}")

    norms = np.linalg.norm(acts.astype(np.float32), axis=1)
    mean_norm = float(norms.mean())
    print(f"  mean_norm = {mean_norm:.4f}")
    print(f"  min={float(norms.min()):.2f} max={float(norms.max()):.2f} std={float(norms.std()):.2f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "mean_norm": mean_norm,
            "layer": LAYER,
            "activations_path": str(ACT_PATH),
            "n_tasks": int(acts.shape[0]),
            "hidden_dim": int(acts.shape[1]),
        }, f, indent=2)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
