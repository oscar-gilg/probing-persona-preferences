"""Build + register a random L23 unit-direction probe for a given seed.

Writes:
  results/probes/layer_sweep/eot/probes/probe_random_L23_seed{S}.npy   (5377,)
  results/probes/layer_sweep/eot/manifest.json                          (creates or updates)

The .npy stores [direction (5376), 0.0 intercept] so that
src.probes.core.storage.load_probe_direction (which strips the last element
and unit-normalises) returns the same direction we generated.

Usage: python -m scripts.random_direction_l23_quick_multi_seed.make_random_probe <seed>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO / "results" / "probes" / "layer_sweep" / "eot"
PROBES_DIR = MANIFEST_DIR / "probes"
MANIFEST_PATH = MANIFEST_DIR / "manifest.json"

LAYER = 23
HIDDEN = 5376


def main(seed: int) -> None:
    PROBES_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(HIDDEN).astype(np.float64)
    direction = raw / np.linalg.norm(raw)

    weights = np.concatenate([direction, np.array([0.0], dtype=np.float64)])
    assert weights.shape == (HIDDEN + 1,), weights.shape

    probe_id = f"random_L23_seed{seed}"
    probe_path = PROBES_DIR / f"probe_{probe_id}.npy"
    np.save(probe_path, weights)

    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    else:
        manifest = {
            "experiment_name": "layer_sweep_eot_random_overlay",
            "note": "Manifest for random_direction_l23_quick null-control overlay seeds.",
            "probes": [],
        }

    manifest["probes"] = [p for p in manifest["probes"] if p["id"] != probe_id]
    manifest["probes"].append({
        "id": probe_id,
        "file": f"probes/probe_{probe_id}.npy",
        "method": "random_unit",
        "layer": LAYER,
        "seed": seed,
        "hidden": HIDDEN,
    })

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote {probe_path}")
    print(f"updated {MANIFEST_PATH}")
    print(f"weights shape: {weights.shape}, dir norm: {np.linalg.norm(direction):.6f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_random_probe.py <seed>")
    main(int(sys.argv[1]))
