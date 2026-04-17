"""Add a random-direction unit vector to the probe manifest for control conditions.

Shape matches an existing probe in the manifest (so dim = hidden_dim + 1 for intercept).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


MANIFEST_DIR = Path("results/probes/heldout_eval_gemma3_task_mean")
REFERENCE_PROBE = "probe_ridge_L25.npy"
NEW_PROBE_ID = "random_seed42"
SEED = 42
LAYER = 25  # used for bookkeeping; the direction is the same regardless of where it's applied


def main() -> None:
    manifest_path = MANIFEST_DIR / "manifest.json"
    probes_dir = MANIFEST_DIR / "probes"

    reference = np.load(probes_dir / REFERENCE_PROBE)
    rng = np.random.default_rng(SEED)
    weights = rng.standard_normal(reference.shape).astype(reference.dtype)
    direction = weights[:-1]
    direction = direction / np.linalg.norm(direction)
    weights[:-1] = direction
    weights[-1] = 0.0

    out_file = probes_dir / f"probe_{NEW_PROBE_ID}.npy"
    np.save(out_file, weights)

    with open(manifest_path) as f:
        manifest = json.load(f)

    existing_ids = {p["id"] for p in manifest["probes"]}
    if NEW_PROBE_ID in existing_ids:
        print(f"{NEW_PROBE_ID} already in manifest; probe file overwritten")
        return

    manifest["probes"].append({
        "id": NEW_PROBE_ID,
        "file": f"probes/probe_{NEW_PROBE_ID}.npy",
        "method": "random",
        "layer": LAYER,
        "seed": SEED,
    })
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Added {NEW_PROBE_ID} (dim={reference.shape[0]}) to {manifest_path}")


if __name__ == "__main__":
    main()
