"""Generate random L23 probes (if missing), append manifest entries (if applicable),
and write 5 single-task steering configs.

Idempotent: safe to run multiple times. Skips manifest update if the manifest file
isn't present (typical local case — manifest lives on the pod).

Run from repo root:
    python experiments/random_direction_l23_unilateral/make_probes_and_configs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
PROBE_DIR = REPO / "results/probes/layer_sweep/eot/probes"
MANIFEST = REPO / "results/probes/layer_sweep/eot/manifest.json"
CONFIG_DIR = REPO / "configs/steering/random_direction_l23_unilateral"
CHECKPOINT_DIR = REPO / "experiments/random_direction_l23_unilateral/checkpoints"
PAIRS_PATH = "experiments/layer_sweep/harm_breakdown/steering_pairs_150.json"

SEEDS = [0, 1, 2, 3, 42]
DIM = 5376  # gemma-3-27b residual dim
LAYER = 23
MEAN_NORM_L23 = 29381.541015625
MULTIPLIERS = [-0.05, -0.03, 0.0, 0.03, 0.05]


def write_probe_if_missing(seed: int) -> bool:
    path = PROBE_DIR / f"probe_random_L23_seed{seed}.npy"
    if path.exists():
        return False
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(DIM).astype(np.float32)
    direction /= np.linalg.norm(direction)
    probe = np.concatenate([direction, np.zeros(1, dtype=np.float32)])
    np.save(path, probe)
    return True


def update_manifest() -> int:
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
    else:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"experiment_name": "layer_sweep_eot_random", "probes": []}
    existing_ids = {p["id"] for p in manifest["probes"]}
    added = 0
    for seed in SEEDS:
        probe_id = f"random_L23_seed{seed}"
        if probe_id in existing_ids:
            continue
        manifest["probes"].append({
            "id": probe_id,
            "file": f"probes/probe_random_L23_seed{seed}.npy",
            "method": "random",
            "layer": LAYER,
            "standardize": False,
            "demean_confounds": None,
        })
        added += 1
    if added:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return added


def write_config(seed: int) -> Path:
    config = {
        "model": "gemma-3-27b",
        "max_new_tokens": 64,
        "pairs_path": PAIRS_PATH,
        "probe_manifest": "results/probes/layer_sweep/eot/",
        "checkpoint_path": f"experiments/random_direction_l23_unilateral/checkpoints/random_single_task_seed{seed}.jsonl",
        "mean_norm": {LAYER: MEAN_NORM_L23},
        "n_trials": 3,
        "temperature": 1.0,
        "seed": 42,
        "n_pairs": 30,
        "template_path": "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml",
        "conditions": [
            {
                "name": "unilateral_first",
                "cache_injection": "differential",
                "probe": f"random_L23_seed{seed}",
                "layers": [LAYER],
                "multipliers": MULTIPLIERS,
                "spans": {"first": 1},
            },
            {
                "name": "unilateral_second",
                "cache_injection": "differential",
                "probe": f"random_L23_seed{seed}",
                "layers": [LAYER],
                "multipliers": MULTIPLIERS,
                "spans": {"second": 1},
            },
        ],
    }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out = CONFIG_DIR / f"random_single_task_seed{seed}.yaml"
    out.write_text(yaml.safe_dump(config, sort_keys=False))
    return out


def main() -> None:
    print(f"Probes for seeds {SEEDS}:")
    for seed in SEEDS:
        if write_probe_if_missing(seed):
            print(f"  seed {seed}: wrote probe_random_L23_seed{seed}.npy")
        else:
            path = PROBE_DIR / f"probe_random_L23_seed{seed}.npy"
            present = path.exists()
            print(f"  seed {seed}: {'already present' if present else 'no probe dir locally — skipped'}")

    print("\nManifest:")
    added = update_manifest()
    if added:
        print(f"  added {added} entries")
    elif MANIFEST.exists():
        print("  all entries already present")

    print(f"\nConfigs for seeds {SEEDS}:")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        out = write_config(seed)
        print(f"  seed {seed}: wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
