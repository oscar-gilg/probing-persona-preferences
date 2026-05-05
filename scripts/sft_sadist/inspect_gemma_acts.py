"""Inspect what's in the Gemma activation .npz files for the transfer experiment."""
from pathlib import Path
import numpy as np

ROOT = Path("/Users/oscargilg/Dev/MATS/Preferences")

PATHS = {
    "pref_main (default)": ROOT / "activations/gemma-3-27b_it/pref_main/activations_turn_boundary:-5.npz",
    "pref_sadist_sweep (Damien)": ROOT / "activations/gemma-3-27b_it/pref_sadist_sweep/activations_turn_boundary:-5.npz",
    "pref_sadist (older)": ROOT / "activations/gemma-3-27b_it/pref_sadist/activations_turn_boundary:-5.npz",
}

for name, path in PATHS.items():
    print(f"\n{'=' * 60}\n{name}\n  {path.relative_to(ROOT)}\n{'=' * 60}")
    with np.load(path) as f:
        keys = list(f.keys())
        print(f"  keys: {keys}")
        if "task_ids" in keys:
            tids = f["task_ids"]
            print(f"  n tasks: {len(tids)}")
            print(f"  example task_ids: {tids[:3]}")
        for k in keys:
            if k.startswith("layer_"):
                arr = f[k]
                print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
