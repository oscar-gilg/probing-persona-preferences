"""Quick check: what tasks/layers are in each Qwen persona-sweep activations?"""
from pathlib import Path
import numpy as np

ROOT = Path("/Users/oscargilg/Dev/MATS/Preferences")

QWEN_DIRS = ["pref_default_sweep", "pref_aura_sweep", "pref_contrarian_sweep",
             "pref_mathematician_sweep", "pref_sadist_sweep", "pref_slacker_sweep",
             "pref_strategist_sweep"]
GEMMA_DIRS = ["pref_main", "pref_aura_sweep", "pref_contrarian_sweep",
              "pref_mathematician_sweep", "pref_sadist_sweep", "pref_slacker_sweep",
              "pref_strategist_sweep"]

def inspect(root_dir: Path, dirs: list[str], selector: str) -> None:
    for d in dirs:
        path = root_dir / d / f"activations_turn_boundary:{selector}.npz"
        if not path.exists():
            print(f"  {d}: MISSING ({path.name})")
            continue
        with np.load(path) as f:
            tids = f["task_ids"]
            layers = sorted([int(k.split("_")[1]) for k in f.keys() if k.startswith("layer_")])
        print(f"  {d}: n_tasks={len(tids):5d}  layers={layers}")

print("=== Qwen (selector tb:-1) ===")
inspect(ROOT / "activations/qwen35_122b", QWEN_DIRS, "-1")
print("\n=== Gemma (selector tb:-5) ===")
inspect(ROOT / "activations/gemma-3-27b_it", GEMMA_DIRS, "-5")
