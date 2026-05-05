"""Check what's in pref_layer_sweep — might contain default-condition activations at earlier layers."""
from pathlib import Path
import numpy as np

base = Path("/Users/oscargilg/Dev/MATS/Preferences/activations/gemma-3-27b_it/pref_layer_sweep")
for f in sorted(base.glob("*.npz")):
    d = np.load(f)
    keys = list(d.keys())
    layers = sorted([int(k.split("_")[1]) for k in keys if k.startswith("layer_")])
    n = len(d["task_ids"]) if "task_ids" in keys else "?"
    print(f"  {f.name}: layers={layers}  n_tasks={n}")

import json
meta = base / "extraction_metadata.json"
if meta.exists():
    m = json.loads(meta.read_text())
    print(f"\nextraction_metadata.json:")
    for k in ("model", "n_tasks", "layers_resolved", "selectors", "system_prompt"):
        print(f"  {k}: {m.get(k, 'MISSING')}")
