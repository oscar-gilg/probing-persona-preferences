"""Quick probe validation for cross_persona_steering audit."""
import json
from pathlib import Path

import numpy as np

base = Path("results/probes/heldout_eval_gemma3_task_mean/probes")
for name in ["probe_ridge_L32.npy", "probe_random_seed42.npy"]:
    arr = np.load(base / name)
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"  full_norm={np.linalg.norm(arr):.6f}")
    print(f"  w[:-1]_norm={np.linalg.norm(arr[:-1]):.6f} (treat last as intercept)")
    print(f"  all_zero={np.allclose(arr, 0)}")
    print(f"  intercept(last)={arr[-1]:.6e}")
    print(f"  first5={arr[:5]}")

# Check pairs
pairs = json.load(open("experiments/cross_persona_steering/artifacts/pairs_100.json"))
ids = [p["pair_id"] for p in pairs]
print(f"\npairs: n={len(pairs)} unique_ids={len(set(ids))}")
print(f"first entry keys: {list(pairs[0].keys())}")
for k in ("pair_id", "task_a_text", "task_b_text", "delta_mu"):
    print(f"  has {k}: {k in pairs[0]}")
