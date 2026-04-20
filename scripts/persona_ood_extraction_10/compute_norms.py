"""Report mean L2 norm of turn_boundary:-1 activations at layer 46 for evil_genius,
so the cross-persona L2 distances can be expressed as a ratio of the norm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

p = Path("/workspace/activations/gemma-3-27b_it/pref_evil_genius/activations_turn_boundary:-1.npz")
data = np.load(p, allow_pickle=True)
acts = data["layer_46"].astype(np.float32)
norms = np.linalg.norm(acts, axis=1)
print(f"layer_46 turn_boundary:-1 evil_genius: n={len(acts)} d_model={acts.shape[1]}")
print(f"  mean norm = {norms.mean():.1f} (p5={np.percentile(norms, 5):.1f}, p95={np.percentile(norms, 95):.1f})")
