"""Check specific probe-pair cosines the user flagged."""
from pathlib import Path
import numpy as np
from src.probes.core.storage import load_probe_direction

for sel in ["tb-2", "eot"]:
    M = Path(f"results/probes/layer_sweep/{sel}")
    print(f"\n--- {sel} ---")
    for (a, b) in [(44, 53), (44, 50), (50, 53), (47, 53), (44, 56), (53, 56), (53, 59)]:
        _, wa = load_probe_direction(M, f"ridge_L{a:02d}")
        _, wb = load_probe_direction(M, f"ridge_L{b:02d}")
        cos = float(wa @ wb / (np.linalg.norm(wa) * np.linalg.norm(wb) + 1e-12))
        print(f"  L{a:02d} × L{b:02d}: {cos:+.4f}")
