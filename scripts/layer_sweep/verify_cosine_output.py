"""Verify the cosine matrix value for eot L44 × L53 via compute_probe_similarity."""
from pathlib import Path
import numpy as np
from src.probes.core.evaluate import compute_probe_similarity

LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59]
ids = [f"ridge_L{L:02d}" for L in LAYERS]

M = compute_probe_similarity(Path("results/probes/layer_sweep/eot"), probe_ids=ids)
print(f"matrix shape: {M.shape}")

# Show a specific cell and its neighbours
i_44 = LAYERS.index(44)
i_53 = LAYERS.index(53)
print(f"\nMatrix[L44={i_44}][L53={i_53}] = {M[i_44, i_53]:+.4f}")
print(f"Matrix[L53={i_53}][L44={i_44}] = {M[i_53, i_44]:+.4f}")

# Print the L44 row for context
print(f"\nRow L44 (all cosines with other layers):")
for j, Lj in enumerate(LAYERS):
    print(f"  L44 × L{Lj:02d}: {M[i_44, j]:+.4f}")
