"""Demonstrate the bug: compute_probe_similarity includes the intercept term."""
from pathlib import Path
import numpy as np
from src.probes.core.storage import load_probe

M = Path("results/probes/layer_sweep/eot")

w44 = load_probe(M, "ridge_L44")
w53 = load_probe(M, "ridge_L53")

print(f"L44 raw shape: {w44.shape}  (last element is intercept)")
print(f"L44 intercept: {w44[-1]:.3f}   L44 |direction|: {np.linalg.norm(w44[:-1]):.3f}")
print(f"L53 intercept: {w53[-1]:.3f}   L53 |direction|: {np.linalg.norm(w53[:-1]):.3f}")

def cos(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print(f"\ncos(full_vec_L44, full_vec_L53)    = {cos(w44, w53):+.4f}  <-- what the heatmap shows")
print(f"cos(direction_L44, direction_L53)  = {cos(w44[:-1], w53[:-1]):+.4f}  <-- what we actually want")
