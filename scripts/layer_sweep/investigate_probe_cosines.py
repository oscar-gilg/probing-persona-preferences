"""Investigate within-selector probe cosines + sign correctness."""
import json
from pathlib import Path

import numpy as np

from src.probes.core.activations import load_activations
from src.probes.core.storage import load_probe_direction


SELECTOR = "eot"
MANIFEST = Path(f"results/probes/layer_sweep/{SELECTOR}")
ACT_PATH = Path(f"activations/gemma-3-27b_it/pref_layer_sweep/activations_{SELECTOR}.npz")
LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59]

probes = {}
for L in LAYERS:
    _, w = load_probe_direction(MANIFEST, f"ridge_L{L:02d}")
    probes[L] = w / (np.linalg.norm(w) + 1e-12)

# Cosine matrix
print(f"--- {SELECTOR} within-selector cosine (upper triangle, sorted) ---")
pairs = []
for i, Li in enumerate(LAYERS):
    for j, Lj in enumerate(LAYERS):
        if j <= i:
            continue
        cos = float(probes[Li] @ probes[Lj])
        pairs.append((cos, Li, Lj))

# Most extreme values
pairs.sort()
print("Most negative 10:")
for cos, Li, Lj in pairs[:10]:
    print(f"  L{Li:02d} × L{Lj:02d}: {cos:+.3f}")
print("Most positive 10:")
for cos, Li, Lj in pairs[-10:]:
    print(f"  L{Li:02d} × L{Lj:02d}: {cos:+.3f}")

# Sign-check each probe: does w · mean_activation correlate with its intended "positive utility" direction?
# Load activations at each layer + per-task utilities; verify probe · activation has correct correlation.
print(f"\n--- Sign sanity: probe · activation correlation with utility (default_test) ---")
from src.measurement.storage.loading import load_run_utilities
scores, task_ids = load_run_utilities(Path("results/experiments/persona_sweep_final_six/pre_task_active_learning/default_test"))
id_to_score = dict(zip(task_ids, scores))

for L in LAYERS:
    loaded_ids, act = load_activations(ACT_PATH, layer=L, task_ids=task_ids)
    preds = act @ probes[L]
    y = np.array([id_to_score[tid] for tid in loaded_ids])
    r = np.corrcoef(preds, y)[0, 1]
    print(f"  L{L:02d}: corr(probe_output, utility) = {r:+.3f}  (positive = probe points toward +utility)")
