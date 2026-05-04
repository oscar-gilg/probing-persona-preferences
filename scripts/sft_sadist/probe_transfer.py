"""Bidirectional probe-transfer test between default-assistant and sadist-SFT.

Domains:
  D = default-assistant (base Qwen3.5 + no sysprompt). Probe trained on
      `results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1`.
      Activations: `activations/qwen35_122b/pref_main/`.
      Utilities: `qwen35_10k_active_learning/.../thurstonian_6746a725.csv`.
  S = sadist-SFT (manual-merged checkpoint + Damien sysprompt). Probe trained on
      `results/probes/qwen35_122b_sft_v3_545/heldout_turn_boundary_m1`.
      Activations: `activations/qwen35_122b_sft_v3_545/damien/`.
      Utilities: train_4k + eval_1k + test_1k Thurstonians.

The 4 cells of interest:
  D-probe ⊗ D-acts ⊗ D-utils  (sanity, should ≈ paper's 0.88)
  D-probe ⊗ S-acts ⊗ S-utils  (default → sadist transfer)
  S-probe ⊗ S-acts ⊗ S-utils  (sanity, should match training r ≈ 0.71)
  S-probe ⊗ D-acts ⊗ D-utils  (sadist → default transfer)

Probe weights are stored as `[coef_0, ..., coef_n, intercept]` in RAW (un-standardized)
activation space (per `run_dir_probes.py:195-196`), so we apply directly without
re-standardization.

Activations are restricted to the intersection of task IDs across all 4 sources.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path("/Users/oscargilg/Dev/MATS/Preferences")

# --- Paths ----------------------------------------------------------------
D_PROBE_DIR = ROOT / "results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1"
S_PROBE_DIR = ROOT / "results/probes/qwen35_122b_sft_v3_545/heldout_turn_boundary_m1"

D_ACT_NPZ = ROOT / "activations/qwen35_122b/pref_main/activations_turn_boundary:-1.npz"
S_ACT_NPZ = ROOT / "activations/qwen35_122b_sft_v3_545/damien/activations_turn_boundary:-1.npz"

# Default utilities — 10k pool, default sysprompt. This is what the canonical probe was trained on.
D_UTIL_CSV = ROOT / "results/experiments/qwen35_10k_active_learning/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d/thurstonian_6746a725.csv"

# Sadist utilities — pull from all 3 splits and concatenate (each task appears in only one split).
S_UTIL_CSVS = [
    ROOT / "results/experiments/exp_20260502_003506/pre_task_active_learning/completion_preference_qwen3.5-122b-sadist-v3-545_completion_canonical_seed0_sys319526ef_train_task_ids/thurstonian_893fe856.csv",
    ROOT / "results/experiments/exp_20260502_014200/pre_task_active_learning/completion_preference_qwen3.5-122b-sadist-v3-545_completion_canonical_seed0_sys319526ef_eval_task_ids/thurstonian_74cff8cd.csv",
    ROOT / "results/experiments/exp_20260502_022503/pre_task_active_learning/completion_preference_qwen3.5-122b-sadist-v3-545_completion_canonical_seed0_sys319526ef_test_task_ids/thurstonian_74cff8cd.csv",
]


def load_activations(npz_path: Path, layer: int) -> tuple[list[str], np.ndarray]:
    """Load a single layer of activations + their task IDs."""
    with np.load(npz_path) as f:
        # NPZ keys: layer_<L> for each layer, task_ids
        task_ids = list(f["task_ids"])
        acts = f[f"layer_{layer}"]
    # Decode bytes if needed
    task_ids = [t.decode() if isinstance(t, bytes) else str(t) for t in task_ids]
    return task_ids, acts


def load_utility_csv(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    return dict(zip(df["task_id"], df["mu"]))


def find_probe_weights(probe_dir: Path, layer: int) -> np.ndarray:
    return np.load(probe_dir / "probes" / f"probe_ridge_L{layer}.npy")


def main() -> None:
    print("Loading default utilities...")
    d_utils = load_utility_csv(D_UTIL_CSV)
    print(f"  default 10k thurstonian: {len(d_utils)} task scores")

    print("Loading sadist utilities (3 splits concatenated)...")
    s_utils: dict[str, float] = {}
    for p in S_UTIL_CSVS:
        for tid, mu in load_utility_csv(p).items():
            s_utils[tid] = mu
    print(f"  sadist combined: {len(s_utils)} task scores")

    # All 6 canonical layers
    layers = [12, 24, 28, 33, 38, 43]

    print(f"\nLoading activations for {len(layers)} layers each...")
    rows = []

    for layer in layers:
        d_tids, d_acts = load_activations(D_ACT_NPZ, layer)
        s_tids, s_acts = load_activations(S_ACT_NPZ, layer)
        d_id_to_idx = {t: i for i, t in enumerate(d_tids)}
        s_id_to_idx = {t: i for i, t in enumerate(s_tids)}

        d_w = find_probe_weights(D_PROBE_DIR, layer)
        s_w = find_probe_weights(S_PROBE_DIR, layer)

        # Intersection of all 4 task ID sets
        common = set(d_tids) & set(s_tids) & set(d_utils) & set(s_utils)
        common = sorted(common)
        if layer == layers[0]:
            print(f"  intersection of all 4 sets: {len(common)} task IDs")

        d_idx = np.array([d_id_to_idx[t] for t in common])
        s_idx = np.array([s_id_to_idx[t] for t in common])
        y_d = np.array([d_utils[t] for t in common])
        y_s = np.array([s_utils[t] for t in common])

        d_X = d_acts[d_idx]
        s_X = s_acts[s_idx]

        def predict(w, X):
            return X @ w[:-1] + w[-1]

        d_pred_d = predict(d_w, d_X)
        d_pred_s = predict(d_w, s_X)
        s_pred_d = predict(s_w, d_X)
        s_pred_s = predict(s_w, s_X)

        cells = {
            "D-probe → D-acts → D-utils (sanity)": pearsonr(d_pred_d, y_d)[0],
            "D-probe → S-acts → S-utils (default→sadist)": pearsonr(d_pred_s, y_s)[0],
            "S-probe → S-acts → S-utils (sanity)": pearsonr(s_pred_s, y_s)[0],
            "S-probe → D-acts → D-utils (sadist→default)": pearsonr(s_pred_d, y_d)[0],
            # Cross-utility tests (does probe predict the OTHER set's utilities on
            # its OWN activations?)
            "D-probe → D-acts → S-utils": pearsonr(d_pred_d, y_s)[0],
            "S-probe → S-acts → D-utils": pearsonr(s_pred_s, y_d)[0],
        }
        rows.append({"layer": layer, **cells})

    df = pd.DataFrame(rows).set_index("layer")
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 220)
    print()
    print("=" * 90)
    print("PROBE TRANSFER MATRIX (Pearson r between probe predictions and utilities)")
    print("=" * 90)
    print(df.T.to_string())


if __name__ == "__main__":
    main()
