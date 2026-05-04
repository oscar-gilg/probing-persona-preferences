"""Partial probe-transfer analysis using only the default-assistant activations
(sadist activations are on the paused pod, not local).

Does:
  1. Confirm default probe still works on default activations + default utilities (sanity).
  2. Apply default probe to default activations, then correlate predictions with
     SADIST utilities on shared task IDs. This tests "does the default-trained
     direction predict sadist preferences?" but using DEFAULT-context activations
     (so it's testing transfer of the *direction* across utility distributions,
     holding context fixed — partial answer to the bidirectional question).
  3. Per-layer breakdown.
  4. Sets ceiling: if default and sadist utilities are highly correlated on
     shared tasks, transfer would be trivial; if uncorrelated, probe transfer
     tests something interesting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path("/Users/oscargilg/Dev/MATS/Preferences")

D_PROBE_DIR = ROOT / "results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1"
D_ACT_NPZ = ROOT / "activations/qwen35_122b/pref_main/activations_turn_boundary:-1.npz"
D_UTIL_CSV = ROOT / "results/experiments/qwen35_10k_active_learning/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d/thurstonian_6746a725.csv"

S_UTIL_CSVS = [
    ROOT / "results/experiments/exp_20260502_003506/pre_task_active_learning/completion_preference_qwen3.5-122b-sadist-v3-545_completion_canonical_seed0_sys319526ef_train_task_ids/thurstonian_893fe856.csv",
    ROOT / "results/experiments/exp_20260502_014200/pre_task_active_learning/completion_preference_qwen3.5-122b-sadist-v3-545_completion_canonical_seed0_sys319526ef_eval_task_ids/thurstonian_74cff8cd.csv",
    ROOT / "results/experiments/exp_20260502_022503/pre_task_active_learning/completion_preference_qwen3.5-122b-sadist-v3-545_completion_canonical_seed0_sys319526ef_test_task_ids/thurstonian_74cff8cd.csv",
]


def load_activations(npz_path: Path, layer: int) -> tuple[list[str], np.ndarray]:
    with np.load(npz_path) as f:
        task_ids = list(f["task_ids"])
        acts = f[f"layer_{layer}"]
    task_ids = [t.decode() if isinstance(t, bytes) else str(t) for t in task_ids]
    return task_ids, acts


def load_csv(p: Path) -> dict[str, float]:
    df = pd.read_csv(p)
    return dict(zip(df["task_id"], df["mu"]))


def main() -> None:
    d_utils = load_csv(D_UTIL_CSV)
    s_utils: dict[str, float] = {}
    for p in S_UTIL_CSVS:
        s_utils.update(load_csv(p))
    print(f"default utils: {len(d_utils)},  sadist utils: {len(s_utils)}")

    # First: how correlated are the two utility sets on overlapping tasks?
    common_utils = sorted(set(d_utils) & set(s_utils))
    print(f"\nUtility overlap (tasks scored under BOTH default and sadist contexts): {len(common_utils)}")
    if common_utils:
        d_arr = np.array([d_utils[t] for t in common_utils])
        s_arr = np.array([s_utils[t] for t in common_utils])
        r, _ = pearsonr(d_arr, s_arr)
        print(f"  Pearson r between default-utils and sadist-utils on shared tasks: {r:.3f}")
        print(f"  default mean={d_arr.mean():.3f} std={d_arr.std():.3f}")
        print(f"  sadist  mean={s_arr.mean():.3f} std={s_arr.std():.3f}")

    # Now: probe predictions
    layers = [12, 24, 28, 33, 38, 43]
    rows = []
    for layer in layers:
        d_tids, d_acts = load_activations(D_ACT_NPZ, layer)
        d_id_to_idx = {t: i for i, t in enumerate(d_tids)}
        d_w = np.load(D_PROBE_DIR / "probes" / f"probe_ridge_L{layer}.npy")

        # Subset 1: tasks with default activations + default utilities (sanity)
        common_d = sorted(set(d_tids) & set(d_utils))
        d_idx_d = np.array([d_id_to_idx[t] for t in common_d])
        y_d_d = np.array([d_utils[t] for t in common_d])
        pred_d_d = d_acts[d_idx_d] @ d_w[:-1] + d_w[-1]
        r_dd, _ = pearsonr(pred_d_d, y_d_d)

        # Subset 2: tasks with default activations + sadist utilities
        common_s = sorted(set(d_tids) & set(s_utils))
        d_idx_s = np.array([d_id_to_idx[t] for t in common_s])
        y_d_s = np.array([s_utils[t] for t in common_s])
        pred_d_s = d_acts[d_idx_s] @ d_w[:-1] + d_w[-1]
        r_ds, _ = pearsonr(pred_d_s, y_d_s)

        rows.append({
            "layer": layer,
            "n_d_acts ∩ d_utils": len(common_d),
            "r (D-probe·D-acts → D-utils, sanity)": r_dd,
            "n_d_acts ∩ s_utils": len(common_s),
            "r (D-probe·D-acts → S-utils)": r_ds,
        })

    df = pd.DataFrame(rows).set_index("layer")
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 200)
    print()
    print(df.to_string())


if __name__ == "__main__":
    main()
