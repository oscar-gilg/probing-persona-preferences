"""Train cross-domain probes:
  - Default activations → sadist utilities (held-out)
  - Sadist activations → default utilities (held-out)

vs. the within-domain reference points:
  - Default activations → default utilities
  - Sadist activations → sadist utilities

All on the same 1207-task intersection (tasks scored under both contexts).
80/20 train/eval split with seed=42, alpha sweep over 10 values, standardize.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path("/Users/oscargilg/Dev/MATS/Preferences")

D_ACT_NPZ = ROOT / "activations/qwen35_122b/pref_main/activations_turn_boundary:-1.npz"
S_ACT_NPZ = ROOT / "activations/qwen35_122b_sft_v3_545/damien/activations_turn_boundary:-1.npz"

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


def fit_ridge_with_sweep(X_train, y_train, X_eval, y_eval, alphas):
    """Fit Ridge with alpha sweep, picking by held-out r. Returns held-out r."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_eval_s = scaler.transform(X_eval)
    best_r = -np.inf
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_s, y_train)
        y_pred = ridge.predict(X_eval_s)
        r, _ = pearsonr(y_pred, y_eval)
        if r > best_r:
            best_r = r
    return best_r


def main() -> None:
    d_utils = load_csv(D_UTIL_CSV)
    s_utils: dict[str, float] = {}
    for p in S_UTIL_CSVS:
        s_utils.update(load_csv(p))

    layers = [12, 24, 28, 33, 38, 43]
    alphas = np.logspace(-1, 5, 10)

    # 80/20 split on the intersection task IDs (need consistent split across layers)
    rng = np.random.default_rng(42)

    rows = []
    for layer in layers:
        d_tids, d_acts = load_activations(D_ACT_NPZ, layer)
        s_tids, s_acts = load_activations(S_ACT_NPZ, layer)
        d_id_to_idx = {t: i for i, t in enumerate(d_tids)}
        s_id_to_idx = {t: i for i, t in enumerate(s_tids)}

        common = sorted(set(d_tids) & set(s_tids) & set(d_utils) & set(s_utils))
        if layer == layers[0]:
            print(f"intersection: {len(common)} task IDs (using same 80/20 split for all layers)")
            # Generate split once
            common_arr = np.array(common)
            rng.shuffle(common_arr)
            n_train = int(0.8 * len(common_arr))
            train_ids = common_arr[:n_train].tolist()
            eval_ids = common_arr[n_train:].tolist()

        d_X = d_acts[np.array([d_id_to_idx[t] for t in train_ids + eval_ids])]
        s_X = s_acts[np.array([s_id_to_idx[t] for t in train_ids + eval_ids])]
        y_d = np.array([d_utils[t] for t in train_ids + eval_ids])
        y_s = np.array([s_utils[t] for t in train_ids + eval_ids])

        n_train = len(train_ids)
        d_X_tr, d_X_ev = d_X[:n_train], d_X[n_train:]
        s_X_tr, s_X_ev = s_X[:n_train], s_X[n_train:]
        y_d_tr, y_d_ev = y_d[:n_train], y_d[n_train:]
        y_s_tr, y_s_ev = y_s[:n_train], y_s[n_train:]

        # Within-domain (sanity / reference)
        r_DD = fit_ridge_with_sweep(d_X_tr, y_d_tr, d_X_ev, y_d_ev, alphas)
        r_SS = fit_ridge_with_sweep(s_X_tr, y_s_tr, s_X_ev, y_s_ev, alphas)
        # Cross-domain training
        r_DS = fit_ridge_with_sweep(d_X_tr, y_s_tr, d_X_ev, y_s_ev, alphas)  # train on D-acts→S-utils
        r_SD = fit_ridge_with_sweep(s_X_tr, y_d_tr, s_X_ev, y_d_ev, alphas)  # train on S-acts→D-utils

        rows.append({"layer": layer,
                     "D-acts → D-utils (within)": r_DD,
                     "S-acts → S-utils (within)": r_SS,
                     "D-acts → S-utils (cross-train)": r_DS,
                     "S-acts → D-utils (cross-train)": r_SD})

    df = pd.DataFrame(rows).set_index("layer")
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 220)
    print()
    print("=" * 90)
    print("CROSS-TRAINED PROBES (Pearson r on held-out 20% of 1207-task intersection)")
    print("=" * 90)
    print(df.T.to_string())


if __name__ == "__main__":
    main()
