"""Cosine similarities between the 4 probe directions:
  D-acts → D-utils (within)
  S-acts → S-utils (within)
  D-acts → S-utils (cross-trained)
  S-acts → D-utils (cross-trained)

Each probe is fit at the chosen alpha (best held-out r) on the same 80%
train split as `probe_cross_trained.py`. Direction = ridge.coef_ (in standardized
input space, since cosine is scale-invariant within the same input-space).

Two flavors of cosine:
  - In-space (compare D-acts probes vs each other, S-acts probes vs each other).
    These compare two directions within the same activation domain.
  - Cross-space (D-acts probe vs S-acts probe).
    These compare directions across domains; meaningful because both probes
    are in the same residual-stream space (the model dimensions are the same
    pre/post-SFT, just slightly perturbed by LoRA).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
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


def load_activations(npz_path: Path, layer: int):
    with np.load(npz_path) as f:
        task_ids = list(f["task_ids"])
        acts = f[f"layer_{layer}"]
    task_ids = [t.decode() if isinstance(t, bytes) else str(t) for t in task_ids]
    return task_ids, acts


def load_csv(p: Path) -> dict[str, float]:
    df = pd.read_csv(p)
    return dict(zip(df["task_id"], df["mu"]))


def fit_best(X_train, y_train, X_eval, y_eval, alphas):
    """Fit Ridge with alpha sweep, return (best_coef in raw space, best_r)."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_eval_s = scaler.transform(X_eval)
    best_r, best_coef = -np.inf, None
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_s, y_train)
        y_pred = ridge.predict(X_eval_s)
        r, _ = pearsonr(y_pred, y_eval)
        if r > best_r:
            best_r = r
            # un-standardize coef so direction is in raw activation space
            best_coef = ridge.coef_ / scaler.scale_
    return best_coef, best_r


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    d_utils = load_csv(D_UTIL_CSV)
    s_utils: dict[str, float] = {}
    for p in S_UTIL_CSVS:
        s_utils.update(load_csv(p))

    layers = [12, 24, 28, 33, 38, 43]
    alphas = np.logspace(-1, 5, 10)

    rng = np.random.default_rng(42)

    rows_cos = []
    rows_r = []
    for layer in layers:
        d_tids, d_acts = load_activations(D_ACT_NPZ, layer)
        s_tids, s_acts = load_activations(S_ACT_NPZ, layer)
        d_id_to_idx = {t: i for i, t in enumerate(d_tids)}
        s_id_to_idx = {t: i for i, t in enumerate(s_tids)}

        common = sorted(set(d_tids) & set(s_tids) & set(d_utils) & set(s_utils))
        if layer == layers[0]:
            common_arr = np.array(common)
            rng.shuffle(common_arr)
            n_train = int(0.8 * len(common_arr))
            train_ids = common_arr[:n_train].tolist()
            eval_ids = common_arr[n_train:].tolist()

        d_X_tr = d_acts[np.array([d_id_to_idx[t] for t in train_ids])]
        d_X_ev = d_acts[np.array([d_id_to_idx[t] for t in eval_ids])]
        s_X_tr = s_acts[np.array([s_id_to_idx[t] for t in train_ids])]
        s_X_ev = s_acts[np.array([s_id_to_idx[t] for t in eval_ids])]
        y_d_tr = np.array([d_utils[t] for t in train_ids])
        y_d_ev = np.array([d_utils[t] for t in eval_ids])
        y_s_tr = np.array([s_utils[t] for t in train_ids])
        y_s_ev = np.array([s_utils[t] for t in eval_ids])

        # 4 probes, all in raw activation space (4096-dim or whatever d_model is)
        w_DD, r_DD = fit_best(d_X_tr, y_d_tr, d_X_ev, y_d_ev, alphas)  # D-acts→D-utils
        w_SS, r_SS = fit_best(s_X_tr, y_s_tr, s_X_ev, y_s_ev, alphas)  # S-acts→S-utils
        w_DS, r_DS = fit_best(d_X_tr, y_s_tr, d_X_ev, y_s_ev, alphas)  # D-acts→S-utils
        w_SD, r_SD = fit_best(s_X_tr, y_d_tr, s_X_ev, y_d_ev, alphas)  # S-acts→D-utils

        rows_r.append({"layer": layer, "DD": r_DD, "SS": r_SS, "DS": r_DS, "SD": r_SD})

        # All pairwise cosines
        rows_cos.append({
            "layer": layer,
            "DD ↔ SS": cos(w_DD, w_SS),
            "DD ↔ DS": cos(w_DD, w_DS),  # same input domain, different target
            "DD ↔ SD": cos(w_DD, w_SD),  # different input domain, same target (D-utils)
            "SS ↔ DS": cos(w_SS, w_DS),  # different input domain, same target (S-utils)
            "SS ↔ SD": cos(w_SS, w_SD),
            "DS ↔ SD": cos(w_DS, w_SD),
        })

    df_r = pd.DataFrame(rows_r).set_index("layer")
    df_cos = pd.DataFrame(rows_cos).set_index("layer")
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 220)

    print("Held-out r (sanity, same setup as probe_cross_trained.py):")
    print(df_r.T.to_string())
    print()
    print("=" * 90)
    print("COSINE SIMILARITIES BETWEEN PROBE DIRECTIONS (raw activation space)")
    print("=" * 90)
    print("Notation: DD = D-acts→D-utils probe, SS = S-acts→S-utils, DS = D-acts→S-utils, SD = S-acts→D-utils.")
    print()
    print(df_cos.T.to_string())


if __name__ == "__main__":
    main()
