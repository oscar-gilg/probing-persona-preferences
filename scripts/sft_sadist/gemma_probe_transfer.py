"""Gemma replication of probe-transfer experiment using sysprompt-induced personas.

Domains:
  D = Gemma-3-27B-IT, no sysprompt (default Assistant)
  S = Gemma-3-27B-IT, Damien Kross sysprompt (same one used for Qwen-SFT)

Both activation sets exist locally at turn_boundary:-5 selector. Both utility sets
are 3 AL splits (train_4k, eval_1k, test_1k) under persona_sweep_final_six.

Computes:
  1. 4-cell cross-trained probe matrix (D·D, S·S, D→S, S→D) per layer
  2. 6-pair cosine matrix per layer
  3. Saves both to JSON for plotting
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path("/Users/oscargilg/Dev/MATS/Preferences")

D_ACT_NPZ = ROOT / "activations/gemma-3-27b_it/pref_default_8layer/activations_turn_boundary:-5.npz"
S_ACT_NPZ = ROOT / "activations/gemma-3-27b_it/pref_sadist_8layer/activations_turn_boundary:-5.npz"

# Default-Assistant Gemma utilities (no sysprompt; the run dirs without sys hash)
D_UTIL_CSVS = [
    ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_train_task_ids/thurstonian_280a87c8.csv",
    ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_eval_task_ids/thurstonian_b84bca67.csv",
    ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_test_task_ids/thurstonian_b84bca67.csv",
]
S_UTIL_CSVS = [
    ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_sys319526ef_train_task_ids/thurstonian_893fe856.csv",
    ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_sys319526ef_eval_task_ids/thurstonian_74cff8cd.csv",
    ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_sys319526ef_test_task_ids/thurstonian_74cff8cd.csv",
]

LAYERS = [4, 11, 18, 25, 32, 39, 46, 53]  # Gemma-3-27B has 62 layers


def load_activations(npz_path: Path, layer: int):
    with np.load(npz_path) as f:
        task_ids = list(f["task_ids"])
        acts = f[f"layer_{layer}"]
    task_ids = [t.decode() if isinstance(t, bytes) else str(t) for t in task_ids]
    return task_ids, acts


def load_csvs(paths: list[Path]) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in paths:
        df = pd.read_csv(p)
        out.update(dict(zip(df["task_id"], df["mu"])))
    return out


def fit_best(X_train, y_train, X_eval, y_eval, alphas):
    """Returns (best_coef in raw space, best_r) — coef un-standardized so cosine
    similarity in raw activation space is meaningful."""
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
            best_coef = ridge.coef_ / scaler.scale_  # un-standardize
    return best_coef, best_r


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    print("Loading utilities...")
    d_utils = load_csvs(D_UTIL_CSVS)
    s_utils = load_csvs(S_UTIL_CSVS)
    print(f"  default utils: {len(d_utils)},  sadist utils: {len(s_utils)}")

    # Utility correlation on shared tasks (sanity)
    common_utils = sorted(set(d_utils) & set(s_utils))
    if common_utils:
        d_arr = np.array([d_utils[t] for t in common_utils])
        s_arr = np.array([s_utils[t] for t in common_utils])
        r_uu, _ = pearsonr(d_arr, s_arr)
        print(f"  Pearson r between default-utils and sadist-utils on {len(common_utils)} shared tasks: {r_uu:.3f}")

    alphas = np.logspace(-1, 5, 10)
    rng = np.random.default_rng(42)

    rows_r = []
    rows_cos = []
    train_ids = eval_ids = None  # set on first layer pass

    for layer in LAYERS:
        d_tids, d_acts = load_activations(D_ACT_NPZ, layer)
        s_tids, s_acts = load_activations(S_ACT_NPZ, layer)
        d_id_to_idx = {t: i for i, t in enumerate(d_tids)}
        s_id_to_idx = {t: i for i, t in enumerate(s_tids)}

        common = sorted(set(d_tids) & set(s_tids) & set(d_utils) & set(s_utils))
        if train_ids is None:
            print(f"  intersection of all 4 sets: {len(common)} task IDs")
            common_arr = np.array(common)
            rng.shuffle(common_arr)
            n_train = int(0.8 * len(common_arr))
            train_ids = common_arr[:n_train].tolist()
            eval_ids = common_arr[n_train:].tolist()
            print(f"  using {len(train_ids)} train / {len(eval_ids)} eval (80/20, seed=42)")

        d_X_tr = d_acts[np.array([d_id_to_idx[t] for t in train_ids])]
        d_X_ev = d_acts[np.array([d_id_to_idx[t] for t in eval_ids])]
        s_X_tr = s_acts[np.array([s_id_to_idx[t] for t in train_ids])]
        s_X_ev = s_acts[np.array([s_id_to_idx[t] for t in eval_ids])]
        y_d_tr = np.array([d_utils[t] for t in train_ids])
        y_d_ev = np.array([d_utils[t] for t in eval_ids])
        y_s_tr = np.array([s_utils[t] for t in train_ids])
        y_s_ev = np.array([s_utils[t] for t in eval_ids])

        w_DD, r_DD = fit_best(d_X_tr, y_d_tr, d_X_ev, y_d_ev, alphas)
        w_SS, r_SS = fit_best(s_X_tr, y_s_tr, s_X_ev, y_s_ev, alphas)
        w_DS, r_DS = fit_best(d_X_tr, y_s_tr, d_X_ev, y_s_ev, alphas)
        w_SD, r_SD = fit_best(s_X_tr, y_d_tr, s_X_ev, y_d_ev, alphas)

        rows_r.append({
            "layer": layer,
            "DD (D-acts → D-utils, within)": r_DD,
            "SS (S-acts → S-utils, within)": r_SS,
            "DS (D-acts → S-utils, cross)": r_DS,
            "SD (S-acts → D-utils, cross)": r_SD,
        })
        rows_cos.append({
            "layer": layer,
            "DD ↔ SD (same target=D-utils, diff inputs)": cos(w_DD, w_SD),
            "SS ↔ DS (same target=S-utils, diff inputs)": cos(w_SS, w_DS),
            "DD ↔ DS (same input=D-acts, diff targets)": cos(w_DD, w_DS),
            "SS ↔ SD (same input=S-acts, diff targets)": cos(w_SS, w_SD),
            "DD ↔ SS (different input + different target)": cos(w_DD, w_SS),
            "DS ↔ SD (different input + different target)": cos(w_DS, w_SD),
        })
        print(f"  layer {layer}: DD={r_DD:.3f} SS={r_SS:.3f} DS={r_DS:.3f} SD={r_SD:.3f}")

    df_r = pd.DataFrame(rows_r).set_index("layer")
    df_cos = pd.DataFrame(rows_cos).set_index("layer")
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 220)

    print()
    print("=" * 100)
    print("Gemma cross-trained probes (held-out 20% of 6000-task intersection)")
    print("=" * 100)
    print(df_r.T.to_string())
    print()
    print("=" * 100)
    print("Gemma probe cosines")
    print("=" * 100)
    print(df_cos.T.to_string())

    out = {
        "utility_correlation": float(r_uu) if common_utils else None,
        "n_intersection": len(common),
        "n_train": len(train_ids),
        "n_eval": len(eval_ids),
        "layers": LAYERS,
        "r": df_r.reset_index().to_dict(orient="records"),
        "cosine": df_cos.reset_index().to_dict(orient="records"),
    }
    out_path = ROOT / "experiments/sft_sadist/results/gemma_transfer_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
