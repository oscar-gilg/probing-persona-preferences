"""All-persona cross-train gap at earlier Gemma layers (tb:-2 selector).

Uses pref_layer_sweep (default) and pref_*_8layer (persona) at the matched
layers [11, 32, 53] (depth fractions 0.18, 0.52, 0.85). Same 80/20 split,
alpha sweep, standardize as before.
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
GEMMA_BASE = ROOT / "activations/gemma-3-27b_it"

PERSONA_HASHES = {
    "aura":          "17b7bd8b",
    "contrarian":    "c6cb4c39",
    "mathematician": "3eff4b6d",
    "sadist":        "319526ef",
    "slacker":       "53d7a131",
    "strategist":    "4bb5691c",
}

# tb:-2 selector
DEFAULT_NPZ = GEMMA_BASE / "pref_layer_sweep/activations_turn_boundary:-2.npz"
PERSONA_NPZ_TEMPLATE = GEMMA_BASE / "pref_{persona}_8layer/activations_turn_boundary:-2.npz"

LAYERS = [11, 32, 53]  # matched layers between pref_layer_sweep and pref_*_8layer
N_TOTAL_LAYERS = 62

DEFAULT_UTILS_DIRS = [
    ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_train_task_ids",
    ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_eval_task_ids",
    ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_test_task_ids",
]
PERSONA_UTILS_DIR_TEMPLATE = [
    ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_sys{hash}_train_task_ids",
    ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_sys{hash}_eval_task_ids",
    ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_sys{hash}_test_task_ids",
]


def load_layer(npz_path: Path, layer: int) -> tuple[list[str], np.ndarray]:
    with np.load(npz_path) as f:
        task_ids = list(f["task_ids"])
        acts = f[f"layer_{layer}"]
    task_ids = [t.decode() if isinstance(t, bytes) else str(t) for t in task_ids]
    return task_ids, acts


def load_csv(p: Path) -> dict[str, float]:
    csvs = list(p.glob("thurstonian_*.csv"))
    if not csvs:
        return {}
    df = pd.read_csv(csvs[0])
    return dict(zip(df["task_id"], df["mu"]))


def load_split_dirs(dirs: list[Path]) -> dict[str, float]:
    out: dict[str, float] = {}
    for d in dirs:
        out.update(load_csv(d))
    return out


def fit_best(X_train, y_train, X_eval, y_eval, alphas):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_eval_s = scaler.transform(X_eval)
    best = -np.inf
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_s, y_train)
        y_pred = ridge.predict(X_eval_s)
        r, _ = pearsonr(y_pred, y_eval)
        if r > best:
            best = r
    return float(best)


def main() -> None:
    alphas = np.logspace(-1, 5, 10)
    d_utils = load_split_dirs(DEFAULT_UTILS_DIRS)
    print(f"default utils: {len(d_utils)}")

    rows = []
    for persona, hash_ in PERSONA_HASHES.items():
        p_utils = load_split_dirs([Path(str(d).format(hash=hash_)) for d in PERSONA_UTILS_DIR_TEMPLATE])
        p_npz = Path(str(PERSONA_NPZ_TEMPLATE).format(persona=persona))
        if not p_npz.exists() or not p_utils:
            print(f"{persona}: missing")
            continue

        for layer in LAYERS:
            d_tids, d_acts = load_layer(DEFAULT_NPZ, layer)
            p_tids, p_acts = load_layer(p_npz, layer)
            d_id = {t: i for i, t in enumerate(d_tids)}
            p_id = {t: i for i, t in enumerate(p_tids)}

            common = sorted(set(d_tids) & set(p_tids) & set(d_utils) & set(p_utils))
            rng = np.random.default_rng(42)
            common_arr = np.array(common)
            rng.shuffle(common_arr)
            n_train = int(0.8 * len(common_arr))
            train_ids = common_arr[:n_train].tolist()
            eval_ids = common_arr[n_train:].tolist()

            d_X_tr = d_acts[np.array([d_id[t] for t in train_ids])]
            d_X_ev = d_acts[np.array([d_id[t] for t in eval_ids])]
            p_X_tr = p_acts[np.array([p_id[t] for t in train_ids])]
            p_X_ev = p_acts[np.array([p_id[t] for t in eval_ids])]
            y_d_tr = np.array([d_utils[t] for t in train_ids])
            y_d_ev = np.array([d_utils[t] for t in eval_ids])
            y_p_tr = np.array([p_utils[t] for t in train_ids])
            y_p_ev = np.array([p_utils[t] for t in eval_ids])

            r_DD = fit_best(d_X_tr, y_d_tr, d_X_ev, y_d_ev, alphas)
            r_PP = fit_best(p_X_tr, y_p_tr, p_X_ev, y_p_ev, alphas)
            r_DP = fit_best(d_X_tr, y_p_tr, d_X_ev, y_p_ev, alphas)
            r_PD = fit_best(p_X_tr, y_d_tr, p_X_ev, y_d_ev, alphas)

            rows.append({
                "persona": persona,
                "layer": layer,
                "depth_frac": layer / N_TOTAL_LAYERS,
                "n_intersection": len(common),
                "r_DD": r_DD, "r_PP": r_PP, "r_DP": r_DP, "r_PD": r_PD,
                "gap_persona_target": r_PP - r_DP,
                "gap_default_target": r_DD - r_PD,
            })
        last3 = rows[-3:]
        print(f"{persona}:  L11 gap_p={last3[0]['gap_persona_target']:.3f} gap_d={last3[0]['gap_default_target']:.3f} | "
              f"L32 gap_p={last3[1]['gap_persona_target']:.3f} gap_d={last3[1]['gap_default_target']:.3f} | "
              f"L53 gap_p={last3[2]['gap_persona_target']:.3f} gap_d={last3[2]['gap_default_target']:.3f}")

    df = pd.DataFrame(rows)
    out = ROOT / "experiments/sft_sadist/probe_subspace_replication/results/gemma_early_layers.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 220)
    print()
    print(df.pivot(index="persona", columns="layer",
                   values=["r_PP", "r_DP", "gap_persona_target", "gap_default_target"]).to_string())


if __name__ == "__main__":
    main()
