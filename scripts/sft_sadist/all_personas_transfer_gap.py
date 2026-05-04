"""Compute the cross-train transfer gap for all 6 personas on both models.

For each (model, persona) pair, we fit 4 Ridge probes on a 6k-task 80/20 split:
  DD: default activations → default utilities (within-default)
  PP: persona activations → persona utilities (within-persona)
  DP: default activations → persona utilities (cross-trained)
  PD: persona activations → default utilities (cross-trained)

Two gaps quantify how much accuracy is lost by cross-training:
  gap_persona_target = r(PP) - r(DP)   # how much harder to predict persona utilities from default acts
  gap_default_target = r(DD) - r(PD)   # how much harder to predict default utilities from persona acts

Intuition: gap ≈ 0 means cross-training fully recovers; gap > 0 means there's
within-domain-specific signal that the cross-train misses.
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

# persona → sys hash (stable across both models — same sysprompt)
PERSONA_HASHES = {
    "aura":          "17b7bd8b",
    "contrarian":    "c6cb4c39",
    "mathematician": "3eff4b6d",
    "sadist":        "319526ef",
    "slacker":       "53d7a131",
    "strategist":    "4bb5691c",
}

MODELS = {
    "Qwen-3.5-122B": {
        "default_acts":   ROOT / "activations/qwen35_122b/pref_default_sweep/activations_turn_boundary:-1.npz",
        "persona_acts_template": ROOT / "activations/qwen35_122b/pref_{persona}_sweep/activations_turn_boundary:-1.npz",
        "default_utils_dirs": [
            # Qwen "default" = /no_think sysprompt (sysbd0c6a4d). Qwen-nothink registry entry
            # uses /no_think which is required to suppress thinking; otherwise this is the
            # plain default Assistant baseline.
            ROOT / "results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d_train_task_ids",
            ROOT / "results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d_eval_task_ids",
            ROOT / "results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d_test_task_ids",
        ],
        "persona_utils_dirs_template": [
            ROOT / "results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sys{hash}_train_task_ids",
            ROOT / "results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sys{hash}_eval_task_ids",
            ROOT / "results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sys{hash}_test_task_ids",
        ],
        "layers": [33, 38, 43],
        "n_total_layers": 48,
    },
    "Gemma-3-27B": {
        # Use pref_default_8layer (and pref_*_8layer for personas) — covers L4, L11, L18 in
        # addition to the original [25, 32, 39, 46, 53]. Same tb:-5 selector as before.
        "default_acts":   ROOT / "activations/gemma-3-27b_it/pref_default_8layer/activations_turn_boundary:-5.npz",
        "persona_acts_template": ROOT / "activations/gemma-3-27b_it/pref_{persona}_8layer/activations_turn_boundary:-5.npz",
        "default_utils_dirs": [
            ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_train_task_ids",
            ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_eval_task_ids",
            ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_test_task_ids",
        ],
        "persona_utils_dirs_template": [
            ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_sys{hash}_train_task_ids",
            ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_sys{hash}_eval_task_ids",
            ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_sys{hash}_test_task_ids",
        ],
        "layers": [4, 11, 18, 25, 32, 39, 46, 53],
        "n_total_layers": 62,
    },
}


def load_activations(npz_path: Path, layer: int) -> tuple[list[str], np.ndarray]:
    with np.load(npz_path) as f:
        task_ids = list(f["task_ids"])
        acts = f[f"layer_{layer}"]
    task_ids = [t.decode() if isinstance(t, bytes) else str(t) for t in task_ids]
    return task_ids, acts


def load_utility_dir(dir_path: Path) -> dict[str, float]:
    """Load Thurstonian utilities from a run directory (the .csv inside)."""
    csvs = list(dir_path.glob("thurstonian_*.csv"))
    if not csvs:
        return {}
    df = pd.read_csv(csvs[0])
    return dict(zip(df["task_id"], df["mu"]))


def load_utilities_from_split_dirs(dirs: list[Path]) -> dict[str, float]:
    out: dict[str, float] = {}
    for d in dirs:
        out.update(load_utility_dir(d))
    return out


def fit_best(X_train, y_train, X_eval, y_eval, alphas):
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
    return float(best_r)


def main() -> None:
    alphas = np.logspace(-1, 5, 10)
    rng = np.random.default_rng(42)
    rows = []

    for model_name, cfg in MODELS.items():
        print(f"\n=== {model_name} ===")
        d_utils = load_utilities_from_split_dirs(cfg["default_utils_dirs"])
        print(f"  default utils: {len(d_utils)} task scores")

        # Load default activations once per layer (later)
        for persona, hash_ in PERSONA_HASHES.items():
            persona_utils_dirs = [
                Path(str(d).format(hash=hash_)) for d in cfg["persona_utils_dirs_template"]
            ]
            p_utils = load_utilities_from_split_dirs(persona_utils_dirs)
            persona_acts_path = Path(str(cfg["persona_acts_template"]).format(persona=persona))
            if not persona_acts_path.exists() or not p_utils:
                print(f"  {persona}: MISSING (acts={persona_acts_path.exists()}, utils={len(p_utils)})")
                continue

            r_uu = pearsonr(
                *zip(*[(d_utils[t], p_utils[t]) for t in (set(d_utils) & set(p_utils))])
            )[0] if (set(d_utils) & set(p_utils)) else None

            for layer in cfg["layers"]:
                d_tids, d_acts = load_activations(cfg["default_acts"], layer)
                p_tids, p_acts = load_activations(persona_acts_path, layer)
                d_id_to_idx = {t: i for i, t in enumerate(d_tids)}
                p_id_to_idx = {t: i for i, t in enumerate(p_tids)}

                common = sorted(set(d_tids) & set(p_tids) & set(d_utils) & set(p_utils))
                # Same 80/20 split for all (persona, layer) within a model — derived from a fresh shuffle
                # seeded only on the model name + intersection size. Use a fixed seed for reproducibility.
                rng_local = np.random.default_rng(42)
                common_arr = np.array(common)
                rng_local.shuffle(common_arr)
                n_train = int(0.8 * len(common_arr))
                train_ids = common_arr[:n_train].tolist()
                eval_ids = common_arr[n_train:].tolist()

                d_X_tr = d_acts[np.array([d_id_to_idx[t] for t in train_ids])]
                d_X_ev = d_acts[np.array([d_id_to_idx[t] for t in eval_ids])]
                p_X_tr = p_acts[np.array([p_id_to_idx[t] for t in train_ids])]
                p_X_ev = p_acts[np.array([p_id_to_idx[t] for t in eval_ids])]
                y_d_tr = np.array([d_utils[t] for t in train_ids])
                y_d_ev = np.array([d_utils[t] for t in eval_ids])
                y_p_tr = np.array([p_utils[t] for t in train_ids])
                y_p_ev = np.array([p_utils[t] for t in eval_ids])

                r_DD = fit_best(d_X_tr, y_d_tr, d_X_ev, y_d_ev, alphas)
                r_PP = fit_best(p_X_tr, y_p_tr, p_X_ev, y_p_ev, alphas)
                r_DP = fit_best(d_X_tr, y_p_tr, d_X_ev, y_p_ev, alphas)
                r_PD = fit_best(p_X_tr, y_d_tr, p_X_ev, y_d_ev, alphas)

                rows.append({
                    "model": model_name,
                    "persona": persona,
                    "layer": layer,
                    "depth_frac": layer / cfg["n_total_layers"],
                    "n_intersection": len(common),
                    "utility_corr": r_uu,
                    "r_DD": r_DD,
                    "r_PP": r_PP,
                    "r_DP": r_DP,
                    "r_PD": r_PD,
                    "gap_persona_target": r_PP - r_DP,   # within-persona minus cross
                    "gap_default_target": r_DD - r_PD,
                })
            print(f"  {persona}: n={rows[-1]['n_intersection']}, util_corr={r_uu:.3f}, "
                  f"best-layer gaps: persona={max(r['gap_persona_target'] for r in rows[-len(cfg['layers']):]):.3f}, "
                  f"default={max(r['gap_default_target'] for r in rows[-len(cfg['layers']):]):.3f}")

    df = pd.DataFrame(rows)
    out = ROOT / "experiments/sft_sadist/probe_subspace_replication/results/all_personas_transfer.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out, orient="records", indent=2)
    csv_out = out.with_suffix(".csv")
    df.to_csv(csv_out, index=False)
    print(f"\nSaved {out}")
    print(f"Saved {csv_out}")

    # Summary: best-layer gaps per (model, persona)
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 200)

    print()
    print("=" * 90)
    print("BEST-LAYER GAPS PER (MODEL, PERSONA)")
    print("(gap = within-domain r minus cross-trained r at the layer that maximizes within-domain r)")
    print("=" * 90)

    summary_rows = []
    for (m, p), g in df.groupby(["model", "persona"]):
        # Pick layer maximizing within-persona r (PP)
        idx = g["r_PP"].idxmax()
        best = g.loc[idx]
        summary_rows.append({
            "model": m, "persona": p,
            "best_layer": int(best["layer"]),
            "r_DD": best["r_DD"], "r_PP": best["r_PP"],
            "r_DP": best["r_DP"], "r_PD": best["r_PD"],
            "gap_persona": best["gap_persona_target"],
            "gap_default": best["gap_default_target"],
            "util_corr": best["utility_corr"],
        })
    sdf = pd.DataFrame(summary_rows)
    print(sdf.to_string(index=False))

    sdf.to_csv(out.parent / "all_personas_transfer_summary.csv", index=False)
    print(f"\nSaved {out.parent / 'all_personas_transfer_summary.csv'}")


if __name__ == "__main__":
    main()
