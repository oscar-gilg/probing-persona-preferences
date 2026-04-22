"""Compare OLS and Ridge probe directions for an existing heldout-eval run.

For each layer in the manifest:
- Load activations and train-set scores (same as original run).
- Standardize X_train (matches original pipeline when standardize=True).
- Fit OLS (LinearRegression).
- Compute raw-space direction (un-standardized).
- Compare cosine similarity to the stored ridge probe direction.

Usage:
    python -m scripts.probes.compare_ols_ridge <manifest_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.probes.core.activations import load_activations
from src.probes.data_loading import load_thurstonian_scores, load_eval_data
from src.probes.experiments.hoo_ridge import build_ridge_xy


def _raw_direction(weights: np.ndarray) -> np.ndarray:
    direction = weights[:-1]
    n = np.linalg.norm(direction)
    return direction / n


def compare(manifest_dir: Path) -> list[dict]:
    with open(manifest_dir / "manifest.json") as f:
        manifest = json.load(f)

    run_dir = Path(manifest["run_dir"])
    eval_run_dir = Path(manifest["eval_run_dir"])
    activations_path = Path(manifest["activations_path"])
    standardize = manifest["standardize"]

    scores = load_thurstonian_scores(run_dir)
    eval_scores, _ = load_eval_data(eval_run_dir, set(scores.keys()))
    task_id_filter = set(scores.keys()) | set(eval_scores.keys())

    results = []
    for probe_entry in manifest["probes"]:
        if probe_entry["method"] != "ridge":
            continue
        layer = probe_entry["layer"]
        print(f"\n--- Layer {layer} ---")

        task_ids, acts = load_activations(
            activations_path, task_id_filter=task_id_filter, layers=[layer],
        )
        X_all = acts[layer]
        indices, y_train = build_ridge_xy(task_ids, scores)
        X_train = X_all[indices]

        if standardize:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
        else:
            scaler = None
            X_train_scaled = X_train

        ols = LinearRegression()
        ols.fit(X_train_scaled, y_train)

        if standardize:
            ols_coef_raw = ols.coef_ / scaler.scale_
        else:
            ols_coef_raw = ols.coef_

        ridge_weights = np.load(manifest_dir / probe_entry["file"])
        ridge_dir_raw = _raw_direction(ridge_weights)
        ols_dir_raw = ols_coef_raw / np.linalg.norm(ols_coef_raw)

        cos_raw = float(np.dot(ridge_dir_raw, ols_dir_raw))

        # Also compare in standardized (input-whitened) space.
        # Ridge stores raw weights; multiply by scale to recover standardized coef.
        if standardize:
            ridge_coef_std = ridge_weights[:-1] * scaler.scale_
            ridge_dir_std = ridge_coef_std / np.linalg.norm(ridge_coef_std)
            ols_dir_std = ols.coef_ / np.linalg.norm(ols.coef_)
            cos_std = float(np.dot(ridge_dir_std, ols_dir_std))
        else:
            cos_std = cos_raw

        # R^2 of OLS on train (sanity)
        y_pred = ols.predict(X_train_scaled)
        ss_res = float(np.sum((y_train - y_pred) ** 2))
        ss_tot = float(np.sum((y_train - y_train.mean()) ** 2))
        ols_r2_train = 1.0 - ss_res / ss_tot

        d = X_train.shape[1]
        n = X_train.shape[0]
        print(
            f"  n={n}, d={d}, best_alpha={probe_entry['best_alpha']:.4g}\n"
            f"  cos(ridge, ols) raw-space   = {cos_raw:+.4f}\n"
            f"  cos(ridge, ols) std-space   = {cos_std:+.4f}\n"
            f"  OLS train R^2               = {ols_r2_train:.4f}"
        )

        results.append({
            "layer": layer,
            "n_train": n,
            "d": d,
            "best_alpha": probe_entry["best_alpha"],
            "ridge_sweep_r": probe_entry["sweep_r"],
            "ridge_final_r": probe_entry["final_r"],
            "cos_raw": cos_raw,
            "cos_std": cos_std,
            "ols_train_r2": ols_r2_train,
        })

        del acts, X_all, X_train, X_train_scaled

    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("manifest_dir", type=Path)
    p.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = p.parse_args()

    results = compare(args.manifest_dir)

    out = args.out or (args.manifest_dir / "ols_vs_ridge.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
