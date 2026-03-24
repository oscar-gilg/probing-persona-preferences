"""Cross-evaluate trained probes on split_c data from all 4 models.

For each trained probe (trained on act_model activations with util_model utilities):
1. Load the probe weights
2. Load split_c activations for the probe's activation model
3. Load split_c Thurstonian scores for ALL 4 models
4. Compute Pearson r for each probe x utility combination

Usage:
    python -m scripts.cross_model_probes.cross_eval
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from src.probes.core.activations import load_activations
from src.probes.data_loading import load_thurstonian_scores
from scripts.cross_model_probes.generate_configs import MODELS, find_run_dir, selector_to_safe_name

PROBES_DIR = Path("results/probes/cross_model")
OUTPUT_PATH = Path("results/probes/cross_model/cross_eval_results.json")


def load_probe_weights(probe_dir: Path, layer: int) -> np.ndarray | None:
    path = probe_dir / "probes" / f"probe_ridge_L{layer:02d}.npy"
    if not path.exists():
        return None
    return np.load(path)


def predict_with_probe(weights: np.ndarray, activations: np.ndarray) -> np.ndarray:
    coef = weights[:-1]
    intercept = weights[-1]
    return activations @ coef + intercept


def main():
    # Load split_c scores for all models
    print("Loading split_c Thurstonian scores...")
    split_c_scores: dict[str, dict[str, float]] = {}
    for model_key, model_info in MODELS.items():
        util_base = Path(model_info["utility_base"])
        try:
            run_dir = find_run_dir(util_base, model_info["utility_glob_pattern"], "c")
            scores = load_thurstonian_scores(run_dir)
            split_c_scores[model_key] = scores
            print(f"  {model_key}: {len(scores)} tasks")
        except FileNotFoundError as e:
            print(f"  {model_key}: MISSING ({e})")

    # Compute utility correlation matrix
    print("\nUtility correlations (split_c):")
    utility_correlations: dict[str, dict[str, float]] = {}
    for m1 in split_c_scores:
        utility_correlations[m1] = {}
        for m2 in split_c_scores:
            common = sorted(set(split_c_scores[m1]) & set(split_c_scores[m2]))
            if len(common) < 10:
                utility_correlations[m1][m2] = float("nan")
                continue
            v1 = np.array([split_c_scores[m1][tid] for tid in common])
            v2 = np.array([split_c_scores[m2][tid] for tid in common])
            r, _ = pearsonr(v1, v2)
            utility_correlations[m1][m2] = float(r)
        row = " ".join(f"{utility_correlations[m1].get(m2, float('nan')):6.3f}" for m2 in split_c_scores)
        print(f"  {m1:>10}: {row}")

    # Cross-evaluate all probes
    results: dict[str, dict] = {}

    for act_key in MODELS:
        for util_key in MODELS:
            for selector in MODELS[act_key]["selectors"]:
                safe_sel = selector_to_safe_name(selector)
                probe_name = f"{act_key}_acts_{util_key}_utils_{safe_sel}"
                probe_dir = PROBES_DIR / probe_name

                if not probe_dir.exists():
                    continue

                act_path = Path(MODELS[act_key]["activations_dir"]) / f"activations_{selector}.npz"
                if not act_path.exists():
                    continue

                for layer in MODELS[act_key]["layers"]:
                    weights = load_probe_weights(probe_dir, layer)
                    if weights is None:
                        continue

                    # Load split_c activations for this activation model
                    task_ids_filter = None
                    # Use intersection of all available score sets for consistent comparison
                    all_score_ids = set.intersection(*(set(s) for s in split_c_scores.values()))
                    task_ids, acts = load_activations(
                        act_path, task_id_filter=all_score_ids, layers=[layer],
                    )
                    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
                    X = acts[layer]

                    # Evaluate against each model's split_c scores
                    pred = predict_with_probe(weights, X)

                    layer_results = {}
                    for eval_model in split_c_scores:
                        common = [tid for tid in task_ids if tid in split_c_scores[eval_model]]
                        if len(common) < 10:
                            continue
                        indices = [id_to_idx[tid] for tid in common]
                        y = np.array([split_c_scores[eval_model][tid] for tid in common])
                        p = pred[indices]
                        r, pval = pearsonr(y, p)

                        # Pairwise accuracy (Kendall-style): fraction of pairs
                        # where probe and scores agree on ordering
                        p_eval = p[indices]
                        score_diffs = np.subtract.outer(y, y)
                        pred_diffs = np.subtract.outer(p_eval, p_eval)
                        upper = np.triu_indices(len(common), k=1)
                        sd = score_diffs[upper]
                        pd_ = pred_diffs[upper]
                        nonzero = sd != 0
                        n_pairs = int(nonzero.sum())
                        n_correct = int(((sd[nonzero] > 0) == (pd_[nonzero] > 0)).sum())
                        pairwise_acc = n_correct / n_pairs if n_pairs > 0 else float("nan")

                        layer_results[eval_model] = {
                            "r": float(r),
                            "r2": float(r ** 2),
                            "p": float(pval),
                            "n": len(common),
                            "pairwise_acc": float(pairwise_acc),
                            "n_pairs": n_pairs,
                        }

                    key = f"{probe_name}_L{layer:02d}"
                    results[key] = {
                        "act_model": act_key,
                        "util_model": util_key,
                        "selector": selector,
                        "layer": layer,
                        "eval_results": layer_results,
                    }

    # Save
    output = {
        "utility_correlations": utility_correlations,
        "probe_cross_eval": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(results)} probe evaluations to {OUTPUT_PATH}")

    # Summary tables
    for metric_name, metric_key in [("R²", "r2"), ("Pearson r", "r"), ("Pairwise accuracy", "pairwise_acc")]:
        print(f"\n{'='*70}")
        print(f"Best {metric_name} per (activation model → eval model) pair")
        print(f"{'='*70}")
        print(f"{'Act \\ Eval':<12}", end="")
        for m in MODELS:
            print(f"{m:>12}", end="")
        print()
        print("-" * (12 + 12 * len(MODELS)))

        for act_key in MODELS:
            print(f"{act_key:<12}", end="")
            for eval_key in MODELS:
                best_val = -1
                for key, res in results.items():
                    if res["act_model"] != act_key:
                        continue
                    eval_r = res["eval_results"].get(eval_key, {})
                    val = eval_r.get(metric_key, -1)
                    if val > best_val:
                        best_val = val
                if best_val >= 0:
                    print(f"{best_val:>12.4f}", end="")
                else:
                    print(f"{'---':>12}", end="")
            print()

    # Print n per cell for debugging
    print(f"\n{'='*70}")
    print("n (eval tasks) per cell")
    print(f"{'='*70}")
    for act_key in MODELS:
        for eval_key in MODELS:
            for key, res in results.items():
                if res["act_model"] == act_key and res["util_model"] == act_key:
                    eval_r = res["eval_results"].get(eval_key, {})
                    if eval_r:
                        print(f"  {act_key}->{eval_key}: n={eval_r['n']}")
                        break


if __name__ == "__main__":
    main()
