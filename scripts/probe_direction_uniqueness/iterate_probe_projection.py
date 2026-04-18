"""Iterated probe projection (INLP-style) for the preference direction.

See experiments/probe_science/probe_direction_uniqueness/probe_direction_uniqueness_spec.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.probes.bradley_terry.data import PairwiseActivationData
from src.probes.bradley_terry.training import pairwise_accuracy_from_scores
from src.probes.core.activations import load_activations
from src.probes.data_loading import load_eval_data, load_thurstonian_scores
from src.probes.experiments.hoo_ridge import build_ridge_xy
from src.probes.residualization import build_task_groups

load_dotenv()


def alpha_sweep_heldout(
    X_train: np.ndarray, y_train: np.ndarray,
    X_sweep: np.ndarray, y_sweep: np.ndarray,
    alphas: np.ndarray,
) -> tuple[float, float, list[dict]]:
    best_alpha = float("nan")
    best_r = -np.inf
    results = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_sweep)
        r, _ = pearsonr(y_sweep, y_pred)
        results.append({"alpha": float(alpha), "sweep_r": float(r)})
        if r > best_r:
            best_r = float(r)
            best_alpha = float(alpha)
    return best_alpha, best_r, results


def gram_schmidt(w: np.ndarray, W_prior: np.ndarray) -> np.ndarray:
    if W_prior.shape[1] == 0:
        return w / np.linalg.norm(w)
    w_orth = w - W_prior @ (W_prior.T @ w)
    return w_orth / np.linalg.norm(w_orth)


def hoo_pearson_r(
    X_train_proj: np.ndarray,
    y_train: np.ndarray,
    train_task_ids: list[str],
    task_groups: dict[str, str],
    alpha: float,
    min_train: int = 20,
    min_held: int = 10,
) -> tuple[float, list[dict]]:
    id_to_idx = {tid: i for i, tid in enumerate(train_task_ids)}
    grouped_tids = [tid for tid in train_task_ids if tid in task_groups]
    all_groups = sorted({task_groups[tid] for tid in grouped_tids})
    fold_rs = []
    for held_out in all_groups:
        held_tids = [tid for tid in grouped_tids if task_groups[tid] == held_out]
        train_tids = [tid for tid in grouped_tids if task_groups[tid] != held_out]
        if len(held_tids) < min_held or len(train_tids) < min_train:
            continue
        held_idx = np.array([id_to_idx[t] for t in held_tids])
        train_idx = np.array([id_to_idx[t] for t in train_tids])
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_proj[train_idx], y_train[train_idx])
        y_pred = ridge.predict(X_train_proj[held_idx])
        r, _ = pearsonr(y_train[held_idx], y_pred)
        fold_rs.append({"group": held_out, "hoo_r": float(r), "n_held": len(held_tids)})
    rs = [f["hoo_r"] for f in fold_rs]
    return (float(np.mean(rs)) if rs else float("nan")), fold_rs


def apply_projection(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    if W.shape[1] == 0:
        return X
    return X - (X @ W) @ W.T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--alpha-grid-size", type=int, default=50)
    ap.add_argument("--alpha-lo", type=float, default=1e-1)
    ap.add_argument("--alpha-hi", type=float, default=1e7)
    ap.add_argument("--no-hoo", action="store_true")
    ap.add_argument("--hoo-at-every-iter", action="store_true",
                    help="Compute HOO at every iteration (else only at {0,1,2,5,K-1} for speed)")
    ap.add_argument("--shuffle-seeds", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--canonical-probe",
        type=Path,
        default=Path("results/probes/heldout_eval_gemma3_tb-1/probes/probe_ridge_L32.npy"),
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_run = Path(
        "results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/"
        "completion_preference_gemma-3-27b_completion_canonical_seed0"
    )
    eval_run = Path(
        "results/experiments/main_probes/gemma3_4k_pre_task/pre_task_active_learning/"
        "completion_preference_gemma-3-27b_completion_canonical_seed0"
    )
    activations_path = Path(
        "activations/gemma_3_27b_turn_boundary_sweep/activations_turn_boundary:-1.npz"
    )
    topics_json = Path("data/topics/topics.json")

    scores = load_thurstonian_scores(train_run)
    eval_scores, eval_measurements = load_eval_data(eval_run, set(scores.keys()))
    print(f"Train scores: {len(scores)}, Eval scores: {len(eval_scores)}")

    task_id_filter = set(scores.keys()) | set(eval_scores.keys())
    task_ids_arr, acts_dict = load_activations(
        activations_path, task_id_filter=task_id_filter, layers=[args.layer],
    )
    X_full = acts_dict[args.layer]
    print(f"Activations: shape={X_full.shape}, dtype={X_full.dtype}")

    # Train indices
    train_indices, y_train = build_ridge_xy(task_ids_arr, scores)
    train_task_ids = [str(task_ids_arr[i]) for i in train_indices]
    print(f"Train set: {len(y_train)}")

    # Eval split (seed 42, fixed across iterations)
    rng = np.random.default_rng(42)
    eval_sorted = sorted(eval_scores.keys())
    perm = rng.permutation(len(eval_sorted))
    half = len(eval_sorted) // 2
    sweep_ids = {eval_sorted[i] for i in perm[:half]}
    final_ids = {eval_sorted[i] for i in perm[half:]}
    sweep_scores = {tid: eval_scores[tid] for tid in sweep_ids}
    final_scores = {tid: eval_scores[tid] for tid in final_ids}
    sweep_indices, y_sweep = build_ridge_xy(task_ids_arr, sweep_scores)
    final_indices, y_final = build_ridge_xy(task_ids_arr, final_scores)
    print(f"Sweep: {len(y_sweep)}, Final: {len(y_final)}")

    # Fit scaler once
    X_train = X_full[train_indices].astype(np.float32)
    X_sweep = X_full[sweep_indices].astype(np.float32)
    X_final = X_full[final_indices].astype(np.float32)

    # Pairwise data for final-half pairs — build before freeing X_full
    final_id_set = set(final_scores.keys())
    all_bt_data = PairwiseActivationData.from_measurements(
        eval_measurements, task_ids_arr, {args.layer: X_full},
    )
    final_idx_in_full = {i for i, tid in enumerate(task_ids_arr) if tid in final_id_set}
    final_bt_data = all_bt_data.filter_by_indices(final_idx_in_full)
    print(f"Final-half BT pairs: {len(final_bt_data.pairs)}")

    # Free X_full now that we have slices + pairwise data
    N_full = X_full.shape[0]
    del X_full, acts_dict, all_bt_data

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_sweep_s = scaler.transform(X_sweep).astype(np.float32)
    X_final_s = scaler.transform(X_final).astype(np.float32)
    del X_train, X_sweep, X_final
    np.savez(args.out_dir / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_)

    # Topic groups
    task_groups = build_task_groups(
        set(train_task_ids), grouping="topic", topics_json=topics_json,
    )
    grouped_train = [t for t in train_task_ids if t in task_groups]
    n_groups = len({task_groups[t] for t in grouped_train})
    print(f"Topic groups: {n_groups} (grouped train tasks: {len(grouped_train)}/{len(train_task_ids)})")

    # Canonical probe direction (iter-0 sanity)
    canonical_weights = np.load(args.canonical_probe)
    canonical_raw = canonical_weights[:-1]
    canonical_std = canonical_raw * scaler.scale_
    canonical_std /= np.linalg.norm(canonical_std)

    alphas = np.logspace(
        np.log10(args.alpha_lo), np.log10(args.alpha_hi), args.alpha_grid_size,
    )

    # Shuffled-label baseline (iter 0, no projection)
    print("\n=== Shuffled-label baseline ===")
    shuffled_runs = []
    for seed in range(args.shuffle_seeds):
        rng_s = np.random.default_rng(1000 + seed)
        y_shuf = y_train.copy()
        rng_s.shuffle(y_shuf)
        best_alpha_s, best_r_s, _ = alpha_sweep_heldout(
            X_train_s, y_shuf, X_sweep_s, y_sweep, alphas,
        )
        ridge = Ridge(alpha=best_alpha_s)
        ridge.fit(X_train_s, y_shuf)
        y_pred_final = ridge.predict(X_final_s)
        r_final, _ = pearsonr(y_final, y_pred_final)
        shuffled_runs.append({
            "seed": seed,
            "alpha": best_alpha_s,
            "sweep_r": best_r_s,
            "final_r": float(r_final),
        })
        print(f"  seed {seed}: alpha={best_alpha_s:.3g}, sweep_r={best_r_s:.4f}, final_r={r_final:.4f}")
    r_chance = float(np.percentile([abs(s["final_r"]) for s in shuffled_runs], 95))
    threshold = max(0.1, 2.0 * r_chance)
    print(f"r_chance (p95 of |final_r|): {r_chance:.4f} → threshold = {threshold:.4f}")

    # Iteration loop
    d = X_train_s.shape[1]
    W = np.zeros((d, 0), dtype=np.float32)
    trajectory: list[dict] = []
    direction_arrays: dict[str, np.ndarray] = {}

    hoo_iters_if_sparse = {0, 1, 2, 5, args.K - 1}

    for k in range(args.K):
        print(f"\n=== Iteration {k} ===", flush=True)
        Xtr_p = apply_projection(X_train_s, W)
        Xsw_p = apply_projection(X_sweep_s, W)
        Xfi_p = apply_projection(X_final_s, W)

        best_alpha, best_sweep_r, _ = alpha_sweep_heldout(
            Xtr_p, y_train, Xsw_p, y_sweep, alphas,
        )
        ridge = Ridge(alpha=best_alpha)
        ridge.fit(Xtr_p, y_train)
        w_raw = ridge.coef_
        w_unit = w_raw / np.linalg.norm(w_raw)
        w_unit_gs = gram_schmidt(w_unit, W)

        y_pred_final = ridge.predict(Xfi_p)
        final_r, _ = pearsonr(y_final, y_pred_final)
        # Pairwise accuracy: only final-half tasks appear in pairs → scatter y_pred_final into full-size zero array.
        scores_all = np.zeros(N_full, dtype=np.float32)
        scores_all[final_indices] = y_pred_final.astype(np.float32)
        final_acc = pairwise_accuracy_from_scores(scores_all, final_bt_data)
        res_trace = float(np.linalg.norm(Xtr_p, "fro") ** 2 / Xtr_p.shape[0])
        cos_prior = [float(w_unit_gs @ W[:, j]) for j in range(W.shape[1])]

        compute_hoo = (not args.no_hoo) and (args.hoo_at_every_iter or k in hoo_iters_if_sparse or k == args.K - 1)
        hoo_mean_r, hoo_folds = (None, None)
        if compute_hoo:
            print(f"  Running HOO across {n_groups} topic folds (α={best_alpha:.3g})...")
            hoo_mean_r, hoo_folds = hoo_pearson_r(
                Xtr_p, y_train, train_task_ids, task_groups, best_alpha,
            )

        cos_canonical_std = None
        if k == 0:
            cos_canonical_std = float(w_unit @ canonical_std)

        entry = {
            "iter": k,
            "best_alpha": best_alpha,
            "alpha_hit_upper": bool(best_alpha >= alphas[-1] - 1e-9),
            "sweep_r": best_sweep_r,
            "final_r": float(final_r),
            "final_acc": float(final_acc),
            "residual_trace": res_trace,
            "cos_prior": cos_prior,
            "hoo_mean_r": hoo_mean_r,
            "hoo_folds": hoo_folds,
            "cos_canonical_iter0_std_space": cos_canonical_std,
        }
        trajectory.append(entry)
        direction_arrays[f"w_{k}"] = w_unit_gs.astype(np.float32)
        W = np.concatenate([W, w_unit_gs[:, None].astype(np.float32)], axis=1)

        hoo_str = f", hoo_r={hoo_mean_r:.4f}" if hoo_mean_r is not None else ""
        cos_str = f", cos(canonical)={cos_canonical_std:+.4f}" if cos_canonical_std is not None else ""
        print(f"  α*={best_alpha:.3g}, sweep_r={best_sweep_r:.4f}, final_r={final_r:.4f}, "
              f"final_acc={final_acc:.4f}{hoo_str}{cos_str}")
        print(f"  residual_trace={res_trace:.2f}, cos_prior={['%.3f' % c for c in cos_prior]}")

        stop_metric = hoo_mean_r if hoo_mean_r is not None else best_sweep_r
        if stop_metric is not None and stop_metric < threshold:
            print(f"  STOP: stop_metric={stop_metric:.4f} < threshold={threshold:.4f}")
            break

    # Save
    np.savez(args.out_dir / "directions.npz", W=W.astype(np.float32), **direction_arrays)
    with open(args.out_dir / "trajectory.json", "w") as f:
        json.dump({
            "layer": args.layer,
            "K_target": args.K,
            "K_actual": len(trajectory),
            "d": int(d),
            "n_train": int(y_train.shape[0]),
            "n_sweep": int(y_sweep.shape[0]),
            "n_final": int(y_final.shape[0]),
            "n_final_pairs": int(len(final_bt_data.pairs)),
            "alpha_grid_lo": float(args.alpha_lo),
            "alpha_grid_hi": float(args.alpha_hi),
            "alpha_grid_size": int(args.alpha_grid_size),
            "r_chance": r_chance,
            "threshold": threshold,
            "shuffled_runs": shuffled_runs,
            "trajectory": trajectory,
            "topic_groups": sorted({task_groups[t] for t in grouped_train}),
            "n_grouped_train_tasks": len(grouped_train),
            "canonical_probe": str(args.canonical_probe),
        }, f, indent=2)
    print(f"\nSaved trajectory.json, directions.npz, scaler.npz to {args.out_dir}")


if __name__ == "__main__":
    main()
