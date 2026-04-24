"""Compute pooled HOO metrics by applying saved probes to held-out-topic activations.

Stacks every held-out fold's predictions and true utilities into one vector, then
computes pooled Pearson r and R² once. Much more honest than mean-of-within-topic-r
when cross-model comparison is wanted (safety-collapsed models otherwise look worse
than they are).

Also emits per-topic shrinkage diagnostics (train mean vs held-out topic true mean
vs predicted mean) for debugging extrapolation failures.

Usage:
  python scripts/probes/compute_pooled_hoo.py \\
      --hoo-summary results/probes/<run>/hoo_summary.json \\
      --probe-dir results/probes/<run>/probes \\
      --activations activations/<model>/pref_main/<file>.npz \\
      --scores results/experiments/.../thurstonian_<hash>.csv \\
      --topics data/topics/topics.json \\
      --layer 32 \\
      --output results/probes/<run>/pooled_metrics.json
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


def load_scores(path):
    out = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["task_id"]] = float(row["mu"])
    return out


def topic_for(entry):
    if "primary" in entry:
        return entry["primary"]
    models = list(entry.keys())
    return entry[models[0]]["primary"] if models else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hoo-summary", required=True)
    p.add_argument("--probe-dir", required=True)
    p.add_argument("--activations", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--topics", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.hoo_summary) as f:
        summary = json.load(f)
    with open(args.topics) as f:
        topics_raw = json.load(f)
    scores = load_scores(args.scores)

    print(f"Loading activations from {args.activations} (layer {args.layer})...")
    data = np.load(args.activations, allow_pickle=True)
    task_ids_arr = data["task_ids"]
    X = data[f"layer_{args.layer}"]
    print(f"  activations shape: {X.shape}")

    id_to_idx = {tid: i for i, tid in enumerate(task_ids_arr)}

    task_topic = {}
    for tid in scores:
        if tid not in topics_raw:
            continue
        t = topic_for(topics_raw[tid])
        if t is None:
            continue
        task_topic[tid] = t

    pooled_y = []
    pooled_pred = []
    per_topic = {}

    probe_dir = Path(args.probe_dir)
    for fold in summary["folds"]:
        key = f"ridge_L{args.layer}"
        if not fold["layers"] or key not in fold["layers"]:
            continue
        held_out = set(fold["held_out_groups"])
        probe_id = fold["layers"][key]["probe_id"]
        weights = np.load(probe_dir / f"probe_{probe_id}.npy")

        eval_tids = [
            tid for tid, t in task_topic.items()
            if t in held_out and tid in id_to_idx
        ]
        if len(eval_tids) < 10:
            continue
        indices = np.array([id_to_idx[tid] for tid in eval_tids])
        y = np.array([scores[tid] for tid in eval_tids])
        y_pred = X[indices] @ weights[:-1] + weights[-1]

        train_mean = float(np.mean([
            scores[tid] for tid, t in task_topic.items()
            if t not in held_out and tid in id_to_idx
        ]))
        per_topic[next(iter(held_out))] = {
            "n_eval": len(eval_tids),
            "true_mean": float(y.mean()),
            "true_std": float(y.std()),
            "pred_mean": float(y_pred.mean()),
            "pred_std": float(y_pred.std()),
            "train_mean": train_mean,
            "within_topic_r": float(pearsonr(y, y_pred)[0]) if y.std() > 1e-8 else None,
        }

        pooled_y.append(y)
        pooled_pred.append(y_pred)

        print(f"  fold {fold['fold_idx']} [{','.join(held_out)}]: "
              f"n={len(eval_tids)}, within-r={per_topic[next(iter(held_out))]['within_topic_r']}")

    y_all = np.concatenate(pooled_y)
    p_all = np.concatenate(pooled_pred)
    r = float(pearsonr(y_all, p_all)[0])
    r2 = float(r2_score(y_all, p_all))
    p_shifted = p_all - p_all.mean() + y_all.mean()
    r2_shifted = float(r2_score(y_all, p_shifted))

    out = {
        "layer": args.layer,
        "activations": args.activations,
        "probe_dir": args.probe_dir,
        "n_pooled": int(len(y_all)),
        "n_folds": len(per_topic),
        "pooled_pearson_r": r,
        "pooled_r2_raw": r2,
        "pooled_r2_mean_adjusted": r2_shifted,
        "y_std": float(y_all.std()),
        "pred_std": float(p_all.std()),
        "pred_spread_ratio": float(p_all.std() / y_all.std()),
        "per_topic": per_topic,
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Pooled metrics ===")
    print(f"  N_pooled = {out['n_pooled']} across {out['n_folds']} folds")
    print(f"  pooled Pearson r = {r:.4f}")
    print(f"  pooled R² (raw) = {r2:.4f}")
    print(f"  pooled R² (mean-shifted) = {r2_shifted:.4f}")
    print(f"  pred spread / true spread = {out['pred_spread_ratio']:.3f}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
