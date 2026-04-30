"""Investigate why LOO outperforms within-distribution heldout in Pearson r but
underperforms in pairwise accuracy for Gemma.

Reconstructs LOO predictions on the 9995-task pool by applying each fold's saved
ridge probe (L32) to its held-out topic's activations. Then computes:

  - Pooled Pearson r (matches existing pooled_metrics)
  - Mean per-topic Pearson r (the within-topic variant; expected lower)
  - Pooled accuracy on cross-topic random pairs (NEW)
  - Pooled accuracy on within-topic-only pairs (matches mean uniform_hoo_acc spirit)

The point: ``pooled_pearson_r`` mixes within- and between-topic variation, so it
benefits from topic identity. ``mean_uniform_hoo_acc`` is per-topic only.
Switching the LOO accuracy bar to a pooled-cross-topic measure makes the LOO
Pearson r and accuracy bars apples-to-apples.
"""

from __future__ import annotations

import csv
import json
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[2]
HOO_DIR = REPO / "results/probes/gemma3_10k_hoo_topic_tb-5"
PROBE_DIR = HOO_DIR / "probes"
ACT_PATH = REPO / "activations/gemma-3-27b_it/pref_main/activations_turn_boundary:-5.npz"
SCORES = REPO / "results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_80fa9dc8.csv"
TOPICS = REPO / "data/topics/topics.json"
LAYER = 32
RNG_SEED = 0
N_PAIRS = 5000  # match the within-distribution test set's pair count


def load_scores(path: Path) -> dict[str, float]:
    out = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["task_id"]] = float(row["mu"])
    return out


def topic_for(entry):
    if isinstance(entry, dict) and "primary" in entry:
        return entry["primary"]
    if isinstance(entry, dict):
        for v in entry.values():
            if isinstance(v, dict) and "primary" in v:
                return v["primary"]
    return None


def main() -> None:
    summary = json.loads((HOO_DIR / "hoo_summary.json").read_text())
    topics_raw = json.loads(TOPICS.read_text())
    scores = load_scores(SCORES)

    print(f"Loading activations layer {LAYER}…")
    data = np.load(ACT_PATH, allow_pickle=True)
    task_ids = data["task_ids"]
    X = data[f"layer_{LAYER}"]
    print(f"  X.shape = {X.shape}")

    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    task_topic = {}
    for tid in scores:
        if tid in topics_raw:
            t = topic_for(topics_raw[tid])
            if t is not None:
                task_topic[tid] = t

    pooled_y: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    pooled_topic: list[np.ndarray] = []
    per_topic_r: dict[str, float] = {}
    per_topic_acc: dict[str, float] = {}
    rng = np.random.default_rng(RNG_SEED)

    for fold in summary["folds"]:
        key = f"ridge_L{LAYER}"
        if key not in fold["layers"]:
            continue
        held_out = set(fold["held_out_groups"])
        probe_id = fold["layers"][key]["probe_id"]
        weights = np.load(PROBE_DIR / f"probe_{probe_id}.npy")

        eval_tids = [tid for tid, t in task_topic.items()
                     if t in held_out and tid in id_to_idx]
        if len(eval_tids) < 10:
            continue
        idx = np.array([id_to_idx[tid] for tid in eval_tids])
        y = np.array([scores[tid] for tid in eval_tids])
        y_pred = X[idx] @ weights[:-1] + weights[-1]
        pooled_y.append(y)
        pooled_pred.append(y_pred)
        pooled_topic.append(np.array([next(iter(held_out))] * len(eval_tids)))

        if y.std() > 1e-8:
            per_topic_r[next(iter(held_out))] = float(pearsonr(y, y_pred)[0])
        # Within-topic random-pair accuracy
        n = len(y)
        n_pairs_topic = min(N_PAIRS // summary["n_folds"], n * (n - 1) // 2)
        i_idx = rng.integers(0, n, size=n_pairs_topic)
        j_idx = rng.integers(0, n, size=n_pairs_topic)
        keep = i_idx != j_idx
        i_idx, j_idx = i_idx[keep], j_idx[keep]
        if len(i_idx) > 0:
            agree = ((y_pred[i_idx] - y_pred[j_idx]) * (y[i_idx] - y[j_idx])) > 0
            per_topic_acc[next(iter(held_out))] = float(agree.mean())

    y_all = np.concatenate(pooled_y)
    p_all = np.concatenate(pooled_pred)
    t_all = np.concatenate(pooled_topic)

    pooled_r = float(pearsonr(y_all, p_all)[0])
    mean_per_topic_r = float(np.mean(list(per_topic_r.values())))

    # Pooled cross-topic accuracy: random pairs from the full pool.
    n = len(y_all)
    rng2 = np.random.default_rng(RNG_SEED + 1)
    i = rng2.integers(0, n, size=N_PAIRS)
    j = rng2.integers(0, n, size=N_PAIRS)
    keep = (i != j) & (y_all[i] != y_all[j])
    i, j = i[keep], j[keep]
    agree_cross = ((p_all[i] - p_all[j]) * (y_all[i] - y_all[j])) > 0
    pooled_acc_cross_topic = float(agree_cross.mean())

    # Pooled within-topic-only accuracy: random pairs but restricted to same topic.
    same_topic = t_all[i] == t_all[j]
    if same_topic.sum() > 0:
        agree_within = ((p_all[i][same_topic] - p_all[j][same_topic])
                        * (y_all[i][same_topic] - y_all[j][same_topic])) > 0
        pooled_acc_within_topic = float(agree_within.mean())
        n_within = int(same_topic.sum())
    else:
        pooled_acc_within_topic = float("nan")
        n_within = 0

    # Pooled different-topic-only accuracy: pairs across topics only.
    diff_topic = ~same_topic
    if diff_topic.sum() > 0:
        agree_diff = ((p_all[i][diff_topic] - p_all[j][diff_topic])
                      * (y_all[i][diff_topic] - y_all[j][diff_topic])) > 0
        pooled_acc_cross_only = float(agree_diff.mean())
        n_cross_only = int(diff_topic.sum())
    else:
        pooled_acc_cross_only = float("nan")
        n_cross_only = 0

    mean_per_topic_acc = float(np.mean(list(per_topic_acc.values())))

    print()
    print(f"=== Gemma-3-27B Ridge L{LAYER}, LOO across {len(per_topic_r)} folds ===")
    print(f"  N_pooled                       = {len(y_all)}")
    print(f"  pooled Pearson r               = {pooled_r:.4f}  (mixes within + between topic)")
    print(f"  mean per-topic Pearson r       = {mean_per_topic_r:.4f}  (within-topic only)")
    print()
    print(f"  pooled accuracy, all pairs     = {pooled_acc_cross_topic:.4f}  ({len(agree_cross)} pairs; cross+within)")
    print(f"  pooled accuracy, cross-topic   = {pooled_acc_cross_only:.4f}  ({n_cross_only} pairs)")
    print(f"  pooled accuracy, within-topic  = {pooled_acc_within_topic:.4f}  ({n_within} pairs)")
    print(f"  mean per-topic random-pair acc = {mean_per_topic_acc:.4f}  (within-topic, per-fold)")
    print()
    print("Within-distribution test (for reference):")
    print(f"  Pearson r          = 0.867 on n=2019")
    print(f"  uniform pair acc   = 0.802 on n_pairs=5091 (pairs from the standard 2k uniform-eval pool)")
    print()
    print("Per-topic breakdown (LOO):")
    print(f"  {'topic':<25s} {'r':>8s} {'acc':>8s} {'n':>6s}")
    for t in sorted(per_topic_r.keys()):
        n_t = int((t_all == t).sum())
        print(f"  {t:<25s} {per_topic_r[t]:>8.3f} {per_topic_acc.get(t, float('nan')):>8.3f} {n_t:>6d}")


if __name__ == "__main__":
    main()
