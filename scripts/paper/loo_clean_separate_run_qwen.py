"""LOO eval with run-2 labels for Qwen-3.5-122B.

Mirror of `loo_clean_separate_run.py` for Gemma. Same setup: per-fold L38 probes
trained on 10k pool (run 1, sysbd0c6a4d), evaluated on the 4k separate
measurement run.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[2]
HOO_DIR = REPO / "results/probes/qwen35_122b/qwen35_122b_hoo_topic_turn_boundary_m1_L38_uniform"
PROBE_DIR = HOO_DIR / "probes"
ACT_PATH = REPO / "activations/qwen35_122b/pref_main/activations_turn_boundary:-1.npz"
SCORES_RUN1 = REPO / "results/experiments/qwen35_10k_active_learning/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d/thurstonian_6746a725.csv"
SCORES_RUN2 = REPO / "results/experiments/qwen35_4k_active_learning/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d_qwen35_4k_task_ids/thurstonian_b5bc6473.csv"
TOPICS = REPO / "data/topics/topics.json"
LAYER = 38
N_PAIRS = 5000
RNG_SEED = 0


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


def evaluate(scores_path: Path, label: str) -> None:
    summary = json.loads((HOO_DIR / "hoo_summary.json").read_text())
    topics_raw = json.loads(TOPICS.read_text())
    scores = load_scores(scores_path)

    data = np.load(ACT_PATH, allow_pickle=True)
    task_ids = data["task_ids"]
    X = data[f"layer_{LAYER}"]
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}

    task_topic = {}
    for tid in scores:
        if tid in topics_raw:
            t = topic_for(topics_raw[tid])
            if t is not None:
                task_topic[tid] = t

    pooled_y, pooled_pred = [], []
    rows = []
    for fold in summary["folds"]:
        key = f"ridge_L{LAYER}"
        if key not in fold["layers"]:
            continue
        held_out = set(fold["held_out_groups"])
        probe_id = fold["layers"][key]["probe_id"]
        weights = np.load(PROBE_DIR / f"probe_{probe_id}.npy")

        eval_tids = [tid for tid, t in task_topic.items()
                     if t in held_out and tid in id_to_idx]
        if len(eval_tids) < 5:
            continue
        idx = np.array([id_to_idx[tid] for tid in eval_tids])
        y = np.array([scores[tid] for tid in eval_tids])
        y_pred = X[idx] @ weights[:-1] + weights[-1]
        pooled_y.append(y)
        pooled_pred.append(y_pred)
        rows.append({
            "topic": next(iter(held_out)),
            "n": len(eval_tids),
            "within_topic_r": float(pearsonr(y, y_pred)[0]) if y.std() > 1e-8 else float("nan"),
        })

    y_all = np.concatenate(pooled_y)
    p_all = np.concatenate(pooled_pred)

    pooled_r = float(pearsonr(y_all, p_all)[0])
    rng = np.random.default_rng(RNG_SEED)
    n = len(y_all)
    i = rng.integers(0, n, size=N_PAIRS)
    j = rng.integers(0, n, size=N_PAIRS)
    keep = (i != j) & (y_all[i] != y_all[j])
    i, j = i[keep], j[keep]
    agree = ((p_all[i] - p_all[j]) * (y_all[i] - y_all[j])) > 0
    pooled_acc = float(agree.mean())

    print(f"\n=== Qwen-3.5-122B L{LAYER} LOO, labels from {label} ===")
    print(f"  N_pooled                     = {n}")
    print(f"  pooled Pearson r             = {pooled_r:.4f}")
    print(f"  pooled accuracy ({len(agree)} pairs) = {pooled_acc:.4f}")
    print(f"  mean per-topic Pearson r     = {np.mean([r['within_topic_r'] for r in rows]):.4f}")
    print()
    print(f"  {'topic':<25s} {'n':>6s} {'within_r':>10s}")
    for r in sorted(rows, key=lambda r: r["topic"]):
        print(f"  {r['topic']:<25s} {r['n']:>6d} {r['within_topic_r']:>10.3f}")


def main() -> None:
    evaluate(SCORES_RUN1, "run 1 (same as training, 10k pool)")
    evaluate(SCORES_RUN2, "run 2 (separate measurement, 4k pool)")


if __name__ == "__main__":
    main()
