"""Compute clean separate-run LOO pooled metrics for all 4 cross-model bar LOO bars.

For each (model, probe-type), apply each fold's saved probe to its held-out
topic's tasks in the SEPARATE 4k measurement run, using run-2 thurstonian
utilities as truth. Pool predictions across folds, compute Pearson r.

Pooled pairwise accuracy is computed on a topic-stratified pair sample that
matches the within-distribution uniform_eval set's within/cross-topic ratio.
Random pairing on a 14-topic pool would be ~85% cross-topic; uniform_eval is
~55% within-topic. Matching that ratio puts LOO and within-dist accuracy on
the same distributional footing, so the LOO bars don't get a free win from
oversampling easy cross-topic pairs.

Saves results to `pooled_metrics_clean.json` next to each LOO output.
"""

from __future__ import annotations

import csv
import json
from math import atanh, sqrt, tanh
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[2]
TOPICS = REPO / "data/topics/topics.json"
N_PAIRS = 5000
RNG_SEED = 0
# Match within-dist uniform_eval's distribution (55% within-topic, 45% cross-topic
# on Gemma; near-identical on Qwen).
TARGET_FRAC_WITHIN_TOPIC = 0.55


CASES = [
    {
        "name": "gemma_probe",
        "hoo_dir": REPO / "results/probes/gemma3_10k_hoo_topic_tb-5",
        "activations": REPO / "activations/gemma-3-27b_it/pref_main/activations_turn_boundary:-5.npz",
        "scores_run1": REPO / "results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_80fa9dc8.csv",
        "scores_run2": REPO / "results/experiments/main_probes/gemma3_4k_pre_task/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_a67822c5.csv",
        "layer": 32,
    },
    {
        "name": "qwen_probe",
        "hoo_dir": REPO / "results/probes/qwen35_122b/qwen35_122b_hoo_topic_turn_boundary_m1_L38_uniform",
        "activations": REPO / "activations/qwen35_122b/pref_main/activations_turn_boundary:-1.npz",
        "scores_run1": REPO / "results/experiments/qwen35_10k_active_learning/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d/thurstonian_6746a725.csv",
        "scores_run2": REPO / "results/experiments/qwen35_4k_active_learning/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d_qwen35_4k_task_ids/thurstonian_b5bc6473.csv",
        "layer": 38,
    },
    {
        "name": "gemma_emb_baseline",
        "hoo_dir": REPO / "results/probes/qwen3_emb_8b_hoo_topic",
        "activations": REPO / "activations/qwen3-emb_8b/pref_main/activations_prompt_last.npz",
        "scores_run1": REPO / "results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_80fa9dc8.csv",
        "scores_run2": REPO / "results/experiments/main_probes/gemma3_4k_pre_task/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_a67822c5.csv",
        "layer": 0,
    },
    {
        "name": "qwen_emb_baseline",
        "hoo_dir": REPO / "results/probes/qwen3_emb_8b_qwen35_hoo_topic",
        "activations": REPO / "activations/qwen3-emb_8b/qwen35_pool/activations_prompt_last.npz",
        "scores_run1": REPO / "results/experiments/qwen35_10k_active_learning/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d/thurstonian_6746a725.csv",
        "scores_run2": REPO / "results/experiments/qwen35_4k_active_learning/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d_qwen35_4k_task_ids/thurstonian_b5bc6473.csv",
        "layer": 0,
    },
]


def fisher_z_ci(r: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n is None or n < 4:
        return float("nan"), float("nan")
    r = max(min(r, 0.999999), -0.999999)
    zhat = atanh(r)
    se = 1.0 / sqrt(n - 3)
    return tanh(zhat - z * se), tanh(zhat + z * se)


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n is None or n <= 0:
        return float("nan"), float("nan")
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def topic_for(entry):
    if isinstance(entry, dict) and "primary" in entry:
        return entry["primary"]
    if isinstance(entry, dict):
        for v in entry.values():
            if isinstance(v, dict) and "primary" in v:
                return v["primary"]
    return None


def load_scores(path: Path) -> dict[str, float]:
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["task_id"]] = float(row["mu"])
    return out


def evaluate_case(case: dict, scores_path: Path, label: str) -> dict:
    summary = json.loads((case["hoo_dir"] / "hoo_summary.json").read_text())
    topics_raw = json.loads(TOPICS.read_text())
    scores = load_scores(scores_path)

    data = np.load(case["activations"], allow_pickle=True)
    task_ids = data["task_ids"]
    X = data[f"layer_{case['layer']}"]
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}

    task_topic = {}
    for tid in scores:
        if tid in topics_raw:
            t = topic_for(topics_raw[tid])
            if t is not None:
                task_topic[tid] = t

    pooled_y, pooled_pred, pooled_topic = [], [], []
    for fold in summary["folds"]:
        key = f"ridge_L{case['layer']}"
        if key not in fold["layers"]:
            continue
        held_out = set(fold["held_out_groups"])
        probe_id = fold["layers"][key]["probe_id"]
        weights = np.load(case["hoo_dir"] / "probes" / f"probe_{probe_id}.npy")

        eval_tids = [tid for tid, t in task_topic.items()
                     if t in held_out and tid in id_to_idx]
        if len(eval_tids) < 5:
            continue
        idx = np.array([id_to_idx[tid] for tid in eval_tids])
        y = np.array([scores[tid] for tid in eval_tids])
        y_pred = X[idx] @ weights[:-1] + weights[-1]
        pooled_y.append(y)
        pooled_pred.append(y_pred)
        pooled_topic.append(np.array([next(iter(held_out))] * len(eval_tids)))

    y_all = np.concatenate(pooled_y)
    p_all = np.concatenate(pooled_pred)
    t_all = np.concatenate(pooled_topic)
    pooled_r = float(pearsonr(y_all, p_all)[0])
    pooled_r_lo, pooled_r_hi = fisher_z_ci(pooled_r, len(y_all))

    # Topic-stratified pair sampling: match TARGET_FRAC_WITHIN_TOPIC.
    rng = np.random.default_rng(RNG_SEED)
    n = len(y_all)
    n_within_target = int(round(N_PAIRS * TARGET_FRAC_WITHIN_TOPIC))
    n_cross_target = N_PAIRS - n_within_target

    def sample_within_topic(n_target: int) -> tuple[np.ndarray, np.ndarray]:
        out_i, out_j = [], []
        # Per-topic pair budget proportional to topic size² (favouring big topics).
        unique_topics, counts = np.unique(t_all, return_counts=True)
        weights_per_topic = counts.astype(float) ** 2
        weights_per_topic /= weights_per_topic.sum()
        per_topic_n = (weights_per_topic * n_target).astype(int)
        leftover = n_target - per_topic_n.sum()
        per_topic_n[:leftover] += 1
        for topic, n_t in zip(unique_topics, per_topic_n):
            if n_t == 0:
                continue
            idx_in_topic = np.flatnonzero(t_all == topic)
            if len(idx_in_topic) < 2:
                continue
            for _ in range(n_t):
                a, b = rng.choice(idx_in_topic, size=2, replace=False)
                out_i.append(a)
                out_j.append(b)
        return np.array(out_i), np.array(out_j)

    def sample_cross_topic(n_target: int) -> tuple[np.ndarray, np.ndarray]:
        out_i, out_j = [], []
        attempts = 0
        max_attempts = n_target * 50
        while len(out_i) < n_target and attempts < max_attempts:
            a, b = rng.integers(0, n, size=2)
            attempts += 1
            if t_all[a] != t_all[b]:
                out_i.append(a)
                out_j.append(b)
        return np.array(out_i), np.array(out_j)

    i_w, j_w = sample_within_topic(n_within_target)
    i_c, j_c = sample_cross_topic(n_cross_target)
    i = np.concatenate([i_w, i_c])
    j = np.concatenate([j_w, j_c])
    keep = (i != j) & (y_all[i] != y_all[j])
    i, j = i[keep], j[keep]
    agree = ((p_all[i] - p_all[j]) * (y_all[i] - y_all[j])) > 0
    pooled_acc = float(agree.mean())
    pooled_acc_lo, pooled_acc_hi = wilson_ci(pooled_acc, len(agree))

    # Also report unstratified for transparency.
    rng2 = np.random.default_rng(RNG_SEED + 1)
    ii = rng2.integers(0, n, size=N_PAIRS)
    jj = rng2.integers(0, n, size=N_PAIRS)
    keep_un = (ii != jj) & (y_all[ii] != y_all[jj])
    ii, jj = ii[keep_un], jj[keep_un]
    agree_un = ((p_all[ii] - p_all[jj]) * (y_all[ii] - y_all[jj])) > 0
    same_topic_un = t_all[ii] == t_all[jj]

    return {
        "label": label,
        "scores_path": str(scores_path.relative_to(REPO)),
        "n_pooled": int(n),
        "n_pairs": int(len(agree)),
        "pair_strata_target_frac_within_topic": TARGET_FRAC_WITHIN_TOPIC,
        "pooled_pearson_r": round(pooled_r, 4),
        "pooled_r_ci_lo": round(pooled_r_lo, 4),
        "pooled_r_ci_hi": round(pooled_r_hi, 4),
        "pooled_pairwise_acc": round(pooled_acc, 4),
        "pooled_acc_ci_lo": round(pooled_acc_lo, 4),
        "pooled_acc_ci_hi": round(pooled_acc_hi, 4),
        "diagnostics_unstratified": {
            "n_pairs": int(len(agree_un)),
            "frac_within_topic": float(same_topic_un.mean()),
            "pooled_pairwise_acc": float(agree_un.mean()),
            "within_topic_acc": float(agree_un[same_topic_un].mean()) if same_topic_un.any() else None,
            "cross_topic_acc": float(agree_un[~same_topic_un].mean()) if (~same_topic_un).any() else None,
        },
    }


def main() -> None:
    for case in CASES:
        print(f"\n=== {case['name']} ===")
        out = {
            "case": case["name"],
            "hoo_dir": str(case["hoo_dir"].relative_to(REPO)),
            "layer": case["layer"],
            "same_run": evaluate_case(case, case["scores_run1"], "run 1 (same as training)"),
            "separate_run": evaluate_case(case, case["scores_run2"], "run 2 (separate measurement, 4k pool)"),
        }
        for k in ("same_run", "separate_run"):
            v = out[k]
            print(f"  {k}: r={v['pooled_pearson_r']} CI=[{v['pooled_r_ci_lo']}, {v['pooled_r_ci_hi']}]  "
                  f"acc={v['pooled_pairwise_acc']} CI=[{v['pooled_acc_ci_lo']}, {v['pooled_acc_ci_hi']}]  n={v['n_pooled']}")
        out_path = case["hoo_dir"] / "pooled_metrics_clean.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  wrote {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
