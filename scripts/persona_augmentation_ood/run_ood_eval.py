"""Evaluate persona-augmented probes on OOD system prompt experiments.

Trains baseline (10k noprompt) and augmented (10k noprompt + 1500 persona) Ridge
probes, then scores OOD turn_boundary activations to see if the augmented probe
better tracks system-prompt-induced preference shifts.

OOD task IDs are excluded from training to avoid contamination.

Usage: python -m scripts.persona_augmentation_ood.run_ood_eval
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.ood.analysis import (
    compute_p_choose_from_pairwise,
    correlate_deltas,
)
from src.probes.core.activations import load_activations
from src.probes.data_loading import load_thurstonian_scores

# --- Paths ---

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

NOPROMPT_10K_RUN = REPO_ROOT / Path(
    "results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0"
)
NOPROMPT_4K_RUN = REPO_ROOT / Path(
    "results/experiments/main_probes/gemma3_4k_pre_task/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0"
)
NOPROMPT_ACT_DIR = REPO_ROOT / "activations/gemma-3-27b_it/pref_main"

PERSONA_RUNS = {
    "villain": (REPO_ROOT / "results/experiments/mra_exp2/pre_task_active_learning", "syse8f24ac6"),
    "sadist": (REPO_ROOT / "results/experiments/mra_exp3/pre_task_active_learning", "sys39e01d59"),
}

PERSONA_ACT_DIRS = {
    "villain": REPO_ROOT / "activations/gemma-3-27b_it/pref_villain",
    "sadist": REPO_ROOT / "activations/gemma-3-27b_it/pref_sadist",
}

SPLIT_FILES = {
    "a": REPO_ROOT / "configs/measurement/active_learning/mra_exp2_split_a_1000_task_ids.txt",
    "b": REPO_ROOT / "configs/measurement/active_learning/mra_exp2_split_b_500_task_ids.txt",
}

OOD_ACTS_DIR = REPO_ROOT / "activations/gemma-3-27b_it/pref_ood_prompts_parent_removed"
RESULTS_OOD = REPO_ROOT / "results/ood"

SELECTORS = ["turn_boundary:-2", "turn_boundary:-5"]
LAYERS = [25, 32, 39, 46, 53]
ALPHAS = np.logspace(-1, 5, 20)
EVAL_SPLIT_SEED = 42

OUTPUT_DIR = REPO_ROOT / "results/experiments/persona_augmentation_ood"


# --- Helpers ---

def load_split_ids(split: str) -> set[str]:
    with open(SPLIT_FILES[split]) as f:
        return {line.strip() for line in f if line.strip()}


def get_ood_task_ids() -> set[str]:
    """Collect all task IDs used in OOD experiments."""
    ids = set()
    # exp1a: from task IDs file
    ids_file = REPO_ROOT / "configs/measurement/active_learning/ood_exp1a_task_ids.txt"
    with open(ids_file) as f:
        ids.update(line.strip() for line in f if line.strip())
    # exp1b-d: from custom task files
    for fname in ["target_tasks.json", "crossed_tasks.json"]:
        p = REPO_ROOT / "configs/ood/tasks" / fname
        if p.exists():
            tasks = json.load(open(p))
            for t in tasks:
                ids.add(t["task_id"])
    return ids


def persona_run_dir(persona: str, split: str) -> Path:
    results_dir, sys_hash = PERSONA_RUNS[persona]
    n = {"a": 1000, "b": 500}[split]
    prefix = "completion_preference_gemma-3-27b_completion_canonical_seed0"
    suffix = f"mra_exp2_split_{split}_{n}_task_ids"
    return results_dir / f"{prefix}_{sys_hash}_{suffix}"


def align_scores_activations(
    scores: dict[str, float],
    task_ids: np.ndarray,
    act_matrix: np.ndarray,
    restrict_to: set[str] | None = None,
    exclude: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    matched_ids, indices, y_vals = [], [], []
    for tid in sorted(scores.keys()):
        if restrict_to and tid not in restrict_to:
            continue
        if exclude and tid in exclude:
            continue
        if tid in id_to_idx:
            matched_ids.append(tid)
            indices.append(id_to_idx[tid])
            y_vals.append(scores[tid])
    return act_matrix[indices], np.array(y_vals), matched_ids


def train_and_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_sweep: np.ndarray,
    y_sweep: np.ndarray,
) -> tuple[Ridge, StandardScaler, float]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_sweep_s = scaler.transform(X_sweep)

    best_alpha, best_r = None, -np.inf
    for alpha in ALPHAS:
        probe = Ridge(alpha=alpha)
        probe.fit(X_train_s, y_train)
        y_pred = probe.predict(X_sweep_s)
        r, _ = pearsonr(y_sweep, y_pred)
        if r > best_r:
            best_r = float(r)
            best_alpha = float(alpha)

    best_probe = Ridge(alpha=best_alpha)
    best_probe.fit(X_train_s, y_train)
    return best_probe, scaler, best_alpha


def probe_to_npy(probe: Ridge, scaler: StandardScaler) -> np.ndarray:
    """Convert Ridge probe + scaler into a single weight vector + bias for scoring.

    The scaler transforms x -> (x - mean) / std, so:
        score = (x - mean)/std @ w + b = x @ (w/std) + (b - mean @ (w/std))
    """
    w_scaled = probe.coef_ / scaler.scale_
    b_scaled = probe.intercept_ - scaler.mean_ @ w_scaled
    return np.concatenate([w_scaled, [b_scaled]])


def score_activations_with_probe(
    npz_path: Path,
    layer: int,
    probe_npy: np.ndarray,
) -> dict[str, float]:
    data = np.load(npz_path, allow_pickle=True)
    acts = data[f"layer_{layer}"]
    weights, bias = probe_npy[:-1], float(probe_npy[-1])
    scores = acts @ weights + bias
    task_ids = list(data["task_ids"])
    return {tid: float(s) for tid, s in zip(task_ids, scores)}


def compute_ood_deltas(
    rates: dict[str, dict[str, float]],
    acts_dir: Path,
    probe_npy: np.ndarray,
    layer: int,
    acts_filename: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline_rates = rates["baseline"]
    baseline_npz = acts_dir / "baseline" / acts_filename
    baseline_scores = score_activations_with_probe(baseline_npz, layer, probe_npy)

    all_beh, all_probe, all_labels = [], [], []
    for cid in rates:
        if cid == "baseline":
            continue
        cond_npz = acts_dir / cid / acts_filename
        if not cond_npz.exists():
            continue
        cond_scores = score_activations_with_probe(cond_npz, layer, probe_npy)
        for tid, cond_rate in rates[cid].items():
            if tid not in baseline_rates or tid not in baseline_scores or tid not in cond_scores:
                continue
            all_beh.append(cond_rate - baseline_rates[tid])
            all_probe.append(cond_scores[tid] - baseline_scores[tid])
            all_labels.append(cid)

    return np.array(all_beh), np.array(all_probe), np.array(all_labels)


# --- Experiment definitions ---

def load_exp_rates(exp_key: str) -> tuple[dict[str, dict[str, float]], Path]:
    if exp_key == "exp1a":
        pairwise = json.load(open(RESULTS_OOD / "category_preference" / "pairwise.json"))
        rates = compute_p_choose_from_pairwise(pairwise["results"])
        return rates, OOD_ACTS_DIR / "exp1_category"
    elif exp_key == "exp1b":
        pairwise = json.load(open(RESULTS_OOD / "hidden_preference" / "pairwise.json"))
        rates = compute_p_choose_from_pairwise(pairwise["results"])
        rates = {k: v for k, v in rates.items() if not k.startswith("compete_")}
        rates = {k: {tid: v for tid, v in rd.items() if tid.startswith("hidden_")} for k, rd in rates.items()}
        return rates, OOD_ACTS_DIR / "exp1_prompts"
    elif exp_key == "exp1c":
        pairwise = json.load(open(RESULTS_OOD / "crossed_preference" / "pairwise.json"))
        rates = compute_p_choose_from_pairwise(pairwise["results"])
        rates = {k: v for k, v in rates.items() if not k.startswith("compete_")}
        rates = {k: {tid: v for tid, v in rd.items() if tid.startswith("crossed_")} for k, rd in rates.items()}
        return rates, OOD_ACTS_DIR / "exp1_prompts"
    elif exp_key == "exp1d":
        pairwise = json.load(open(RESULTS_OOD / "crossed_preference" / "pairwise.json"))
        rates = compute_p_choose_from_pairwise(pairwise["results"])
        rates = {k: v for k, v in rates.items() if k.startswith("compete_") or k == "baseline"}
        rates = {k: {tid: v for tid, v in rd.items() if tid.startswith("crossed_")} for k, rd in rates.items()}
        return rates, OOD_ACTS_DIR / "exp1_prompts"
    else:
        raise ValueError(f"Unknown experiment: {exp_key}")


# --- Main ---

def run_experiment():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ood_task_ids = get_ood_task_ids()
    print(f"OOD task IDs to exclude from training: {len(ood_task_ids)}")

    # Load noprompt data
    print("Loading noprompt scores...")
    noprompt_10k_scores = load_thurstonian_scores(NOPROMPT_10K_RUN)
    noprompt_4k_scores = load_thurstonian_scores(NOPROMPT_4K_RUN)

    # Split 4k into sweep/test
    rng = np.random.default_rng(EVAL_SPLIT_SEED)
    eval_ids = sorted(noprompt_4k_scores.keys())
    perm = rng.permutation(len(eval_ids))
    half = len(eval_ids) // 2
    noprompt_sweep_scores = {eval_ids[i]: noprompt_4k_scores[eval_ids[i]] for i in perm[:half]}
    noprompt_test_scores = {eval_ids[i]: noprompt_4k_scores[eval_ids[i]] for i in perm[half:]}

    # Count how many 10k tasks overlap with OOD
    overlap = set(noprompt_10k_scores.keys()) & ood_task_ids
    print(f"  10k noprompt tasks overlapping with OOD: {len(overlap)} (will be excluded)")

    # Load persona data (splits A + B = 1500 total)
    split_a_ids = load_split_ids("a")
    split_b_ids = load_split_ids("b")
    all_persona_train_ids = split_a_ids | split_b_ids

    persona_scores = {}
    for persona in PERSONA_RUNS:
        persona_scores[persona] = {}
        for split in ["a", "b"]:
            run_dir = persona_run_dir(persona, split)
            if run_dir.exists():
                persona_scores[persona][split] = load_thurstonian_scores(run_dir)
            else:
                print(f"  WARNING: {persona} split {split} not found at {run_dir}")

    # Load behavioral data for all experiments
    exp_keys = ["exp1a", "exp1b", "exp1c", "exp1d"]
    exp_data = {}
    for key in exp_keys:
        try:
            rates, acts_dir = load_exp_rates(key)
            exp_data[key] = (rates, acts_dir)
        except FileNotFoundError as e:
            print(f"  WARNING: {key} behavioral data not found: {e}")

    all_results = []

    for selector in SELECTORS:
        print(f"\n{'='*60}")
        print(f"Selector: {selector}")
        print(f"{'='*60}")

        acts_filename = f"activations_{selector}.npz"

        # Load noprompt activations
        noprompt_act_path = NOPROMPT_ACT_DIR / acts_filename
        noprompt_task_ids, noprompt_acts = load_activations(noprompt_act_path, layers=LAYERS)

        for layer in LAYERS:
            print(f"\n--- Layer {layer} ---")
            noprompt_act_matrix = noprompt_acts[layer]

            # Baseline: 10k noprompt (excluding OOD tasks)
            X_train_base, y_train_base, train_ids_base = align_scores_activations(
                noprompt_10k_scores, noprompt_task_ids, noprompt_act_matrix,
                exclude=ood_task_ids,
            )
            X_sweep, y_sweep, _ = align_scores_activations(
                noprompt_sweep_scores, noprompt_task_ids, noprompt_act_matrix,
            )
            X_test_np, y_test_np, _ = align_scores_activations(
                noprompt_test_scores, noprompt_task_ids, noprompt_act_matrix,
            )

            probe_base, scaler_base, alpha_base = train_and_sweep(
                X_train_base, y_train_base, X_sweep, y_sweep,
            )
            # Noprompt heldout r
            y_pred_test = probe_base.predict(scaler_base.transform(X_test_np))
            r_test_base, _ = pearsonr(y_test_np, y_pred_test)
            print(f"  Baseline: n_train={len(y_train_base)}, alpha={alpha_base:.1f}, "
                  f"noprompt test r={r_test_base:.3f}")

            probe_base_npy = probe_to_npy(probe_base, scaler_base)

            # Evaluate baseline on OOD experiments
            base_ood_results = {}
            for exp_key in exp_data:
                rates, acts_dir = exp_data[exp_key]
                try:
                    beh, prb, labels = compute_ood_deltas(
                        rates, acts_dir, probe_base_npy, layer, acts_filename,
                    )
                    if len(beh) > 0:
                        stats_result = correlate_deltas(beh, prb)
                        base_ood_results[exp_key] = {
                            "pearson_r": stats_result["pearson_r"],
                            "sign_agreement": stats_result["sign_agreement"],
                            "n": stats_result["n"],
                            "behavioral_deltas": beh.tolist(),
                            "probe_deltas": prb.tolist(),
                            "condition_labels": labels.tolist(),
                        }
                        print(f"    {exp_key}: r={stats_result['pearson_r']:.3f}, "
                              f"sign={stats_result['sign_agreement']:.1%}, n={stats_result['n']}")
                except Exception as e:
                    print(f"    {exp_key}: ERROR — {e}")

            all_results.append({
                "condition": "baseline",
                "donor": None,
                "selector": selector,
                "layer": layer,
                "best_alpha": alpha_base,
                "n_train": len(y_train_base),
                "noprompt_test_r": float(r_test_base),
                "ood_results": base_ood_results,
            })

            # Augmented: per donor persona
            for persona in PERSONA_RUNS:
                persona_act_dir = PERSONA_ACT_DIRS[persona]
                persona_act_path = persona_act_dir / acts_filename
                if not persona_act_path.exists():
                    print(f"  SKIP {persona}: {persona_act_path} not found")
                    continue

                p_task_ids, p_acts = load_activations(persona_act_path, layers=LAYERS)
                p_act_matrix = p_acts[layer]

                # Train: noprompt (excl OOD) + 1500 persona (splits A+B)
                X_p_train, y_p_train, _ = align_scores_activations(
                    {**persona_scores[persona].get("a", {}),
                     **persona_scores[persona].get("b", {})},
                    p_task_ids, p_act_matrix,
                    restrict_to=all_persona_train_ids,
                )
                X_train_aug = np.concatenate([X_train_base, X_p_train])
                y_train_aug = np.concatenate([y_train_base, y_p_train])

                # Sweep on same noprompt heldout
                probe_aug, scaler_aug, alpha_aug = train_and_sweep(
                    X_train_aug, y_train_aug, X_sweep, y_sweep,
                )
                y_pred_aug = probe_aug.predict(scaler_aug.transform(X_test_np))
                r_test_aug, _ = pearsonr(y_test_np, y_pred_aug)
                print(f"  +{persona}: n_train={len(y_train_aug)}, alpha={alpha_aug:.1f}, "
                      f"noprompt test r={r_test_aug:.3f}")

                probe_aug_npy = probe_to_npy(probe_aug, scaler_aug)

                # Evaluate augmented on OOD
                aug_ood_results = {}
                for exp_key in exp_data:
                    rates, acts_dir = exp_data[exp_key]
                    try:
                        beh, prb, labels = compute_ood_deltas(
                            rates, acts_dir, probe_aug_npy, layer, acts_filename,
                        )
                        if len(beh) > 0:
                            stats_result = correlate_deltas(beh, prb)
                            aug_ood_results[exp_key] = {
                                "pearson_r": stats_result["pearson_r"],
                                "sign_agreement": stats_result["sign_agreement"],
                                "n": stats_result["n"],
                                "behavioral_deltas": beh.tolist(),
                                "probe_deltas": prb.tolist(),
                                "condition_labels": labels.tolist(),
                            }
                            print(f"    {exp_key}: r={stats_result['pearson_r']:.3f}, "
                                  f"sign={stats_result['sign_agreement']:.1%}, n={stats_result['n']}")
                    except Exception as e:
                        print(f"    {exp_key}: ERROR — {e}")

                all_results.append({
                    "condition": "augmented",
                    "donor": persona,
                    "selector": selector,
                    "layer": layer,
                    "best_alpha": alpha_aug,
                    "n_train": len(y_train_aug),
                    "noprompt_test_r": float(r_test_aug),
                    "ood_results": aug_ood_results,
                })

    # Save results (strip large arrays for summary)
    summary = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != "ood_results"}
        entry["ood_results"] = {}
        for exp_key, ood in r["ood_results"].items():
            entry["ood_results"][exp_key] = {
                k: v for k, v in ood.items()
                if k not in ("behavioral_deltas", "probe_deltas", "condition_labels")
            }
        summary.append(entry)

    with open(OUTPUT_DIR / "ood_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save full results (with arrays for plotting)
    with open(OUTPUT_DIR / "ood_eval_full.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_experiment()
