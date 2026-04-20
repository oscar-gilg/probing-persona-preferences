"""Persona augmentation experiment: does adding persona data to the 10k noprompt probe help?

Trains two probe conditions per (persona, selector, layer):
1. Baseline: 10k noprompt only
2. Augmented: 10k noprompt + 1000 persona (split A)

Alpha sweep on heldout noprompt (+ persona split B for augmented).
Evaluation on noprompt test set + persona split C for all personas.

Usage: python -m scripts.persona_augmentation.run_experiment
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.probes.core.activations import load_activations
from src.probes.data_loading import load_thurstonian_scores

# --- Paths ---

NOPROMPT_10K_RUN = Path(
    "results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0"
)
NOPROMPT_4K_RUN = Path(
    "results/experiments/main_probes/gemma3_4k_pre_task/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0"
)
NOPROMPT_ACT_DIR = Path("activations/gemma-3-27b_it/pref_main")

PERSONA_RUNS = {
    "villain": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "syse8f24ac6"),
    "midwest": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "sys5d504504"),
    "sadist": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys39e01d59"),
}

PERSONA_ACT_DIRS = {
    "villain": Path("activations/gemma-3-27b_it/pref_villain"),
    "midwest": Path("activations/gemma-3-27b_it/pref_midwest"),
    "sadist": Path("activations/gemma-3-27b_it/pref_sadist"),
}

SPLIT_FILES = {
    "a": Path("configs/measurement/active_learning/mra_exp2_split_a_1000_task_ids.txt"),
    "b": Path("configs/measurement/active_learning/mra_exp2_split_b_500_task_ids.txt"),
    "c": Path("configs/measurement/active_learning/mra_exp2_split_c_1000_task_ids.txt"),
}

# All non-default personas for cross-persona eval
ALL_PERSONA_RUNS = {
    "villain": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "syse8f24ac6"),
    "aesthete": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "sys021d8ca1"),
    "midwest": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "sys5d504504"),
    "provocateur": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sysf4d93514"),
    "trickster": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys09a42edc"),
    "autocrat": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys1c18219a"),
    "sadist": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys39e01d59"),
}

SELECTORS = ["turn_boundary:-2", "turn_boundary:-5"]
LAYERS = [25, 32, 39, 46, 53]
ALPHAS = np.logspace(-1, 5, 20)
EVAL_SPLIT_SEED = 42

OUTPUT_DIR = Path("results/experiments/persona_augmentation")


# --- Helpers ---

def load_split_ids(split: str) -> set[str]:
    with open(SPLIT_FILES[split]) as f:
        return {line.strip() for line in f if line.strip()}


def persona_run_dir(persona: str, split: str) -> Path:
    results_dir, sys_hash = ALL_PERSONA_RUNS[persona]
    n = {"a": 1000, "b": 500, "c": 1000}[split]
    prefix = "completion_preference_gemma-3-27b_completion_canonical_seed0"
    suffix = f"mra_exp2_split_{split}_{n}_task_ids"
    dirname = f"{prefix}_{sys_hash}_{suffix}"
    return results_dir / dirname


def load_act_file(act_dir: Path, selector: str) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    path = act_dir / f"activations_{selector}.npz"
    task_ids, act_dict = load_activations(path, layers=LAYERS)
    return task_ids, act_dict


def align_scores_activations(
    scores: dict[str, float],
    task_ids: np.ndarray,
    act_matrix: np.ndarray,
    restrict_to: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Match scores to activation rows. Returns (X, y, matched_ids)."""
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    matched_ids, indices, y_vals = [], [], []
    for tid in sorted(scores.keys()):
        if restrict_to and tid not in restrict_to:
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
) -> tuple[Ridge, StandardScaler, float, list[dict]]:
    """Train Ridge with alpha sweep. Returns (best_probe, scaler, best_alpha, sweep_results)."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_sweep_s = scaler.transform(X_sweep)

    best_alpha, best_r = None, -np.inf
    sweep_results = []
    for alpha in ALPHAS:
        probe = Ridge(alpha=alpha)
        probe.fit(X_train_s, y_train)
        y_pred = probe.predict(X_sweep_s)
        r, _ = pearsonr(y_sweep, y_pred)
        r = float(r)
        sweep_results.append({"alpha": float(alpha), "r": r})
        if r > best_r:
            best_r = r
            best_alpha = float(alpha)

    best_probe = Ridge(alpha=best_alpha)
    best_probe.fit(X_train_s, y_train)
    return best_probe, scaler, best_alpha, sweep_results


def evaluate_probe(
    probe: Ridge,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    y_pred = probe.predict(scaler.transform(X_test))
    r, p = pearsonr(y_test, y_pred)
    ss_res = float(np.sum((y_test - y_pred) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"pearson_r": float(r), "r2": float(r2), "n": len(y_test)}


# --- Main ---

def run_experiment():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load noprompt data (shared across all conditions)
    print("Loading noprompt 10k scores...")
    noprompt_10k_scores = load_thurstonian_scores(NOPROMPT_10K_RUN)
    noprompt_4k_scores = load_thurstonian_scores(NOPROMPT_4K_RUN)

    # Split 4k heldout into sweep/test halves (same seed as existing probe)
    rng = np.random.default_rng(EVAL_SPLIT_SEED)
    eval_ids = sorted(noprompt_4k_scores.keys())
    perm = rng.permutation(len(eval_ids))
    half = len(eval_ids) // 2
    noprompt_sweep_ids = {eval_ids[i] for i in perm[:half]}
    noprompt_test_ids = {eval_ids[i] for i in perm[half:]}
    noprompt_sweep_scores = {tid: noprompt_4k_scores[tid] for tid in noprompt_sweep_ids}
    noprompt_test_scores = {tid: noprompt_4k_scores[tid] for tid in noprompt_test_ids}

    print(f"  10k train: {len(noprompt_10k_scores)}, sweep: {len(noprompt_sweep_scores)}, test: {len(noprompt_test_scores)}")

    # Load persona split IDs
    split_a_ids = load_split_ids("a")
    split_b_ids = load_split_ids("b")
    split_c_ids = load_split_ids("c")

    # Load persona scores
    persona_scores = {}
    for persona in ALL_PERSONA_RUNS:
        persona_scores[persona] = {}
        for split in ["a", "b", "c"]:
            run_dir = persona_run_dir(persona, split)
            if run_dir.exists():
                persona_scores[persona][split] = load_thurstonian_scores(run_dir)
            else:
                print(f"  WARNING: {persona} split {split} not found at {run_dir}")

    # Check which donor personas have activations
    available_donors = []
    for persona in ["sadist", "villain", "midwest"]:
        act_dir = PERSONA_ACT_DIRS[persona]
        if act_dir.exists():
            available_donors.append(persona)
        else:
            print(f"  SKIPPING {persona}: activations not found at {act_dir}")

    if not available_donors:
        print("\nNo donor activations available. Run extraction first (see spec).")
        return

    all_results = []

    for selector in SELECTORS:
        print(f"\n{'='*60}")
        print(f"Selector: {selector}")
        print(f"{'='*60}")

        # Load noprompt activations
        noprompt_task_ids, noprompt_acts = load_act_file(NOPROMPT_ACT_DIR, selector)

        for layer in LAYERS:
            print(f"\n--- Layer {layer} ---")
            noprompt_act_matrix = noprompt_acts[layer]

            # Baseline: 10k noprompt
            X_train_base, y_train_base, _ = align_scores_activations(
                noprompt_10k_scores, noprompt_task_ids, noprompt_act_matrix,
            )
            X_sweep_base, y_sweep_base, _ = align_scores_activations(
                noprompt_sweep_scores, noprompt_task_ids, noprompt_act_matrix,
            )
            X_test_noprompt, y_test_noprompt, _ = align_scores_activations(
                noprompt_test_scores, noprompt_task_ids, noprompt_act_matrix,
            )

            probe_base, scaler_base, alpha_base, sweep_base = train_and_sweep(
                X_train_base, y_train_base, X_sweep_base, y_sweep_base,
            )
            eval_noprompt_base = evaluate_probe(probe_base, scaler_base, X_test_noprompt, y_test_noprompt)
            print(f"  Baseline: alpha={alpha_base:.1f}, noprompt test r={eval_noprompt_base['pearson_r']:.3f}")

            # Evaluate baseline on persona test sets (using noprompt activations for split C tasks)
            base_persona_evals = {}
            for persona in ALL_PERSONA_RUNS:
                if "c" not in persona_scores[persona]:
                    continue
                X_p, y_p, _ = align_scores_activations(
                    persona_scores[persona]["c"], noprompt_task_ids, noprompt_act_matrix,
                    restrict_to=split_c_ids,
                )
                if len(y_p) < 10:
                    continue
                base_persona_evals[persona] = evaluate_probe(probe_base, scaler_base, X_p, y_p)

            result_base = {
                "condition": "baseline",
                "donor": None,
                "selector": selector,
                "layer": layer,
                "best_alpha": alpha_base,
                "alpha_sweep": sweep_base,
                "n_train": len(y_train_base),
                "noprompt_test": eval_noprompt_base,
                "persona_test": base_persona_evals,
            }
            all_results.append(result_base)

            # Augmented: per donor persona
            for persona in available_donors:
                if "a" not in persona_scores[persona]:
                    continue

                persona_act_ids, persona_acts = load_act_file(PERSONA_ACT_DIRS[persona], selector)
                persona_act_matrix = persona_acts[layer]

                # Train: 10k noprompt + 1000 persona split A
                X_p_train, y_p_train, _ = align_scores_activations(
                    persona_scores[persona]["a"], persona_act_ids, persona_act_matrix,
                    restrict_to=split_a_ids,
                )
                X_train_aug = np.concatenate([X_train_base, X_p_train])
                y_train_aug = np.concatenate([y_train_base, y_p_train])

                # Sweep: noprompt sweep + persona split B
                X_sweep_parts = [X_sweep_base]
                y_sweep_parts = [y_sweep_base]
                if "b" in persona_scores[persona]:
                    X_p_sweep, y_p_sweep, _ = align_scores_activations(
                        persona_scores[persona]["b"], persona_act_ids, persona_act_matrix,
                        restrict_to=split_b_ids,
                    )
                    X_sweep_parts.append(X_p_sweep)
                    y_sweep_parts.append(y_p_sweep)
                X_sweep_aug = np.concatenate(X_sweep_parts)
                y_sweep_aug = np.concatenate(y_sweep_parts)

                probe_aug, scaler_aug, alpha_aug, sweep_aug = train_and_sweep(
                    X_train_aug, y_train_aug, X_sweep_aug, y_sweep_aug,
                )

                # Eval on noprompt test
                eval_noprompt_aug = evaluate_probe(probe_aug, scaler_aug, X_test_noprompt, y_test_noprompt)

                # Eval on persona test sets
                # For the donor persona: use persona activations
                aug_persona_evals = {}
                if "c" in persona_scores[persona]:
                    X_p_test, y_p_test, _ = align_scores_activations(
                        persona_scores[persona]["c"], persona_act_ids, persona_act_matrix,
                        restrict_to=split_c_ids,
                    )
                    if len(y_p_test) >= 10:
                        aug_persona_evals[persona] = evaluate_probe(probe_aug, scaler_aug, X_p_test, y_p_test)

                # For other personas: use noprompt activations (we don't have their persona activations)
                for other_persona in ALL_PERSONA_RUNS:
                    if other_persona == persona or "c" not in persona_scores[other_persona]:
                        continue
                    X_o, y_o, _ = align_scores_activations(
                        persona_scores[other_persona]["c"], noprompt_task_ids, noprompt_act_matrix,
                        restrict_to=split_c_ids,
                    )
                    if len(y_o) < 10:
                        continue
                    aug_persona_evals[other_persona] = evaluate_probe(probe_aug, scaler_aug, X_o, y_o)

                print(f"  +{persona}: alpha={alpha_aug:.1f}, noprompt r={eval_noprompt_aug['pearson_r']:.3f}, "
                      f"{persona} r={aug_persona_evals.get(persona, {}).get('pearson_r', 'n/a')}")

                result_aug = {
                    "condition": "augmented",
                    "donor": persona,
                    "selector": selector,
                    "layer": layer,
                    "best_alpha": alpha_aug,
                    "alpha_sweep": sweep_aug,
                    "n_train": len(y_train_aug),
                    "noprompt_test": eval_noprompt_aug,
                    "persona_test": aug_persona_evals,
                }
                all_results.append(result_aug)

    output_path = OUTPUT_DIR / "persona_augmentation_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_experiment()
