"""Score persona/prompt activations with iter-0 / iter-1 / control directions.

See experiments/probe_science/probe_direction_uniqueness/persona_prompt_tracking/persona_prompt_tracking_spec.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from scipy.stats import pearsonr

from src.probes.core.activations import load_activations
from src.probes.data_loading import load_thurstonian_scores

load_dotenv()


def sha256_hash8(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:8]


@dataclass
class Condition:
    experiment: str
    persona: str
    split: str | None
    activations_npz: Path
    thurstonian_run_dir: Path
    baseline: bool = False


def load_sys_prompt(cfg_path: Path) -> str:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["measurement_system_prompt"]


def enumerate_exp1b(repo_root: Path, activations_filename: str) -> list[Condition]:
    conds = []
    topics = [
        "ancient_history", "astronomy", "cats", "cheese",
        "classical_music", "cooking", "gardening", "rainy_weather",
    ]
    configs_dir = repo_root / "configs/measurement/active_learning/ood_exp1b"
    runs_dir = repo_root / "results/experiments/ood_exp1b/pre_task_active_learning"
    acts_root = repo_root / "activations/ood_tb/exp1_prompts"

    baseline_npz = acts_root / "baseline" / activations_filename
    baseline_run = runs_dir / "completion_preference_gemma-3-27b_completion_canonical_seed0"
    if baseline_npz.exists() and baseline_run.exists():
        conds.append(Condition(
            experiment="1b", persona="baseline", split=None,
            activations_npz=baseline_npz, thurstonian_run_dir=baseline_run,
            baseline=True,
        ))

    for topic in topics:
        for polarity in ["pos_persona", "neg_persona"]:
            persona = f"{topic}_{polarity}"
            act_npz = acts_root / persona / activations_filename
            cfg_path = configs_dir / f"{persona}.yaml"
            if not (act_npz.exists() and cfg_path.exists()):
                continue
            sys_hash = sha256_hash8(load_sys_prompt(cfg_path))
            run_dir = runs_dir / f"completion_preference_gemma-3-27b_completion_canonical_seed0_sys{sys_hash}"
            if not run_dir.exists():
                print(f"  [1b] {persona}: no run dir for sys{sys_hash}")
                continue
            conds.append(Condition(
                experiment="1b", persona=persona, split=None,
                activations_npz=act_npz, thurstonian_run_dir=run_dir,
            ))
    return conds


def enumerate_mra(
    repo_root: Path, exp_name: str, personas: list[str], activations_filename: str,
) -> list[Condition]:
    conds = []
    configs_dir = repo_root / f"configs/measurement/active_learning/{exp_name}"
    runs_dir = repo_root / f"results/experiments/{exp_name}/pre_task_active_learning"
    exp_num = "2" if exp_name == "mra_exp2" else "3"

    for persona in personas:
        cfg_files = sorted(configs_dir.glob(f"{persona}_split_*.yaml"))
        if not cfg_files:
            continue
        sys_hash = sha256_hash8(load_sys_prompt(cfg_files[0]))
        act_npz = repo_root / f"activations/gemma_3_27b_{persona}_tb" / activations_filename
        if not act_npz.exists():
            continue
        for split in ["a", "b", "c"]:
            matches = list(runs_dir.glob(
                f"*_sys{sys_hash}_mra_exp2_split_{split}_*_task_ids"
            ))
            if not matches:
                print(f"  [{exp_name}] {persona} split {split}: no run dir for sys{sys_hash}")
                continue
            conds.append(Condition(
                experiment=exp_num, persona=persona, split=split,
                activations_npz=act_npz, thurstonian_run_dir=matches[0],
            ))
    return conds


def score_direction(
    activations: np.ndarray, scaler_mean: np.ndarray, scaler_scale: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    X_std = (activations - scaler_mean) / scaler_scale
    return X_std @ direction


def match_and_score(
    cond: Condition, layer: int, scaler_mean: np.ndarray, scaler_scale: np.ndarray,
    directions: dict[str, np.ndarray], min_tasks: int, train_tids: set[str],
) -> tuple[dict, dict[str, np.ndarray], np.ndarray, list[str]] | None:
    """Returns (summary_row, scores_per_direction, y_array, matched_task_ids) or None."""
    thurstonian = load_thurstonian_scores(cond.thurstonian_run_dir)
    task_ids, acts_dict = load_activations(cond.activations_npz, layers=[layer])
    X = acts_dict[layer].astype(np.float32)
    tid_to_idx = {str(tid): i for i, tid in enumerate(task_ids)}
    matched = [t for t in thurstonian if t in tid_to_idx]
    if len(matched) < min_tasks:
        print(f"  skip ({len(matched)} < {min_tasks})")
        return None
    idx = np.array([tid_to_idx[t] for t in matched])
    y = np.array([thurstonian[t] for t in matched])
    Xm = X[idx]
    r_values = {}
    scores_per_dir = {}
    for name, direction in directions.items():
        score = score_direction(Xm, scaler_mean, scaler_scale, direction)
        r_values[f"r_{name}"] = float(pearsonr(y, score)[0])
        scores_per_dir[name] = score
    overlap = sum(1 for t in matched if t in train_tids)
    summary = {
        "experiment": cond.experiment,
        "persona": cond.persona,
        "split": cond.split,
        "is_baseline": cond.baseline,
        "n_matched": len(matched),
        "n_overlap_train": overlap,
        **r_values,
    }
    return summary, scores_per_dir, y, matched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--directions-dir", type=Path, default=Path(
        "experiments/probe_science/probe_direction_uniqueness/persona_prompt_tracking/output/L32_tb-2"
    ))
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--random-seeds", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, default=Path(
        "experiments/probe_science/probe_direction_uniqueness/persona_prompt_tracking"
    ))
    ap.add_argument(
        "--activations-filename", type=str, required=True,
        help="Filename (not path) of the NPZ inside each persona's activations dir, "
             "e.g. 'activations_turn_boundary:-2.npz'",
    )
    ap.add_argument("--train-run", type=Path, default=Path(
        "results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/"
        "completion_preference_gemma-3-27b_completion_canonical_seed0"
    ))
    ap.add_argument("--min-tasks", type=int, default=30)
    args = ap.parse_args()

    repo_root = Path.cwd()

    dirs = np.load(args.directions_dir / "directions.npz")
    w_keys = sorted(k for k in dirs.keys() if k.startswith("w_"))
    probe_dirs = {k.replace("_", ""): dirs[k].astype(np.float32) for k in w_keys}
    d = probe_dirs[list(probe_dirs)[0]].shape[0]
    print(f"Loaded probe directions: {list(probe_dirs)}")

    sc = np.load(args.directions_dir / "scaler.npz")
    scaler_mean = sc["mean"]
    scaler_scale = sc["scale"].clip(min=1e-6)

    rng = np.random.default_rng(2000)
    random_dirs = {}
    for s in range(args.random_seeds):
        v = rng.standard_normal(d)
        random_dirs[f"random_{s}"] = (v / np.linalg.norm(v)).astype(np.float32)

    directions = {**probe_dirs, **random_dirs}

    train_scores = load_thurstonian_scores(args.train_run)
    train_tids = set(train_scores.keys())
    print(f"Train tasks: {len(train_tids)}")

    conditions = []
    conditions += enumerate_exp1b(repo_root, args.activations_filename)
    conditions += enumerate_mra(repo_root, "mra_exp2", ["villain", "midwest"], args.activations_filename)
    conditions += enumerate_mra(repo_root, "mra_exp3", ["sadist"], args.activations_filename)
    print(f"Conditions enumerated: {len(conditions)}")

    per_condition = []
    raw_per_cond: list[tuple[dict, dict[str, np.ndarray], np.ndarray]] = []
    baseline_cond: Condition | None = None
    for cond in conditions:
        print(f"[{cond.experiment}] {cond.persona}{'/'+cond.split if cond.split else ''}")
        result = match_and_score(
            cond, args.layer, scaler_mean, scaler_scale, directions,
            args.min_tasks, train_tids,
        )
        if result is None:
            continue
        row, scores_per_dir, y_arr, _ = result
        r_rand = [row[f"r_random_{s}"] for s in range(args.random_seeds)]
        row["r_random_mean"] = float(np.mean(r_rand))
        row["r_random_std"] = float(np.std(r_rand))
        row["r_random_p95_abs"] = float(np.percentile(np.abs(r_rand), 95))
        per_condition.append(row)
        raw_per_cond.append((row, scores_per_dir, y_arr))
        probe_r_str = ", ".join(f"{name}={row[f'r_{name}']:+.3f}" for name in probe_dirs)
        print(f"  n={row['n_matched']}, overlap={row['n_overlap_train']}, "
              f"{probe_r_str}, r_rand={row['r_random_mean']:+.3f}±{row['r_random_std']:.3f}")
        if cond.baseline:
            baseline_cond = cond

    # Aggregate per persona: pool splits for exp 2/3 (disjoint task sets under same system prompt).
    # For exp 1b each persona is already one condition.
    from collections import defaultdict
    persona_groups: dict[tuple[str, str], list[tuple[dict, dict[str, np.ndarray], np.ndarray]]] = defaultdict(list)
    for entry in raw_per_cond:
        row, _, _ = entry
        key = (row["experiment"], row["persona"])
        persona_groups[key].append(entry)

    per_persona = []
    for (experiment, persona), entries in persona_groups.items():
        pooled_y = np.concatenate([e[2] for e in entries])
        pooled_overlap = sum(e[0]["n_overlap_train"] for e in entries)
        per_persona_row = {
            "experiment": experiment,
            "persona": persona,
            "is_baseline": entries[0][0]["is_baseline"],
            "n_splits_pooled": len(entries),
            "n_total": int(pooled_y.shape[0]),
            "n_overlap_train": pooled_overlap,
        }
        for name in directions:
            pooled_scores = np.concatenate([e[1][name] for e in entries])
            per_persona_row[f"r_{name}"] = float(pearsonr(pooled_y, pooled_scores)[0])
        r_rand = [per_persona_row[f"r_random_{s}"] for s in range(args.random_seeds)]
        per_persona_row["r_random_mean"] = float(np.mean(r_rand))
        per_persona_row["r_random_std"] = float(np.std(r_rand))
        per_persona_row["r_random_p95_abs"] = float(np.percentile(np.abs(r_rand), 95))
        per_persona.append(per_persona_row)
        probe_r_str = ", ".join(f"{name}={per_persona_row[f'r_{name}']:+.3f}" for name in probe_dirs)
        print(f"[persona] [{experiment}] {persona} (n={per_persona_row['n_total']}, "
              f"{len(entries)} split(s)): {probe_r_str}")

    # Baseline halves null (split the baseline tasks 5 times × 2 halves)
    halves: list[dict] = []
    if baseline_cond is not None:
        print(f"\nBaseline halves null: splitting '{baseline_cond.persona}' 5 times")
        thurstonian = load_thurstonian_scores(baseline_cond.thurstonian_run_dir)
        task_ids, acts_dict = load_activations(baseline_cond.activations_npz, layers=[args.layer])
        X = acts_dict[args.layer].astype(np.float32)
        tid_to_idx = {str(tid): i for i, tid in enumerate(task_ids)}
        matched = [t for t in thurstonian if t in tid_to_idx]
        idx = np.array([tid_to_idx[t] for t in matched])
        y = np.array([thurstonian[t] for t in matched])
        Xm = X[idx]
        probe_scores = {
            name: score_direction(Xm, scaler_mean, scaler_scale, probe_dirs[name])
            for name in probe_dirs
        }
        for seed in range(5):
            rng_s = np.random.default_rng(3000 + seed)
            perm = rng_s.permutation(len(matched))
            half = len(matched) // 2
            for label, sl in [("h1", perm[:half]), ("h2", perm[half:])]:
                halves.append({
                    "seed": seed, "half": label, "n": int(len(sl)),
                    **{
                        f"r_{name}": float(pearsonr(y[sl], s[sl])[0])
                        for name, s in probe_scores.items()
                    },
                })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "results.json", "w") as f:
        json.dump({
            "directions_dir": str(args.directions_dir),
            "layer": args.layer,
            "d": int(d),
            "random_seeds": args.random_seeds,
            "min_tasks": args.min_tasks,
            "n_train_tasks": len(train_tids),
            "conditions": per_condition,
            "personas": per_persona,
            "baseline_halves_null": halves,
        }, f, indent=2)
    print(f"\nSaved results.json ({len(per_condition)} conditions, "
          f"{len(per_persona)} personas, {len(halves)} baseline-half samples)")


if __name__ == "__main__":
    main()
