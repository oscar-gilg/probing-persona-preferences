"""Analyse E1a for Qwen (full per-task) and Gemma (from pre-computed arrays).

For Qwen: reads freshly extracted activations + the 16 AL Thurstonian fits to compute
per-task (probe_delta, utility_delta, behavioral_delta).

For Gemma: reads pre-computed behavioral_delta + probe_delta arrays from
experiments/probe_generalization/ood_system_prompts/analysis_results_full.json
(no task_ids are stored there, so utility_delta for Gemma is omitted).

Output: experiments/qwen_replication/e1a/e1a_per_task.json
Shape:
  {
    "qwen-3.5-122b": {
      <selector>: {
        "pooled_all": {n, pearson_r, ci95},
        "pooled_on_target": {...},
        "per_condition": [{condition_id, topic, tasks: [{task_id, probe_delta,
                                                         utility_delta,
                                                         behavioral_delta,
                                                         on_target}]}]}},
    "gemma-3-27b": {
      <selector>: {
        "pooled_all": {...},
        "pooled_on_target": {...},
        "per_condition": [{condition_id, topic, tasks: [{probe_delta,
                                                         behavioral_delta,
                                                         on_target}]}]}}
  }
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from src.measurement.storage.loading import load_run_utilities
from src.ood.analysis import _score_activations, _split_probe

ROOT = Path(".")
OUTPUT_JSON = ROOT / "experiments/qwen_replication/e1a/e1a_per_task.json"

TOPICS = [
    "ancient_history",
    "astronomy",
    "cats",
    "cheese",
    "classical_music",
    "cooking",
    "gardening",
    "rainy_weather",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config_dir: Path
    baseline_config: str
    run_dirs: list[Path]  # search run_dirs for matching utility fits by system_prompt
    activations_dir: Path
    probes: dict[str, tuple[Path, str]]  # selector_name -> (probe_path, npz_filename)
    layer: int


QWEN = ModelSpec(
    name="qwen-3.5-122b",
    config_dir=ROOT / "configs/measurement/active_learning/qwen35_ood_exp1b",
    baseline_config="baseline.yaml",
    run_dirs=[
        ROOT / "results/experiments/qwen35_ood_e1a_final/pre_task_active_learning",
        ROOT / "results/experiments/exp_20260422_124719/pre_task_active_learning",
    ],
    activations_dir=ROOT / "activations/qwen35_122b_ood/e1a",
    probes={
        "tb-1": (
            ROOT / "results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1/probes/probe_ridge_L38.npy",
            "activations_turn_boundary:-1.npz",
        ),
        "tb-4": (
            ROOT / "results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m4/probes/probe_ridge_L38.npy",
            "activations_turn_boundary:-4.npz",
        ),
    },
    layer=38,
)

GEMMA = ModelSpec(
    name="gemma-3-27b",
    config_dir=ROOT / "configs/measurement/active_learning/ood_exp1b",
    baseline_config="baseline.yaml",
    run_dirs=[ROOT / "results/experiments/ood_exp1b/pre_task_active_learning"],
    activations_dir=ROOT / "activations/gemma-3-27b_it/pref_ood_e1a",
    probes={
        "prompt_last": (
            ROOT / "results/probes/gemma3_10k_heldout_std_raw/probes/probe_ridge_L31.npy",
            "activations_prompt_last.npz",
        ),
    },
    layer=31,
)


def topic_of(condition_id: str) -> str:
    for t in TOPICS:
        if condition_id.startswith(t + "_"):
            return t
    return "baseline" if condition_id == "baseline" else "unknown"


_BASELINE_SENTINELS = {None, "", "/no_think"}


def find_utility_dir(system_prompt: str | None, run_dirs: list[Path]) -> Path | None:
    """Find the measurement output dir matching the given system prompt.

    Qwen's runner auto-injects "/no_think" as the system prompt when the config has
    none, so we treat it as equivalent to None for baseline matching.
    """
    target_is_baseline = system_prompt is None or system_prompt.strip() in {"", "/no_think"}
    for run_dir in run_dirs:
        if not run_dir.exists():
            continue
        for subdir in sorted(run_dir.iterdir()):
            if not subdir.is_dir():
                continue
            al_yaml = subdir / "active_learning.yaml"
            if not al_yaml.exists():
                continue
            cfg = yaml.safe_load(al_yaml.read_text())
            sp = cfg.get("system_prompt")
            sp_is_baseline = sp is None or sp.strip() in {"", "/no_think"}
            if target_is_baseline and sp_is_baseline:
                return subdir
            if not target_is_baseline and not sp_is_baseline and sp.strip() == system_prompt.strip():
                return subdir
    return None


def load_conditions(spec: ModelSpec) -> list[tuple[str, str | None]]:
    """Return (condition_id, system_prompt) for each YAML in the config dir."""
    conditions: list[tuple[str, str | None]] = []
    for yaml_path in sorted(spec.config_dir.glob("*.yaml")):
        cfg = yaml.safe_load(yaml_path.read_text())
        cid = yaml_path.stem
        sp = cfg.get("measurement_system_prompt")
        conditions.append((cid, sp))
    return conditions


def load_choice_rates(run_dir: Path) -> dict[str, float]:
    """Load measurements.yaml from a run dir and compute per-task P(choose).

    For each task t, P(choose t) = #pairs where t was chosen / #pairs involving t.
    """
    measurements_path = run_dir / "measurements.yaml"
    if not measurements_path.exists():
        return {}
    pairs = yaml.safe_load(measurements_path.read_text())
    if not pairs:
        return {}
    wins: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for p in pairs:
        ta, tb = p["task_a"], p["task_b"]
        choice = p.get("choice")
        total[ta] += 1
        total[tb] += 1
        if choice == "a":
            wins[ta] += 1
        elif choice == "b":
            wins[tb] += 1
    return {tid: wins[tid] / total[tid] for tid in total if total[tid] > 0}


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    rs: list[float] = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r, _ = stats.pearsonr(x[idx], y[idx])
        rs.append(r)
    rs.sort()
    return rs[int(0.025 * n_boot)], rs[int(0.975 * n_boot)]


def analyse_model(spec: ModelSpec) -> dict:
    """Compute per-task probe_delta and utility_delta for each selector × condition."""
    conditions = load_conditions(spec)
    baseline_cid = spec.baseline_config.replace(".yaml", "")
    print(f"\n{'=' * 60}\n{spec.name}: {len(conditions)} conditions\n{'=' * 60}")

    # Load baseline utilities and choice rates
    baseline_prompt = next((sp for cid, sp in conditions if cid == baseline_cid), None)
    baseline_dir = find_utility_dir(baseline_prompt, spec.run_dirs)
    if baseline_dir is None:
        raise FileNotFoundError(f"No baseline utility dir for {spec.name}")
    baseline_mu, baseline_task_ids = load_run_utilities(baseline_dir)
    baseline_util = dict(zip(baseline_task_ids, baseline_mu))
    baseline_pchoose = load_choice_rates(baseline_dir)
    print(f"Baseline utilities: {len(baseline_util)} tasks from {baseline_dir}")
    print(f"Baseline choice rates: {len(baseline_pchoose)} tasks")

    out = {}
    for selector, (probe_path, npz_name) in spec.probes.items():
        weights, bias = _split_probe(probe_path)

        baseline_act_dir = spec.activations_dir / baseline_cid
        if not (baseline_act_dir / npz_name).exists():
            print(f"  [{selector}] baseline activations missing at {baseline_act_dir / npz_name}")
            continue

        baseline_probe = _score_activations(baseline_act_dir / npz_name, spec.layer, weights, bias)
        print(f"\nSelector {selector}: baseline probe scored on {len(baseline_probe)} tasks")

        per_condition = []
        for cid, sp in conditions:
            if cid == baseline_cid:
                continue

            utility_dir = find_utility_dir(sp, spec.run_dirs)
            if utility_dir is None:
                print(f"  skip {cid}: no utility dir found")
                continue

            cond_mu, cond_task_ids = load_run_utilities(utility_dir)
            cond_util = dict(zip(cond_task_ids, cond_mu))
            cond_pchoose = load_choice_rates(utility_dir)

            cond_npz = spec.activations_dir / cid / npz_name
            if not cond_npz.exists():
                print(f"  skip {cid}: activations missing at {cond_npz}")
                continue
            cond_probe = _score_activations(cond_npz, spec.layer, weights, bias)

            shared = sorted(
                set(cond_probe.keys()) & set(baseline_probe.keys()) &
                set(cond_util.keys()) & set(baseline_util.keys())
            )
            if not shared:
                print(f"  skip {cid}: no shared tasks")
                continue

            topic = topic_of(cid)
            tasks_out = []
            probe_deltas, util_deltas, on_target_flags = [], [], []
            for tid in shared:
                pd = cond_probe[tid] - baseline_probe[tid]
                ud = cond_util[tid] - baseline_util[tid]
                bd = float("nan")
                if tid in cond_pchoose and tid in baseline_pchoose:
                    bd = cond_pchoose[tid] - baseline_pchoose[tid]
                on_tgt = tid.startswith(f"hidden_{topic}_") if topic in TOPICS else False
                tasks_out.append({
                    "task_id": tid,
                    "probe_delta": float(pd),
                    "utility_delta": float(ud),
                    "behavioral_delta": float(bd),
                    "on_target": bool(on_tgt),
                })
                probe_deltas.append(pd)
                util_deltas.append(ud)
                on_target_flags.append(on_tgt)

            probe_deltas = np.array(probe_deltas)
            util_deltas = np.array(util_deltas)
            on_target_flags = np.array(on_target_flags)

            r_all, _ = stats.pearsonr(probe_deltas, util_deltas)
            r_on = float("nan")
            if on_target_flags.sum() >= 3:
                r_on, _ = stats.pearsonr(probe_deltas[on_target_flags], util_deltas[on_target_flags])

            per_condition.append({
                "condition_id": cid,
                "topic": topic,
                "n_all": len(shared),
                "n_on_target": int(on_target_flags.sum()),
                "r_all": float(r_all),
                "r_on_target": float(r_on),
                "tasks": tasks_out,
            })
            print(f"  {cid}: n_all={len(shared)} r_all={r_all:+.3f} n_on={on_target_flags.sum()} r_on={r_on:+.3f}")

        # Pool
        all_probe, all_util, all_on = [], [], []
        for c in per_condition:
            for t in c["tasks"]:
                all_probe.append(t["probe_delta"])
                all_util.append(t["utility_delta"])
                all_on.append(t["on_target"])
        all_probe = np.array(all_probe)
        all_util = np.array(all_util)
        all_on = np.array(all_on)

        pooled_all_r, pooled_all_p = stats.pearsonr(all_probe, all_util)
        pooled_all_ci = bootstrap_ci(all_probe, all_util)
        pooled_on_r, pooled_on_p = stats.pearsonr(all_probe[all_on], all_util[all_on])
        pooled_on_ci = bootstrap_ci(all_probe[all_on], all_util[all_on])

        out[selector] = {
            "layer": spec.layer,
            "probe_path": str(probe_path),
            "npz_name": npz_name,
            "pooled_all": {
                "n": int(len(all_probe)),
                "pearson_r": float(pooled_all_r),
                "p": float(pooled_all_p),
                "ci95": [float(pooled_all_ci[0]), float(pooled_all_ci[1])],
            },
            "pooled_on_target": {
                "n": int(all_on.sum()),
                "pearson_r": float(pooled_on_r),
                "p": float(pooled_on_p),
                "ci95": [float(pooled_on_ci[0]), float(pooled_on_ci[1])],
            },
            "per_condition": per_condition,
        }

        print(f"  POOLED all: n={len(all_probe)} r={pooled_all_r:.3f} CI=[{pooled_all_ci[0]:.2f}, {pooled_all_ci[1]:.2f}]")
        print(f"  POOLED on-target: n={all_on.sum()} r={pooled_on_r:.3f} CI=[{pooled_on_ci[0]:.2f}, {pooled_on_ci[1]:.2f}]")

    return out


def analyse_gemma_from_stored() -> dict:
    """Load Gemma pooled arrays from the stored OOD analysis JSON.

    The stored JSON has (probe_delta, behavioural_delta, condition_label) rows aligned
    but no task_ids. We reconstruct task_ids by replaying the same iteration
    `_recompute_with_task_ids` uses: for each condition (in rates-dict order),
    iterate rates[cid] and emit tids. Every exp1b condition has the same 40
    hidden tasks, so this recovers task_ids deterministically and lets us attach
    on_target flags to each Gemma row.
    """
    from src.ood.analysis import compute_p_choose_from_pairwise

    stored_path = ROOT / "experiments/probe_generalization/ood_system_prompts/analysis_results_full.json"
    data = json.loads(stored_path.read_text())["exp1b"]["L31"]

    probe_arr = data["probe_deltas"]
    beh_arr = data["behavioural_deltas"] if "behavioural_deltas" in data else data["behavioral_deltas"]
    labels = data["condition_labels"]

    pairwise = json.loads((ROOT / "results/ood/hidden_preference/pairwise.json").read_text())
    rates = compute_p_choose_from_pairwise(pairwise["results"])
    rates = {k: v for k, v in rates.items() if not k.startswith("compete_")}
    rates = {
        k: {tid: v for tid, v in rd.items() if tid.startswith("hidden_")}
        for k, rd in rates.items()
    }
    baseline_rates = rates["baseline"]

    # Map (cid, behavioural_delta) -> task_id so we can recover task ids from the
    # stored arrays — the stored JSON's per-condition ordering differs from the
    # rates-dict iteration order, but behavioural_delta is a unique key within a
    # condition (up to ties, which don't happen in this data).
    lookup_by_delta: dict[str, dict[float, str]] = {}
    for cid, rd in rates.items():
        if cid == "baseline":
            continue
        cmap: dict[float, str] = {}
        for tid, rate in rd.items():
            if tid not in baseline_rates:
                continue
            delta = round(rate - baseline_rates[tid], 9)
            cmap[delta] = tid
        lookup_by_delta[cid] = cmap

    # Load task->topic map for on-target flag.
    task_to_topic: dict[str, str] = {}
    tasks_json = ROOT / "configs/ood/tasks/target_tasks.json"
    for t in json.loads(tasks_json.read_text()):
        task_to_topic[t["task_id"]] = t["topic"]

    per_cond_map: dict[str, dict] = {}
    all_probe: list[float] = []
    all_beh: list[float] = []
    all_on: list[bool] = []

    for p, b, cid in zip(probe_arr, beh_arr, labels):
        key = round(float(b), 9)
        tid = lookup_by_delta.get(cid, {}).get(key)

        cond_topic = topic_of(cid)
        task_topic = task_to_topic.get(tid) if tid is not None else None
        on_tgt = task_topic is not None and task_topic == cond_topic

        rec = per_cond_map.setdefault(cid, {
            "condition_id": cid,
            "topic": cond_topic,
            "tasks": [],
        })
        rec["tasks"].append({
            "task_id": tid,
            "probe_delta": float(p),
            "behavioral_delta": float(b),
            "on_target": bool(on_tgt),
        })
        all_probe.append(p)
        all_beh.append(b)
        all_on.append(on_tgt)

    for rec in per_cond_map.values():
        rec["n_all"] = len(rec["tasks"])
        rec["n_on_target"] = sum(1 for t in rec["tasks"] if t["on_target"])

    per_condition = list(per_cond_map.values())

    all_probe = np.array(all_probe)
    all_beh = np.array(all_beh)
    all_on = np.array(all_on)
    r_all, p_all = stats.pearsonr(all_probe, all_beh)
    ci_all = bootstrap_ci(all_probe, all_beh)
    r_on, p_on = stats.pearsonr(all_probe[all_on], all_beh[all_on])
    ci_on = bootstrap_ci(all_probe[all_on], all_beh[all_on])

    return {
        "prompt_last": {
            "layer": 31,
            "note": (
                "behavioural_delta and probe_delta loaded from stored OOD JSON; "
                "task_ids reconstructed from the rates iteration in "
                "scripts/ood_system_prompts/plot_ground_truth._recompute_experiment('exp1b'). "
                "utility_delta is not available for Gemma (no per-condition Thurstonian fit in this pipeline)."
            ),
            "pooled_all": {
                "n": int(len(all_probe)),
                "pearson_r": float(r_all),
                "p": float(p_all),
                "ci95": [float(ci_all[0]), float(ci_all[1])],
            },
            "pooled_on_target": {
                "n": int(all_on.sum()),
                "pearson_r": float(r_on),
                "p": float(p_on),
                "ci95": [float(ci_on[0]), float(ci_on[1])],
            },
            "per_condition": per_condition,
        }
    }


def main() -> None:
    results = {}
    try:
        results[QWEN.name] = analyse_model(QWEN)
    except FileNotFoundError as e:
        print(f"[{QWEN.name}] SKIPPED: {e}")

    gemma_activations_exist = (GEMMA.activations_dir / GEMMA.baseline_config.replace(".yaml", "") /
                                "activations_prompt_last.npz").exists()
    if gemma_activations_exist:
        try:
            results[GEMMA.name] = analyse_model(GEMMA)
        except FileNotFoundError as e:
            print(f"[{GEMMA.name}] SKIPPED: {e}")
    else:
        print(f"\n[{GEMMA.name}] activations missing — loading stored behavioral/probe arrays")
        results[GEMMA.name] = analyse_gemma_from_stored()

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
