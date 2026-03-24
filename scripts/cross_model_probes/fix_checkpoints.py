"""Reconstruct checkpoint.yaml from measurements.yaml for aborted runs."""

import yaml
from pathlib import Path

BASE = Path("results/experiments/gptoss_qwen_mra2/pre_task_active_learning")

ABORTED_DIRS = [
    "completion_preference_gpt-oss-120b_completion_canonical_seed0_mra_exp2_split_a_1000_task_ids",
    "completion_preference_gpt-oss-120b_completion_canonical_seed0_mra_exp2_split_c_1000_task_ids",
    "completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d_mra_exp2_split_c_1000_task_ids",
]

for dirname in ABORTED_DIRS:
    run_dir = BASE / dirname
    measurements_path = run_dir / "measurements.yaml"
    summary_path = run_dir / "active_learning.yaml"

    with open(measurements_path) as f:
        measurements = yaml.safe_load(f)

    with open(summary_path) as f:
        summary = yaml.safe_load(f)

    comparisons = [
        {"task_a": m["task_a"], "task_b": m["task_b"], "choice": m["choice"]}
        for m in measurements
        if m["choice"] in ("a", "b")
    ]

    checkpoint = {
        "iteration": summary["n_iterations"],
        "comparisons": comparisons,
        "rank_correlations": summary["rank_correlations"],
    }

    ckpt_path = run_dir / "checkpoint.yaml"
    with open(ckpt_path, "w") as f:
        yaml.dump(checkpoint, f, default_flow_style=False)

    print(f"{dirname}: {len(comparisons)} comparisons, iteration {summary['n_iterations']}")
