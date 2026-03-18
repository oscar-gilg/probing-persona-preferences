"""Convert an existing active learning run to checkpoint format for --resume.

Usage:
    python scripts/convert_run_to_checkpoint.py <run_dir>

Example:
    python scripts/convert_run_to_checkpoint.py results/experiments/exp_20260316_224618/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d
"""

import sys
from pathlib import Path

from src.measurement.storage.base import load_yaml
from src.fitting.thurstonian_fitting.active_learning_checkpoint import save_checkpoint, checkpoint_exists

run_dir = Path(sys.argv[1])

measurements_path = run_dir / "measurements.yaml"
al_path = run_dir / "active_learning.yaml"

if not measurements_path.exists():
    print(f"No measurements.yaml in {run_dir}")
    sys.exit(1)
if not al_path.exists():
    print(f"No active_learning.yaml in {run_dir}")
    sys.exit(1)
if checkpoint_exists(run_dir):
    print(f"checkpoint.yaml already exists in {run_dir}")
    sys.exit(1)

measurements = load_yaml(measurements_path)
al_summary = load_yaml(al_path)

# Strip origin fields — checkpoint only needs task_a, task_b, choice
comparisons = [
    {"task_a": m["task_a"], "task_b": m["task_b"], "choice": m["choice"]}
    for m in measurements
]

path = save_checkpoint(
    run_dir,
    iteration=al_summary["n_iterations"],
    comparisons_dicts=comparisons,
    rank_correlations=al_summary["rank_correlations"],
)

print(f"Created {path}")
print(f"  iteration: {al_summary['n_iterations']}")
print(f"  comparisons: {len(comparisons)}")
print(f"  rank_correlations: {al_summary['rank_correlations']}")
