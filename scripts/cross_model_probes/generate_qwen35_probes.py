"""Generate probe configs for Qwen 3.5 122B: all 5 turn boundary selectors.

Train on 10k, eval on 4k (heldout).

Usage:
    python -m scripts.cross_model_probes.generate_qwen35_probes
"""

from pathlib import Path

import yaml

OUTPUT_DIR = Path("configs/probes/qwen35_122b")

RUN_DIR = "results/experiments/qwen35_10k_active_learning/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d"
EVAL_RUN_DIR = "results/experiments/qwen35_4k_active_learning/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d_qwen35_4k_task_ids"
ACTIVATIONS_DIR = "activations/qwen35_122b_turn_boundary_sweep"

SELECTORS = [
    "turn_boundary:-1",
    "turn_boundary:-2",
    "turn_boundary:-3",
    "turn_boundary:-4",
    "turn_boundary:-5",
]
LAYERS = [12, 24, 28, 33, 38, 43]


def selector_to_safe_name(selector: str) -> str:
    return selector.replace(":", "_").replace("-", "m")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for selector in SELECTORS:
        safe = selector_to_safe_name(selector)
        name = f"qwen35_122b_heldout_{safe}"

        config = {
            "experiment_name": name,
            "run_dir": RUN_DIR,
            "eval_run_dir": EVAL_RUN_DIR,
            "activations_path": f"{ACTIVATIONS_DIR}/activations_{selector}.npz",
            "output_dir": f"results/probes/qwen35_122b/{name}",
            "layers": LAYERS,
            "modes": ["ridge"],
            "standardize": True,
            "alpha_sweep_size": 10,
            "eval_split_seed": 42,
        }

        path = OUTPUT_DIR / f"{name}.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"  {path}")

    print(f"\nGenerated {len(SELECTORS)} configs")


if __name__ == "__main__":
    main()
