"""Generate active learning YAML configs from sweep_personas.json."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PERSONAS_FILE = ROOT / "experiments/persona_sweep/sweep_personas.json"
OUTPUT_DIR = ROOT / "configs/measurement/persona_sweep"

BASE_CONFIG = {
    "preference_mode": "pre_task_active_learning",
    "model": "gemma-3-27b",
    "temperature": 1.0,
    "n_tasks": 500,
    "task_origins": ["wildchat", "alpaca", "math", "bailbench", "stress_test"],
    "stratified_sampling": True,
    "task_sampling_seed": 42,
    "templates": "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml",
    "response_formats": ["completion"],
    "n_samples": 3,
    "pair_order_seed": 42,
    "active_learning": {
        "initial_degree": 2,
        "batch_size": 300,
        "max_iterations": 10,
        "p_threshold": 0.3,
        "q_threshold": 0.3,
        "convergence_threshold": 0.98,
        "seed": 42,
    },
}


def main():
    with open(PERSONAS_FILE) as f:
        data = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for f in OUTPUT_DIR.glob("*.yaml"):
        f.unlink()

    baseline = {**BASE_CONFIG, "experiment_id": "persona_sweep_baseline"}
    with open(OUTPUT_DIR / "baseline.yaml", "w") as f:
        yaml.dump(baseline, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  baseline.yaml")

    for persona in data["personas"]:
        name = persona["name"]
        config = {
            **BASE_CONFIG,
            "experiment_id": f"persona_sweep_{name}",
            "measurement_system_prompt": persona["system_prompt"],
        }
        filename = f"{name}.yaml"
        with open(OUTPUT_DIR / filename, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"  {filename}")

    total = len(data["personas"]) + 1
    print(f"\nGenerated {total} configs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
