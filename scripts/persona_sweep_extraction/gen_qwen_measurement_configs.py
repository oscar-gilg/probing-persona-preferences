"""Generate active-learning measurement configs for the final-six personas + default
across the three canonical splits (train/eval/test) on Qwen-3.5-122B.

Mirrors gen_measurement_configs.py with Qwen-specific model and default-persona
handling:
- model: qwen3.5-122b-nothink (auto-injects /no_think, matching all existing Qwen AL runs)
- default persona: no measurement_system_prompt (lets /no_think be auto-injected)
- AL hyperparameters identical to Gemma's persona sweep.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PERSONAS_JSON = REPO / "experiments/persona_sweep/sweep_personas.json"
CONFIG_DIR = REPO / "configs/measurement/qwen_persona_sweep/final_six"

SPLITS = {
    "train": ("data/canonical_splits/train_task_ids.txt", 4000, 1200),
    "eval":  ("data/canonical_splits/eval_task_ids.txt",  1000,  300),
    "test":  ("data/canonical_splits/test_task_ids.txt",  1000,  300),
}


def main() -> None:
    with open(PERSONAS_JSON) as f:
        data = json.load(f)

    final_six = data["metadata"]["final_six"]
    by_name = {p["name"]: p["system_prompt"] for p in data["personas"]}
    missing = [p for p in final_six if p not in by_name]
    if missing:
        raise SystemExit(f"Missing personas in sweep_personas.json: {missing}")

    personas = ["default", *final_six]

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    for persona in personas:
        for split, (task_ids_file, n_tasks, batch_size) in SPLITS.items():
            cfg = {
                "preference_mode": "pre_task_active_learning",
                "model": "qwen3.5-122b-nothink",
                "temperature": 1.0,
                "n_tasks": n_tasks,
                "task_origins": ["wildchat", "alpaca", "math", "bailbench", "stress_test"],
                "stratified_sampling": False,
                "task_sampling_seed": 42,
                "include_task_ids_file": task_ids_file,
                "templates": "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml",
                "response_formats": ["completion"],
                "n_samples": 3,
                "pair_order_seed": 42,
                "active_learning": {
                    "initial_degree": 3,
                    "batch_size": batch_size,
                    "max_iterations": 10,
                    "p_threshold": 0.3,
                    "q_threshold": 0.3,
                    "convergence_threshold": 0.98,
                    "seed": 42,
                },
                "experiment_id": f"qwen_persona_sweep_final_six_{persona}_{split}",
            }
            if persona != "default":
                cfg["measurement_system_prompt"] = by_name[persona]

            out = CONFIG_DIR / f"{persona}_{split}.yaml"
            with open(out, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, width=10_000)
            print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
