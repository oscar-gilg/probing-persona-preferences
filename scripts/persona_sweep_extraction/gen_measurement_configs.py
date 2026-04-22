"""Generate active-learning measurement configs for the final-six personas plus
the default (no system prompt) across the three canonical splits.

Writes configs/measurement/persona_sweep/final_six/<persona>_<split>.yaml. Each
config pins include_task_ids_file to the corresponding canonical split so that
measurements align byte-exactly with the extraction run.

The default-persona configs omit measurement_system_prompt so the runner sends
the raw assistant prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PERSONAS_JSON = REPO / "experiments/persona_sweep/sweep_personas.json"
CONFIG_DIR = REPO / "configs/measurement/persona_sweep/final_six"

SPLITS = {
    "train": ("data/canonical_splits/train_task_ids.txt", 4000, 1200),
    "eval":  ("data/canonical_splits/eval_task_ids.txt",  1000,  300),
    "test":  ("data/canonical_splits/test_task_ids.txt",  1000,  300),
}


def main() -> None:
    with open(PERSONAS_JSON) as f:
        data = json.load(f)

    selected = data["metadata"]["final_six"]
    by_name = {p["name"]: p["system_prompt"] for p in data["personas"]}
    missing = [p for p in selected if p not in by_name]
    if missing:
        raise SystemExit(f"Missing personas in sweep_personas.json: {missing}")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Final six + default. Default gets no measurement_system_prompt.
    personas_with_prompt = [(p, by_name[p]) for p in selected] + [("default", None)]

    for persona, system_prompt in personas_with_prompt:
        for split, (task_ids_file, n_tasks, batch_size) in SPLITS.items():
            cfg = {
                "preference_mode": "pre_task_active_learning",
                "model": "gemma-3-27b",
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
                    "initial_degree": 2,
                    "batch_size": batch_size,
                    "max_iterations": 10,
                    "p_threshold": 0.3,
                    "q_threshold": 0.3,
                    "convergence_threshold": 0.98,
                    "seed": 42,
                },
                "experiment_id": f"persona_sweep_final_six_{persona}_{split}",
            }
            if system_prompt is not None:
                cfg["measurement_system_prompt"] = system_prompt
            out = CONFIG_DIR / f"{persona}_{split}.yaml"
            with open(out, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, width=10_000)
            print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
