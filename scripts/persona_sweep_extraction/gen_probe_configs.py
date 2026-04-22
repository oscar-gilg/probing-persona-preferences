"""Generate probe configs for the final-six + default persona set, two selectors.

Writes configs/probes/persona_sweep_final_six/<persona>_tb-{2,5}.yaml. Each
config trains on <persona>_train, alpha-sweeps + eval on <persona>_eval, and
outputs one Ridge probe per layer in [25, 32, 39, 46, 53].
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PERSONAS_JSON = REPO / "experiments/persona_sweep/sweep_personas.json"
CONFIG_DIR = REPO / "configs/probes/persona_sweep_final_six"
AL_DIR = REPO / "results/experiments/persona_sweep_final_six/pre_task_active_learning"
OUT_ROOT = REPO / "results/probes/persona_sweep_final_six"

SELECTORS = ["turn_boundary:-2", "turn_boundary:-5"]
LAYERS = [25, 32, 39, 46, 53]


def activations_dir(persona: str) -> Path:
    if persona == "default":
        return REPO / "activations/gemma-3-27b_it/pref_main"
    return REPO / f"activations/gemma-3-27b_it/pref_{persona}_sweep"


def main() -> None:
    with open(PERSONAS_JSON) as f:
        data = json.load(f)
    personas = list(data["metadata"]["final_six"]) + ["default"]

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    for persona in personas:
        for selector in SELECTORS:
            selector_tag = selector.replace("turn_boundary:", "tb")  # tb-2, tb-5
            name = f"{persona}_{selector_tag}"
            cfg = {
                "experiment_name": f"persona_sweep_final_six_{name}",
                "run_dir": str((AL_DIR / f"{persona}_train").relative_to(REPO)),
                "eval_run_dir": str((AL_DIR / f"{persona}_eval").relative_to(REPO)),
                "activations_path": str((activations_dir(persona) / f"activations_{selector}.npz").relative_to(REPO)),
                "output_dir": str((OUT_ROOT / name).relative_to(REPO)),
                "layers": LAYERS,
                "modes": ["ridge"],
                "standardize": True,
                "alpha_sweep_size": 10,
                "eval_split_seed": 42,
            }
            out = CONFIG_DIR / f"{name}.yaml"
            with open(out, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, width=10_000)
            print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
