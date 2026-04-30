"""Generate Qwen probe configs for the final-six + default persona set, two selectors.

Writes configs/probes/qwen_persona_sweep_final_six/<persona>_tb-{1,4}.yaml. Each
config trains on <persona>_train, alpha-sweeps + eval on <persona>_eval, and
outputs one Ridge probe per layer in [33, 38, 43].

Adapted from `scripts/persona_sweep_extraction/gen_probe_configs.py` on the
research-loop/persona_probe_transfer branch (Gemma version).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PERSONAS_JSON = REPO / "experiments/persona_sweep/sweep_personas.json"
CONFIG_DIR = REPO / "configs/probes/qwen_persona_sweep_final_six"
AL_DIR = REPO / "results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning"
OUT_ROOT = REPO / "results/probes/qwen_persona_sweep_final_six"

SELECTORS = ["turn_boundary:-1", "turn_boundary:-4"]
LAYERS = [33, 38, 43]


def activations_dir(persona: str) -> Path:
    return REPO / f"activations/qwen35_122b/pref_{persona}_sweep"


def main() -> None:
    with open(PERSONAS_JSON) as f:
        data = json.load(f)
    personas = ["default"] + list(data["metadata"]["final_six"])

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    for persona in personas:
        for selector in SELECTORS:
            selector_tag = selector.replace("turn_boundary:", "tb")
            name = f"{persona}_{selector_tag}"
            cfg = {
                "experiment_name": f"qwen_persona_sweep_final_six_{name}",
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
