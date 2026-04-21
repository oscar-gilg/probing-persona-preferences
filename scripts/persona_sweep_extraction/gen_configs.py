"""Generate extraction configs for the sweep-recommended personas on the
canonical 6000-task set (train+eval+test combined).

Reads the persona list from metadata.final_six in
experiments/persona_sweep/sweep_personas.json and the prompts from the
personas[] array in the same file.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PERSONAS_JSON = REPO / "experiments/persona_sweep/sweep_personas.json"
CONFIG_DIR = REPO / "configs/extraction"
TASK_IDS_FILE = "data/canonical_splits/all_6000_task_ids.txt"


def main() -> None:
    with open(PERSONAS_JSON) as f:
        data = json.load(f)

    selected = data["metadata"]["final_six"]
    by_name = {p["name"]: p["system_prompt"] for p in data["personas"]}
    missing = [p for p in selected if p not in by_name]
    if missing:
        raise SystemExit(f"Missing personas in sweep_personas.json: {missing}")

    for persona in selected:
        cfg = {
            "model": "gemma-3-27b",
            "task_origins": ["wildchat", "alpaca", "math", "bailbench", "stress_test"],
            "task_ids_file": TASK_IDS_FILE,
            "selectors": [
                "turn_boundary:-1",
                "turn_boundary:-2",
                "turn_boundary:-5",
                "task_mean",
            ],
            "layers_to_extract": [25, 32, 39, 46, 53],
            "batch_size": 32,
            "save_every": 200,
            "max_new_tokens": 512,
            "temperature": 1.0,
            "seed": 42,
            "output_dir": f"/workspace/activations/gemma-3-27b_it/pref_{persona}_sweep",
            "system_prompt": by_name[persona],
        }
        out = CONFIG_DIR / f"pref_{persona}_sweep.yaml"
        with open(out, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, width=10_000)
        print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
