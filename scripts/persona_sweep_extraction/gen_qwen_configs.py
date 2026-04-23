"""Generate Qwen-3.5-122B extraction configs for the final-six personas + default
on the canonical 6000-task set (same task ids as the Gemma persona sweep).

Mirrors gen_configs.py. Differences vs Gemma:
- model: qwen3.5-122b (non-nothink; no /no_think injection during extraction)
- selectors: [tb-1, tb-4] — the established Qwen probe selectors (L38 probe)
- layers: [33, 38, 43] — L38 + one before + one after for a minimal layer sweep
- batch_size: 4 (matches the proven E1c extraction; MoE is memory-tight)
- max_new_tokens: 1 (skip generation; persona-transfer analysis uses activations only)
- adds a 7th config (default) with empty system_prompt
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

    final_six = data["metadata"]["final_six"]
    by_name = {p["name"]: p["system_prompt"] for p in data["personas"]}
    missing = [p for p in final_six if p not in by_name]
    if missing:
        raise SystemExit(f"Missing personas in sweep_personas.json: {missing}")

    personas = ["default", *final_six]

    for persona in personas:
        cfg = {
            "model": "qwen3.5-122b",
            "task_origins": ["wildchat", "alpaca", "math", "bailbench", "stress_test"],
            "task_ids_file": TASK_IDS_FILE,
            "selectors": [
                "turn_boundary:-1",
                "turn_boundary:-4",
            ],
            "layers_to_extract": [33, 38, 43],
            "device": "auto",
            "batch_size": 4,
            "save_every": 200,
            "max_new_tokens": 1,
            "temperature": 1.0,
            "seed": 42,
            "output_dir": f"/workspace/activations/qwen35_122b/pref_{persona}_sweep",
        }
        if persona != "default":
            cfg["system_prompt"] = by_name[persona]

        out = CONFIG_DIR / f"qwen35_pref_{persona}_sweep.yaml"
        with open(out, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, width=10_000)
        print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
