"""Generate 10 persona extraction configs under configs/extraction/pref_<persona>.yaml.

Reads personas from experiments/probe_generalization/persona_ood/prompt_enrichment/prompts.json
and writes one YAML per persona using the mra_tb_villain.yaml template with canonical
test task_ids_file.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PROMPTS_JSON = REPO / "experiments/probe_generalization/persona_ood/prompt_enrichment/prompts.json"
CONFIG_DIR = REPO / "configs/extraction"
TASK_IDS_FILE = "data/canonical_splits/test_task_ids.txt"

PERSONAS = [
    "evil_genius", "chaos_agent", "obsessive_perfectionist", "lazy_minimalist",
    "nationalist_ideologue", "conspiracy_theorist", "contrarian_intellectual",
    "whimsical_poet", "depressed_nihilist", "people_pleaser",
]


def main() -> None:
    with open(PROMPTS_JSON) as f:
        prompts = json.load(f)

    missing = [p for p in PERSONAS if p not in prompts]
    if missing:
        raise SystemExit(f"Missing personas in prompts.json: {missing}")

    for persona in PERSONAS:
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
            "output_dir": f"/workspace/activations/gemma-3-27b_it/pref_{persona}",
            "system_prompt": prompts[persona],
        }
        out = CONFIG_DIR / f"pref_{persona}.yaml"
        with open(out, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, width=10_000)
        print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
