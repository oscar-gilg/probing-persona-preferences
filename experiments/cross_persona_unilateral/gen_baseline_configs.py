"""Generate 6 no-steering baseline configs (multipliers=[0]) for each persona.

Each baseline config runs 100 pairs × 3 trials × 2 orderings × 1 condition × 1 mult
= 600 generations. The coef=0 hook adds a zero tensor, so the output is identical
to no steering — gives an empirical baseline of P(pick first-span task) and
P(pick second-span task) under each persona's system prompt.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv


load_dotenv()

PERSONAS = ["aura", "contrarian", "mathematician", "sadist", "slacker", "strategist"]
INJECT_LAYER = 25
TEMPLATE_PATH = "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml"

CONFIG_DIR = Path("configs/steering/cross_persona_unilateral")
PAIRS_PATH = Path("experiments/cross_persona_unilateral/steering_pairs.json")
CHECKPOINTS_DIR = Path("experiments/cross_persona_unilateral/checkpoints")
PROBE_MANIFEST = "results/probes/persona_sweep_final_six/default_tb-5/"
PERSONA_CONFIG_TMPL = "configs/measurement/persona_sweep/final_six/{persona}_train.yaml"


def _load_system_prompt(persona: str) -> str:
    path = Path(PERSONA_CONFIG_TMPL.format(persona=persona))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg["measurement_system_prompt"]


def _load_existing_mean_norm(persona: str) -> float:
    """Pull mean_norm from the existing {persona}.yaml (already computed)."""
    path = CONFIG_DIR / f"{persona}.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return float(cfg["mean_norm"])


def _config_for_persona(persona: str) -> dict:
    return {
        "model": "gemma-3-27b",
        "max_new_tokens": 64,
        "pairs_path": str(PAIRS_PATH),
        "probe_manifest": PROBE_MANIFEST,
        "checkpoint_path": str(CHECKPOINTS_DIR / f"{persona}_baseline.jsonl"),
        "mean_norm": _load_existing_mean_norm(persona),
        "n_trials": 3,
        "temperature": 1.0,
        "seed": 42,
        "n_pairs": None,
        "template_path": TEMPLATE_PATH,
        "system_prompt": _load_system_prompt(persona),
        "conditions": [
            {
                "name": "baseline",
                "cache_injection": "differential",
                "probe": f"ridge_L{INJECT_LAYER:02d}",
                "layers": [INJECT_LAYER],
                "multipliers": [0],
                "spans": {"first": 1},
            },
        ],
    }


def _dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def main() -> None:
    written: list[Path] = []
    for persona in PERSONAS:
        cfg = _config_for_persona(persona)
        path = CONFIG_DIR / f"{persona}_baseline.yaml"
        _dump_yaml(path, cfg)
        written.append(path)

    print(f"Wrote {len(written)} baseline configs:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
