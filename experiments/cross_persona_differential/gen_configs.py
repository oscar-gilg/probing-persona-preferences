"""Generate steering YAML configs for cross-persona differential steering.

One config per persona. Each config runs a single differential condition at L25
with `spans: {first: 1, second: -1}` and multipliers [-0.05, -0.03, 0.03, 0.05].

v2: all personas steer along the *Assistant (default)* probe — same probe used
in the open-ended evil steering experiment and in the original §3.4 default
steering — so the cross-persona claim is "the Assistant probe drives every
persona's choices" rather than "each persona has its own usable probe".

mean_norm stays per-persona (the steering acts on that persona's activations,
so the coefficient is calibrated to the persona's own activation magnitudes).
system_prompt is pulled from the persona's measurement config.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv


load_dotenv()

PERSONAS = ["aura", "contrarian", "mathematician", "sadist", "slacker", "strategist"]
INJECT_LAYER = 25
MULTIPLIERS = [-0.05, -0.03, 0.03, 0.05]
TEMPLATE_PATH = "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml"

CONFIG_DIR = Path("configs/steering/cross_persona_differential")
UNILATERAL_CONFIG_DIR = Path("configs/steering/cross_persona_unilateral")
PAIRS_PATH = Path("experiments/cross_persona_unilateral/steering_pairs.json")
CHECKPOINTS_DIR = Path("experiments/cross_persona_differential/checkpoints")
PROBE_MANIFEST = "results/probes/persona_sweep_final_six/default_tb-5/"
PERSONA_CONFIG_TMPL = "configs/measurement/persona_sweep/final_six/{persona}_train.yaml"


def _load_system_prompt(persona: str) -> str:
    path = Path(PERSONA_CONFIG_TMPL.format(persona=persona))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg["measurement_system_prompt"]


def _load_existing_mean_norm(persona: str) -> float:
    path = UNILATERAL_CONFIG_DIR / f"{persona}.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return float(cfg["mean_norm"])


def _config_for_persona(persona: str, mean_norm_L25: float, system_prompt: str) -> dict:
    return {
        "model": "gemma-3-27b",
        "max_new_tokens": 64,
        "pairs_path": str(PAIRS_PATH),
        "probe_manifest": PROBE_MANIFEST,
        "checkpoint_path": str(CHECKPOINTS_DIR / f"{persona}.jsonl"),
        "mean_norm": float(mean_norm_L25),
        "n_trials": 3,
        "temperature": 1.0,
        "seed": 42,
        "n_pairs": None,
        "template_path": TEMPLATE_PATH,
        "system_prompt": system_prompt,
        "conditions": [
            {
                "name": "differential",
                "cache_injection": "differential",
                "probe": f"ridge_L{INJECT_LAYER:02d}",
                "layers": [INJECT_LAYER],
                "multipliers": list(MULTIPLIERS),
                "spans": {"first": 1, "second": -1},
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
        mean_norm = _load_existing_mean_norm(persona)
        system_prompt = _load_system_prompt(persona)
        print(f"[{persona}] mean_norm(L{INJECT_LAYER}) = {mean_norm:.1f}")
        print(f"  system_prompt: {system_prompt[:80]}...")

        cfg = _config_for_persona(persona, mean_norm, system_prompt)
        path = CONFIG_DIR / f"{persona}.yaml"
        _dump_yaml(path, cfg)
        written.append(path)

    print(f"\nWrote {len(written)} configs under {CONFIG_DIR.resolve()}:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
