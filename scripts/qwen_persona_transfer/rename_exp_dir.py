"""Add legible <persona>_<split> symlinks to the Qwen persona-sweep AL dir.

Differences from the Gemma version (`scripts/persona_sweep_extraction/rename_exp_dir.py`):
- No top-level dir rename — the experiment_id was set to `qwen_persona_sweep_final_six`
  in the AL configs, so the dir is already legibly named.
- Includes `default` (uses `/no_think` as its system prompt hash, since the `-nothink`
  variant injects `/no_think` at runtime when no `measurement_system_prompt` is set).

Usage:
    python -m scripts.qwen_persona_transfer.rename_exp_dir
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PERSONAS_JSON = REPO / "experiments/persona_sweep/sweep_personas.json"
AL_DIR = REPO / "results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning"
SPLITS = {
    "train_task_ids": "train",
    "eval_task_ids": "eval",
    "test_task_ids": "test",
}
DEFAULT_SYSTEM_PROMPT = "/no_think"


def sys_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


def main() -> None:
    if not AL_DIR.is_dir():
        raise SystemExit(f"{AL_DIR} does not exist")

    with open(PERSONAS_JSON) as f:
        data = json.load(f)
    names = data["metadata"]["final_six"]
    by_name = {p["name"]: p["system_prompt"] for p in data["personas"]}
    hash_to_persona = {sys_hash(by_name[n]): n for n in names}
    hash_to_persona[sys_hash(DEFAULT_SYSTEM_PROMPT)] = "default"

    for sub in sorted(AL_DIR.iterdir()):
        if not sub.is_dir() or sub.is_symlink():
            continue
        parts = sub.name.split("_")
        sys_part = next((p for p in parts if p.startswith("sys") and len(p) == 11), None)
        split_key = next((k for k in SPLITS if sub.name.endswith(k)), None)
        if sys_part is None or split_key is None:
            print(f"skipping {sub.name} (no sys<hash>_<split> pattern)")
            continue
        h = sys_part[3:]
        if h not in hash_to_persona:
            print(f"skipping {sub.name} (unknown hash {h})")
            continue
        legible = f"{hash_to_persona[h]}_{SPLITS[split_key]}"
        link = AL_DIR / legible
        if link.exists():
            print(f"skipping symlink for {sub.name}: {link.name} already exists")
            continue
        link.symlink_to(sub.name)
        print(f"symlinked {legible} -> {sub.name}")


if __name__ == "__main__":
    main()
