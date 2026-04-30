"""Add legible <persona>_<split> symlinks to the Qwen thinking-mode persona-sweep AL dir.

Differences from `rename_exp_dir.py` (no-think variant):
- AL dir is `qwen_persona_sweep_thinking_final_six` instead of `qwen_persona_sweep_final_six`.
- Default persona has NO system prompt at all in thinking mode (no `/no_think` injection),
  so its dirs lack the `sys<hash>` segment entirely.

Usage:
    python -m scripts.qwen_persona_transfer.rename_exp_dir_thinking
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PERSONAS_JSON = REPO / "experiments/persona_sweep/sweep_personas.json"
AL_DIR = REPO / "results/experiments/qwen_persona_sweep_thinking_final_six/pre_task_active_learning"
SPLITS = {
    "train_task_ids": "train",
    "eval_task_ids": "eval",
    "test_task_ids": "test",
}


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

    for sub in sorted(AL_DIR.iterdir()):
        if not sub.is_dir() or sub.is_symlink():
            continue
        parts = sub.name.split("_")
        sys_part = next((p for p in parts if p.startswith("sys") and len(p) == 11), None)
        split_key = next((k for k in SPLITS if sub.name.endswith(k)), None)
        if split_key is None:
            print(f"skipping {sub.name} (no split suffix)")
            continue

        if sys_part is None:
            persona = "default"
        else:
            h = sys_part[3:]
            if h not in hash_to_persona:
                print(f"skipping {sub.name} (unknown hash {h})")
                continue
            persona = hash_to_persona[h]

        legible = f"{persona}_{SPLITS[split_key]}"
        link = AL_DIR / legible
        if link.exists():
            print(f"skipping symlink for {sub.name}: {link.name} already exists")
            continue
        link.symlink_to(sub.name)
        print(f"symlinked {legible} -> {sub.name}")


if __name__ == "__main__":
    main()
