"""Rename the auto-generated active-learning experiment folder to a legible name
and add legible symlinks to each sys<hash>_<split> subdir.

Usage:
    python -m scripts.persona_sweep_extraction.rename_exp_dir <exp_id>
e.g. python -m scripts.persona_sweep_extraction.rename_exp_dir exp_20260421_215515
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results/experiments"
PERSONAS_JSON = REPO / "experiments/persona_sweep/sweep_personas.json"
TARGET_NAME = "persona_sweep_final_six"
SPLITS = {
    "train_task_ids": "train",
    "eval_task_ids": "eval",
    "test_task_ids": "test",
}


def sys_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <exp_id>")
    exp_id = sys.argv[1]

    src = RESULTS / exp_id
    dst = RESULTS / TARGET_NAME
    if not src.is_dir():
        raise SystemExit(f"{src} does not exist")
    if dst.exists():
        raise SystemExit(f"{dst} already exists; refusing to overwrite")

    with open(PERSONAS_JSON) as f:
        data = json.load(f)
    names = data["metadata"]["final_six"]
    by_name = {p["name"]: p["system_prompt"] for p in data["personas"]}
    hash_to_persona = {sys_hash(by_name[n]): n for n in names}

    src.rename(dst)
    print(f"renamed {src.relative_to(REPO)} -> {dst.relative_to(REPO)}")

    al = dst / "pre_task_active_learning"
    for sub in sorted(al.iterdir()):
        if not sub.is_dir():
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
        link = al / legible
        if link.exists():
            print(f"skipping symlink for {sub.name}: {link.name} already exists")
            continue
        link.symlink_to(sub.name)
        print(f"symlinked {legible} -> {sub.name}")


if __name__ == "__main__":
    main()
