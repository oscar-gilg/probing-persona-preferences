"""Verify one persona's extraction outputs against spec success criteria.

Usage: python scripts/persona_ood_extraction_10/verify_extraction.py <persona>
Exit 0 on success, 1 on any failure. Prints a short per-file table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EXPECTED_SELECTORS = ["turn_boundary:-1", "turn_boundary:-2", "turn_boundary:-5", "task_mean"]
EXPECTED_LAYERS = [25, 32, 39, 46, 53]
MIN_TASK_IDS = 990


def check_npz(path: Path) -> tuple[bool, int, list[int]]:
    data = np.load(path, allow_pickle=True)
    n = len(data["task_ids"])
    layer_keys = sorted(
        int(k.split("_")[1]) for k in data.keys() if k.startswith("layer_")
    )
    return n >= MIN_TASK_IDS and layer_keys == EXPECTED_LAYERS, n, layer_keys


def main() -> None:
    persona = sys.argv[1]
    out_dir = Path(f"/workspace/activations/gemma-3-27b_it/pref_{persona}")
    failures: list[str] = []

    print(f"Verifying {out_dir}")
    if not out_dir.is_dir():
        print(f"FAIL: directory missing")
        sys.exit(1)

    for sel in EXPECTED_SELECTORS:
        p = out_dir / f"activations_{sel}.npz"
        if not p.exists():
            print(f"  FAIL missing {p.name}")
            failures.append(p.name)
            continue
        ok, n, layers = check_npz(p)
        status = "OK" if ok else "FAIL"
        print(f"  {status} {p.name} task_ids={n} layers={layers}")
        if not ok:
            failures.append(p.name)

    for name in ("completions_with_activations.json", "extraction_metadata.json"):
        p = out_dir / name
        if p.exists() and p.stat().st_size > 0:
            print(f"  OK {name} ({p.stat().st_size} B)")
        else:
            print(f"  FAIL missing or empty {name}")
            failures.append(name)

    # Check metadata system_prompt
    meta_path = out_dir / "extraction_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        sp = meta.get("system_prompt")
        if not sp:
            print("  FAIL: extraction_metadata.system_prompt missing/empty")
            failures.append("system_prompt")
        else:
            print(f"  OK system_prompt set ({len(sp)} chars)")

    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
