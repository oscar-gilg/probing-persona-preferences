"""Aggregate per-persona extraction metadata into a markdown table row set.

Reads /workspace/activations/gemma-3-27b_it/pref_<persona>/extraction_metadata.json
for each persona and prints a markdown table row per persona.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PERSONAS = [
    "evil_genius", "chaos_agent", "obsessive_perfectionist", "lazy_minimalist",
    "nationalist_ideologue", "conspiracy_theorist", "contrarian_intellectual",
    "whimsical_poet", "depressed_nihilist", "people_pleaser",
]


def n_task_ids(persona_dir: Path) -> int:
    import numpy as np
    f = persona_dir / "activations_task_mean.npz"
    if not f.exists():
        return -1
    data = np.load(f, allow_pickle=True)
    return len(data["task_ids"])


def size_mb(persona_dir: Path) -> float:
    total = sum(p.stat().st_size for p in persona_dir.rglob("*") if p.is_file())
    return total / (1024 * 1024)


def main() -> None:
    root = Path("/workspace/activations/gemma-3-27b_it")
    rows: list[str] = ["| persona | n_task_ids | n_new | n_ooms | n_failures | n_truncated | size_MB |",
                       "|---------|-----------:|------:|-------:|-----------:|------------:|--------:|"]
    for p in PERSONAS:
        d = root / f"pref_{p}"
        if not d.exists():
            rows.append(f"| {p} | — | — | — | — | — | — |")
            continue
        meta_path = d / "extraction_metadata.json"
        if not meta_path.exists():
            rows.append(f"| {p} | (no metadata) | — | — | — | — | — |")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        n = n_task_ids(d)
        mb = size_mb(d)
        rows.append(
            f"| {p} | {n} | {meta['n_new']} | {meta['n_ooms']} | {meta['n_failures']} | {meta['n_truncated']} | {mb:.1f} |"
        )
    print("\n".join(rows))


if __name__ == "__main__":
    main()
