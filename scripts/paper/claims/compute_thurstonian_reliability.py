"""Compute Thurstonian seed-to-seed reliability.

The §2.1 methodology paragraph reports `seed-to-seed r = 0.94`: the
correlation between Thurstonian utilities fit from two independent
active-learning runs on the same task pool.

Two runs on Gemma-3-27B default persona live under
`results/experiments/main_probes/`:
  - `gemma3_10k_run1/.../thurstonian_80fa9dc8.csv`  (10k tasks)
  - `gemma3_3k_run2/.../thurstonian_a1ebd06e.csv`   (3k tasks)

We compute Pearson r on the intersection of task_ids.

Run:
  python scripts/paper/claims/compute_thurstonian_reliability.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from src.paper.claims import ClaimSet


REPO_ROOT = Path(__file__).resolve().parents[3]

RUN1 = (
    REPO_ROOT / "results/experiments/main_probes/gemma3_10k_run1/"
    "pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0/"
    "thurstonian_80fa9dc8.csv"
)
RUN2 = (
    REPO_ROOT / "results/experiments/main_probes/gemma3_3k_run2/"
    "pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0/"
    "thurstonian_a1ebd06e.csv"
)


def _load(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["task_id"]] = float(row["mu"])
    return out


def main() -> None:
    run1 = _load(RUN1)
    run2 = _load(RUN2)
    shared = sorted(set(run1) & set(run2))
    if not shared:
        raise RuntimeError("No overlapping task_ids between the two runs.")

    xs = np.array([run1[t] for t in shared])
    ys = np.array([run2[t] for t in shared])
    r, _ = pearsonr(xs, ys)

    claims = ClaimSet(source="scripts/paper/claims/compute_thurstonian_reliability.py")
    claims.register(
        "Thurstonian seed-to-seed r",
        round(float(r), 3),
        "Pearson correlation between Thurstonian utilities fit from two "
        "independent active-learning runs on the default Gemma-3-27B persona "
        f"(gemma3_10k_run1 vs gemma3_3k_run2; {len(shared)} overlapping task_ids). "
        "Interpreted in the paper as a seed-to-seed reliability estimate for the "
        "utility-extraction pipeline.",
        used_in=["sec:method-revealed"],
    )
    claims.register(
        "Thurstonian seed-to-seed overlap count",
        len(shared),
        f"{len(shared)} task_ids appear in both Gemma-3-27B active-learning runs "
        "used for the Thurstonian seed-to-seed reliability estimate.",
        used_in=["sec:method-revealed"],
    )

    sidecar = REPO_ROOT / "paper" / "claims" / "thurstonian_reliability.json"
    claims.save(sidecar)
    print(f"Saved {sidecar.relative_to(REPO_ROOT)}  ({len(claims.claims)} claims)")
    print(f"  seed-to-seed r = {r:.4f}  (n = {len(shared)})")


if __name__ == "__main__":
    main()
