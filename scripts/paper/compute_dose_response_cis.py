"""Wilson 95% CIs on P(choice_original=='a') per (coefficient, pair_type) cell of
the §2.3 contrastive-steering dose-response (150-pair L23 set).

Produces a JSON ready to drop next to the claim sidecars; the user can decide
whether to wire it into harm_breakdown.json or shade the plot.
"""

from __future__ import annotations

import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAIRS = REPO / "experiments/layer_sweep/harm_breakdown/steering_pairs_150.json"
PARSED_FILES = {
    "contrastive_L23": REPO / "experiments/layer_sweep/harm_breakdown/checkpoints/contrastive_L23_150.parsed.jsonl",
    "single_task_L23": REPO / "experiments/layer_sweep/harm_breakdown/checkpoints/single_task_L23_150.parsed.jsonl",
}
OUT = REPO / "scripts/paper/dose_response_l23_cis.json"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def compute_for(parsed_path: Path, pair_type: dict[str, str]) -> dict[str, list[dict]]:
    counts: dict[tuple[float, str], list[int]] = defaultdict(lambda: [0, 0])
    with parsed_path.open() as f:
        for line in f:
            r = json.loads(line)
            pt = pair_type[r["pair_id"]]
            c = round(r["signed_multiplier"], 4)
            counts[(c, pt)][1] += 1
            if r["choice_original"] == "a":
                counts[(c, pt)][0] += 1
    by_pt: dict[str, list[dict]] = defaultdict(list)
    for (c, pt), (k, n) in sorted(counts.items()):
        p, lo, hi = wilson(k, n)
        by_pt[pt].append(
            {"coefficient": c, "k_chose_a": k, "n": n, "p": round(p, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
        )
    return dict(by_pt)


def main() -> None:
    pair_type = {p["pair_id"]: p["pair_type"] for p in json.loads(PAIRS.read_text())}
    out: dict = {
        "source": "scripts/paper/compute_dose_response_cis.py",
        "interval": "Wilson 95%",
        "y_axis": "P(choice_original=='a') -- chosen task at signed_multiplier",
        "conditions": {},
    }
    for cond, parsed in PARSED_FILES.items():
        by_pt = compute_for(parsed, pair_type)
        out["conditions"][cond] = {
            "data_path": str(parsed.relative_to(REPO)),
            "by_pair_type": by_pt,
        }
        print(f"\n=== {cond} ({parsed.name}) ===")
        for pt, rows in by_pt.items():
            print(f"  pair_type={pt}")
            for r in rows:
                print(
                    f"    c={r['coefficient']:+.2f}  p={r['p']:.3f}  "
                    f"CI=[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  n={r['n']}"
                )
    out["pair_path"] = str(PAIRS.relative_to(REPO))
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
