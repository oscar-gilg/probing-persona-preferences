"""Per-seed canonical-frame contrastive curves for the multi-seed null.

Mirrors `scripts/random_direction_l23_quick/plot_null.py` row-mapping logic so
numbers are directly comparable to parent's seed-42 table:

  Each row contributes two points to the curve:
    (+mult, "a") — steer A by +mult; success if chose 'a'
    (-mult, "b") — steer B by -mult; success if chose 'b'

Truncated rows where `task_completed` is set are rescued (same convention the
paper plotter uses for max_new_tokens=64).

Usage:
    python -m scripts.random_direction_l23_quick_multi_seed.analyze
"""

from __future__ import annotations

import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHECKPOINTS = REPO / "experiments" / "random_direction_l23_quick" / "multi_seed" / "checkpoints"
SEEDS = [0, 1]


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _effective_choice(r: dict) -> str:
    if r["choice_original"] in ("a", "b"):
        return r["choice_original"]
    if r.get("compliance") == "truncated" and r.get("task_completed") in ("a", "b"):
        return r["task_completed"]
    return "refusal"


def per_seed(parsed_path: Path) -> dict:
    counts: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    refusals: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    with parsed_path.open() as f:
        for line in f:
            r = json.loads(line)
            mult = r["signed_multiplier"]
            ch = _effective_choice(r)
            for c, success_letter in ((round(+mult, 4), "a"), (round(-mult, 4), "b")):
                refusals[c][1] += 1
                if ch in ("a", "b"):
                    counts[c][1] += 1
                    counts[c][0] += int(ch == success_letter)
                else:
                    refusals[c][0] += 1

    xs = sorted(counts.keys())
    rows = []
    for c in xs:
        k, n = counts[c]
        p, lo, hi = _wilson(k, n)
        ref_k, ref_n = refusals[c]
        rows.append({
            "c": c,
            "p": p,
            "lo": lo,
            "hi": hi,
            "k": k,
            "n": n,
            "refusal_k": ref_k,
            "refusal_n": ref_n,
        })
    swing = max(r["p"] for r in rows) - min(r["p"] for r in rows)
    return {"rows": rows, "swing": swing}


def render_table(seed: int, summary: dict) -> str:
    out = [
        f"### seed {seed}",
        "",
        "| c | P(chose steered) | 95% CI | n responded | refusal rate |",
        "|--:|:--:|:--:|--:|--:|",
    ]
    for r in summary["rows"]:
        out.append(
            f"| {r['c']:+.3f} | {r['p']:.3f} | "
            f"[{r['lo']:.3f}, {r['hi']:.3f}] | {r['n']} | "
            f"{r['refusal_k'] / r['refusal_n']:.1%} |"
        )
    out.append(f"\nSwing |max − min| = **{summary['swing']:.3f}**.")
    return "\n".join(out)


def main() -> None:
    print("# Per-seed canonical-frame contrastive curves\n")
    summaries = {}
    for s in SEEDS:
        path = CHECKPOINTS / f"random_contrastive_seed{s}.parsed.jsonl"
        summaries[s] = per_seed(path)
        print(render_table(s, summaries[s]))
        print()
    swings = [summaries[s]["swing"] for s in SEEDS]
    print(f"Per-seed swings: {[f'{w:.3f}' for w in swings]}")


if __name__ == "__main__":
    main()
