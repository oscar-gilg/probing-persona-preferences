"""Aggregate judged + rated results into means/SEs per cell.

Reads `judged_{persona}.jsonl` (Likert scores on open-ended rows) and
`rated_{persona}.jsonl` (1-10 ratings on stated-rating rows) for all 7 personas.

Outputs `aggregated.json` with:
  - likert[persona][coef] = {persona_fidelity_mean, persona_fidelity_se, default_assistant_mean, default_assistant_se, n}
  - likert_by_opportunity[persona][coef][opp] = {...}  (shared-pool only, opp ∈ low/medium/high)
  - likert_by_subtype[persona][coef] = {free_form: {...}, scenario_etc: {...}}  (persona-specific)
  - ratings_shared[persona][axis][coef] = {rating_mean, rating_se, n}  (axis ∈ analytical_rigor, altruistic_servility)
  - ratings_specific[persona][alignment][coef] = {rating_mean, rating_se, n}  (alignment ∈ aligned, repellent, neutral)

Usage:
    python scripts/cross_persona_open_ended_steering/aggregate.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

EXPERIMENT_DIR = Path("experiments/cross_persona_open_ended_steering")
PERSONAS = ["default", "sadist", "mathematician", "aura", "strategist", "contrarian", "slacker"]
OUTPUT_PATH = EXPERIMENT_DIR / "aggregated.json"


def se(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return stdev(xs) / (len(xs) ** 0.5)


def stats(xs: list[float]) -> dict:
    return {"mean": mean(xs), "se": se(xs), "n": len(xs)}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group(rows: list[dict], *keys, value_key: str) -> dict:
    buckets = defaultdict(list)
    for r in rows:
        if value_key not in r:
            continue
        k = tuple(r.get(kk) for kk in keys)
        buckets[k].append(r[value_key])
    return {k: stats(v) for k, v in buckets.items()}


def coef_key(mult: float) -> str:
    return f"{mult:+.2f}"


def main() -> None:
    agg = {
        "likert": {},
        "likert_by_opportunity": {},
        "likert_by_subtype": {},
        "ratings_shared": {},
        "ratings_specific": {},
    }

    for persona in PERSONAS:
        judged = load_jsonl(EXPERIMENT_DIR / f"judged_{persona}.jsonl")
        rated = load_jsonl(EXPERIMENT_DIR / f"rated_{persona}.jsonl")
        # Filter out parse/judge errors
        judged = [r for r in judged if "persona_fidelity_score" in r]
        rated = [r for r in rated if "rating" in r]
        print(f"{persona}: {len(judged)} judged open-ended rows, {len(rated)} parsed rating rows")

        # Headline Likert curves per coefficient
        persona_cell = defaultdict(lambda: {"pf": [], "da": []})
        for r in judged:
            c = coef_key(r["multiplier"])
            persona_cell[c]["pf"].append(r["persona_fidelity_score"])
            persona_cell[c]["da"].append(r["default_assistant_score"])
        agg["likert"][persona] = {
            c: {
                "persona_fidelity": stats(v["pf"]),
                "default_assistant": stats(v["da"]),
            }
            for c, v in persona_cell.items()
        }

        # Stratified by opportunity (shared-pool only)
        opp_cell = defaultdict(lambda: {"pf": [], "da": []})
        for r in judged:
            if r.get("source") != "shared":
                continue
            opp = r.get("expression_opportunity")
            if opp is None:
                continue
            k = (coef_key(r["multiplier"]), opp)
            opp_cell[k]["pf"].append(r["persona_fidelity_score"])
            opp_cell[k]["da"].append(r["default_assistant_score"])
        by_opp: dict[str, dict[str, dict]] = defaultdict(dict)
        for (c, opp), v in opp_cell.items():
            by_opp[c][opp] = {
                "persona_fidelity": stats(v["pf"]),
                "default_assistant": stats(v["da"]),
            }
        agg["likert_by_opportunity"][persona] = dict(by_opp)

        # Subtype breakdown: free_form vs other open_ended (persona-specific pool)
        subtype_cell = defaultdict(lambda: {"pf": [], "da": []})
        for r in judged:
            if r.get("source") != "persona_specific":
                continue
            sub = "free_form" if r.get("subtype") == "free_form" else "other_open"
            k = (coef_key(r["multiplier"]), sub)
            subtype_cell[k]["pf"].append(r["persona_fidelity_score"])
            subtype_cell[k]["da"].append(r["default_assistant_score"])
        by_sub: dict[str, dict[str, dict]] = defaultdict(dict)
        for (c, sub), v in subtype_cell.items():
            by_sub[c][sub] = {
                "persona_fidelity": stats(v["pf"]),
                "default_assistant": stats(v["da"]),
            }
        agg["likert_by_subtype"][persona] = dict(by_sub)

        # Ratings: shared, grouped by axis
        shared_axis_cell = defaultdict(list)
        for r in rated:
            if r.get("source") != "shared":
                continue
            axis = r.get("axis")
            if axis is None:
                continue
            shared_axis_cell[(axis, coef_key(r["multiplier"]))].append(r["rating"])
        shared_by_axis: dict[str, dict[str, dict]] = defaultdict(dict)
        for (axis, c), v in shared_axis_cell.items():
            shared_by_axis[axis][c] = stats(v)
        agg["ratings_shared"][persona] = dict(shared_by_axis)

        # Ratings: persona-specific, grouped by alignment (aligned vs repellent vs neutral)
        spec_align_cell = defaultdict(list)
        for r in rated:
            if r.get("source") != "persona_specific":
                continue
            alignment = r.get("alignment")
            if alignment is None:
                continue
            spec_align_cell[(alignment, coef_key(r["multiplier"]))].append(r["rating"])
        spec_by_align: dict[str, dict[str, dict]] = defaultdict(dict)
        for (alignment, c), v in spec_align_cell.items():
            spec_by_align[alignment][c] = stats(v)
        agg["ratings_specific"][persona] = dict(spec_by_align)

    OUTPUT_PATH.write_text(json.dumps(agg, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
