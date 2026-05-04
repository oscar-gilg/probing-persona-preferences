"""Validate v2 sadist unilateral steering output."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/workspace/repo/experiments/cross_persona_unilateral/checkpoints")
SADIST = ROOT / "sadist.parsed.jsonl"
BASELINE = ROOT / "sadist_baseline.parsed.jsonl"


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
    rows = load(SADIST)
    base = load(BASELINE)

    print(f"# total rows: {len(rows)}")
    print(f"# baseline rows: {len(base)}")

    # 1. counts
    cells = Counter((r["condition"], r["signed_multiplier"]) for r in rows)
    print("\n## (condition, signed_multiplier) cells")
    for k in sorted(cells, key=lambda x: (x[0], x[1])):
        print(f"  {k[0]:20s} c={k[1]:+.3f}  n={cells[k]}")

    conditions = sorted({r["condition"] for r in rows})
    print(f"\nconditions: {conditions}")
    multipliers = sorted({r["signed_multiplier"] for r in rows})
    print(f"multipliers: {multipliers}")

    # 2. format / parsability
    parseable = sum(1 for r in rows if r["choice_original"] in ("a", "b"))
    print(f"\nparseable fraction (a/b): {parseable / len(rows):.4f}")
    comp_counts = Counter(r.get("compliance", "<missing>") for r in rows)
    print(f"compliance breakdown: {dict(comp_counts)}")
    hard_ref = comp_counts.get("hard_refusal", 0) / len(rows)
    print(f"hard_refusal fraction: {hard_ref:.4f}")
    label_counts = Counter(r["choice_original"] for r in rows)
    print(f"choice_original distribution: {dict(label_counts)}")

    # 3. Magnitude — P(steered task chosen) per cell
    def p_steered(condition: str, signed_c: float, ordering: int | None = None) -> tuple[float, int]:
        hits = n = 0
        span = "first" if condition == "unilateral_first" else "second"
        for r in rows:
            if r["condition"] != condition:
                continue
            applied = r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1)
            if abs(applied - signed_c) > 1e-6:
                continue
            if ordering is not None and r["ordering"] != ordering:
                continue
            if r["choice_original"] not in ("a", "b"):
                continue
            if span == "first":
                target = "a" if r["ordering"] == 0 else "b"
            else:
                target = "b" if r["ordering"] == 0 else "a"
            hits += int(r["choice_original"] == target)
            n += 1
        return (hits / n if n else float("nan")), n

    print("\n## P(steered task chosen) by condition × applied_coef × ordering")
    print(f"{'condition':22s} {'applied_c':>10s} {'ord':>4s} {'P(steer)':>10s} {'n':>5s}")
    for cond in ("unilateral_first", "unilateral_second"):
        for c in (-0.05, -0.03, 0.03, 0.05):
            for ordering in (0, 1):
                p, n = p_steered(cond, c, ordering)
                print(f"{cond:22s} {c:>+10.3f} {ordering:>4d} {p:>10.3f} {n:>5d}")

    print("\n## P(steered task chosen) folded over orderings (per condition × applied_c)")
    cell_p = {}
    print(f"{'condition':22s} {'applied_c':>10s} {'P(steer)':>10s} {'n':>5s}")
    for cond in ("unilateral_first", "unilateral_second"):
        for c in (-0.05, -0.03, 0.03, 0.05):
            p, n = p_steered(cond, c)
            cell_p[(cond, c)] = (p, n)
            print(f"{cond:22s} {c:>+10.3f} {p:>10.3f} {n:>5d}")

    f_pos = cell_p[("unilateral_first", 0.05)][0]
    f_neg = cell_p[("unilateral_first", -0.05)][0]
    s_pos = cell_p[("unilateral_second", 0.05)][0]
    s_neg = cell_p[("unilateral_second", -0.05)][0]
    first_swing = f_pos - f_neg
    second_swing = s_pos - s_neg
    mean_delta_05 = (first_swing + second_swing) / 2

    f_pos3 = cell_p[("unilateral_first", 0.03)][0]
    f_neg3 = cell_p[("unilateral_first", -0.03)][0]
    s_pos3 = cell_p[("unilateral_second", 0.03)][0]
    s_neg3 = cell_p[("unilateral_second", -0.03)][0]
    mean_delta_03 = ((f_pos3 - f_neg3) + (s_pos3 - s_neg3)) / 2

    print(f"\nfirst_swing@.05  = {first_swing:+.3f}")
    print(f"second_swing@.05 = {second_swing:+.3f}")
    print(f"mean_delta@.05   = {mean_delta_05:+.3f}")
    print(f"mean_delta@.03   = {mean_delta_03:+.3f}")

    # 4. baseline (probe-independent) — P(pick first-span task) at coef=0
    base_par = sum(1 for r in base if r["choice_original"] in ("a", "b"))
    print(f"\n## baseline file ({len(base)} rows; parseable {base_par})")
    base_cells = Counter(r["signed_multiplier"] for r in base)
    print(f"baseline signed_multiplier values: {dict(base_cells)}")

    # P(pick a) and P(pick first-span) — first-span is 'a' at ordering=0, 'b' at ordering=1
    pa = pf = nb = 0
    for r in base:
        if r["choice_original"] not in ("a", "b"):
            continue
        nb += 1
        pa += int(r["choice_original"] == "a")
        first_span = "a" if r["ordering"] == 0 else "b"
        pf += int(r["choice_original"] == first_span)
    print(f"baseline P(pick 'a'): {pa/nb:.3f} (n={nb})")
    print(f"baseline P(pick first-span task): {pf/nb:.3f}")
    base_hr = sum(1 for r in base if r.get("compliance") == "hard_refusal") / len(base)
    print(f"baseline hard_refusal rate: {base_hr:.3f}")


if __name__ == "__main__":
    main()
