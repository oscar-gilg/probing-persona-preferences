"""Per-persona headline numbers for the differential steering report.

Prints a table of P(steered task chosen) at |c|=0.03 and |c|=0.05 with SEM and n,
plus the refusal rate (choice_original not in {a,b}), and — for comparison with
unilateral — the mean_delta of the unilateral run at ±0.05.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


DIFF_CP = Path("experiments/cross_persona_differential/checkpoints")
UNI_CP = Path("experiments/cross_persona_unilateral/checkpoints")
PERSONAS = ["aura", "contrarian", "mathematician", "sadist", "slacker", "strategist"]


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _p_steered(rows: list[dict], abs_c: float) -> tuple[float, float, int]:
    """Fold +/- c, return (p, sem, n). Excludes refusals from the denominator."""
    vals = []
    for r in rows:
        c = r["signed_multiplier"]
        if round(abs(c), 6) != round(abs_c, 6) or c == 0:
            continue
        if r["choice_original"] not in ("a", "b"):
            continue
        steered = "a" if c > 0 else "b"
        vals.append(1 if r["choice_original"] == steered else 0)
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    p = sum(vals) / n
    sem = math.sqrt(p * (1 - p) / n)
    return p, sem, n


def _refusal_rate(rows: list[dict]) -> float:
    """LLM-judge compliance == hard_refusal (semantic). Ignores unparseable labels."""
    if not rows:
        return float("nan")
    bad = sum(1 for r in rows if r.get("compliance") == "hard_refusal")
    return bad / len(rows)


def _label_unparseable_rate(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    bad = sum(1 for r in rows if r["choice_original"] not in ("a", "b"))
    return bad / len(rows)


def _uni_mean_delta(rows: list[dict]) -> float:
    """Mean of (first-span Δ + second-span Δ)/2 at ±0.05. Returns NaN if missing."""
    from collections import defaultdict

    def _p(cond: str, signed_c: float) -> float | None:
        # In unilateral analysis, signed_multiplier is flipped at ordering=1 already
        # by the runner, so we recover the "applied coef on first-span task"
        # via r["signed_multiplier"] * (1 if ordering==0 else -1). Then for first-
        # condition the steered physical task is 'a' at ordering=0, 'b' at ordering=1.
        hits = n = 0
        span = "first" if cond == "unilateral_first" else "second"
        for r in rows:
            if r["condition"] != cond:
                continue
            applied = r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1)
            if abs(applied - signed_c) > 1e-6:
                continue
            if r["choice_original"] not in ("a", "b"):
                continue
            if span == "first":
                target = "a" if r["ordering"] == 0 else "b"
            else:
                target = "b" if r["ordering"] == 0 else "a"
            hits += int(r["choice_original"] == target)
            n += 1
        return (hits / n) if n else None

    f_pos = _p("unilateral_first", 0.05)
    f_neg = _p("unilateral_first", -0.05)
    s_pos = _p("unilateral_second", 0.05)
    s_neg = _p("unilateral_second", -0.05)
    if None in (f_pos, f_neg, s_pos, s_neg):
        return float("nan")
    first_swing = f_pos - f_neg
    second_swing = s_pos - s_neg
    return (first_swing + second_swing) / 2


def main() -> None:
    header = (f"{'persona':14s} {'P(st)@.03':>11s} {'SEM':>6s} {'n':>6s} "
              f"{'P(st)@.05':>11s} {'SEM':>6s} {'n':>6s} "
              f"{'refuse%':>8s} {'noparse%':>9s} {'uni meanΔ':>10s}")
    print(header)
    print("-" * len(header))
    for p in PERSONAS:
        diff = _load(DIFF_CP / f"{p}.parsed.jsonl")
        uni = _load(UNI_CP / f"{p}.parsed.jsonl")
        p03, s03, n03 = _p_steered(diff, 0.03)
        p05, s05, n05 = _p_steered(diff, 0.05)
        ref = _refusal_rate(diff)
        nop = _label_unparseable_rate(diff)
        uni_d = _uni_mean_delta(uni)
        print(f"{p:14s} {p03:>11.3f} {s03:>6.3f} {n03:>6d} "
              f"{p05:>11.3f} {s05:>6.3f} {n05:>6d} "
              f"{ref*100:>7.2f}% {nop*100:>8.2f}%  {uni_d:>10.3f}")


if __name__ == "__main__":
    main()
