"""Compute per-persona swing magnitudes + baselines for the report."""

from __future__ import annotations

import json
from pathlib import Path


CHECKPOINTS = Path("experiments/cross_persona_unilateral/checkpoints")
PERSONAS = ["aura", "contrarian", "mathematician", "sadist", "slacker", "strategist"]


def _load(name: str) -> list[dict]:
    path = CHECKPOINTS / f"{name}.parsed.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _physical_task(span: str, ordering: int) -> str:
    if span == "first":
        return "a" if ordering == 0 else "b"
    return "b" if ordering == 0 else "a"


def _p_steered(rows: list[dict], cond: str, coef: float) -> float | None:
    span = "first" if cond == "unilateral_first" else "second"
    hits = n = 0
    for r in rows:
        if r["condition"] != cond:
            continue
        applied = r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1)
        if abs(applied - coef) > 1e-6:
            continue
        target = _physical_task(span, r["ordering"])
        hits += int(r["choice_original"] == target)
        n += 1
    return (hits / n) if n else None


def _baseline_p(rows: list[dict], span: str) -> float:
    hits = n = 0
    for r in rows:
        if r["choice_original"] not in ("a", "b"):
            continue
        target = _physical_task(span, r["ordering"])
        hits += int(r["choice_original"] == target)
        n += 1
    return hits / n


def main() -> None:
    print(f"{'persona':14s} {'first Δ':>9s} {'second Δ':>9s} {'mean Δ':>9s} "
          f"{'base1':>7s} {'base2':>7s} {'posbias':>8s}")
    print("-" * 70)
    for p in PERSONAS:
        steer = _load(p)
        base = _load(f"{p}_baseline")
        first_swing = _p_steered(steer, "unilateral_first", 0.05) - _p_steered(steer, "unilateral_first", -0.05)
        second_swing = _p_steered(steer, "unilateral_second", 0.05) - _p_steered(steer, "unilateral_second", -0.05)
        mean_swing = (first_swing + second_swing) / 2
        b1 = _baseline_p(base, "first")
        b2 = _baseline_p(base, "second")
        print(f"{p:14s} {first_swing:>9.3f} {second_swing:>9.3f} {mean_swing:>9.3f} "
              f"{b1:>7.3f} {b2:>7.3f} {b1-b2:>+8.3f}")


if __name__ == "__main__":
    main()
