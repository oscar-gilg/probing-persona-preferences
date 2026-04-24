"""Compute per-persona swing magnitudes for the report."""

from __future__ import annotations

import json
from pathlib import Path


CHECKPOINTS = Path("experiments/cross_persona_unilateral/checkpoints")
PERSONAS = ["aura", "contrarian", "mathematician", "sadist", "slacker", "strategist"]


def _p_steered(rows: list[dict], cond: str, coef: float) -> float | None:
    span = "first" if cond == "unilateral_first" else "second"
    hits = n = 0
    for r in rows:
        if r["condition"] != cond:
            continue
        applied = r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1)
        if abs(applied - coef) > 1e-6:
            continue
        if span == "first":
            target = "a" if r["ordering"] == 0 else "b"
        else:
            target = "b" if r["ordering"] == 0 else "a"
        hits += int(r["choice_original"] == target)
        n += 1
    return (hits / n) if n else None


def _refusal(rows: list[dict]) -> float:
    return sum(1 for r in rows if r["choice_original"] not in ("a", "b")) / len(rows)


def main() -> None:
    print(f"{'persona':14s} {'first Δ':>9s} {'second Δ':>9s} {'mean Δ':>9s} {'refusal':>8s}")
    print("-" * 60)
    for p in PERSONAS:
        path = CHECKPOINTS / f"{p}.parsed.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        first_swing = _p_steered(rows, "unilateral_first", 0.05) - _p_steered(rows, "unilateral_first", -0.05)
        second_swing = _p_steered(rows, "unilateral_second", 0.05) - _p_steered(rows, "unilateral_second", -0.05)
        mean_swing = (first_swing + second_swing) / 2
        refusal = _refusal(rows)
        print(f"{p:14s} {first_swing:>9.3f} {second_swing:>9.3f} {mean_swing:>9.3f} {refusal:>7.1%}")


if __name__ == "__main__":
    main()
