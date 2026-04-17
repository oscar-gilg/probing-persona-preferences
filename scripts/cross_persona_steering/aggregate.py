"""Aggregate cross-persona steering checkpoints into per-(persona, condition, coef) stats.

Reads parsed checkpoints (`checkpoint_{persona}.parsed.jsonl`) written by the
steering runner. Emits a single aggregated JSON with:
  - P(choose default-preferred task) per (persona, condition, coefficient)
  - alignment-shift Δ(c) = P_default-pref(c) - P_default-pref(0)
  - counts, baseline-P(A) terciles, per-pair fractions for stratification
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


EXP_DIR = Path("experiments/cross_persona_steering")
PERSONAS = ["sadist", "villain", "aesthete", "stem_obsessive"]


def load_parsed(persona: str) -> list[dict]:
    path = EXP_DIR / f"checkpoint_{persona}.parsed.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def default_preferred_task(delta_mu: float) -> str:
    """Return which task is preferred by the default-persona probe (higher utility)."""
    return "a" if delta_mu >= 0 else "b"


def row_chose_default_preferred(row: dict) -> bool | None:
    """Did the model (steered or not) choose the default-preferred task? None if choice not parseable."""
    choice = row.get("choice_original")
    if choice not in ("a", "b"):
        return None
    return choice == default_preferred_task(row["delta_mu"])


def aggregate_persona(rows: list[dict]) -> dict:
    """Group by (condition, signed_multiplier), compute mean and counts per pair + overall."""
    by_cell = defaultdict(list)  # (condition, mult) -> list of 1/0
    by_pair = defaultdict(list)  # (condition, mult, pair_id) -> list of 1/0
    baseline_by_pair = defaultdict(list)  # pair_id at mult=0 -> list of 1/0
    for row in rows:
        pref = row_chose_default_preferred(row)
        if pref is None:
            continue
        key = (row["condition"], row["signed_multiplier"])
        by_cell[key].append(pref)
        by_pair[(row["condition"], row["signed_multiplier"], row["pair_id"])].append(pref)
        if row["signed_multiplier"] == 0:
            baseline_by_pair[row["pair_id"]].append(pref)

    cells = {}
    for (cond, mult), vals in by_cell.items():
        cells[f"{cond}|{mult}"] = {
            "condition": cond,
            "coefficient": mult,
            "n_rows": len(vals),
            "mean_default_pref": float(np.mean(vals)),
            "std_default_pref": float(np.std(vals)),
        }

    # Per-pair baseline P(A)
    baseline_pair_mean = {pid: float(np.mean(v)) for pid, v in baseline_by_pair.items()}
    return {
        "cells": cells,
        "baseline_pair_mean_default_pref": baseline_pair_mean,
    }


def main() -> None:
    out = {"personas": {}}
    for persona in PERSONAS:
        rows = load_parsed(persona)
        if not rows:
            print(f"[skip] {persona}: no parsed rows")
            continue
        agg = aggregate_persona(rows)
        n_cells = len(agg["cells"])
        n_rows = sum(c["n_rows"] for c in agg["cells"].values())
        print(f"[ok]   {persona}: {n_rows} rows across {n_cells} (cond, coef) cells")
        out["personas"][persona] = agg

    out_path = EXP_DIR / "aggregated.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
