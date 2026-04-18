"""Aggregate cross-persona steering checkpoints.

Primary metric (differential-steering validation): P(steered task was chosen),
folded across +/- coefficient. For each (persona, condition, |coef|) with |coef|>0:
  - At c>0 the "steered" task is Task A (first span gets +direction).
  - At c<0 the "steered" task is Task B (first span gets -direction).
  - We mean over all trials on both sides.

At |coef|=0 the "steered task" is undefined (no steering); we report P(choose A)
as a baseline diagnostic instead.

Secondary metric kept for backward compatibility: P(choose default-preferred task)
stratified per (condition, signed_coef). Not the headline in the rewritten report.
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
    return "a" if delta_mu >= 0 else "b"


def aggregate_persona(rows: list[dict]) -> dict:
    """Compute validation-style P(steered) and legacy P(default-preferred) cells."""
    # Validation metric: P(steered was chosen), folded by |coef|
    val_cells = defaultdict(list)          # (condition, |coef|>0) -> list of 1/0
    baseline_at_zero = defaultdict(list)   # condition -> list of 1/0 (P(A) at c=0)

    # Legacy metric: P(default-preferred chosen) per signed coefficient
    def_cells = defaultdict(list)          # (condition, signed_coef) -> list of 1/0

    for row in rows:
        choice = row["choice_original"]
        if choice not in ("a", "b"):
            continue
        c = row["signed_multiplier"]
        cond = row["condition"]

        # Validation metric
        if c == 0:
            baseline_at_zero[cond].append(1 if choice == "a" else 0)
        else:
            steered_chosen = (c > 0 and choice == "a") or (c < 0 and choice == "b")
            val_cells[(cond, abs(c))].append(1 if steered_chosen else 0)

        # Legacy metric
        def_pref = default_preferred_task(row["delta_mu"])
        def_cells[(cond, c)].append(1 if choice == def_pref else 0)

    validation = {}
    for (cond, abs_c), vals in val_cells.items():
        validation[f"{cond}|{abs_c}"] = {
            "condition": cond,
            "abs_coefficient": abs_c,
            "n_rows": len(vals),
            "mean_steered_chosen": float(np.mean(vals)),
            "sem": float(np.std(vals) / np.sqrt(len(vals))),
        }
    baseline = {
        cond: {
            "condition": cond,
            "n_rows": len(v),
            "mean_p_a": float(np.mean(v)),
        }
        for cond, v in baseline_at_zero.items()
    }

    legacy = {}
    for (cond, coef), vals in def_cells.items():
        legacy[f"{cond}|{coef}"] = {
            "condition": cond,
            "coefficient": coef,
            "n_rows": len(vals),
            "mean_default_pref": float(np.mean(vals)),
        }

    return {
        "validation_cells": validation,
        "baseline_at_zero": baseline,
        "legacy_default_pref_cells": legacy,
    }


def main() -> None:
    out = {"personas": {}}
    for persona in PERSONAS:
        rows = load_parsed(persona)
        if not rows:
            print(f"[skip] {persona}: no parsed rows")
            continue
        agg = aggregate_persona(rows)
        n_val = sum(c["n_rows"] for c in agg["validation_cells"].values())
        n_base = sum(c["n_rows"] for c in agg["baseline_at_zero"].values())
        print(f"[ok]   {persona}: {n_val} steered-direction rows across "
              f"{len(agg['validation_cells'])} (cond, |coef|) cells; "
              f"{n_base} baseline rows (c=0)")
        out["personas"][persona] = agg

    out_path = EXP_DIR / "aggregated.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
