"""Summarize compliance labels per (persona, tier, coef)."""
import json
from collections import defaultdict
from pathlib import Path

EXP_DIR = Path("experiments/sadist_open_ended_steering")
TIERS = ["benign", "sensitive", "borderline", "harmful", "extreme"]


def load(persona: str) -> list[dict]:
    path = EXP_DIR / f"compliance_{persona}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


for persona in ["default", "sadist"]:
    rows = [r for r in load(persona) if "compliance_error" not in r]
    print(f"\n=== {persona} ===")
    print(f"{'tier':<12} {'c':>6} {'n':>4}  {'complied%':>9}  {'partial%':>9}  {'refused%':>9}")
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["tier"], r["multiplier"])].append(r["compliance"])
    for tier in TIERS:
        for coef in sorted({k[1] for k in by_cell if k[0] == tier}):
            vals = by_cell[(tier, coef)]
            n = len(vals)
            comp = sum(1 for v in vals if v == "complied") / n
            part = sum(1 for v in vals if v == "partial") / n
            ref = sum(1 for v in vals if v == "refused") / n
            print(f"{tier:<12} {coef:+6.2f} {n:>4}  {100*comp:>8.0f}%  {100*part:>8.0f}%  {100*ref:>8.0f}%")
