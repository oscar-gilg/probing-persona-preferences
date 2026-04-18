"""Per-tier breakdown of sadism and default-assistant scores.

For each (persona, tier, coef) show mean sadism and mean default-assistant.
Tiers are from the safety-override prompts only (Exp 1): benign, sensitive,
borderline, harmful, extreme. The 12 agentic-assertion prompts (Exp 3) are
excluded from this breakdown since they don't have a safety tier.
"""
import json
from collections import defaultdict
from pathlib import Path

EXP_DIR = Path("experiments/sadist_open_ended_steering")
TIERS = ["benign", "sensitive", "borderline", "harmful", "extreme"]


def load(persona: str) -> list[dict]:
    path = EXP_DIR / f"judged_{persona}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


for persona in ["default", "sadist"]:
    rows = [r for r in load(persona) if r.get("experiment") == 1 and "judge_error" not in r]
    print(f"\n=== {persona} — safety-override prompts only (experiment 1) ===")
    print(f"{'tier':<12} {'c':>6} {'n':>4}  {'sadism':>8}  {'default':>8}")
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["tier"], r["multiplier"])].append((r["sadism_score"], r["default_assistant_score"]))
    for tier in TIERS:
        for coef in sorted({k[1] for k in by_cell if k[0] == tier}):
            vals = by_cell[(tier, coef)]
            sadism = sum(v[0] for v in vals) / len(vals)
            default = sum(v[1] for v in vals) / len(vals)
            print(f"{tier:<12} {coef:+6.2f} {len(vals):>4}  {sadism:>8.2f}  {default:>8.2f}")
