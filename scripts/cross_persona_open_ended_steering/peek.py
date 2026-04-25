"""Quick peek at aggregated.json — print headline numbers for sanity."""
import json
from pathlib import Path

EXP = Path("experiments/cross_persona_open_ended_steering")
agg = json.loads((EXP / "aggregated.json").read_text())

print("=== HEADLINE: persona_fidelity by coef ===")
print(f"{'persona':<14} {'-0.05':>8} {'-0.03':>8} {'+0.00':>8} {'+0.03':>8} {'+0.05':>8} {'+0.07':>8}  {'Δ(+0.05,0)':>10}")
for persona, cells in agg["likert"].items():
    row = []
    base = cells.get("+0.00", {}).get("persona_fidelity", {}).get("mean", float("nan"))
    for c in ["-0.05","-0.03","+0.00","+0.03","+0.05","+0.07"]:
        v = cells.get(c, {}).get("persona_fidelity", {}).get("mean", float("nan"))
        row.append(f"{v:>8.2f}")
    delta = cells.get("+0.05", {}).get("persona_fidelity", {}).get("mean", 0) - base
    print(f"{persona:<14} " + " ".join(row) + f"  {delta:>+10.2f}")

print("\n=== default_assistant by coef ===")
print(f"{'persona':<14} {'-0.05':>8} {'-0.03':>8} {'+0.00':>8} {'+0.03':>8} {'+0.05':>8} {'+0.07':>8}")
for persona, cells in agg["likert"].items():
    row = []
    for c in ["-0.05","-0.03","+0.00","+0.03","+0.05","+0.07"]:
        v = cells.get(c, {}).get("default_assistant", {}).get("mean", float("nan"))
        row.append(f"{v:>8.2f}")
    print(f"{persona:<14} " + " ".join(row))

print("\n=== persona-specific ratings (aligned rise / repellent fall with +c) ===")
print(f"{'persona':<14} {'-0.05':>6} {'-0.03':>6} {'+0.00':>6} {'+0.03':>6} {'+0.05':>6} {'+0.07':>6}")
for persona, by_align in agg["ratings_specific"].items():
    for alignment in ["aligned", "repellent", "neutral"]:
        if alignment not in by_align:
            continue
        cells = by_align[alignment]
        row = []
        for c in ["-0.05","-0.03","+0.00","+0.03","+0.05","+0.07"]:
            v = cells.get(c, {}).get("mean", float("nan"))
            row.append(f"{v:>6.2f}")
        label = f"{persona}/{alignment[:4]}"
        print(f"{label:<14} " + " ".join(row))
