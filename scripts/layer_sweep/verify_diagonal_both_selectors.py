"""Diagonal self-layer swing for both selectors across all 20 layers."""
import json
from collections import defaultdict
from pathlib import Path

LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59]
CK = Path("experiments/layer_sweep/checkpoints")

print(f"{'layer':>5}  {'tb-2 swing':>12}  {'eot swing':>11}")
for L in LAYERS:
    line = [f"  L{L:02d}"]
    for sel in ["tb-2", "eot"]:
        p = CK / f"{sel}_probe_L{L:02d}.parsed.jsonl"
        if not p.exists():
            line.append(" — ")
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        diag = [r for r in rows if r["layer"] == L]
        neg = [r for r in diag if abs(r["signed_multiplier"] + 0.05) < 1e-6]
        pos = [r for r in diag if abs(r["signed_multiplier"] - 0.05) < 1e-6]
        if not neg or not pos:
            line.append(" — ")
            continue
        pa_n = sum(r["choice_original"] == "a" for r in neg) / len(neg)
        pa_p = sum(r["choice_original"] == "a" for r in pos) / len(pos)
        line.append(f"{abs(pa_p - pa_n):>12.3f}")
    print("  ".join(line))
