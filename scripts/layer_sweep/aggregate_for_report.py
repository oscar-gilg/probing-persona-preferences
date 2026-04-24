"""Aggregate steering results from all checkpoints for the report."""
import json
import re
from collections import defaultdict
from pathlib import Path

def p_a(rows):
    return sum(r["choice_original"] == "a" for r in rows) / len(rows) if rows else float("nan")

def p_refuse(rows):
    return sum(r["choice_original"] not in ("a", "b") for r in rows) / len(rows) if rows else float("nan")


CHECKPOINTS = Path("experiments/layer_sweep/checkpoints")

# ---- Differential diagonal (probe at own layer) per selector ----
print("=" * 78)
print("DIFFERENTIAL DIAGONAL — probe at own layer, inject at own layer")
print("=" * 78)
for sel in ["tb-2", "eot"]:
    print(f"\n--- {sel} ---")
    print(f"{'layer':>5}  {'P(a)@-5%':>10}  {'P(a)@+5%':>10}  {'|effect|':>10}  {'refuse':>8}")
    for L in [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59]:
        cp = CHECKPOINTS / f"{sel}_probe_L{L:02d}.parsed.jsonl"
        if not cp.exists():
            continue
        rows = [json.loads(l) for l in cp.read_text().splitlines()]
        diag = [r for r in rows if r["layer"] == L]
        if not diag:
            print(f"  L{L:02d}  (no self-cell in spine config)")
            continue
        neg5 = [r for r in diag if abs(r["signed_multiplier"] + 0.05) < 1e-6]
        pos5 = [r for r in diag if abs(r["signed_multiplier"] - 0.05) < 1e-6]
        p_n = p_a(neg5)
        p_p = p_a(pos5)
        eff = abs(p_p - p_n)
        print(f"  L{L:02d}  {p_n:>10.3f}  {p_p:>10.3f}  {eff:>10.3f}  {p_refuse(diag):>8.3f}")

# ---- Unilateral diagonal (both spans) ----
print()
print("=" * 78)
print("UNILATERAL DIAGONAL — eot probe at own layer, single span at a time")
print("=" * 78)
for part in ["early", "late"]:
    cp = CHECKPOINTS / f"eot_unilateral_diagonal_{part}.parsed.jsonl"
    if not cp.exists():
        print(f"  {part}: not found")
        continue
    rows = [json.loads(l) for l in cp.read_text().splitlines()]
    layers = sorted({r["layer"] for r in rows})
    conditions = sorted({r["condition"] for r in rows})
    print(f"\n--- {part} ({len(rows)} rows, layers {layers[:3]}...{layers[-3:]}, conditions {conditions}) ---")
    print(f"{'layer':>5}  {'condition':>20}  {'P(a)@-5%':>10}  {'P(a)@+5%':>10}  {'|swing|':>8}  {'refuse':>8}")
    for L in layers:
        for cond in conditions:
            cell = [r for r in rows if r["layer"] == L and r["condition"] == cond]
            neg5 = [r for r in cell if abs(r["signed_multiplier"] + 0.05) < 1e-6]
            pos5 = [r for r in cell if abs(r["signed_multiplier"] - 0.05) < 1e-6]
            p_n = p_a(neg5)
            p_p = p_a(pos5)
            swing = abs(p_p - p_n)
            print(f"  L{L:02d}  {cond:>20}  {p_n:>10.3f}  {p_p:>10.3f}  {swing:>8.3f}  {p_refuse(cell):>8.3f}")

# ---- Spine × injection heatmap data (differential) ----
print()
print("=" * 78)
print("SPINE STEERING — tb-2 + eot spine probe × injection layer, peak at |mult|=0.05")
print("=" * 78)
SPINE = [11, 23, 32, 44, 53]
for sel in ["tb-2", "eot"]:
    print(f"\n--- {sel} spine probes ---")
    inject_layers = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
    header = "  ".join(f"L{L:02d}" for L in inject_layers)
    print(f"{'probe':>6}  {header}")
    for p in SPINE:
        cp = CHECKPOINTS / f"{sel}_probe_L{p:02d}.parsed.jsonl"
        if not cp.exists():
            continue
        rows = [json.loads(l) for l in cp.read_text().splitlines()]
        cells = []
        for L in inject_layers:
            rs = [r for r in rows if r["layer"] == L]
            if not rs:
                cells.append(" —   ")
                continue
            neg5 = [r for r in rs if abs(r["signed_multiplier"] + 0.05) < 1e-6]
            pos5 = [r for r in rs if abs(r["signed_multiplier"] - 0.05) < 1e-6]
            eff = abs(p_a(pos5) - p_a(neg5))
            cells.append(f"{eff:.3f}")
        print(f"  L{p:02d}  {'  '.join(cells)}")
