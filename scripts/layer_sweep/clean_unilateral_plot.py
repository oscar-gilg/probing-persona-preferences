"""Unilateral dose-response across all 20 layers.

x-axis: actual signed coefficient applied to the steered task's tokens
        (= signed_multiplier × (+1 if ordering==0 else -1); span_coef = +1 here).
y-axis: P(model picked THAT task).

Baselines:
  - first-span and second-span baselines at coef=0 are derived empirically from
    "dead" layers (L≤14 and L≥35) where steering has no effect. Each panel gets
    a black point at x=0 showing these baselines.
  - The aggregate (first + second) / 2 dashed line removes the position-bias
    signature, showing the pure steering effect averaged across which span got
    the push.
"""
from datetime import datetime
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

CHECKPOINTS = Path("experiments/layer_sweep/checkpoints")
ASSETS = Path("experiments/layer_sweep/assets")
LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59]
DEAD_LAYERS = [2, 5, 8, 11, 14, 35, 38, 41, 44, 47, 50, 53, 56, 59]

def load():
    rows = []
    for f in ["eot_unilateral_diagonal_early.parsed.jsonl", "eot_unilateral_diagonal_late.parsed.jsonl"]:
        p = CHECKPOINTS / f
        if p.exists():
            rows.extend(json.loads(l) for l in p.read_text().splitlines())
    return rows

def steered_span(cond):
    return "first" if cond == "unilateral_first" else "second"

def physical_task_in_span(span, ordering):
    if span == "first":
        return "a" if ordering == 0 else "b"
    return "b" if ordering == 0 else "a"

def p_steered_at(rows, layer, cond, coef):
    span = steered_span(cond)
    hits = 0
    n = 0
    for r in rows:
        if r["layer"] != layer or r["condition"] != cond:
            continue
        applied = r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1)
        if abs(applied - coef) > 1e-6:
            continue
        tgt = physical_task_in_span(span, r["ordering"])
        hits += int(r["choice_original"] == tgt)
        n += 1
    return (hits / n) if n else None

def baselines_from_dead_layers(rows):
    """Return (first_baseline, second_baseline) averaged across dead layers × all coefs."""
    out = {}
    for cond in ("unilateral_first", "unilateral_second"):
        span = steered_span(cond)
        hits = n = 0
        for r in rows:
            if r["layer"] not in DEAD_LAYERS or r["condition"] != cond:
                continue
            tgt = physical_task_in_span(span, r["ordering"])
            hits += int(r["choice_original"] == tgt)
            n += 1
        out[span] = hits / n
    return out["first"], out["second"]

def main():
    rows = load()
    first_baseline, second_baseline = baselines_from_dead_layers(rows)
    aggregate_baseline = (first_baseline + second_baseline) / 2
    print(f"Baselines (from L{DEAD_LAYERS[0]}-L{DEAD_LAYERS[4]} and L{DEAD_LAYERS[5]}-L{DEAD_LAYERS[-1]}):")
    print(f"  first-span  = {first_baseline:.3f}")
    print(f"  second-span = {second_baseline:.3f}")
    print(f"  aggregate   = {aggregate_baseline:.3f}")

    coefs = sorted({r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1) for r in rows})
    ncols, nrows = 5, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.3 * nrows), sharey=True, sharex=True)

    for idx, L in enumerate(LAYERS):
        ax = axes[idx // ncols][idx % ncols]

        # Compute y for each span line at each coef
        y_by_span = {"first": [], "second": []}
        for coef in coefs:
            for cond in ("unilateral_first", "unilateral_second"):
                span = steered_span(cond)
                y = p_steered_at(rows, L, cond, coef)
                y_by_span[span].append(y)

        # Insert baseline at x=0 and sort
        xs_first = list(coefs) + [0.0]
        ys_first = y_by_span["first"] + [first_baseline]
        ordered_first = sorted(zip(xs_first, ys_first))
        ax.plot([p[0] for p in ordered_first], [p[1] for p in ordered_first],
                "o-", color="C0", label="first-span", markersize=4, linewidth=1.5)

        xs_second = list(coefs) + [0.0]
        ys_second = y_by_span["second"] + [second_baseline]
        ordered_second = sorted(zip(xs_second, ys_second))
        ax.plot([p[0] for p in ordered_second], [p[1] for p in ordered_second],
                "o-", color="C1", label="second-span", markersize=4, linewidth=1.5)

        # Baseline dots at x=0 — black for visibility
        ax.plot(0, first_baseline, "o", color="black", markersize=4)
        ax.plot(0, second_baseline, "o", color="black", markersize=4)

        # Aggregate dashed line: (first + second) / 2 at each coef, plus baseline at 0
        agg_xs = list(coefs) + [0.0]
        agg_ys = [(f + s) / 2 for f, s in zip(y_by_span["first"], y_by_span["second"])] + [aggregate_baseline]
        ordered_agg = sorted(zip(agg_xs, agg_ys))
        ax.plot([p[0] for p in ordered_agg], [p[1] for p in ordered_agg],
                "--", color="gray", label="aggregate", linewidth=1.5)

        ax.axvline(0, color="gray", linestyle="-", alpha=0.2, linewidth=0.5)
        ax.set_title(f"L{L}", fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.06, 0.06)
        ax.grid(True, alpha=0.3)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("coef on steered task (× N)", fontsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("P(picked steered task)", fontsize=8)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=7)

    fig.suptitle("Unilateral steering — P(picked steered task) vs per-task signed coefficient (eot probes)", fontsize=11)
    fig.tight_layout()
    stamp = datetime.now().strftime("%m%d%y")
    out = ASSETS / f"plot_{stamp}_unilateral_dose_response.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
