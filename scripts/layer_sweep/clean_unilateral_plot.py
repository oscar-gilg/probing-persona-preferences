"""Clean unilateral plot: x = applied coefficient on the steered task, y = P(picked that task).

Applied coefficient = signed_multiplier × span_coef × ordering_sign
                    = signed_multiplier × (+1 if ordering==0 else -1)    [span_coef=+1 for both uni conditions]

This un-mixes the ordering correction so each x-value corresponds to a single
physical operation: a coefficient of that sign and magnitude applied to one
task's tokens.
"""
from datetime import datetime
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CHECKPOINTS = Path("experiments/layer_sweep/checkpoints")
ASSETS = Path("experiments/layer_sweep/assets")

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

def main():
    rows = load()
    layers = [17, 20, 23, 26, 29]

    fig, axes = plt.subplots(1, len(layers), figsize=(3 * len(layers), 3.5), sharey=True)
    for ax, L in zip(axes, layers):
        for cond, color in [("unilateral_first", "C0"), ("unilateral_second", "C1")]:
            span = steered_span(cond)
            by_coef = defaultdict(list)
            for r in rows:
                if r["layer"] != L or r["condition"] != cond:
                    continue
                applied_coef = r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1)
                tgt = physical_task_in_span(span, r["ordering"])
                picked = r["choice_original"] == tgt
                by_coef[round(applied_coef, 4)].append(picked)

            xs, ys, ns = [], [], []
            for c in sorted(by_coef):
                picks = by_coef[c]
                xs.append(c)
                ys.append(sum(picks) / len(picks))
                ns.append(len(picks))
            label = f"{span}-span steered"
            ax.plot(xs, ys, "o-", color=color, label=label, markersize=6)

        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)
        ax.set_xlabel("coef applied to steered task  (× N(L))")
        ax.set_title(f"L{L}")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("P(model picked the steered task)")
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Unilateral dose-response — eot, per-task signed coefficient")
    fig.tight_layout()
    stamp = datetime.now().strftime("%m%d%y")
    out = ASSETS / f"plot_{stamp}_unilateral_dose_response.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
