"""Unilateral dose-response across all 20 layers.

x-axis: actual signed coefficient applied to the steered task's tokens
        (= signed_multiplier × (+1 if ordering==0 else -1); span_coef = +1 here).
y-axis: P(model picked THAT task).

Two lines per panel: first-span steered, second-span steered.

The no-steering baseline is exactly 0.5 for this metric by construction — the
steered task is task_a half the time (ordering=0) and task_b half the time
(ordering=1), so averaging across orderings cancels the natural P(a) > 0.5
preference bias.
"""
from datetime import datetime
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

CHECKPOINTS = Path("experiments/layer_sweep/checkpoints")
ASSETS = Path("experiments/layer_sweep/assets")
LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59]

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

    ncols, nrows = 5, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.3 * nrows), sharey=True, sharex=True)

    for idx, L in enumerate(LAYERS):
        ax = axes[idx // ncols][idx % ncols]
        for cond, color in [("unilateral_first", "C0"), ("unilateral_second", "C1")]:
            span = steered_span(cond)
            by_coef = defaultdict(list)
            for r in rows:
                if r["layer"] != L or r["condition"] != cond:
                    continue
                applied_coef = r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1)
                tgt = physical_task_in_span(span, r["ordering"])
                by_coef[round(applied_coef, 4)].append(r["choice_original"] == tgt)
            xs = sorted(by_coef)
            ys = [sum(by_coef[x]) / len(by_coef[x]) for x in xs]
            if xs:
                ax.plot(xs, ys, "o-", color=color, label=f"{span}-span", markersize=4, linewidth=1.5)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.axvline(0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)
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
