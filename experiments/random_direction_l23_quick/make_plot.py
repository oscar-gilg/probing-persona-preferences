"""Render the random-vs-validated L23 contrastive comparison plot.

Pools pair types for the validated probe so it sits on the same axes as the
random direction (which has only the pooled curve).
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CIS_JSON = ROOT / "scripts" / "paper" / "dose_response_l23_cis.json"
RANDOM_PARSED = HERE / "checkpoints" / "random_contrastive.parsed.jsonl"
OUT = HERE / "assets" / "plot_050626_random_L23_contrastive_null.png"

BG = "#FAF9F6"
PROBE_COLOR = "#1f77b4"   # validated probe (blue, matches paper Fig 3a hue)
RANDOM_COLOR = "#888888"  # random direction


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def pooled_validated_probe() -> tuple[list[float], list[float], list[float], list[float]]:
    """Pool bb/hb/hh at each coefficient. Add c=0 = 0.5 (canonical-frame identity)."""
    data = json.loads(CIS_JSON.read_text())
    by_pair = data["conditions"]["contrastive_L23"]["by_pair_type"]
    by_coef: dict[float, tuple[int, int]] = {}
    for pair_type in ("bb", "hb", "hh"):
        for row in by_pair[pair_type]:
            c = row["coefficient"]
            k, n = row["k_chose_a"], row["n"]
            kk, nn = by_coef.get(c, (0, 0))
            by_coef[c] = (kk + k, nn + n)
    coefs = sorted(by_coef.keys())
    xs, ps, los, his = [], [], [], []
    for c in coefs:
        k, n = by_coef[c]
        p, lo, hi = _wilson(k, n)
        xs.append(c)
        ps.append(p)
        los.append(lo)
        his.append(hi)
    # canonical-frame identity at c=0
    xs.insert(len([x for x in xs if x < 0]), 0.0)
    ps.insert(len([x for x in coefs if x < 0]), 0.5)
    los.insert(len([x for x in coefs if x < 0]), 0.5)
    his.insert(len([x for x in coefs if x < 0]), 0.5)
    return xs, ps, los, his


def _effective_choice(r: dict) -> str:
    if r["choice_original"] in ("a", "b"):
        return r["choice_original"]
    if r.get("compliance") == "truncated" and r.get("task_completed") in ("a", "b"):
        return r["task_completed"]
    return "refusal"


def random_direction_curve() -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Canonical frame (matches build_steering_integrated.load_random_contrastive):
    each row contributes two points — at +mult we count chose_a, at -mult we
    count chose_b. c=0 is exactly 0.5 by construction.
    Refusal rate is per signed_multiplier in the raw rows.
    """
    counts: dict[float, list[int]] = {}
    refusal_counts: dict[float, list[int]] = {}
    with RANDOM_PARSED.open() as f:
        for line in f:
            r = json.loads(line)
            mult = round(float(r["signed_multiplier"]), 4)
            refusal_counts.setdefault(mult, [0, 0])
            refusal_counts[mult][1] += 1
            ch = _effective_choice(r)
            if ch not in ("a", "b"):
                refusal_counts[mult][0] += 1
                continue
            for c, target in [(round(+mult, 4), "a"), (round(-mult, 4), "b")]:
                counts.setdefault(c, [0, 0])
                counts[c][1] += 1
                counts[c][0] += int(ch == target)
    xs = sorted(counts.keys())
    ps, los, his, refusals = [], [], [], []
    for c in xs:
        k, n = counts[c]
        p, lo, hi = _wilson(k, n)
        ps.append(p)
        los.append(lo)
        his.append(hi)
        # refusal rate at |c|: average of mult and -mult raw rows
        m = abs(c)
        if m in refusal_counts:
            ref = refusal_counts[m][0] / refusal_counts[m][1]
        else:
            ref = 0.0
        refusals.append(ref)
    return xs, ps, los, his, refusals


def main() -> None:
    px, pp, pl, ph = pooled_validated_probe()
    rx, rp, rl, rh, refusals = random_direction_curve()

    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=BG)
    ax.set_facecolor(BG)

    # 0.5 chance line + c=0 vertical
    ax.axhline(0.5, color="#888", lw=0.8, ls=":", zorder=1)
    ax.axvline(0.0, color="#888", lw=0.5, ls="-", alpha=0.4, zorder=1)

    # Random direction
    rx_a = np.array(rx)
    rp_a = np.array(rp)
    yerr_r = np.vstack([rp_a - np.array(rl), np.array(rh) - rp_a])
    ax.errorbar(
        rx_a, rp_a, yerr=yerr_r,
        marker="s", color=RANDOM_COLOR, ls="--", lw=1.6, capsize=3,
        label="random direction (this experiment)",
        zorder=3,
    )

    # Validated probe (pooled bb+hb+hh)
    px_a = np.array(px)
    pp_a = np.array(pp)
    yerr_p = np.vstack([pp_a - np.array(pl), np.array(ph) - pp_a])
    ax.errorbar(
        px_a, pp_a, yerr=yerr_p,
        marker="o", color=PROBE_COLOR, lw=1.8, capsize=3,
        label="validated probe (Fig 3a, pooled across pair types)",
        zorder=4,
    )

    # Refusal rate as faint bars at the bottom
    bar_h = 0.04
    ax.bar(rx_a, np.array(refusals) * 0.4, width=0.004,
           color="#bbbbbb", alpha=0.55, zorder=2,
           label=f"refusal rate (random; ~{int(round(np.mean(refusals)*100))}%)")

    ax.set_xlabel(r"steering coefficient $c$ (× mean activation norm at L23)")
    ax.set_ylabel(r"$P(\mathrm{chose\ steered\ task}\mid\mathrm{responded})$")
    ax.set_title(
        "Only the preference direction moves choice — random L23 direction is null",
        fontsize=12, pad=10,
    )
    ax.set_xlim(-0.065, 0.065)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="center left", frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=BG)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
