"""Plot the random-direction null-control curve from the parsed checkpoint.

y = P(chose steered task | responded), x = c (signed multiplier from the
canonical contrastive frame). Each row contributes two points (task_a steered
at +mult → chose=='a' is success; task_b steered at -mult → chose=='b' is
success). Wilson 95% CI bars. A faint refusal-rate band is drawn at the
bottom of the panel so the dropped trials stay visible.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
PARSED = REPO / "experiments" / "random_direction_l23_quick" / "checkpoints" / "random_contrastive.parsed.jsonl"
ASSETS = REPO / "experiments" / "random_direction_l23_quick" / "assets"
TODAY = date.today().strftime("%m%d%y")
OUT = ASSETS / f"plot_{TODAY}_random_L23_contrastive_null.png"


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _effective_choice(r: dict) -> str:
    """Same rescue logic the paper plotter uses for max_new_tokens=64 truncation."""
    if r["choice_original"] in ("a", "b"):
        return r["choice_original"]
    if r.get("compliance") == "truncated" and r.get("task_completed") in ("a", "b"):
        return r["task_completed"]
    return "refusal"


def main() -> None:
    counts: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    refusals: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    with PARSED.open() as f:
        for line in f:
            r = json.loads(line)
            mult = r["signed_multiplier"]
            ch = _effective_choice(r)
            for c, success_letter in ((round(+mult, 4), "a"), (round(-mult, 4), "b")):
                refusals[c][1] += 1
                if ch in ("a", "b"):
                    counts[c][1] += 1
                    counts[c][0] += int(ch == success_letter)
                else:
                    refusals[c][0] += 1

    xs = sorted(counts.keys())
    ys, elo, ehi = [], [], []
    for c in xs:
        k, n = counts[c]
        p, lo, hi = _wilson(k, n)
        ys.append(p)
        elo.append(max(0.0, p - lo))
        ehi.append(max(0.0, hi - p))

    refusal_rates = [refusals[c][0] / refusals[c][1] for c in xs]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    fig.patch.set_facecolor("#FAF9F6")
    ax.set_facecolor("white")

    ax.errorbar(
        xs, ys, yerr=[elo, ehi],
        color="#374151", marker="s", markersize=5, linewidth=1.4,
        capsize=2.5, alpha=0.9, label="random direction",
    )

    ax.bar(
        xs, refusal_rates, width=0.004,
        bottom=0.0, color="#9CA3AF", alpha=0.45,
        label="refusal rate",
    )

    ax.set_xlim(-0.07, 0.07)
    ax.set_ylim(0, 1)
    ax.set_xticks([-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.axhline(0.5, color="#9CA3AF", linestyle=":", alpha=0.6, linewidth=0.8)
    ax.axvline(0, color="#9CA3AF", linestyle="-", alpha=0.4, linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors="#6B7280", labelsize=9)
    ax.set_ylabel("P(chose steered task | responded)", fontsize=10, color="#374151")
    ax.set_xlabel("c", fontsize=11, color="#374151", style="italic")
    ax.set_title(
        "Random L23 unit direction — contrastive null control",
        fontsize=12, color="#374151", weight="bold", pad=8,
    )
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=180, facecolor="#FAF9F6")
    plt.close(fig)
    print(f"saved {OUT}")

    swing = max(ys) - min(ys)
    n_total = sum(counts[c][1] for c in xs)
    n_refused = sum(refusals[c][0] for c in xs)
    print("\nSummary:")
    for c, y, lo, hi, r_rate in zip(xs, ys, elo, ehi, refusal_rates):
        k, n = counts[c]
        print(f"  c={c:+.3f}  P(steered)={y:.3f} [{y - lo:.3f}, {y + hi:.3f}]  k={k}/{n}  refused={r_rate:.2%}")
    print(f"  swing |max - min| = {swing:.3f}")
    print(f"  total responded: {n_total}, refused: {n_refused}")


if __name__ == "__main__":
    main()
