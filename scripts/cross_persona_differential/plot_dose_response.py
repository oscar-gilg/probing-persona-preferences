"""Cross-persona differential steering dose-response.

One panel per persona (2x3). x = |signed coef| (fraction of L25 mean_norm).
y = P(steered task chosen), folded across +/- coef:
  - c > 0: steered task = original Task A (the +probe side of the differential).
  - c < 0: steered task = original Task B.

Anchor at (0, 0.5). The baseline `{persona}_baseline.parsed.jsonl` is no-steering
with no notion of "steered task", so we do not use it here (shown in unilateral).
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


CHECKPOINTS = Path("experiments/cross_persona_differential/checkpoints")
ASSETS = Path("experiments/cross_persona_differential/assets")
PERSONAS = ["aura", "contrarian", "mathematician", "sadist", "slacker", "strategist"]


def _load_parsed(persona: str) -> list[dict]:
    path = CHECKPOINTS / f"{persona}.parsed.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _steered_task(signed_c: float) -> str:
    return "a" if signed_c > 0 else "b"


def _p_steered_by_absc(rows: list[dict]) -> dict[float, tuple[float, float, int]]:
    """Return {|c|: (p, sem, n)} folding +/- signed_multiplier."""
    buckets: dict[float, list[int]] = {}
    for r in rows:
        c = r["signed_multiplier"]
        if c == 0 or r["choice_original"] not in ("a", "b"):
            continue
        steered = _steered_task(c)
        hit = 1 if r["choice_original"] == steered else 0
        key = round(abs(c), 6)
        buckets.setdefault(key, []).append(hit)
    out = {}
    for k, vs in buckets.items():
        n = len(vs)
        p = sum(vs) / n
        sem = math.sqrt(p * (1 - p) / n) if n > 0 else 0.0
        out[k] = (p, sem, n)
    return out


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    stats = {p: _p_steered_by_absc(_load_parsed(p)) for p in PERSONAS}
    for p in PERSONAS:
        print(f"  {p}: {stats[p]}")

    all_coefs = sorted({c for s in stats.values() for c in s.keys()})
    print(f"|coefs|: {all_coefs}")

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharey=True, sharex=True)
    axes = axes.flatten()

    for idx, persona in enumerate(PERSONAS):
        ax = axes[idx]
        s = stats[persona]
        if not s:
            ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes,
                    ha="center", va="center", alpha=0.6)
            ax.set_title(persona, fontsize=10)
            continue
        xs = [0.0] + sorted(s.keys())
        ys = [0.5] + [s[k][0] for k in sorted(s.keys())]
        es = [0.0] + [s[k][1] for k in sorted(s.keys())]
        ax.errorbar(xs, ys, yerr=es, marker="o", color="C0",
                    linewidth=1.8, markersize=6, capsize=3)
        ax.axhline(0.5, color="gray", linestyle="-", alpha=0.3, linewidth=0.6)
        ax.set_title(persona, fontsize=10)
        ax.set_ylim(0.4, 1.0)
        ax.set_xlim(-0.003, 0.06)
        ax.grid(True, alpha=0.3)
        if idx // 3 == 1:
            ax.set_xlabel("|coef| (× mean_norm(L25))", fontsize=8)
        if idx % 3 == 0:
            ax.set_ylabel("P(steered task chosen)", fontsize=8)

    fig.suptitle(
        "Cross-persona differential steering at L25 — P(steered task chosen) vs |coef|",
        fontsize=11,
    )
    fig.tight_layout()
    stamp = dt.date.today().strftime("%m%d%y")
    out = ASSETS / f"plot_{stamp}_cross_persona_differential_dose_response.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out.resolve()}")


if __name__ == "__main__":
    main()
