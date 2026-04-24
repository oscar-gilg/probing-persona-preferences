"""Cross-persona unilateral steering dose-response.

One panel per persona (2x3 grid), two lines per panel: first-span and
second-span unilateral. x = actual signed coefficient applied to the steered
task (un-mixing the ordering flip); y = P(model picked that span's task).

Schema of parsed checkpoint row:
  condition: "unilateral_first" | "unilateral_second"
  layer: int (always 25 here)
  signed_multiplier: float (-0.05, -0.03, 0.03, 0.05)
  ordering: int (0 or 1 — flips which physical task is in each span)
  choice_original: "a" | "b" | refusal tag

For (condition, mult, ordering):
  - span = "first" if condition == "unilateral_first" else "second"
  - physical task in span: (span=first, ord=0) → a; (first, 1) → b; (second, 0) → b; (second, 1) → a
  - applied_coef on that task = signed_multiplier × (+1 if ordering==0 else -1)
  - target = physical task in span
  - y = P(choice_original == target)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


CHECKPOINTS = Path("experiments/cross_persona_unilateral/checkpoints")
ASSETS = Path("experiments/cross_persona_unilateral/assets")
PERSONAS = ["aura", "contrarian", "mathematician", "sadist", "slacker", "strategist"]


def _load(persona: str) -> list[dict]:
    path = CHECKPOINTS / f"{persona}.parsed.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _physical_task(span: str, ordering: int) -> str:
    if span == "first":
        return "a" if ordering == 0 else "b"
    return "b" if ordering == 0 else "a"


def _p_steered(rows: list[dict], cond: str, coef: float) -> float | None:
    span = "first" if cond == "unilateral_first" else "second"
    hits = n = 0
    for r in rows:
        if r["condition"] != cond:
            continue
        applied = r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1)
        if abs(applied - coef) > 1e-6:
            continue
        target = _physical_task(span, r["ordering"])
        hits += int(r["choice_original"] == target)
        n += 1
    return (hits / n) if n else None


def _refusal_rate(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    refusals = sum(1 for r in rows if r["choice_original"] not in ("a", "b"))
    return refusals / len(rows)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    data = {p: _load(p) for p in PERSONAS}

    for p in PERSONAS:
        n = len(data[p])
        refusal = _refusal_rate(data[p])
        print(f"  {p}: {n} rows, refusal={refusal:.1%}")

    missing = [p for p, rows in data.items() if not rows]
    if missing:
        print(f"\nMissing checkpoints for: {missing}. Plot will skip empty panels.")

    # Collect the coef axis from whoever has data (all configs share the same multiplier set)
    coefs = sorted({
        r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1)
        for rows in data.values() for r in rows
    })
    print(f"Applied coefs: {coefs}")

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharey=True, sharex=True)
    axes = axes.flatten()

    for idx, persona in enumerate(PERSONAS):
        ax = axes[idx]
        rows = data[persona]
        if not rows:
            ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes, ha="center", va="center", alpha=0.6)
            ax.set_title(persona, fontsize=10)
            continue

        for cond, color, label in [
            ("unilateral_first", "C0", "first-span"),
            ("unilateral_second", "C1", "second-span"),
        ]:
            xs, ys = [], []
            for coef in coefs:
                y = _p_steered(rows, cond, coef)
                if y is not None:
                    xs.append(coef)
                    ys.append(y)
            ax.plot(xs, ys, "o-", color=color, label=label, markersize=5, linewidth=1.5)

        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.axvline(0, color="gray", linestyle="-", alpha=0.2, linewidth=0.5)
        ax.set_title(persona, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.06, 0.06)
        ax.grid(True, alpha=0.3)
        if idx // 3 == 1:
            ax.set_xlabel("coef on steered task (× mean_norm(L25))", fontsize=8)
        if idx % 3 == 0:
            ax.set_ylabel("P(picked steered task)", fontsize=8)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Cross-persona unilateral steering at L25 — P(picked steered task) vs signed coef", fontsize=11)
    fig.tight_layout()
    stamp = datetime.now().strftime("%m%d%y")
    out = ASSETS / f"plot_{stamp}_cross_persona_unilateral_dose_response.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out.resolve()}")


if __name__ == "__main__":
    main()
