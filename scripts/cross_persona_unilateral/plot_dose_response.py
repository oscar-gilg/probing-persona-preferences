"""Cross-persona unilateral steering dose-response.

One panel per persona (2x3 grid). Two colored lines per panel: first-span (blue)
and second-span (orange) unilateral steering. Aggregate dashed gray line at
each coef = (first + second) / 2. Black dots at x=0 = empirical no-steering
baselines from `{persona}_baseline.parsed.jsonl` (run via API).

Mirrors the layer_sweep cross-persona_unilateral plot format so the two are
visually comparable.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


CHECKPOINTS = Path("experiments/cross_persona_unilateral/checkpoints")
ASSETS = Path("experiments/cross_persona_unilateral/assets")
PERSONAS = ["aura", "contrarian", "mathematician", "sadist", "slacker", "strategist"]


def _load(name: str) -> list[dict]:
    path = CHECKPOINTS / f"{name}.parsed.jsonl"
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


def _baseline_p(rows: list[dict], span: str) -> float | None:
    """P(picked the task in <span>) under no steering. Uses choice_original in the original
    task_a/task_b frame: first-span task == task_a if ordering==0 else task_b."""
    hits = n = 0
    for r in rows:
        target = _physical_task(span, r["ordering"])
        if r["choice_original"] not in ("a", "b"):
            continue  # skip refusals
        hits += int(r["choice_original"] == target)
        n += 1
    return (hits / n) if n else None


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    steer = {p: _load(p) for p in PERSONAS}
    base = {p: _load(f"{p}_baseline") for p in PERSONAS}

    for p in PERSONAS:
        ns = len(steer[p])
        nb = len(base[p])
        print(f"  {p}: steering {ns} rows, baseline {nb} rows")

    coefs = sorted({
        r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1)
        for rows in steer.values() for r in rows
    })
    print(f"Applied coefs: {coefs}")

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharey=True, sharex=True)
    axes = axes.flatten()

    for idx, persona in enumerate(PERSONAS):
        ax = axes[idx]
        rows = steer[persona]
        base_rows = base[persona]
        if not rows:
            ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes,
                    ha="center", va="center", alpha=0.6)
            ax.set_title(persona, fontsize=10)
            continue

        first_base = _baseline_p(base_rows, "first") if base_rows else None
        second_base = _baseline_p(base_rows, "second") if base_rows else None
        agg_base = (first_base + second_base) / 2 if (first_base is not None and second_base is not None) else None

        # Per-span lines: include x=0 baseline if available, then sort
        xs_f = list(coefs) + ([0.0] if first_base is not None else [])
        ys_f = [_p_steered(rows, "unilateral_first", c) for c in coefs] + ([first_base] if first_base is not None else [])
        xs_s = list(coefs) + ([0.0] if second_base is not None else [])
        ys_s = [_p_steered(rows, "unilateral_second", c) for c in coefs] + ([second_base] if second_base is not None else [])

        ordered_f = sorted((x, y) for x, y in zip(xs_f, ys_f) if y is not None)
        ordered_s = sorted((x, y) for x, y in zip(xs_s, ys_s) if y is not None)
        ax.plot([p[0] for p in ordered_f], [p[1] for p in ordered_f],
                "o-", color="C0", label="first-span", markersize=5, linewidth=1.5)
        ax.plot([p[0] for p in ordered_s], [p[1] for p in ordered_s],
                "o-", color="C1", label="second-span", markersize=5, linewidth=1.5)

        # Aggregate: (first + second) / 2 at each coef (+ baseline at x=0)
        agg_xs, agg_ys = [], []
        for c in coefs:
            yf = _p_steered(rows, "unilateral_first", c)
            ys = _p_steered(rows, "unilateral_second", c)
            if yf is not None and ys is not None:
                agg_xs.append(c)
                agg_ys.append((yf + ys) / 2)
        if agg_base is not None:
            agg_xs.append(0.0)
            agg_ys.append(agg_base)
        ordered_agg = sorted(zip(agg_xs, agg_ys))
        ax.plot([p[0] for p in ordered_agg], [p[1] for p in ordered_agg],
                "--", color="gray", label="aggregate", linewidth=1.5)

        # Black dots at x=0 for baselines
        if first_base is not None:
            ax.plot(0, first_base, "o", color="black", markersize=4)
        if second_base is not None:
            ax.plot(0, second_base, "o", color="black", markersize=4)

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

    fig.suptitle(
        "Cross-persona unilateral steering at L25 — P(picked steered task) vs signed coef",
        fontsize=11,
    )
    fig.tight_layout()
    stamp = datetime.now().strftime("%m%d%y")
    out = ASSETS / f"plot_{stamp}_cross_persona_unilateral_dose_response.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out.resolve()}")


if __name__ == "__main__":
    main()
