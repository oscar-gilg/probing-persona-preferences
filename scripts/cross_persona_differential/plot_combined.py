"""Combined cross-persona dose-response: contrastive + single-task panels per persona.

3x4 grid. Two personas per row; each persona occupies two columns (contrastive | single-task).
Signed x-axis (mirrors section 2.3 fig:steering-dose-response styling), even though
contrastive is symmetric. Empirical no-steering baselines at x=0 are plotted as
black dots from `{persona}_baseline.parsed.jsonl`.

- Contrastive panel: y = P(picked first-span task) — single line. Goes low at -c
  (first-span gets -probe in differential), high at +c.
- Single-task panel: two lines, P(picked the steered span's task) for each of
  unilateral_first / unilateral_second. Aggregate dashed line removes position bias.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


DIFF_CHECKPOINTS = Path("experiments/cross_persona_differential/checkpoints")
UNI_CHECKPOINTS = Path("experiments/cross_persona_unilateral/checkpoints")
ASSETS = Path("paper/figures/main")
PERSONAS = ["aura", "contrarian", "mathematician", "sadist", "slacker", "strategist"]


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _physical_task(span: str, ordering: int) -> str:
    if span == "first":
        return "a" if ordering == 0 else "b"
    return "b" if ordering == 0 else "a"


def _presented_coef(row: dict) -> float:
    return row["signed_multiplier"] * (1 if row["ordering"] == 0 else -1)


def _p_first_span_picked_diff(rows: list[dict], coef: float) -> tuple[float, float, int] | None:
    """For differential rows: P(picked first-span task) at signed presented coef."""
    hits = n = 0
    for r in rows:
        if abs(_presented_coef(r) - coef) > 1e-6:
            continue
        if r["choice_original"] not in ("a", "b"):
            continue
        target = _physical_task("first", r["ordering"])
        hits += int(r["choice_original"] == target)
        n += 1
    if n == 0:
        return None
    p = hits / n
    sem = math.sqrt(p * (1 - p) / n)
    return p, sem, n


def _p_first_span_uni(rows: list[dict], cond: str, coef: float) -> tuple[float, float, int] | None:
    """P(picked first-span task) for unilateral condition `cond` at signed presented coef.
    Always measures P(first-span), regardless of which span was steered — exposes
    suppression/amplification per side via the line direction (low→high or high→low)."""
    hits = n = 0
    for r in rows:
        if r["condition"] != cond:
            continue
        if abs(_presented_coef(r) - coef) > 1e-6:
            continue
        if r["choice_original"] not in ("a", "b"):
            continue
        target = _physical_task("first", r["ordering"])
        hits += int(r["choice_original"] == target)
        n += 1
    if n == 0:
        return None
    p = hits / n
    sem = math.sqrt(p * (1 - p) / n)
    return p, sem, n


def _baseline_p(rows: list[dict], span: str) -> float | None:
    hits = n = 0
    for r in rows:
        if r["choice_original"] not in ("a", "b"):
            continue
        target = _physical_task(span, r["ordering"])
        hits += int(r["choice_original"] == target)
        n += 1
    return (hits / n) if n else None


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    diff = {p: _load(DIFF_CHECKPOINTS / f"{p}.parsed.jsonl") for p in PERSONAS}
    uni = {p: _load(UNI_CHECKPOINTS / f"{p}.parsed.jsonl") for p in PERSONAS}
    base = {p: _load(UNI_CHECKPOINTS / f"{p}_baseline.parsed.jsonl") for p in PERSONAS}

    coefs = sorted({_presented_coef(r) for rows in uni.values() for r in rows})
    print(f"Applied signed coefs: {coefs}")
    for p in PERSONAS:
        print(f"  {p}: diff {len(diff[p])}, uni {len(uni[p])}, baseline {len(base[p])}")

    fig, axes = plt.subplots(3, 4, figsize=(13, 7.5), sharey=True, sharex=True)

    for row_idx in range(3):
        for col_idx_in_pair in range(2):
            persona = PERSONAS[row_idx * 2 + col_idx_in_pair]
            ax_c = axes[row_idx][col_idx_in_pair * 2]      # contrastive
            ax_s = axes[row_idx][col_idx_in_pair * 2 + 1]  # single-task

            first_base = _baseline_p(base[persona], "first")
            second_base = _baseline_p(base[persona], "second")

            # Contrastive panel
            xs_c, ys_c, es_c = [], [], []
            for c in coefs:
                stat = _p_first_span_picked_diff(diff[persona], c)
                if stat is None:
                    continue
                xs_c.append(c); ys_c.append(stat[0]); es_c.append(stat[1])
            if first_base is not None:
                xs_c.append(0.0); ys_c.append(first_base); es_c.append(0.0)
            order_c = sorted(zip(xs_c, ys_c, es_c))
            ax_c.errorbar([p[0] for p in order_c], [p[1] for p in order_c],
                          yerr=[p[2] for p in order_c],
                          marker="o", color="C2", linewidth=1.7, markersize=5,
                          capsize=2, label="contrastive")
            if first_base is not None:
                ax_c.plot(0, first_base, "o", color="black", markersize=4, zorder=5)

            # Single-task panel: both lines plot P(picked first-span task) vs signed coef.
            # first-span steered: amplifies first-span at +c → low→high.
            # second-span steered: amplifies second-span at +c → first-span suppressed → high→low.
            # The two lines are mirror images about y=0.5 modulo position bias and amp/suppr asymmetry.
            for cond, color, label in [("unilateral_first", "C0", "first-span steered"),
                                        ("unilateral_second", "C1", "second-span steered")]:
                xs, ys, es = [], [], []
                for c in coefs:
                    stat = _p_first_span_uni(uni[persona], cond, c)
                    if stat is None:
                        continue
                    xs.append(c); ys.append(stat[0]); es.append(stat[1])
                if first_base is not None:
                    xs.append(0.0); ys.append(first_base); es.append(0.0)
                order = sorted(zip(xs, ys, es))
                ax_s.errorbar([p[0] for p in order], [p[1] for p in order],
                              yerr=[p[2] for p in order],
                              marker="o", color=color, linewidth=1.5, markersize=4,
                              capsize=2, label=label)
            if first_base is not None:
                ax_s.plot(0, first_base, "o", color="black", markersize=4, zorder=5)

            for ax, sub in [(ax_c, "contrastive"), (ax_s, "single-task")]:
                ax.axvline(0, color="gray", linestyle="-", alpha=0.2, linewidth=0.5)
                ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.5)
                ax.set_title(f"{persona} · {sub}", fontsize=9)
                ax.set_ylim(0, 1)
                ax.set_xlim(-0.06, 0.06)
                ax.grid(True, alpha=0.3)

            if row_idx == 2:
                ax_c.set_xlabel("signed coef (× mean_norm(L25))", fontsize=8)
                ax_s.set_xlabel("signed coef (× mean_norm(L25))", fontsize=8)
            if col_idx_in_pair == 0:
                ax_c.set_ylabel("P(picked first-span task)", fontsize=8)
            if row_idx == 0 and col_idx_in_pair == 0:
                ax_c.legend(loc="lower right", fontsize=7)
                ax_s.legend(loc="lower right", fontsize=7)

    fig.suptitle(
        "Per-persona probe steering at L25 — contrastive (left) and single-task (right) per persona",
        fontsize=11,
    )
    fig.tight_layout()
    stamp = datetime.now().strftime("%m%d%y")
    out = ASSETS / f"plot_{stamp}_cross_persona_perprobe_steering.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out.resolve()}")


if __name__ == "__main__":
    main()
