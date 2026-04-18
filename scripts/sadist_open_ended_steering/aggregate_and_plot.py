"""Aggregate judged responses and plot sadism vs default-assistant curves.

Reads judged_{default,sadist}.jsonl and writes:
  - aggregated.json
  - assets/plot_<date>_sadism_vs_default_by_coef.png  (2-panel, one per persona)
"""

from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXP_DIR = Path("experiments/sadist_open_ended_steering")
ASSETS = EXP_DIR / "assets"
PERSONAS = ["default", "sadist"]


def date_tag() -> str:
    return dt.date.today().strftime("%m%d%y")


def load_judged(persona: str) -> list[dict]:
    path = EXP_DIR / f"judged_{persona}.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return [r for r in rows if "judge_error" not in r]


def aggregate_persona(rows: list[dict]) -> dict:
    by_coef = defaultdict(list)   # mult -> list of (sadism, default_assistant)
    for r in rows:
        by_coef[r["multiplier"]].append((r["sadism_score"], r["default_assistant_score"]))
    cells = {}
    for mult in sorted(by_coef):
        vals = np.array(by_coef[mult])
        sadism = vals[:, 0]
        default_a = vals[:, 1]
        cells[str(mult)] = {
            "multiplier": mult,
            "n": len(vals),
            "sadism_mean": float(sadism.mean()),
            "sadism_sem": float(sadism.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            "default_mean": float(default_a.mean()),
            "default_sem": float(default_a.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
        }
    return cells


def aggregate_all() -> dict:
    out = {"personas": {}}
    for p in PERSONAS:
        rows = load_judged(p)
        if not rows:
            print(f"[skip] {p}: no judged rows")
            continue
        cells = aggregate_persona(rows)
        out["personas"][p] = {"cells": cells, "n_rows": len(rows)}
        print(f"[ok]   {p}: {len(rows)} rows across {len(cells)} coef cells")
    with open(EXP_DIR / "aggregated.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def plot_dose_response(agg: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, persona in zip(axes, PERSONAS):
        data = agg["personas"].get(persona)
        if not data:
            ax.text(0.5, 0.5, f"(no data for {persona})", ha="center", va="center")
            ax.set_title(persona)
            continue
        cells = sorted(data["cells"].values(), key=lambda c: c["multiplier"])
        x = np.array([c["multiplier"] for c in cells])
        sadism = np.array([c["sadism_mean"] for c in cells])
        sadism_sem = np.array([c["sadism_sem"] for c in cells])
        default_a = np.array([c["default_mean"] for c in cells])
        default_sem = np.array([c["default_sem"] for c in cells])
        ax.errorbar(x, sadism, yerr=sadism_sem, marker="o", color="#d62728",
                    linewidth=2, markersize=7, label="sadism", capsize=3)
        ax.errorbar(x, default_a, yerr=default_sem, marker="s", color="#1f77b4",
                    linewidth=2, markersize=7, linestyle="--",
                    label="default-assistant", capsize=3)
        ax.axvline(0, color="#cccccc", linewidth=0.8, zorder=0)
        ax.set_title(f"{persona} persona", fontsize=12, fontweight="bold")
        ax.set_xlabel("steering coefficient  (fraction of L25 mean norm)")
        ax.set_ylim(1, 5)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    axes[0].set_ylabel("judge score (1 = none, 5 = textbook)")
    fig.suptitle(
        "Open-ended steering: sadism and default-assistant scales under each persona",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    agg = aggregate_all()
    if not agg["personas"]:
        print("nothing to plot")
        return
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / f"plot_{date_tag()}_sadism_vs_default_by_coef.png"
    plot_dose_response(agg, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
