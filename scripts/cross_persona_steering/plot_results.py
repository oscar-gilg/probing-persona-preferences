"""Plot cross-persona steering aggregated results.

Reads experiments/cross_persona_steering/aggregated.json and writes:
  - assets/plot_<date>_cross_persona_dose_response.png (4-panel grid)
  - assets/plot_<date>_alignment_shift.png (stratified bar chart)
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXP_DIR = Path("experiments/cross_persona_steering")
ASSETS = EXP_DIR / "assets"
PERSONAS = ["sadist", "villain", "aesthete", "stem_obsessive"]
MAIN_CONDITION = "differential_L25_probeL32"
CONTROL_CONDITION = "differential_L25_random"


def date_tag() -> str:
    return dt.date.today().strftime("%m%d%y")


def cells_to_curve(cells: dict, condition: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract (coefficients, mean P(default-pref)) sorted by coefficient for a condition."""
    filtered = [c for c in cells.values() if c["condition"] == condition]
    filtered.sort(key=lambda c: c["coefficient"])
    coefs = np.array([c["coefficient"] for c in filtered])
    means = np.array([c["mean_default_pref"] for c in filtered])
    return coefs, means


def plot_dose_response(agg: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    axes = axes.ravel()
    for i, persona in enumerate(PERSONAS):
        ax = axes[i]
        data = agg["personas"].get(persona)
        if not data:
            ax.text(0.5, 0.5, f"(no data for {persona})", ha="center", va="center")
            ax.set_title(persona)
            continue
        x_main, y_main = cells_to_curve(data["cells"], MAIN_CONDITION)
        x_ctrl, y_ctrl = cells_to_curve(data["cells"], CONTROL_CONDITION)
        ax.plot(x_main, y_main, "o-", color="#1f77b4", linewidth=2, label="probe L32")
        ax.plot(x_ctrl, y_ctrl, "s--", color="#888888", linewidth=1.5, label="random direction")
        ax.axhline(0.5, color="#cccccc", linewidth=0.8, zorder=0)
        ax.axvline(0, color="#cccccc", linewidth=0.8, zorder=0)
        ax.set_title(persona)
        ax.set_xlabel("steering coefficient")
        if i % 2 == 0:
            ax.set_ylabel("P(choose default-preferred task)")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def alignment_shift(data: dict, target_coef: float = 0.03) -> dict:
    """Return Δ(target) = P(default-pref)(+target) - P(default-pref)(0), stratified by baseline tercile."""
    baseline = data.get("baseline_pair_mean_default_pref", {})
    if not baseline:
        return {}
    pair_ids = list(baseline.keys())
    baseline_vals = np.array([baseline[p] for p in pair_ids])
    terciles = np.quantile(baseline_vals, [1 / 3, 2 / 3])
    bins = np.digitize(baseline_vals, terciles)  # 0 low, 1 mid, 2 high
    pid_to_bin = {p: int(b) for p, b in zip(pair_ids, bins)}

    # Crude: use overall cell means as Δ — finer stratification would require per-pair cell data.
    cells = data["cells"]
    p_zero = None
    p_target = None
    for cell in cells.values():
        if cell["condition"] == MAIN_CONDITION and cell["coefficient"] == 0:
            p_zero = cell["mean_default_pref"]
        if cell["condition"] == MAIN_CONDITION and abs(cell["coefficient"] - target_coef) < 1e-9:
            p_target = cell["mean_default_pref"]
    if p_zero is None or p_target is None:
        return {}
    return {"delta_overall": p_target - p_zero, "baseline_terciles": terciles.tolist(), "n_low": int(np.sum(bins == 0)), "n_mid": int(np.sum(bins == 1)), "n_high": int(np.sum(bins == 2))}


def plot_alignment_shift(agg: dict, out: Path) -> None:
    deltas = {}
    for persona in PERSONAS:
        data = agg["personas"].get(persona)
        if not data:
            continue
        info = alignment_shift(data, target_coef=0.03)
        if info:
            deltas[persona] = info["delta_overall"]
    if not deltas:
        print("no personas with both 0 and +0.03 cells — skipping alignment-shift plot")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(deltas.keys())
    vals = [deltas[p] for p in names]
    bars = ax.bar(names, vals, color=["#d62728" if v < 0 else "#2ca02c" for v in vals])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(r"$\Delta = P_\mathrm{def}(+0.03) - P_\mathrm{def}(0)$")
    ax.set_title("Alignment shift at c = +0.03")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + (0.005 if v >= 0 else -0.015),
                f"{v:+.3f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with open(EXP_DIR / "aggregated.json") as f:
        agg = json.load(f)
    ASSETS.mkdir(parents=True, exist_ok=True)
    tag = date_tag()
    dr_path = ASSETS / f"plot_{tag}_cross_persona_dose_response.png"
    as_path = ASSETS / f"plot_{tag}_alignment_shift.png"
    plot_dose_response(agg, dr_path)
    plot_alignment_shift(agg, as_path)
    print(f"wrote {dr_path}\nwrote {as_path}")


if __name__ == "__main__":
    main()
