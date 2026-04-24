"""Plot cross-persona steering validation results.

Reads experiments/cross_persona_steering/aggregated.json and writes:
  - assets/plot_<date>_cross_persona_steered_dose_response.png (4-panel P(steered) vs |c|)

Also registers claims for P(steered task chosen) at |c|=0.05 and |c|=0.03 for
each persona (probe direction) and at |c|=0.03 for the random-direction control.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from corroborate import ClaimSet


EXP_DIR = Path("experiments/cross_persona_steering")
ASSETS = EXP_DIR / "assets"
PERSONAS = ["sadist", "villain", "aesthete", "stem_obsessive"]
MAIN_CONDITION = "differential_L25_probeL32"
CONTROL_CONDITION = "differential_L25_random"

# Repo-relative path read by the producer (shared across all registered claims).
AGG_PATH = "experiments/cross_persona_steering/aggregated.json"

PERSONA_LABEL = {
    "sadist": "sadist",
    "villain": "villain",
    "aesthete": "aesthete",
    "stem_obsessive": "stem-obsessive",
}


def date_tag() -> str:
    return dt.date.today().strftime("%m%d%y")


def cells_to_curve(cells: dict, condition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (|coef|, P(steered), SEM) arrays sorted by |coef| for a condition."""
    filtered = [c for c in cells.values() if c["condition"] == condition]
    filtered.sort(key=lambda c: c["abs_coefficient"])
    abs_c = np.array([c["abs_coefficient"] for c in filtered])
    means = np.array([c["mean_steered_chosen"] for c in filtered])
    sems = np.array([c["sem"] for c in filtered])
    # Prepend the symmetry anchor at |c|=0: by definition P(steered)=0.5 (neutral).
    abs_c = np.concatenate([[0.0], abs_c])
    means = np.concatenate([[0.5], means])
    sems = np.concatenate([[0.0], sems])
    return abs_c, means, sems


def _collect_persona_values(persona: str, cells: dict) -> dict[str, float]:
    """Return {'probe_at_0.05','probe_at_0.03','random_at_0.03'} → rounded P(steered)."""
    out: dict[str, float] = {}
    for cell in cells.values():
        condition = cell["condition"]
        abs_c = float(cell["abs_coefficient"])
        value = round(float(cell["mean_steered_chosen"]), 3)
        if condition == MAIN_CONDITION and abs_c == 0.05:
            out["probe_at_0.05"] = value
        elif condition == MAIN_CONDITION and abs_c == 0.03:
            out["probe_at_0.03"] = value
        elif condition == CONTROL_CONDITION and abs_c == 0.03:
            out["random_at_0.03"] = value
    return out


def _register_cross_persona_claims(claims: ClaimSet, agg: dict) -> None:
    """Collapse per-(persona, |c|, direction) scalars into two dict-of-dicts claims."""
    collected: dict[str, dict[str, float]] = {}
    for persona in PERSONAS:
        data = agg["personas"].get(persona)
        if not data:
            continue
        collected[persona] = _collect_persona_values(persona, data["validation_cells"])

    # |c| = 0.05: probe direction only. Row dict keyed by "<persona> probe direction"
    # so the emitted slugs end in ...SadistProbeDirection, ...AestheteProbeDirection, etc.
    value_005: dict[str, float] = {}
    for persona in PERSONAS:
        if persona not in collected or "probe_at_0.05" not in collected[persona]:
            continue
        label = PERSONA_LABEL[persona]
        value_005[f"{label} probe direction"] = collected[persona]["probe_at_0.05"]
    claims.register(
        name="Cross persona steering P chosen at C 0.05",
        value=value_005,
        statement=(
            "Under each persona system prompt (sadist, villain, aesthete, stem-obsessive), "
            "contrastive steering along the default-persona Gemma-3-27B probe (ridge, L32) "
            "applied at layer 25 at |c| = 0.05 (fraction of the L25 mean activation norm) "
            "makes Gemma-3-27B pick the positively-steered task at the reported "
            "P(steered task chosen)."
        ),
        used_in=["fig:cross-persona-steering", "sec:shared-steering"],
        data_paths=[AGG_PATH],
        derivation=(
            "For each persona p, read "
            "`personas.p.validation_cells[differential_L25_probeL32|0.05].mean_steered_chosen`; "
            "round to 3dp."
        ),
    )

    # |c| = 0.03: both probe and random directions. 2-D table keyed by persona, then
    # by direction, preserving slugs like ...SadistProbeDirection / ...SadistRandomDirection.
    value_003: dict[str, dict[str, float]] = {}
    for persona in PERSONAS:
        if persona not in collected:
            continue
        label = PERSONA_LABEL[persona]
        row: dict[str, float] = {}
        if "probe_at_0.03" in collected[persona]:
            row["probe direction"] = collected[persona]["probe_at_0.03"]
        if "random_at_0.03" in collected[persona]:
            row["random direction"] = collected[persona]["random_at_0.03"]
        if row:
            value_003[label] = row
    claims.register(
        name="Cross persona steering P chosen at C 0.03",
        value=value_003,
        statement=(
            "Under each persona system prompt (sadist, villain, aesthete, stem-obsessive), "
            "P(steered task chosen) at |c| = 0.03 (fraction of the L25 mean activation norm) "
            "for two directions: `probe direction` = contrastive steering along the "
            "default-persona Gemma-3-27B ridge (L32) probe applied at layer 25; "
            "`random direction` = differential steering along a random direction (control) "
            "at layer 25."
        ),
        used_in=["fig:cross-persona-steering", "sec:shared-steering"],
        data_paths=[AGG_PATH],
        derivation=(
            "For each persona p, read "
            "`personas.p.validation_cells[differential_L25_probeL32|0.03].mean_steered_chosen` "
            "(probe direction) and "
            "`personas.p.validation_cells[differential_L25_random|0.03].mean_steered_chosen` "
            "(random direction); round to 3dp."
        ),
    )


def plot_dose_response(agg: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharey=True, sharex=True)
    axes = axes.ravel()
    for i, persona in enumerate(PERSONAS):
        ax = axes[i]
        data = agg["personas"].get(persona)
        if not data:
            ax.text(0.5, 0.5, f"(no data for {persona})", ha="center", va="center")
            ax.set_title(persona)
            continue
        cells = data["validation_cells"]
        x_main, y_main, e_main = cells_to_curve(cells, MAIN_CONDITION)
        x_ctrl, y_ctrl, e_ctrl = cells_to_curve(cells, CONTROL_CONDITION)
        ax.errorbar(x_main, y_main, yerr=e_main, marker="o", color="#1f77b4",
                    linewidth=2, markersize=7, label="probe direction (ridge_L32)", capsize=3)
        ax.errorbar(x_ctrl, y_ctrl, yerr=e_ctrl, marker="s", color="#888888",
                    linewidth=1.5, markersize=6, linestyle="--",
                    label="random direction (control)", capsize=3)
        ax.axhline(0.5, color="#cccccc", linewidth=0.8, zorder=0)
        ax.set_title(persona, fontsize=12, fontweight="bold")
        ax.set_ylim(0.4, 1.0)
        ax.set_xlim(-0.003, 0.055)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="lower right", fontsize=9)
    for i in [2, 3]:
        axes[i].set_xlabel("|steering coefficient|  (fraction of L25 mean norm)")
    for i in [0, 2]:
        axes[i].set_ylabel("P(steered task was chosen)")
    fig.suptitle(
        "Differential steering under persona system prompts: P(steered task chosen) vs |coefficient|",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with open(EXP_DIR / "aggregated.json") as f:
        agg = json.load(f)
    ASSETS.mkdir(parents=True, exist_ok=True)
    tag = date_tag()
    path = ASSETS / f"plot_{tag}_cross_persona_steered_dose_response.png"
    claims = ClaimSet(source="scripts/cross_persona_steering/plot_results.py")
    plot_dose_response(agg, path)
    _register_cross_persona_claims(claims, agg)
    claims.save("paper/claims/cross_persona_steering.json")
    print(f"wrote {path}")
    print(f"registered {len(claims.claims)} claims to paper/claims/cross_persona_steering.json")


if __name__ == "__main__":
    main()
