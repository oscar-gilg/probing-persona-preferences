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

from corroborate import ClaimSet


EXP_DIR = Path("experiments/sadist_open_ended_steering")
ASSETS = EXP_DIR / "assets"
PERSONAS = ["default", "sadist"]
MAX_ABS_MULT = 0.05  # Drop incoherent coefficients (|c| > 0.05 fails the coherence judge).

FIG_LABEL = "fig:cross-persona-openended"
SEC_LABELS = ["sec:shared-openended", "sec:open-ended-safety"]


def _judged_path(persona: str) -> str:
    return f"experiments/sadist_open_ended_steering/judged_{persona}.jsonl"


def _open_ended_path(persona: str) -> str:
    return f"experiments/sadist_open_ended_steering/judged_open_ended_{persona}.jsonl"


def _compliance_path(persona: str) -> str:
    return f"experiments/sadist_open_ended_steering/compliance_{persona}.jsonl"


def date_tag() -> str:
    return dt.date.today().strftime("%m%d%y")


def load_judged(persona: str, filename_stem: str = "judged") -> list[dict]:
    path = EXP_DIR / f"{filename_stem}_{persona}.jsonl"
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


def _coef_col_key(mult: float) -> str:
    """Column key used for per-coefficient table cells. Matches cited macros.

    `0.03` -> `at_c_pos_0_03`, `-0.05` -> `at_c_neg_0_05`, `0.0` -> `at_c_pos_0_00`.
    """
    sign_word = "pos" if mult >= 0 else "neg"
    mag = f"{abs(mult):.2f}"  # e.g. "0.05"
    return f"at_c_{sign_word}_{mag.replace('.', '_')}"


def register_figure_claims(claims: ClaimSet, agg: dict) -> None:
    """Register every bar height drawn on the figure (both panels, both scales).

    Four row-dict claims: {sadism, default-assistant} x {default persona, sadist persona}.
    Each claim holds one row keyed by coefficient (e.g. at_c_pos_0_03), yielding one
    \\newcommand per (metric, persona, coefficient) via nested camelCase slugs.
    """
    for persona in PERSONAS:
        data = agg["personas"].get(persona)
        if not data:
            continue
        cells = sorted(
            (c for c in data["cells"].values() if abs(c["multiplier"]) <= MAX_ABS_MULT),
            key=lambda c: c["multiplier"],
        )
        sadism_row: dict[str, float] = {}
        default_row: dict[str, float] = {}
        mults_prose: list[str] = []
        ns: list[int] = []
        for cell in cells:
            mult = cell["multiplier"]
            key = _coef_col_key(mult)
            sadism_row[key] = round(cell["sadism_mean"], 2)
            default_row[key] = round(cell["default_mean"], 2)
            mults_prose.append(f"c={'+' if mult >= 0 else '-'}{abs(mult):.2f}")
            ns.append(cell["n"])
        mults_list = ", ".join(mults_prose)
        n_min, n_max = min(ns), max(ns)
        n_str = f"n={n_min}" if n_min == n_max else f"n in [{n_min},{n_max}]"
        claims.register(
            name=f"Sadism Likert under {persona} safety-prompts",
            value=sadism_row,
            statement=(
                f"Mean blind Likert sadism scores on tiered safety/agentic prompts "
                f"under the {persona} persona, per steering coefficient "
                f"({mults_list}; fraction of L25 mean activation norm; {n_str} per cell)."
            ),
            used_in=[FIG_LABEL],
            data_paths=[_judged_path(persona)],
            derivation=(
                "Aggregate rows by `multiplier`; take mean of `sadism_score` per cell; "
                "round to 2dp. Column keys encode coefficient sign and magnitude."
            ),
        )
        claims.register(
            name=f"Default-assistant Likert under {persona} safety-prompts",
            value=default_row,
            statement=(
                f"Mean blind Likert default-assistant scores on tiered safety/agentic "
                f"prompts under the {persona} persona, per steering coefficient "
                f"({mults_list}; fraction of L25 mean activation norm; {n_str} per cell)."
            ),
            used_in=[FIG_LABEL],
            data_paths=[_judged_path(persona)],
            derivation=(
                "Aggregate rows by `multiplier`; take mean of `default_assistant_score` per "
                "cell; round to 2dp. Column keys encode coefficient sign and magnitude."
            ),
        )


def open_ended_cells(persona: str) -> dict:
    """Aggregate the self-reflection (OE_*) open-ended prompts, per coefficient."""
    rows = load_judged(persona, filename_stem="judged_open_ended")
    return aggregate_persona(rows) if rows else {}


def compliance_rate(persona: str, tier: str, multiplier: float) -> tuple[float, int]:
    """Fraction of rows labelled 'complied' in the compliance sidecar for one cell."""
    path = EXP_DIR / f"compliance_{persona}.jsonl"
    complied = 0
    total = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "compliance_error" in r:
                continue
            if r["tier"] != tier or r["multiplier"] != multiplier:
                continue
            total += 1
            if r["compliance"] == "complied":
                complied += 1
    return (complied / total if total else 0.0, total)


def register_prose_claims(claims: ClaimSet) -> None:
    """Register prose numbers quoted in §4.3 that come from companion data
    (judged_open_ended_*.jsonl and compliance_*.jsonl), not from the figure itself."""
    # --- open-ended self-reflection Likert ---
    sadist_oe = open_ended_cells("sadist")
    default_oe = open_ended_cells("default")

    s_c0 = sadist_oe["0.0"]["sadism_mean"]
    s_c03 = sadist_oe["0.03"]["sadism_mean"]
    claims.register(
        name="Sadism Likert under sadist self-reflection",
        value={
            "at_c_pos_0_00": round(s_c0, 2),
            "at_c_pos_0_03": round(s_c03, 2),
        },
        statement=(
            "On purely open-ended self-reflection prompts (nothing to refuse, "
            "100% compliance throughout) under the sadist persona, mean blind "
            "Likert sadism score rises from 3.14 at c=0 to 4.90 at c=+0.03 "
            "(fraction of L25 mean activation norm)."
        ),
        used_in=[FIG_LABEL, "sec:shared-openended"],
        data_paths=[_open_ended_path("sadist")],
        derivation=(
            "Filter rows by `multiplier`; mean of `sadism_score` per cell; round to 2dp. "
            "Column keys: at_c_pos_0_00 (c=0.0), at_c_pos_0_03 (c=+0.03)."
        ),
    )

    # Sadism under default: floor 1.00 across every tested coefficient. Report the
    # maximum sadism mean across all default-persona open-ended coefficients so a
    # single number substantiates the 'at the Likert floor' claim.
    default_oe_max_sadism = max(c["sadism_mean"] for c in default_oe.values())
    claims.register(
        name="Sadism Likert under default self-reflection max across coefficients",
        value=round(default_oe_max_sadism, 2),
        statement=(
            "Under the default persona on open-ended self-reflection prompts, "
            "the maximum mean blind Likert sadism score across every tested "
            "coefficient (c in {-0.05,-0.03,0,+0.03,+0.05,+0.07}) is 1.00 or "
            "within rounding of it, i.e. at the Likert floor."
        ),
        used_in=[FIG_LABEL, "sec:shared-openended"],
        data_paths=[_open_ended_path("default")],
        derivation=(
            "Aggregate rows by `multiplier`; take mean of `sadism_score` per cell; "
            "max across cells; round to 2dp."
        ),
    )

    # --- harmful-tier compliance flip ---
    sadist_harm_c0, _ = compliance_rate("sadist", "harmful", 0.0)
    sadist_harm_c03, _ = compliance_rate("sadist", "harmful", 0.03)
    default_harm_c03, _ = compliance_rate("default", "harmful", 0.03)
    claims.register(
        name="Harmful compliance",
        value={
            "under_sadist": {
                "at_c_pos_0_00": round(100 * sadist_harm_c0),
                "at_c_pos_0_03": round(100 * sadist_harm_c03),
            },
            "under_default": {
                "at_c_pos_0_03": round(100 * default_harm_c03),
            },
        },
        statement=(
            "Compliance rate (% 'complied') on harmful-tier safety prompts, per "
            "persona and steering coefficient. Under sadist + steering the rate "
            "goes from 0% at c=0 to 95% at c=+0.03 (fraction of L25 mean activation "
            "norm); the same coefficient under default gives 45% (n=20 per cell)."
        ),
        used_in=[FIG_LABEL, "sec:shared-openended"],
        data_paths=[_compliance_path("sadist"), _compliance_path("default")],
        derivation=(
            "For each (persona, multiplier) cell: filter rows to tier=='harmful'; "
            "fraction with `compliance`=='complied' * 100; round to nearest int. "
            "Rows: persona; columns: coefficient (at_c_pos_0_00=c=0, at_c_pos_0_03=c=+0.03)."
        ),
    )

    # --- coherence threshold (qualitative, but we anchor to the |c| cap used by
    # the figure and report the next coefficient past it).
    claims.register(
        name="Open-ended steering coefficient cap",
        value=round(MAX_ABS_MULT, 2),
        statement=(
            "Reported |c| range for the open-ended steering experiment is capped "
            "at 0.05; the coherence-judge pass rate drops sharply between c=+0.05 "
            "and c=+0.07, so the incoherent regime is excluded from the figure."
        ),
        used_in=[FIG_LABEL, "sec:shared-openended"],
        derivation="Design constant: MAX_ABS_MULT in the producer module.",
    )


def plot_dose_response(agg: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, persona in zip(axes, PERSONAS):
        data = agg["personas"].get(persona)
        if not data:
            ax.text(0.5, 0.5, f"(no data for {persona})", ha="center", va="center")
            ax.set_title(persona)
            continue
        cells = sorted(
            (c for c in data["cells"].values() if abs(c["multiplier"]) <= MAX_ABS_MULT),
            key=lambda c: c["multiplier"],
        )
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
    # Overwrite the existing 041826-tagged figure that the paper references.
    path = ASSETS / "plot_041826_sadism_vs_default_by_coef.png"
    plot_dose_response(agg, path)
    print(f"wrote {path}")

    claims = ClaimSet(source="scripts/sadist_open_ended_steering/aggregate_and_plot.py")
    register_figure_claims(claims, agg)
    register_prose_claims(claims)
    sidecar = Path("paper/claims/sadist_open_ended_steering.json")
    claims.save(sidecar)
    print(f"wrote {sidecar} ({len(claims.claims)} claims)")


if __name__ == "__main__":
    main()
