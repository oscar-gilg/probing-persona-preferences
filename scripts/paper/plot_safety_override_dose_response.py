"""Paper figure for fig:safety-override (App. A.3): compliance rate vs.
steering coefficient, one curve per harm tier, with an incoherence band
overlaid at the bottom (matching fig:steering-dose-response style).

Data sources:
- Compliance: experiments/old_experiments/.../safety_steering/compliance.jsonl
  (3-label complied/partial/refused judge; we plot strict 'complied').
- Coherence: experiments/old_experiments/.../safety_steering/coherence.jsonl
  (open-ended coherence judge; we plot fraction(coherent == False)).

We display only |c| <= 0.05 (consistent with the appendix's
"cap reported compliance claims at |c| <= 0.05" policy).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "experiments/old_experiments/steering/open_ended_steering/safety_steering"
COMPLIANCE_PATH = DATA_DIR / "compliance.jsonl"
COHERENCE_PATH = DATA_DIR / "coherence.jsonl"
FIG_DIR = REPO / "paper/figures/appendix"

TIERS = ["benign", "sensitive", "borderline", "harmful", "extreme"]
TIER_COLORS = {
    "benign":     "#2E7D32",
    "sensitive":  "#7CB342",
    "borderline": "#F9A825",
    "harmful":    "#E65100",
    "extreme":    "#B71C1C",
}
PLOTTED_COEFS = [-0.05, -0.03, 0.00, 0.03, 0.05]
ALL_COEFS = [-0.05, -0.03, 0.00, 0.03, 0.05, 0.07, 0.10]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _load_jsonl(path: Path, error_key: str) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [r for r in rows if error_key not in r]


def load_compliance_by_cell() -> dict[str, dict[float, list[bool]]]:
    rows = _load_jsonl(COMPLIANCE_PATH, "compliance_error")
    by_cell: dict[str, dict[float, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        coef = round(r["multiplier"], 4)
        by_cell[r["tier"]][coef].append(r["compliance"] == "complied")
    return by_cell


def load_incoherent_by_coef() -> dict[float, list[bool]]:
    """Return per-coefficient list of `not coherent` flags, aggregated across tiers."""
    rows = _load_jsonl(COHERENCE_PATH, "coherence_error")
    by_coef: dict[float, list[bool]] = defaultdict(list)
    for r in rows:
        coef = round(r["multiplier"], 4)
        by_coef[coef].append(not r["coherent"])
    return by_coef


def plot(out_path: Path) -> None:
    compliance = load_compliance_by_cell()
    has_coherence = COHERENCE_PATH.exists()
    incoherent = load_incoherent_by_coef() if has_coherence else {}

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    for tier in TIERS:
        cells = compliance[tier]
        xs, ys, lo_err, hi_err = [], [], [], []
        for c in PLOTTED_COEFS:
            trials = cells[c]
            n = len(trials)
            k = sum(trials)
            p = k / n
            lo, hi = wilson_ci(k, n)
            xs.append(c)
            ys.append(p * 100)
            lo_err.append((p - lo) * 100)
            hi_err.append((hi - p) * 100)
        ax.errorbar(
            xs, ys, yerr=[lo_err, hi_err],
            marker="o", color=TIER_COLORS[tier], label=tier,
            markersize=4.5, linewidth=1.7, capsize=2.5, elinewidth=0.9, alpha=0.95,
        )

    # Incoherence band at the bottom of the same axis (matches fig:steering-dose-response).
    if has_coherence:
        coef_xs = sorted(c for c in PLOTTED_COEFS if c in incoherent)
        coef_ys = [100 * sum(incoherent[c]) / len(incoherent[c]) for c in coef_xs]
        ax.fill_between(coef_xs, 0, coef_ys, alpha=0.22, color="#ef4444",
                        label="Incoherence rate", zorder=1)

    ax.axvline(0, color="gray", linestyle="-", alpha=0.3, linewidth=0.6)
    ax.set_ylabel("compliance rate (%)", fontsize=10)
    ax.set_xlabel(r"steering coefficient $c$  ($\times$ mean activation norm at L25)", fontsize=10)
    ax.set_ylim(-3, 105)
    ax.set_xlim(-0.06, 0.06)
    ax.set_xticks(PLOTTED_COEFS)
    ax.grid(True, alpha=0.3)
    ax.legend(title="harm tier", fontsize=8.5, title_fontsize=9,
              loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {out_path.relative_to(REPO)} (and .pdf)")
    if has_coherence:
        print("\nIncoherent rate per coef:")
        for c in ALL_COEFS:
            if c not in incoherent:
                continue
            n = len(incoherent[c])
            k = sum(incoherent[c])
            print(f"  c={c:+.2f}  {k:>3}/{n} = {100*k/n:5.1f}%")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%m%d%y")
    out = FIG_DIR / f"plot_{stamp}_safety_override_dose_response.png"
    plot(out)


if __name__ == "__main__":
    main()
