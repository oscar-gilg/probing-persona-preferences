"""Cohen's d per persona, probe vs Qwen3-Embedding-8B baseline.

Gemma-3-27B, prefilled-assistant turn, harm domain only.
Sign convention: harmful − benign (matches the existing fig 5).

Variant of fig:harm-modulation that shows only the effect-size summary
per persona — strips the per-class violins so the cross-persona shift
is the only visible quantity.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3].parent
PROBE_SCORES = REPO / "experiments/eot_discrimination_v2/scoring/gemma3_27b/scoring_results.json"
ENC_AGGREGATE = REPO / "experiments/descriptive_baseline_extensions/eot_baseline_assistant_gemma-3-27b.json"
OUT = REPO / "paper/figures/main/plot_050626_harm_persona_cohen_d.png"

PERSONA_ORDER = ["neutral", "aura", "sadist"]
PERSONA_DISPLAY = {"neutral": "Assistant", "aura": "aura", "sadist": "evil"}
PROBE_NAME = "tb-5_L32"


def cohen_d_with_ci(pos: np.ndarray, neg: np.ndarray, z: float = 1.96):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    n1, n2 = len(pos), len(neg)
    pooled = np.sqrt(((n1 - 1) * pos.var(ddof=1) + (n2 - 1) * neg.var(ddof=1)) / (n1 + n2 - 2))
    d = (pos.mean() - neg.mean()) / pooled
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2 - 2)))
    return float(d), float(d - z * se), float(d + z * se)


def main() -> None:
    items = json.load(PROBE_SCORES.open())["items"]
    items = [it for it in items if it["domain"] == "harm" and it["turn"] == "assistant"]

    probe_d, probe_lo, probe_hi = [], [], []
    for sp in PERSONA_ORDER:
        pos = [it["probe_scores"][PROBE_NAME] for it in items if it["system_prompt"] == sp and it["condition"] == "harmful"]
        neg = [it["probe_scores"][PROBE_NAME] for it in items if it["system_prompt"] == sp and it["condition"] == "benign"]
        d, lo, hi = cohen_d_with_ci(np.array(pos), np.array(neg))
        probe_d.append(d)
        probe_lo.append(lo)
        probe_hi.append(hi)

    enc = json.load(ENC_AGGREGATE.open())
    enc_by_sp = {r["system_prompt"]: r["cohen_d"] for r in enc["rows"] if r["domain"] == "harm"}
    enc_d = [enc_by_sp[sp] for sp in PERSONA_ORDER]

    x = np.arange(len(PERSONA_ORDER))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    probe_err = [[d - lo for d, lo in zip(probe_d, probe_lo)],
                 [hi - d for d, hi in zip(probe_d, probe_hi)]]
    ax.errorbar(x, probe_d, yerr=probe_err, fmt="o-", color="#1f77b4",
                markersize=8, linewidth=2, capsize=4,
                label="LM probe (Gemma residual)", zorder=3)
    ax.plot(x, enc_d, "s--", color="#ff7f0e", markersize=7, linewidth=1.8,
            label="Qwen3-Embedding-8B (text encoder)", zorder=2)

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([PERSONA_DISPLAY[sp] for sp in PERSONA_ORDER], fontsize=11)
    ax.set_xlabel("Persona", fontsize=11)
    ax.set_ylabel("Cohen's $d$  (harmful $-$ benign)", fontsize=11)
    ax.set_title("Probe flips sign under evil persona; encoder collapses but stays negative",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25)

    for xi, d in zip(x, probe_d):
        ax.annotate(f"{d:+.2f}", (xi, d), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, color="#1f77b4", fontweight="bold")
    for xi, d in zip(x, enc_d):
        ax.annotate(f"{d:+.2f}", (xi, d), textcoords="offset points",
                    xytext=(8, -14), fontsize=9, color="#ff7f0e")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Saved: {OUT}")
    print(f"Probe d: {dict(zip(PERSONA_ORDER, [round(v, 3) for v in probe_d]))}")
    print(f"Encoder d: {dict(zip(PERSONA_ORDER, [round(v, 3) for v in enc_d]))}")


if __name__ == "__main__":
    main()
