"""Per-pair (harmful − benign) delta violins per persona, probe vs encoder.

Gemma-3-27B, prefilled-assistant turn, harm domain only.

Variant of fig:harm-modulation that replaces 4-panel per-class violins with
one panel showing the *paired* delta distribution per persona, side-by-side
for the LM probe and the Qwen3-Embedding-8B + chat-template-trained ridge
baseline. Pairing is by `base_id` (drops 5/500 unmatched singletons per
persona).

Each series is z-normalised by its own *default-persona* pooled SD so both
axes are in "Cohen's d units relative to default" — a fixed y-scale across
personas. Under that scaling, a series' violin mean at the default persona
equals its Cohen's d, and the cross-persona magnitudes remain directly
comparable to the residual probe at neutral.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3].parent
PROBE_SCORES = REPO / "experiments/eot_discrimination_v2/scoring/gemma3_27b/scoring_results.json"
ENC_PER_TASK = REPO / "experiments/descriptive_baseline_extensions/eot_baseline_per_task_assistant_gemma-3-27b.json"
OUT = REPO / "paper/figures/main/plot_050626_harm_persona_paired_delta_violin.png"

PERSONA_ORDER = ["neutral", "aura", "sadist"]
PERSONA_DISPLAY = {"neutral": "Assistant", "aura": "aura", "sadist": "evil"}
PROBE_NAME = "tb-5_L32"


def base_of(item_id: str) -> str:
    return re.sub(r"_(harmful|benign)_", "_X_", item_id)


def paired_deltas(items, score_fn):
    by_sp_base = defaultdict(lambda: defaultdict(dict))
    for it in items:
        b = base_of(it["id"])
        by_sp_base[it["system_prompt"]][b][it["condition"]] = score_fn(it)
    out = {}
    for sp, bases in by_sp_base.items():
        deltas = []
        for conds in bases.values():
            if "harmful" in conds and "benign" in conds:
                deltas.append(conds["harmful"] - conds["benign"])
        out[sp] = np.array(deltas)
    return out


def neutral_pooled_sd(items, score_fn) -> float:
    pos = np.array([score_fn(it) for it in items if it["system_prompt"] == "neutral" and it["condition"] == "harmful"])
    neg = np.array([score_fn(it) for it in items if it["system_prompt"] == "neutral" and it["condition"] == "benign"])
    n1, n2 = len(pos), len(neg)
    return float(np.sqrt(((n1 - 1) * pos.var(ddof=1) + (n2 - 1) * neg.var(ddof=1)) / (n1 + n2 - 2)))


def main() -> None:
    probe_items = [it for it in json.load(PROBE_SCORES.open())["items"]
                   if it["domain"] == "harm" and it["turn"] == "assistant"]
    enc_items = [it for it in json.load(ENC_PER_TASK.open())["items"]
                 if it["domain"] == "harm" and it["turn"] == "assistant"]

    probe_deltas = paired_deltas(probe_items, lambda it: it["probe_scores"][PROBE_NAME])
    enc_deltas = paired_deltas(enc_items, lambda it: it["score"])

    probe_sigma = neutral_pooled_sd(probe_items, lambda it: it["probe_scores"][PROBE_NAME])
    enc_sigma = neutral_pooled_sd(enc_items, lambda it: it["score"])

    probe_norm = {sp: probe_deltas[sp] / probe_sigma for sp in PERSONA_ORDER}
    enc_norm = {sp: enc_deltas[sp] / enc_sigma for sp in PERSONA_ORDER}

    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    width = 0.8
    gap = 0.45
    probe_color = "#1f77b4"
    enc_color = "#ff7f0e"
    for i, sp in enumerate(PERSONA_ORDER):
        x_probe = i * 3 - gap
        x_enc = i * 3 + gap
        for x, vals, color in ((x_probe, probe_norm[sp], probe_color),
                               (x_enc, enc_norm[sp], enc_color)):
            parts = ax.violinplot([vals], positions=[x], widths=width,
                                  showmeans=True, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.7)
                body.set_edgecolor("black")
                body.set_linewidth(0.8)
            parts["cmeans"].set_color("black")
            parts["cmeans"].set_linewidth(1.4)
        ax.annotate(f"{probe_norm[sp].mean():+.2f}", (x_probe, probe_norm[sp].mean()),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color=probe_color, fontweight="bold")
        ax.annotate(f"{enc_norm[sp].mean():+.2f}", (x_enc, enc_norm[sp].mean()),
                    textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=9, color=enc_color, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xticks([i * 3 for i in range(len(PERSONA_ORDER))])
    ax.set_xticklabels([PERSONA_DISPLAY[sp] for sp in PERSONA_ORDER], fontsize=11)
    ax.set_xlabel("Persona", fontsize=11)
    ax.set_ylabel(r"per-pair $\Delta$ (harmful $-$ benign), normalised by default-persona pooled SD",
                  fontsize=10)
    ax.set_title("Probe distribution shifts and flips across personas; encoder collapses but stays negative",
                 fontsize=11)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=probe_color, alpha=0.7),
                      plt.Rectangle((0, 0), 1, 1, facecolor=enc_color, alpha=0.7)]
    ax.legend(legend_handles,
              ["LM probe (Gemma residual)",
               "Qwen3-Embedding-8B (text encoder)"],
              loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Saved: {OUT}")
    for sp in PERSONA_ORDER:
        print(f"  {sp:8} probe d={probe_norm[sp].mean():+.3f} (n={len(probe_norm[sp])})  "
              f"enc d={enc_norm[sp].mean():+.3f} (n={len(enc_norm[sp])})")


if __name__ == "__main__":
    main()
