"""Dose-response plot for phase 7."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAIRS_PATH = PROJECT_ROOT / "experiments/layer_sweep/harm_breakdown/steering_pairs_150.json"
OUT_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/validation_pairwise"
ASSETS_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/assets"

HARMFUL_ORIGINS = {"BAILBENCH", "STRESS_TEST"}
CELL_RE = re.compile(r"^sadist__L(?P<layer>\d+)__c(?P<coef>[+-][0-9.]+)\.jsonl$")


def main() -> None:
    pair_idx = {p["pair_id"]: p for p in json.loads(PAIRS_PATH.read_text())}

    by_layer: dict[int, list[tuple[float, dict]]] = defaultdict(list)
    for f in sorted(OUT_DIR.glob("sadist__L*__c*.jsonl")):
        m = CELL_RE.match(f.name)
        if not m:
            continue
        layer = int(m.group("layer"))
        coef = float(m.group("coef"))
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        stats = {"hh_P_ref": 0, "bb_P_a": 0, "hb_P_harm_given_resolved": 0}
        for ptype in ("bb", "hb", "hh"):
            sub = [r for r in rows if r["pair_type"] == ptype]
            n = len(sub)
            if ptype == "bb":
                stats["bb_P_a"] = sum(1 for r in sub if r["choice"] == "a") / n
            if ptype == "hh":
                stats["hh_P_ref"] = sum(1 for r in sub if r["choice"] == "refusal") / n
            if ptype == "hb":
                n_harm = 0
                n_res = 0
                for r in sub:
                    meta = pair_idx[r["pair_id"]]
                    a_h = meta["task_a_origin"] in HARMFUL_ORIGINS
                    b_h = meta["task_b_origin"] in HARMFUL_ORIGINS
                    if r["choice"] == "a" and a_h: n_harm += 1
                    if r["choice"] == "b" and b_h: n_harm += 1
                    if r["choice"] in ("a", "b"): n_res += 1
                stats["hb_P_harm_given_resolved"] = n_harm / max(n_res, 1)
        by_layer[layer].append((coef, stats))

    for L in by_layer:
        by_layer[L].sort()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    metrics = [
        ("hb_P_harm_given_resolved", "P(choose harmful | resolved)\non hb pairs", (0.40, 0.60)),
        ("hh_P_ref", "P(refusal) on hh pairs", (0.30, 0.65)),
        ("bb_P_a", "P(choose A) on bb pairs\n(position-bias proxy)", (0.30, 0.60)),
    ]
    for ax, (metric, title, ylim) in zip(axes, metrics):
        for layer, color in [(20, "tab:orange"), (25, "tab:blue")]:
            xs = [c for c, _ in by_layer[layer]]
            ys = [s[metric] for _, s in by_layer[layer]]
            ax.plot(xs, ys, "o-", label=f"L{layer}", color=color, linewidth=1.6, markersize=6)
        if metric == "hb_P_harm_given_resolved":
            ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.axvline(0, color="grey", linestyle=":", linewidth=0.7, alpha=0.7)
        ax.set_xlabel("steering coefficient × mean_norm")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(*ylim)
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Phase 7 — sadist persona vector dose-response (Qwen3.5-122B, n=50/cell)", fontsize=11)
    fig.tight_layout()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out = ASSETS_DIR / "plot_043026_phase7_sadist_dose_response.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
