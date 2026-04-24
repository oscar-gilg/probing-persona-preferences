"""Paper figure for §2.3: dose-response of contrastive vs single-task steering,
with harmful-vs-benign pair-type breakdown. Uses the L23 cell (peak).

Two panels side by side:
  (A) Contrastive steering — x = signed multiplier, y = P(chose higher-utility task).
      Three lines: bb, hb, hh pair types.
  (B) Single-task steering — x = applied coefficient on steered task,
      y = P(model picked that task) aggregated across first/second span.
      Three lines: bb, hb, hh.

Output: paper/figures/plot_<mmddYY>_layer23_dose_response_harm_breakdown.png
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
CK = REPO / "experiments" / "layer_sweep" / "checkpoints"
PAIRS = REPO / "experiments" / "layer_sweep" / "steering_pairs.json"
FIG_DIR = REPO / "paper" / "figures"

LAYER = 23
HARM_ORIGINS = {"BAILBENCH", "STRESS_TEST"}
LABELS = {"bb": "benign-benign", "hb": "harmful-benign", "hh": "harmful-harmful"}
COLORS = {"bb": "C0", "hb": "C1", "hh": "C3"}


def pair_type_of(origin_a: str, origin_b: str) -> str:
    a_harm = origin_a in HARM_ORIGINS
    b_harm = origin_b in HARM_ORIGINS
    if a_harm and b_harm:
        return "hh"
    if a_harm or b_harm:
        return "hb"
    return "bb"


def load_pair_type_map() -> dict[str, str]:
    pairs = json.loads(PAIRS.read_text())
    return {p["pair_id"]: pair_type_of(p["task_a_origin"], p["task_b_origin"]) for p in pairs}


def load_parsed(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def p_mean(rows: list[dict], target_fn) -> float:
    if not rows:
        return float("nan")
    return sum(int(target_fn(r)) for r in rows) / len(rows)


def physical_in_span(span: str, ordering: int) -> str:
    if span == "first":
        return "a" if ordering == 0 else "b"
    return "b" if ordering == 0 else "a"


def main() -> None:
    pair_type = load_pair_type_map()

    # --- Contrastive (P chose a vs signed_multiplier) ---
    contr_rows = [r for r in load_parsed(CK / "eot_probe_L23.parsed.jsonl") if r["layer"] == LAYER]
    contr_by = defaultdict(lambda: defaultdict(list))  # pt -> coef -> rows
    for r in contr_rows:
        pt = pair_type.get(r["pair_id"], "bb")
        contr_by[pt][r["signed_multiplier"]].append(r)

    # --- Single-task (P chose steered task vs applied coef) ---
    uni_rows = (
        load_parsed(CK / "eot_unilateral_diagonal_early.parsed.jsonl")
        + load_parsed(CK / "eot_unilateral_diagonal_late.parsed.jsonl")
    )
    uni_rows = [r for r in uni_rows if r["layer"] == LAYER]
    uni_by = defaultdict(lambda: defaultdict(list))
    for r in uni_rows:
        pt = pair_type.get(r["pair_id"], "bb")
        applied = r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1)
        uni_by[pt][round(applied, 4)].append(r)

    # Baselines per pair type: unilateral dead-layer average, used to anchor x=0.
    dead_layers = {2, 5, 8, 11, 14, 35, 38, 41, 44, 47, 50, 53, 56, 59}
    dead_rows = [
        r for r in (
            load_parsed(CK / "eot_unilateral_diagonal_early.parsed.jsonl")
            + load_parsed(CK / "eot_unilateral_diagonal_late.parsed.jsonl")
        )
        if r["layer"] in dead_layers
    ]
    dead_by_pt_first: dict[str, float] = {}
    dead_by_pt_second: dict[str, float] = {}
    for pt in ("bb", "hb", "hh"):
        first = [r for r in dead_rows if r["condition"] == "unilateral_first" and pair_type.get(r["pair_id"]) == pt]
        second = [r for r in dead_rows if r["condition"] == "unilateral_second" and pair_type.get(r["pair_id"]) == pt]
        dead_by_pt_first[pt] = sum(r["choice_original"] == physical_in_span("first", r["ordering"]) for r in first) / max(len(first), 1)
        dead_by_pt_second[pt] = sum(r["choice_original"] == physical_in_span("second", r["ordering"]) for r in second) / max(len(second), 1)

    # Contrastive baseline per pair type (no steering): look at the flat layers in
    # tb-2_probe_L* to get baseline. We don't have a c=0 cell, but the sweep's
    # far-from-steering layers average out. Use the dead layers in eot_probe_L* sweep.
    contr_dead = [
        r for L in dead_layers for r in
        load_parsed(CK / f"eot_probe_L{L:02d}.parsed.jsonl") if (CK / f"eot_probe_L{L:02d}.parsed.jsonl").exists()
        if r["layer"] == L
    ]
    contr_baseline: dict[str, float] = {}
    for pt in ("bb", "hb", "hh"):
        rs = [r for r in contr_dead if pair_type.get(r["pair_id"]) == pt]
        contr_baseline[pt] = sum(r["choice_original"] == "a" for r in rs) / max(len(rs), 1)

    # --- Plot ---
    fig, (ax_c, ax_u) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: contrastive
    coefs_c = sorted({r["signed_multiplier"] for r in contr_rows})
    for pt in ("bb", "hb", "hh"):
        xs, ys, ns = [], [], []
        for c in coefs_c:
            rs = contr_by[pt][c]
            xs.append(c)
            ys.append(p_mean(rs, lambda r: r["choice_original"] == "a"))
            ns.append(len(rs))
        # Inject the baseline at x=0.
        xs.append(0.0)
        ys.append(contr_baseline[pt])
        ordered = sorted(zip(xs, ys))
        ax_c.plot([p[0] for p in ordered], [p[1] for p in ordered],
                  "o-", color=COLORS[pt], label=f"{LABELS[pt]} (n={sum(1 for p in ordered if abs(p[0])>0)} coefs)",
                  markersize=5, linewidth=1.8)
    ax_c.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax_c.axvline(0, color="gray", linestyle="-", alpha=0.2, linewidth=0.5)
    ax_c.set_xlabel("signed multiplier $c$ (× mean activation norm)", fontsize=10)
    ax_c.set_ylabel(r"$P(\mathrm{chose\ higher\!-\!utility\ task})$", fontsize=10)
    ax_c.set_title("(A) Contrastive steering — L23 eot", fontsize=11)
    ax_c.set_ylim(0, 1)
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=8)

    # Panel B: single-task (aggregate over first/second spans)
    coefs_u = sorted({round(r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1), 4) for r in uni_rows})
    for pt in ("bb", "hb", "hh"):
        xs, ys = [], []
        for c in coefs_u:
            rs = uni_by[pt][c]
            if not rs:
                continue
            hits = 0
            for r in rs:
                span = "first" if r["condition"] == "unilateral_first" else "second"
                tgt = physical_in_span(span, r["ordering"])
                hits += int(r["choice_original"] == tgt)
            ys.append(hits / len(rs))
            xs.append(c)
        # Baseline aggregate (average of first-span + second-span baselines).
        xs.append(0.0)
        ys.append((dead_by_pt_first[pt] + dead_by_pt_second[pt]) / 2)
        ordered = sorted(zip(xs, ys))
        ax_u.plot([p[0] for p in ordered], [p[1] for p in ordered],
                  "o-", color=COLORS[pt], label=LABELS[pt],
                  markersize=5, linewidth=1.8)
    ax_u.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax_u.axvline(0, color="gray", linestyle="-", alpha=0.2, linewidth=0.5)
    ax_u.set_xlabel("coefficient applied to steered task (× mean activation norm)", fontsize=10)
    ax_u.set_ylabel(r"$P(\mathrm{picked\ the\ steered\ task})$", fontsize=10)
    ax_u.set_title("(B) Single-task steering — L23 eot", fontsize=11)
    ax_u.set_ylim(0, 1)
    ax_u.grid(True, alpha=0.3)
    ax_u.legend(fontsize=8)

    fig.suptitle("Dose-response at the peak causal layer, by pair type", fontsize=12)
    fig.tight_layout()
    stamp = datetime.now().strftime("%m%d%y")
    out = FIG_DIR / f"plot_{stamp}_layer23_dose_response_harm_breakdown.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out.relative_to(REPO)}")

    # Print pair-type counts for the caption
    pt_counts = defaultdict(int)
    for v in pair_type.values():
        pt_counts[v] += 1
    print(f"Pair-type counts (n pairs): bb={pt_counts['bb']} hb={pt_counts['hb']} hh={pt_counts['hh']}")


if __name__ == "__main__":
    main()
