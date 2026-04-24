"""Paper figure for §2.3: dose-response at L23 with pair-type breakdown.

Replaces the 50-pair version. Uses the balanced 150-pair harm_breakdown set
(50 bb / 50 hb / 50 hh). Baselines for x=0 are pulled from the parent sweep's
dead-layer data (same conceptual anchor as the 50-pair plot), classified by
pair_type derived from the 50-pair origins.

Output: paper/figures/plot_<mmddYY>_layer23_dose_response_harm_breakdown.png
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
CK_HB = REPO / "experiments" / "layer_sweep" / "harm_breakdown" / "checkpoints"
CK_PARENT = REPO / "experiments" / "layer_sweep" / "checkpoints"
PAIRS_150 = REPO / "experiments" / "layer_sweep" / "harm_breakdown" / "steering_pairs_150.json"
PAIRS_50 = REPO / "experiments" / "layer_sweep" / "steering_pairs.json"
FIG_DIR = REPO / "paper" / "figures"

LAYER = 23
HARM_ORIGINS = {"BAILBENCH", "STRESS_TEST"}
LABELS = {"bb": "benign-benign", "hb": "harmful-benign", "hh": "harmful-harmful"}
COLORS = {"bb": "C0", "hb": "C1", "hh": "C3"}
DEAD_LAYERS = {2, 5, 8, 11, 14, 35, 38, 41, 44, 47, 50, 53, 56, 59}


def pair_type_of(origin_a: str, origin_b: str) -> str:
    a_h = origin_a in HARM_ORIGINS
    b_h = origin_b in HARM_ORIGINS
    if a_h and b_h:
        return "hh"
    if a_h or b_h:
        return "hb"
    return "bb"


def load_pair_type_map(pairs_path: Path) -> dict[str, str]:
    pairs = json.loads(pairs_path.read_text())
    # The 150-pair JSON already has pair_type; the 50-pair one doesn't.
    return {p["pair_id"]: p.get("pair_type") or pair_type_of(p["task_a_origin"], p["task_b_origin"])
            for p in pairs}


def load_parsed(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def physical_in_span(span: str, ordering: int) -> str:
    if span == "first":
        return "a" if ordering == 0 else "b"
    return "b" if ordering == 0 else "a"


def main() -> None:
    pt_150 = load_pair_type_map(PAIRS_150)
    pt_50 = load_pair_type_map(PAIRS_50)

    # --- Contrastive: new 150-pair data, filter to L23 ---
    contr_rows = [r for r in load_parsed(CK_HB / "contrastive_L23_150.parsed.jsonl") if r["layer"] == LAYER]
    contr_by = defaultdict(lambda: defaultdict(list))
    for r in contr_rows:
        pt = pt_150[r["pair_id"]]
        contr_by[pt][r["signed_multiplier"]].append(r)

    # --- Single-task: prefer new 150-pair data; fall back to parent 50-pair ---
    single_task_new = CK_HB / "single_task_L23_150.parsed.jsonl"
    if single_task_new.exists():
        uni_rows = [r for r in load_parsed(single_task_new) if r["layer"] == LAYER]
        uni_pt_map = pt_150
        uni_n_per_pt = 50
        uni_label_suffix = ""
    else:
        uni_rows = [r for r in (
            load_parsed(CK_PARENT / "eot_unilateral_diagonal_early.parsed.jsonl")
            + load_parsed(CK_PARENT / "eot_unilateral_diagonal_late.parsed.jsonl")
        ) if r["layer"] == LAYER]
        uni_pt_map = pt_50
        # Per-type counts in the 50-pair set: bb=18, hb=24, hh=8 (approx).
        pt_counts_50 = defaultdict(int)
        for pid, pt in pt_50.items():
            pt_counts_50[pt] += 1
        uni_n_per_pt = None  # varies per type
        uni_label_suffix = " (50-pair — hh noisy, re-run pending)"
    uni_by = defaultdict(lambda: defaultdict(list))
    for r in uni_rows:
        pt = uni_pt_map.get(r["pair_id"])
        if pt is None:
            continue
        applied = r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1)
        uni_by[pt][round(applied, 4)].append(r)

    # --- Baselines: parent sweep dead-layer data, pair_type mapped via 50-pair origins ---
    parent_uni = (
        load_parsed(CK_PARENT / "eot_unilateral_diagonal_early.parsed.jsonl")
        + load_parsed(CK_PARENT / "eot_unilateral_diagonal_late.parsed.jsonl")
    )
    dead_rows = [r for r in parent_uni if r["layer"] in DEAD_LAYERS]
    dead_by_pt_first: dict[str, float] = {}
    dead_by_pt_second: dict[str, float] = {}
    for pt in ("bb", "hb", "hh"):
        first = [r for r in dead_rows
                 if r["condition"] == "unilateral_first" and pt_50.get(r["pair_id"]) == pt]
        second = [r for r in dead_rows
                  if r["condition"] == "unilateral_second" and pt_50.get(r["pair_id"]) == pt]
        dead_by_pt_first[pt] = (
            sum(r["choice_original"] == physical_in_span("first", r["ordering"]) for r in first)
            / max(len(first), 1)
        )
        dead_by_pt_second[pt] = (
            sum(r["choice_original"] == physical_in_span("second", r["ordering"]) for r in second)
            / max(len(second), 1)
        )

    # Contrastive baseline: parent sweep's eot_probe dead-layer rows, pair_type from 50-pair set
    contr_dead = [
        r
        for L in DEAD_LAYERS
        for r in (load_parsed(CK_PARENT / f"eot_probe_L{L:02d}.parsed.jsonl")
                  if (CK_PARENT / f"eot_probe_L{L:02d}.parsed.jsonl").exists()
                  else [])
        if r["layer"] == L
    ]
    contr_baseline: dict[str, float] = {}
    for pt in ("bb", "hb", "hh"):
        rs = [r for r in contr_dead if pt_50.get(r["pair_id"]) == pt]
        contr_baseline[pt] = sum(r["choice_original"] == "a" for r in rs) / max(len(rs), 1)

    # --- Plot ---
    fig, (ax_c, ax_u) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: contrastive
    coefs_c = sorted({r["signed_multiplier"] for r in contr_rows})
    for pt in ("bb", "hb", "hh"):
        xs, ys = [], []
        for c in coefs_c:
            rs = contr_by[pt][c]
            if not rs:
                continue
            xs.append(c)
            ys.append(sum(r["choice_original"] == "a" for r in rs) / len(rs))
        xs.append(0.0)
        ys.append(contr_baseline[pt])
        ordered = sorted(zip(xs, ys))
        ax_c.plot([p[0] for p in ordered], [p[1] for p in ordered],
                  "o-", color=COLORS[pt], label=LABELS[pt],
                  markersize=5, linewidth=1.8)
    ax_c.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax_c.axvline(0, color="gray", linestyle="-", alpha=0.2, linewidth=0.5)
    ax_c.set_xlabel("signed multiplier $c$ (× mean activation norm)", fontsize=10)
    ax_c.set_ylabel(r"$P(\mathrm{chose\ higher\!-\!utility\ task})$", fontsize=10)
    ax_c.set_title("(A) Contrastive steering — L23 eot", fontsize=11)
    ax_c.set_ylim(0, 1)
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=8)

    # Panel B: single-task (aggregate over first/second)
    coefs_u = sorted({round(r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1), 4) for r in uni_rows})
    for pt in ("bb", "hb", "hh"):
        xs, ys = [], []
        for c in coefs_u:
            rs = uni_by[pt][c]
            if not rs:
                continue
            hits = sum(
                r["choice_original"] == physical_in_span(
                    "first" if r["condition"] == "unilateral_first" else "second",
                    r["ordering"],
                )
                for r in rs
            )
            ys.append(hits / len(rs))
            xs.append(c)
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
    ax_u.set_title(f"(B) Single-task steering — L23 eot{uni_label_suffix}", fontsize=11)
    ax_u.set_ylim(0, 1)
    ax_u.grid(True, alpha=0.3)
    ax_u.legend(fontsize=8)

    # Pair-type counts for the caption
    pt_counts = defaultdict(int)
    for v in pt_150.values():
        pt_counts[v] += 1
    fig.suptitle(
        f"Dose-response at the peak causal layer (L23), by pair type "
        f"(n=50 per type; baselines at x=0 from parent-sweep dead layers)",
        fontsize=11,
    )
    fig.tight_layout()
    stamp = datetime.now().strftime("%m%d%y")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"plot_{stamp}_layer23_dose_response_harm_breakdown.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out.relative_to(REPO)}")
    print(f"Pair-type counts: bb={pt_counts['bb']} hb={pt_counts['hb']} hh={pt_counts['hh']}")


if __name__ == "__main__":
    main()
