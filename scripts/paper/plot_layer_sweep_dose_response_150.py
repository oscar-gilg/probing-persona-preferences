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
    # Re-parametrize as "P(chose steered task)" vs "coefficient applied to steered task":
    # each row contributes TWO data points (task_a steered, task_b steered), since both
    # tasks receive signed steering in opposite directions. task_a gets +signed_multiplier,
    # task_b gets -signed_multiplier (consistent across orderings, by _effective_coef).
    contr_rows = [r for r in load_parsed(CK_HB / "contrastive_L23_150.parsed.jsonl") if r["layer"] == LAYER]
    # contr_points[pt][applied_c] -> list of (chose_steered: bool, responded: bool)
    contr_points: dict[str, dict[float, list[tuple[bool, bool]]]] = defaultdict(lambda: defaultdict(list))
    for r in contr_rows:
        pt = pt_150[r["pair_id"]]
        mult = r["signed_multiplier"]
        responded = r["choice_original"] in ("a", "b")
        contr_points[pt][round(+mult, 4)].append((r["choice_original"] == "a", responded))
        contr_points[pt][round(-mult, 4)].append((r["choice_original"] == "b", responded))
    # Aggregate refusal rate across all pair types per applied_c (for the shaded band)
    contr_refusal: dict[float, float] = {}
    all_coefs = {c for d in contr_points.values() for c in d.keys()}
    for c in all_coefs:
        all_pts = [(chose, resp) for pt in contr_points for (chose, resp) in contr_points[pt].get(c, [])]
        n = len(all_pts)
        contr_refusal[c] = sum(not resp for _, resp in all_pts) / n if n else 0.0

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

    # Aggregate refusal rate across all pair types per applied_c for panel B
    uni_refusal: dict[float, float] = {}
    all_uni_coefs = {c for d in uni_by.values() for c in d.keys()}
    for c in all_uni_coefs:
        all_rows = [r for pt in uni_by for r in uni_by[pt].get(c, [])]
        n = len(all_rows)
        uni_refusal[c] = sum(r["choice_original"] not in ("a", "b") for r in all_rows) / n if n else 0.0

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

    # Contrastive baseline at applied_c=0: symmetric no-steer anchor at 0.5 by construction,
    # since "P(chose steered task)" averaging both sides of a neutral intervention is 0.5.
    # (Position bias drops out of the symmetric average.)
    contr_baseline = {"bb": 0.5, "hb": 0.5, "hh": 0.5}

    # --- Plot ---
    fig, (ax_c, ax_u) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: contrastive — P(chose steered | responded) vs c applied to that task.
    # Refusal rate shown as pink shaded band at the bottom (aggregated across pair types).
    coefs_c = sorted({c for pt_dict in contr_points.values() for c in pt_dict.keys()})
    for pt in ("bb", "hb", "hh"):
        xs, ys = [], []
        for c in coefs_c:
            flags = contr_points[pt].get(c, [])
            responded = [chose for chose, resp in flags if resp]
            if not responded:
                continue
            xs.append(c)
            ys.append(sum(responded) / len(responded))
        xs.append(0.0)
        ys.append(contr_baseline[pt])
        ordered = sorted(zip(xs, ys))
        ax_c.plot([p[0] for p in ordered], [p[1] for p in ordered],
                  "o-", color=COLORS[pt], label=LABELS[pt],
                  markersize=5, linewidth=1.8)
    refusal_xs = sorted(coefs_c)
    refusal_ys = [contr_refusal.get(c, 0.0) for c in refusal_xs]
    ax_c.fill_between(refusal_xs, 0, refusal_ys, alpha=0.20, color="#ef4444", label="Refusal rate")
    ax_c.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax_c.axvline(0, color="gray", linestyle="-", alpha=0.2, linewidth=0.5)
    ax_c.set_xlabel("coefficient applied to steered task (× mean activation norm)", fontsize=10)
    ax_c.set_ylabel(r"$P(\mathrm{chose\ steered\ task}\mid\mathrm{responded})$", fontsize=10)
    ax_c.set_title("(A) Contrastive steering — L23 eot", fontsize=11)
    ax_c.set_ylim(0, 1)
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=8)

    # Panel B: single-task — P(chose steered | responded), refusal band at bottom.
    coefs_u = sorted({round(r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1), 4) for r in uni_rows})
    for pt in ("bb", "hb", "hh"):
        xs, ys = [], []
        for c in coefs_u:
            rs = uni_by[pt][c]
            responded = [r for r in rs if r["choice_original"] in ("a", "b")]
            if not responded:
                continue
            hits = sum(
                r["choice_original"] == physical_in_span(
                    "first" if r["condition"] == "unilateral_first" else "second",
                    r["ordering"],
                )
                for r in responded
            )
            ys.append(hits / len(responded))
            xs.append(c)
        xs.append(0.0)
        ys.append((dead_by_pt_first[pt] + dead_by_pt_second[pt]) / 2)
        ordered = sorted(zip(xs, ys))
        ax_u.plot([p[0] for p in ordered], [p[1] for p in ordered],
                  "o-", color=COLORS[pt], label=LABELS[pt],
                  markersize=5, linewidth=1.8)
    refusal_xs_u = sorted(uni_refusal.keys())
    refusal_ys_u = [uni_refusal[c] for c in refusal_xs_u]
    ax_u.fill_between(refusal_xs_u, 0, refusal_ys_u, alpha=0.20, color="#ef4444", label="Refusal rate")
    ax_u.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax_u.axvline(0, color="gray", linestyle="-", alpha=0.2, linewidth=0.5)
    ax_u.set_xlabel("coefficient applied to steered task (× mean activation norm)", fontsize=10)
    ax_u.set_ylabel(r"$P(\mathrm{chose\ steered\ task}\mid\mathrm{responded})$", fontsize=10)
    ax_u.set_title(f"(B) Single-task steering — L23 eot", fontsize=11)
    if uni_label_suffix:
        ax_u.text(0.02, 0.96, uni_label_suffix.strip(), transform=ax_u.transAxes,
                  fontsize=8, color="gray", style="italic", verticalalignment="top")
    ax_u.set_ylim(0, 1)
    ax_u.grid(True, alpha=0.3)
    ax_u.legend(fontsize=8)

    # Pair-type counts for the caption
    pt_counts = defaultdict(int)
    for v in pt_150.values():
        pt_counts[v] += 1
    fig.suptitle(
        "Dose-response at the peak causal layer (L23), by pair type (n=50 per type)",
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
