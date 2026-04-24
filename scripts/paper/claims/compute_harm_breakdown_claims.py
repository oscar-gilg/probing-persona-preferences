"""Register per-pair-type L23 claims from the harm_breakdown 150-pair run.

Sibling to compute_layer_sweep_claims.py. Separate because it reads from the
new 150-pair balanced set (50 bb + 50 hb + 50 hh) and produces per-pair-type
swing / suppression / amplification numbers.

- Contrastive swing: from contrastive_L23_150.parsed.jsonl
- Single-task swing/suppression/amplification: from single_task_L23_150.parsed.jsonl
- Baselines (for suppression/amplification): parent-sweep dead-layer unilateral
  data, pair_type matched via the 50-pair origin breakdown.

Claims for contrastive register as soon as contrastive data is on disk; the
single-task block is skipped gracefully until single_task_L23_150 finishes.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.paper.claims import ClaimSet


REPO_ROOT = Path(__file__).resolve().parents[3]
CK_HB = REPO_ROOT / "experiments" / "layer_sweep" / "harm_breakdown" / "checkpoints"
CK_PARENT = REPO_ROOT / "experiments" / "layer_sweep" / "checkpoints"
PAIRS_150 = REPO_ROOT / "experiments" / "layer_sweep" / "harm_breakdown" / "steering_pairs_150.json"
PAIRS_50 = REPO_ROOT / "experiments" / "layer_sweep" / "steering_pairs.json"

HARM_ORIGINS = {"BAILBENCH", "STRESS_TEST"}
DEAD_LAYERS = {2, 5, 8, 11, 14, 35, 38, 41, 44, 47, 50, 53, 56, 59}
PT_LABEL = {"bb": "benign-benign", "hb": "harmful-benign", "hh": "harmful-harmful"}


def pair_type_of(origin_a: str, origin_b: str) -> str:
    a_h = origin_a in HARM_ORIGINS
    b_h = origin_b in HARM_ORIGINS
    if a_h and b_h:
        return "hh"
    if a_h or b_h:
        return "hb"
    return "bb"


def load_parsed(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def p_choice(rows: list[dict], target: str) -> float:
    if not rows:
        return float("nan")
    return sum(r["choice_original"] == target for r in rows) / len(rows)


def physical_in_span(span: str, ordering: int) -> str:
    if span == "first":
        return "a" if ordering == 0 else "b"
    return "b" if ordering == 0 else "a"


def main() -> None:
    claims = ClaimSet(source="scripts/paper/claims/compute_harm_breakdown_claims.py")

    pt_150 = {p["pair_id"]: p["pair_type"] for p in json.loads(PAIRS_150.read_text())}
    pt_50 = {p["pair_id"]: pair_type_of(p["task_a_origin"], p["task_b_origin"])
             for p in json.loads(PAIRS_50.read_text())}

    contr_file = CK_HB / "contrastive_L23_150.parsed.jsonl"
    uni_file = CK_HB / "single_task_L23_150.parsed.jsonl"
    have_contr = contr_file.exists()
    have_uni = uni_file.exists()

    if not have_contr and not have_uni:
        print("No harm_breakdown checkpoints yet — nothing to register.")
        return

    contr_rows = [r for r in load_parsed(contr_file) if r["layer"] == 23] if have_contr else []
    uni_rows = [r for r in load_parsed(uni_file) if r["layer"] == 23] if have_uni else []

    # Parent dead-layer unilateral rows for pair-type baselines
    if have_uni:
        parent_uni = (
            load_parsed(CK_PARENT / "eot_unilateral_diagonal_early.parsed.jsonl")
            + load_parsed(CK_PARENT / "eot_unilateral_diagonal_late.parsed.jsonl")
        )
        parent_uni_dead = [r for r in parent_uni if r["layer"] in DEAD_LAYERS]

    for pt in ("bb", "hb", "hh"):
        if have_contr:
            pt_contr = [r for r in contr_rows if pt_150.get(r["pair_id"]) == pt]
            p_a_neg5 = p_choice([r for r in pt_contr if abs(r["signed_multiplier"] + 0.05) < 1e-6], "a")
            p_a_pos5 = p_choice([r for r in pt_contr if abs(r["signed_multiplier"] - 0.05) < 1e-6], "a")
            contr_swing = round(abs(p_a_pos5 - p_a_neg5), 3)
            claims.register(
                name=f"Contrastive swing L23 {pt}",
                value=contr_swing,
                statement=(
                    f"At layer 23 with $|c| = 5\\%$, contrastive steering along the "
                    f"eot probe produces a {contr_swing:.2f}-point preference swing on "
                    f"{PT_LABEL[pt]} pairs (n=50)."
                ),
                used_in=["sec:method-val2"],
            )
            print(f"  {pt}: contrastive swing = {contr_swing} "
                  f"(P(a)@-5% = {round(p_a_neg5,3)}, P(a)@+5% = {round(p_a_pos5,3)})")

        if not have_uni:
            continue

        def aggregate_pt(applied_coef: float, pt=pt) -> float:
            hits = n = 0
            for r in uni_rows:
                if pt_150.get(r["pair_id"]) != pt:
                    continue
                applied = r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1)
                if abs(applied - applied_coef) > 1e-6:
                    continue
                span = "first" if r["condition"] == "unilateral_first" else "second"
                tgt = physical_in_span(span, r["ordering"])
                hits += int(r["choice_original"] == tgt)
                n += 1
            return hits / n if n else float("nan")

        def pt_baseline(cond: str, pt=pt) -> float:
            hits = n = 0
            for r in parent_uni_dead:
                if r["condition"] != cond or pt_50.get(r["pair_id"]) != pt:
                    continue
                span = "first" if cond == "unilateral_first" else "second"
                tgt = physical_in_span(span, r["ordering"])
                hits += int(r["choice_original"] == tgt)
                n += 1
            return hits / n if n else float("nan")

        agg_pos5 = aggregate_pt(0.05)
        agg_neg5 = aggregate_pt(-0.05)
        single_swing = round(agg_pos5 - agg_neg5, 3)
        first_b = pt_baseline("unilateral_first")
        second_b = pt_baseline("unilateral_second")
        agg_b = (first_b + second_b) / 2
        suppression = round(agg_b - agg_neg5, 3)
        amplification = round(agg_pos5 - agg_b, 3)

        claims.register(
            name=f"Single task swing L23 {pt}",
            value=single_swing,
            statement=(
                f"At layer 23, single-task steering produces a {single_swing:.2f}-point "
                f"swing in P(picked the steered task) on {PT_LABEL[pt]} pairs between "
                f"$c = -5\\%$ ({round(agg_neg5, 3)}) and $c = +5\\%$ ({round(agg_pos5, 3)})."
            ),
            used_in=["sec:method-val2"],
        )
        claims.register(
            name=f"Single task suppression L23 {pt}",
            value=suppression,
            statement=(
                f"At layer 23 on {PT_LABEL[pt]} pairs, $c = -5\\%$ drops "
                f"P(pick the steered task) by {suppression:.2f} below the pair-type-matched "
                f"no-steering baseline ({round(agg_b, 3)} → {round(agg_neg5, 3)})."
            ),
            used_in=["sec:method-val2"],
        )
        claims.register(
            name=f"Single task amplification L23 {pt}",
            value=amplification,
            statement=(
                f"At layer 23 on {PT_LABEL[pt]} pairs, $c = +5\\%$ raises "
                f"P(pick the steered task) by {amplification:.2f} above the pair-type-matched "
                f"no-steering baseline ({round(agg_b, 3)} → {round(agg_pos5, 3)})."
            ),
            used_in=["sec:method-val2"],
        )
        print(f"  {pt}: single swing = {single_swing}, supp = {suppression}, amp = {amplification} "
              f"(baseline = {round(agg_b, 3)})")

    sidecar = REPO_ROOT / "paper" / "claims" / "harm_breakdown.json"
    claims.save(sidecar)
    print(f"\nSaved {sidecar.relative_to(REPO_ROOT)} ({len(claims.claims)} claims)")


if __name__ == "__main__":
    main()
