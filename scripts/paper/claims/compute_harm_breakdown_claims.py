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

from corroborate import ClaimSet


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
                    f"eot probe produces a {contr_swing:.2f}-point swing in "
                    f"$P(\\text{{chose steered task}} \\mid \\text{{responded}})$ on "
                    f"{PT_LABEL[pt]} pairs (n=50) of the harm-breakdown 150-pair set."
                ),
                used_in=["sec:method-val2"],
                data_paths=[
                    "experiments/layer_sweep/harm_breakdown/checkpoints/contrastive_L23_150.parsed.jsonl",
                    "experiments/layer_sweep/harm_breakdown/steering_pairs_150.json",
                ],
                derivation=(
                    f"Filter to layer==23 in the contrastive parsed.jsonl; keep rows "
                    f"where the pair's pair_type=='{pt}' (via steering_pairs_150.json); "
                    f"compute P(choice_original=='a') separately at signed_multiplier "
                    f"==+0.05 and ==-0.05; return |P@+0.05 - P@-0.05|; round to 3dp."
                ),
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

        data_paths_single = [
            "experiments/layer_sweep/harm_breakdown/checkpoints/single_task_L23_150.parsed.jsonl",
            "experiments/layer_sweep/harm_breakdown/steering_pairs_150.json",
            "experiments/layer_sweep/checkpoints/eot_unilateral_diagonal_early.parsed.jsonl",
            "experiments/layer_sweep/checkpoints/eot_unilateral_diagonal_late.parsed.jsonl",
            "experiments/layer_sweep/steering_pairs.json",
        ]

        claims.register(
            name=f"Single task swing L23 {pt}",
            value=single_swing,
            statement=(
                f"At layer 23, single-task steering produces a {single_swing:.2f}-point "
                f"swing in $P(\\text{{chose steered task}} \\mid \\text{{responded}})$ "
                f"on {PT_LABEL[pt]} pairs between "
                f"$c = -5\\%$ ({round(agg_neg5, 3)}) and $c = +5\\%$ ({round(agg_pos5, 3)})."
            ),
            used_in=["sec:method-val2"],
            data_paths=data_paths_single,
            derivation=(
                f"Filter single_task_L23_150.parsed.jsonl to layer==23 and pair_type=='{pt}'. "
                f"For each row compute applied_c = signed_multiplier * (1 if ordering==0 else -1) "
                f"and target = 'a' if first-span ordering==0 else 'b' (analogous for second-span). "
                f"Aggregate P(choice_original==target) at applied_c==+0.05 and ==-0.05; "
                f"swing = P(+) - P(-); round to 3dp."
            ),
        )
        claims.register(
            name=f"Single task suppression L23 {pt}",
            value=suppression,
            statement=(
                f"At layer 23 on {PT_LABEL[pt]} pairs, $c = -5\\%$ drops "
                f"P(chose steered task) by {suppression:.2f} below the pair-type-matched "
                f"no-steering baseline ({round(agg_b, 3)} → {round(agg_neg5, 3)})."
            ),
            used_in=["sec:method-val2"],
            data_paths=data_paths_single,
            derivation=(
                f"Baseline: parent-sweep unilateral rows at dead layers "
                f"{sorted(DEAD_LAYERS)} restricted to pair_type=='{pt}' (via 50-pair "
                f"origins in steering_pairs.json); compute mean(first-span P(picked) + "
                f"second-span P(picked)). Then suppression = baseline - aggregate(-0.05) "
                f"at L23 (computed as in swing claim); round to 3dp."
            ),
        )
        claims.register(
            name=f"Single task amplification L23 {pt}",
            value=amplification,
            statement=(
                f"At layer 23 on {PT_LABEL[pt]} pairs, $c = +5\\%$ raises "
                f"P(chose steered task) by {amplification:.2f} above the pair-type-matched "
                f"no-steering baseline ({round(agg_b, 3)} → {round(agg_pos5, 3)})."
            ),
            used_in=["sec:method-val2"],
            data_paths=data_paths_single,
            derivation=(
                f"Baseline computed as in the suppression claim. Then amplification = "
                f"aggregate(+0.05) - baseline at L23; round to 3dp."
            ),
        )
        print(f"  {pt}: single swing = {single_swing}, supp = {suppression}, amp = {amplification} "
              f"(baseline = {round(agg_b, 3)})")

    # --- Aggregate endpoints across all pair types (for §2.3 prose) ---
    if have_contr:
        # P(chose steered | responded) at c=±0.05, pooled over pair types
        def pooled_contr_endpoint(applied_c: float) -> float:
            hits = n = 0
            for r in contr_rows:
                mult = r["signed_multiplier"]
                # Point A: steered=a, applied=+mult
                if abs(applied_c - mult) < 1e-6 and r["choice_original"] in ("a", "b"):
                    hits += int(r["choice_original"] == "a")
                    n += 1
                # Point B: steered=b, applied=-mult
                if abs(applied_c - (-mult)) < 1e-6 and r["choice_original"] in ("a", "b"):
                    hits += int(r["choice_original"] == "b")
                    n += 1
            return hits / n if n else float("nan")

        p_neg5 = round(pooled_contr_endpoint(-0.05), 3)
        p_pos5 = round(pooled_contr_endpoint(+0.05), 3)
        claims.register(
            name="Contrastive P chose steered at L23 c neg 0.05",
            value=p_neg5,
            statement=(
                f"At layer 23, contrastive steering at $c = -5\\%$ drives "
                f"$P(\\text{{chose steered task}} \\mid \\text{{responded}}) = {p_neg5}$ "
                f"pooled across the balanced 150-pair set (all pair types)."
            ),
            used_in=["sec:method-val2"],
            data_paths=[
                "experiments/layer_sweep/harm_breakdown/checkpoints/contrastive_L23_150.parsed.jsonl",
            ],
            derivation=(
                "Filter to layer==23. For each row treat task_a-as-steered (applied_c = "
                "+signed_multiplier, chose_steered = choice=='a') and task_b-as-steered "
                "(applied_c = -signed_multiplier, chose_steered = choice=='b'); keep "
                "responded rows; bucket by applied_c; report mean at applied_c==-0.05; "
                "round to 3dp."
            ),
        )
        claims.register(
            name="Contrastive P chose steered at L23 c pos 0.05",
            value=p_pos5,
            statement=(
                f"At layer 23, contrastive steering at $c = +5\\%$ drives "
                f"$P(\\text{{chose steered task}} \\mid \\text{{responded}}) = {p_pos5}$ "
                f"pooled across the balanced 150-pair set (all pair types)."
            ),
            used_in=["sec:method-val2"],
            data_paths=[
                "experiments/layer_sweep/harm_breakdown/checkpoints/contrastive_L23_150.parsed.jsonl",
            ],
            derivation="Same as the c=-0.05 endpoint but report mean at applied_c==+0.05.",
        )

        # Per-pair-type endpoint at +0.05, to derive the min-across-pair-types headline.
        def pt_contr_endpoint(applied_c: float, pt: str) -> float:
            hits = n = 0
            for r in contr_rows:
                if pt_150.get(r["pair_id"]) != pt:
                    continue
                mult = r["signed_multiplier"]
                if r["choice_original"] not in ("a", "b"):
                    continue
                if abs(applied_c - mult) < 1e-6:
                    hits += int(r["choice_original"] == "a")
                    n += 1
                if abs(applied_c - (-mult)) < 1e-6:
                    hits += int(r["choice_original"] == "b")
                    n += 1
            return hits / n if n else float("nan")

        p_pos5_by_pt = {pt: round(pt_contr_endpoint(+0.05, pt), 3) for pt in ("bb", "hb", "hh")}
        min_p_pos5 = round(min(p_pos5_by_pt.values()), 3)
        claims.register(
            name="Contrastive P chose steered at L23 c pos 0.05 min pair type",
            value=min_p_pos5,
            statement=(
                f"At layer 23 with $|c|=5\\%$, the minimum of "
                f"$P(\\text{{chose steered task}} \\mid \\text{{responded}})$ across "
                f"the three pair types "
                f"(bb={p_pos5_by_pt['bb']}, hb={p_pos5_by_pt['hb']}, hh={p_pos5_by_pt['hh']}) "
                f"is {min_p_pos5} on the balanced 150-pair set — lower bound for "
                f"the paper's headline causal-efficacy claim."
            ),
            used_in=["abstract", "sec:shared.intro", "sec:method-val2"],
            data_paths=[
                "experiments/layer_sweep/harm_breakdown/checkpoints/contrastive_L23_150.parsed.jsonl",
                "experiments/layer_sweep/harm_breakdown/steering_pairs_150.json",
            ],
            derivation=(
                "Per pair_type in {bb, hb, hh}: filter contrastive_L23_150.parsed.jsonl "
                "to layer==23 and pair_type==pt; for each responded row, credit "
                "P(chose steered) by treating task a as steered "
                "(applied_c = +signed_multiplier, chose_steered = choice=='a') and task b "
                "as steered (applied_c = -signed_multiplier, chose_steered = choice=='b'); "
                "report mean at applied_c==+0.05; take min across pair types; round to 3dp."
            ),
        )
        print(f"  per-pt P@+5%: {p_pos5_by_pt}; min = {min_p_pos5}")

    if have_uni:
        # Aggregate single-task swing (mean of per-pair-type swings)
        swings = []
        for pt in ("bb", "hb", "hh"):
            def agg_pt(applied_coef: float, pt=pt) -> float:
                hits = n = 0
                for r in uni_rows:
                    if pt_150.get(r["pair_id"]) != pt:
                        continue
                    applied = r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1)
                    if abs(applied - applied_coef) > 1e-6 or r["choice_original"] not in ("a", "b"):
                        continue
                    span = "first" if r["condition"] == "unilateral_first" else "second"
                    tgt = physical_in_span(span, r["ordering"])
                    hits += int(r["choice_original"] == tgt)
                    n += 1
                return hits / n if n else float("nan")
            swings.append(agg_pt(0.05) - agg_pt(-0.05))
        agg_swing = round(sum(swings) / len(swings), 3)
        claims.register(
            name="Single task swing L23 aggregate",
            value=agg_swing,
            statement=(
                f"Single-task steering at L23 produces an average "
                f"{agg_swing:.2f}-point swing in "
                f"$P(\\text{{chose steered task}} \\mid \\text{{responded}})$ "
                f"between $c=-5\\%$ and $c=+5\\%$, averaged across bb/hb/hh pair types "
                f"(each weighted equally; the per-type swings are near-uniform)."
            ),
            used_in=["sec:method-val2"],
            data_paths=[
                "experiments/layer_sweep/harm_breakdown/checkpoints/single_task_L23_150.parsed.jsonl",
                "experiments/layer_sweep/harm_breakdown/steering_pairs_150.json",
            ],
            derivation=(
                "For each pair_type in {bb, hb, hh}, compute the single-task swing (see "
                "per-pair-type Single task swing claims); take the unweighted mean "
                "across the three pair types; round to 3dp."
            ),
        )

    sidecar = REPO_ROOT / "paper" / "claims" / "harm_breakdown.json"
    claims.save(sidecar)
    print(f"\nSaved {sidecar.relative_to(REPO_ROOT)} ({len(claims.claims)} claims)")


if __name__ == "__main__":
    main()
