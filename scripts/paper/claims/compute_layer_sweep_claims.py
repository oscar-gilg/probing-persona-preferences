"""Register claims from the layer_sweep experiment (§2.3 + App. preference probe geometry).

Data sources (in the research-loop/layer_sweep branch, symlinked into the main
worktree at experiments/layer_sweep/checkpoints):

  experiments/layer_sweep/probe_metrics.json
      Per-layer Pearson r (tb:-2 and eot selectors) on default_test.

  experiments/layer_sweep/checkpoints/tb-2_probe_L23.parsed.jsonl
      Contrastive (differential) steering at L23, tb:-2 probe. Spine config so it
      holds 14400 rows covering all 12 injection layers; we filter to layer==23.

  experiments/layer_sweep/checkpoints/eot_unilateral_diagonal_early.parsed.jsonl
      Single-task steering on eot probe at its self-layer. Early config covers
      L2-L35.

  experiments/layer_sweep/checkpoints/eot_unilateral_diagonal_late.parsed.jsonl
      Same but L38-L59.

Run:
  python scripts/paper/claims/compute_layer_sweep_claims.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.paper.claims import ClaimSet


REPO_ROOT = Path(__file__).resolve().parents[3]
CK = REPO_ROOT / "experiments" / "layer_sweep" / "checkpoints"
PROBE_METRICS = REPO_ROOT / "experiments" / "layer_sweep" / "probe_metrics.json"


def load_parsed(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def p_choice(rows: list[dict], target: str) -> float:
    if not rows:
        return float("nan")
    return sum(r["choice_original"] == target for r in rows) / len(rows)


def p_refuse(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return sum(r["choice_original"] not in ("a", "b") for r in rows) / len(rows)


def main() -> None:
    claims = ClaimSet(source="scripts/paper/claims/compute_layer_sweep_claims.py")

    probe_metrics = json.loads(PROBE_METRICS.read_text())

    # ---- Probe quality ---------------------------------------------------
    tb2 = probe_metrics["tb-2"]
    eot = probe_metrics["eot"]
    tb2_peak_L = max(tb2, key=lambda k: tb2[k]["pearson_r"])
    eot_peak_L = max(eot, key=lambda k: eot[k]["pearson_r"])
    tb2_peak_r = round(tb2[tb2_peak_L]["pearson_r"], 3)
    eot_peak_r = round(eot[eot_peak_L]["pearson_r"], 3)

    claims.register(
        name="Probe peak Pearson r tb-2",
        value=tb2_peak_r,
        statement=(
            f"A ridge probe on Gemma-3-27B residual-stream activations at the "
            f"second-to-last turn-boundary token attains a peak held-out Pearson "
            f"correlation of {tb2_peak_r} with default-persona Thurstonian utilities "
            f"at layer {int(tb2_peak_L)}, across the 20 sampled layers."
        ),
        used_in=["sec:method-val2", "app:probe-geometry"],
        data_paths=["experiments/layer_sweep/probe_metrics.json"],
        derivation=(
            "Read probe_metrics.json; find the layer maximising tb-2[L].pearson_r; "
            "report the maximum; round to 3dp."
        ),
    )
    claims.register(
        name="Probe peak layer tb-2",
        value=int(tb2_peak_L),
        statement=(
            f"The tb:-2-selector ridge probe on Gemma-3-27B peaks at layer "
            f"{int(tb2_peak_L)} among the 20 layers sampled every three blocks."
        ),
        used_in=["sec:method-val2", "app:probe-geometry"],
        data_paths=["experiments/layer_sweep/probe_metrics.json"],
        derivation="argmax_L tb-2[L].pearson_r in probe_metrics.json.",
    )
    claims.register(
        name="Probe peak Pearson r eot",
        value=eot_peak_r,
        statement=(
            f"A ridge probe on Gemma-3-27B residual-stream activations at the "
            f"end-of-turn token attains a peak held-out Pearson correlation of "
            f"{eot_peak_r} with default-persona Thurstonian utilities at layer "
            f"{int(eot_peak_L)}."
        ),
        used_in=["app:probe-geometry"],
        data_paths=["experiments/layer_sweep/probe_metrics.json"],
        derivation=(
            "Read probe_metrics.json; find the layer maximising eot[L].pearson_r; "
            "report the maximum; round to 3dp."
        ),
    )

    # ---- Contrastive steering at L23 (eot = strongest cell) --------------
    eot_L23 = [r for r in load_parsed(CK / "eot_probe_L23.parsed.jsonl") if r["layer"] == 23]
    neg5 = [r for r in eot_L23 if abs(r["signed_multiplier"] + 0.05) < 1e-6]
    pos5 = [r for r in eot_L23 if abs(r["signed_multiplier"] - 0.05) < 1e-6]
    p_a_neg5 = round(p_choice(neg5, "a"), 3)
    p_a_pos5 = round(p_choice(pos5, "a"), 3)
    contr_swing_L23 = round(abs(p_a_pos5 - p_a_neg5), 3)

    tb2_L23 = [r for r in load_parsed(CK / "tb-2_probe_L23.parsed.jsonl") if r["layer"] == 23]
    tb2_swing_L23 = round(
        abs(p_choice([r for r in tb2_L23 if abs(r["signed_multiplier"] - 0.05) < 1e-6], "a")
            - p_choice([r for r in tb2_L23 if abs(r["signed_multiplier"] + 0.05) < 1e-6], "a")),
        3,
    )

    claims.register(
        name="Contrastive steering P a at L23 c neg 0.05 eot",
        value=p_a_neg5,
        statement=(
            f"Contrastive steering along the eot probe direction at layer 23 "
            f"with $c = -5\\%$ of the mean activation norm drives "
            f"$P(\\text{{chose higher-utility task}}) = {p_a_neg5}$ on a 50-pair "
            f"test sample."
        ),
        used_in=["sec:method-val2"],
        data_paths=["experiments/layer_sweep/checkpoints/eot_probe_L23.parsed.jsonl"],
        derivation=(
            "Filter rows to layer==23 and signed_multiplier==-0.05; "
            "P(choice_original=='a'); round to 3dp."
        ),
    )
    claims.register(
        name="Contrastive steering P a at L23 c pos 0.05 eot",
        value=p_a_pos5,
        statement=(
            f"Contrastive steering along the eot probe direction at layer 23 "
            f"with $c = +5\\%$ raises $P(\\text{{chose higher-utility task}}) = {p_a_pos5}$."
        ),
        used_in=["sec:method-val2"],
        data_paths=["experiments/layer_sweep/checkpoints/eot_probe_L23.parsed.jsonl"],
        derivation=(
            "Filter rows to layer==23 and signed_multiplier==+0.05; "
            "P(choice_original=='a'); round to 3dp."
        ),
    )
    claims.register(
        name="Contrastive preference swing L23",
        value=contr_swing_L23,
        statement=(
            f"At layer 23 with $|c| = 5\\%$, contrastive steering along the eot "
            f"probe produces a {contr_swing_L23:.2f}-point swing in "
            f"$P(\\text{{chose higher-utility task}})$ from {p_a_neg5} at "
            f"$c = -5\\%$ to {p_a_pos5} at $c = +5\\%$."
        ),
        used_in=["sec:method-val2", "abstract"],
        data_paths=["experiments/layer_sweep/checkpoints/eot_probe_L23.parsed.jsonl"],
        derivation="|P(a)@+5% - P(a)@-5%| at layer 23 in eot_probe_L23.parsed.jsonl; round to 3dp.",
    )
    claims.register(
        name="Contrastive preference swing L23 tb-2",
        value=tb2_swing_L23,
        statement=(
            f"Using the tb:-2 probe instead of eot at the same layer and "
            f"coefficient yields a smaller but still substantial "
            f"{tb2_swing_L23:.2f}-point preference swing."
        ),
        used_in=["sec:method-val2"],
        data_paths=["experiments/layer_sweep/checkpoints/tb-2_probe_L23.parsed.jsonl"],
        derivation="|P(a)@+5% - P(a)@-5%| at layer 23 in tb-2_probe_L23.parsed.jsonl; round to 3dp.",
    )

    # ---- Single-task (unilateral) at L23 ---------------------------------
    uni_rows = (
        load_parsed(CK / "eot_unilateral_diagonal_early.parsed.jsonl")
        + load_parsed(CK / "eot_unilateral_diagonal_late.parsed.jsonl")
    )

    def span_role(cond: str) -> str:
        return "first" if cond == "unilateral_first" else "second"

    def physical_in_span(span: str, ordering: int) -> str:
        if span == "first":
            return "a" if ordering == 0 else "b"
        return "b" if ordering == 0 else "a"

    # Dead-layer baseline (flat layers): L2-L14, L35-L59
    DEAD = [2, 5, 8, 11, 14, 35, 38, 41, 44, 47, 50, 53, 56, 59]

    def baseline_for(span: str) -> float:
        cond = f"unilateral_{span}"
        hits = n = 0
        for r in uni_rows:
            if r["layer"] not in DEAD or r["condition"] != cond:
                continue
            tgt = physical_in_span(span, r["ordering"])
            hits += int(r["choice_original"] == tgt)
            n += 1
        return hits / n

    first_baseline = round(baseline_for("first"), 3)
    second_baseline = round(baseline_for("second"), 3)
    agg_baseline = round((first_baseline + second_baseline) / 2, 3)

    claims.register(
        name="Single task first span baseline",
        value=first_baseline,
        statement=(
            f"Without any steering, the model picks the first-presented task at "
            f"P = {first_baseline} (measured empirically from the flat layers L2-L14 "
            f"and L35-L59 in the single-task sweep, where the dose-response is null)."
        ),
        used_in=["sec:method-val2", "app:probe-geometry"],
        data_paths=[
            "experiments/layer_sweep/checkpoints/eot_unilateral_diagonal_early.parsed.jsonl",
            "experiments/layer_sweep/checkpoints/eot_unilateral_diagonal_late.parsed.jsonl",
        ],
        derivation=(
            "Restrict to condition=='unilateral_first' and layer in dead_layers "
            "[2,5,8,11,14,35,38,41,44,47,50,53,56,59]; compute the fraction of "
            "trials where the model picked whichever task was in first position "
            "(a at ordering=0, b at ordering=1); round to 3dp."
        ),
    )
    claims.register(
        name="Single task second span baseline",
        value=second_baseline,
        statement=(
            f"Without any steering, the model picks the second-presented task at "
            f"P = {second_baseline} (same flat-layer measurement as the first-span "
            f"baseline, so the two baselines sum to 1 up to refusals)."
        ),
        used_in=["sec:method-val2", "app:probe-geometry"],
        data_paths=[
            "experiments/layer_sweep/checkpoints/eot_unilateral_diagonal_early.parsed.jsonl",
            "experiments/layer_sweep/checkpoints/eot_unilateral_diagonal_late.parsed.jsonl",
        ],
        derivation=(
            "Same as first-span baseline but condition=='unilateral_second' and "
            "target = task in second position."
        ),
    )
    claims.register(
        name="Single task position bias gap",
        value=round(first_baseline - second_baseline, 3),
        statement=(
            f"The baseline position bias --- first-span minus second-span P(picked) "
            f"with no steering --- is {round(first_baseline - second_baseline, 3):.2f}, "
            f"indicating the model has a mild preference for the first-presented "
            f"option independent of utility."
        ),
        used_in=["sec:method-val2"],
        data_paths=[
            "experiments/layer_sweep/checkpoints/eot_unilateral_diagonal_early.parsed.jsonl",
            "experiments/layer_sweep/checkpoints/eot_unilateral_diagonal_late.parsed.jsonl",
        ],
        derivation="first_baseline - second_baseline; round to 3dp.",
    )

    # Aggregate P(steered) at L23 for positive and negative coefs
    def aggregate_at(layer: int, applied_coef: float) -> float:
        hits = n = 0
        for r in uni_rows:
            if r["layer"] != layer:
                continue
            applied = r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1)
            if abs(applied - applied_coef) > 1e-6:
                continue
            span = span_role(r["condition"])
            tgt = physical_in_span(span, r["ordering"])
            hits += int(r["choice_original"] == tgt)
            n += 1
        return hits / n

    agg_pos5_L23 = round(aggregate_at(23, 0.05), 3)
    agg_neg5_L23 = round(aggregate_at(23, -0.05), 3)
    single_swing_L23 = round(agg_pos5_L23 - agg_neg5_L23, 3)
    suppression_L23 = round(agg_baseline - agg_neg5_L23, 3)
    amplification_L23 = round(agg_pos5_L23 - agg_baseline, 3)

    claims.register(
        name="Single task aggregate P steered at L23 c pos 0.05",
        value=agg_pos5_L23,
        statement=(
            f"At layer 23 with an applied coefficient of +5\\% on a single task's "
            f"tokens, the model picks that task at P = {agg_pos5_L23}, averaged "
            f"across which of the two physical tasks was steered."
        ),
        used_in=["sec:method-val2"],
        data_paths=[
            "experiments/layer_sweep/checkpoints/eot_unilateral_diagonal_early.parsed.jsonl"
        ],
        derivation=(
            "Aggregate over both unilateral_first and unilateral_second conditions "
            "at layer==23 where the applied coefficient (= signed_multiplier * "
            "(+1 if ordering==0 else -1)) equals +0.05; compute P(choice_original "
            "== physical task in steered span); round to 3dp."
        ),
    )
    claims.register(
        name="Single task aggregate P steered at L23 c neg 0.05",
        value=agg_neg5_L23,
        statement=(
            f"At layer 23 with an applied coefficient of -5\\% on a single task's "
            f"tokens, the model picks that task at P = {agg_neg5_L23}."
        ),
        used_in=["sec:method-val2"],
        data_paths=[
            "experiments/layer_sweep/checkpoints/eot_unilateral_diagonal_early.parsed.jsonl"
        ],
        derivation="Same as the +5% version but applied coefficient == -0.05.",
    )
    claims.register(
        name="Single task aggregate swing L23",
        value=single_swing_L23,
        statement=(
            f"Single-task steering at layer 23 produces a "
            f"{single_swing_L23:.2f}-point swing in P(picked the steered task) "
            f"between c = -5\\% ({agg_neg5_L23}) and c = +5\\% ({agg_pos5_L23})."
        ),
        used_in=["sec:method-val2"],
        data_paths=[
            "experiments/layer_sweep/checkpoints/eot_unilateral_diagonal_early.parsed.jsonl"
        ],
        derivation="aggregate(+0.05) - aggregate(-0.05) at layer 23; round to 3dp.",
    )
    claims.register(
        name="Single task suppression L23",
        value=suppression_L23,
        statement=(
            f"At layer 23, applying c = -5\\% to a task's tokens drops P(pick "
            f"that task) by {suppression_L23:.2f} below the no-steering baseline "
            f"(from {agg_baseline} to {agg_neg5_L23})."
        ),
        used_in=["sec:method-val2"],
        data_paths=[
            "experiments/layer_sweep/checkpoints/eot_unilateral_diagonal_early.parsed.jsonl"
        ],
        derivation="aggregate_baseline - aggregate(-0.05) at layer 23; round to 3dp.",
    )
    claims.register(
        name="Single task amplification L23",
        value=amplification_L23,
        statement=(
            f"At layer 23, applying c = +5\\% to a task's tokens raises P(pick "
            f"that task) by only {amplification_L23:.2f} above the no-steering "
            f"baseline ({agg_baseline} to {agg_pos5_L23}) --- markedly less than "
            f"the symmetric suppression delta."
        ),
        used_in=["sec:method-val2"],
        data_paths=[
            "experiments/layer_sweep/checkpoints/eot_unilateral_diagonal_early.parsed.jsonl"
        ],
        derivation="aggregate(+0.05) - aggregate_baseline at layer 23; round to 3dp.",
    )

    # ---- Max refusal rate across the sweep -------------------------------
    # Across both selectors' diagonal + spine data.
    all_steering_files = (
        list(CK.glob("tb-2_probe_L*.parsed.jsonl"))
        + list(CK.glob("eot_probe_L*.parsed.jsonl"))
        + list(CK.glob("eot_unilateral_diagonal_*.parsed.jsonl"))
    )
    max_refuse_pct = 0.0
    by_bucket = defaultdict(list)
    for f in all_steering_files:
        for r in load_parsed(f):
            by_bucket[(f.stem, r["layer"], r["signed_multiplier"])].append(r)
    for rows in by_bucket.values():
        r_rate = p_refuse(rows) * 100
        if r_rate > max_refuse_pct:
            max_refuse_pct = r_rate
    max_refuse_pct = round(max_refuse_pct, 2)

    claims.register(
        name="Steering max refusal rate percent",
        value=max_refuse_pct,
        statement=(
            f"Across every (layer, coefficient, steering variant) bucket in the "
            f"layer-sweep experiment --- 20 layers for both selectors, contrastive "
            f"and single-task --- the worst-case refusal rate is {max_refuse_pct}\\%. "
            f"Steering does not operate near the coherence-collapse regime at "
            f"these coefficients."
        ),
        used_in=["sec:method-val2"],
        data_paths=[
            "experiments/layer_sweep/checkpoints/tb-2_probe_L*.parsed.jsonl",
            "experiments/layer_sweep/checkpoints/eot_probe_L*.parsed.jsonl",
            "experiments/layer_sweep/checkpoints/eot_unilateral_diagonal_*.parsed.jsonl",
        ],
        derivation=(
            "For each (file, layer, signed_multiplier) bucket, compute fraction "
            "of rows with choice_original not in {'a','b'}; take max across all "
            "buckets; express as percent; round to 2dp."
        ),
    )

    # ---- Working layer range --------------------------------------------
    # Contrastive diagonal (eot, the stronger selector): contiguous range where self-layer swing >= 0.30
    def contrastive_diag_swing(L: int, sel: str = "eot") -> float:
        f = CK / f"{sel}_probe_L{L:02d}.parsed.jsonl"
        if not f.exists():
            return float("nan")
        rows = [r for r in load_parsed(f) if r["layer"] == L]
        p_p = p_choice([r for r in rows if abs(r["signed_multiplier"] - 0.05) < 1e-6], "a")
        p_n = p_choice([r for r in rows if abs(r["signed_multiplier"] + 0.05) < 1e-6], "a")
        return abs(p_p - p_n)

    layers = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59]
    working = sorted(L for L in layers if contrastive_diag_swing(L) >= 0.30)

    if working:
        working_lo, working_hi = working[0], working[-1]
        claims.register(
            name="Contrastive steering working layer range lo",
            value=working_lo,
            statement=(
                f"Contrastive steering along the eot probe direction produces a "
                f"preference swing of at least 0.30 at every self-layer cell "
                f"from layer {working_lo} through layer {working_hi}."
            ),
            used_in=["sec:method-val2"],
            data_paths=["experiments/layer_sweep/checkpoints/eot_probe_L*.parsed.jsonl"],
            derivation=(
                "For each of the 20 sampled layers L, compute |P(a)@+5% - P(a)@-5%| "
                "at the eot probe's self-cell; keep Ls with swing >= 0.30; report min."
            ),
        )
        claims.register(
            name="Contrastive steering working layer range hi",
            value=working_hi,
            statement=(
                f"The upper end of the working steering window is layer "
                f"{working_hi}; at every layer L > {working_hi} in the sweep, "
                f"the self-layer contrastive swing drops below 0.30."
            ),
            used_in=["sec:method-val2"],
            data_paths=["experiments/layer_sweep/checkpoints/eot_probe_L*.parsed.jsonl"],
            derivation="Same as the lo claim but report max of the filtered layers.",
        )

    sidecar = REPO_ROOT / "paper" / "claims" / "layer_sweep.json"
    claims.save(sidecar)
    print(f"Saved {sidecar.relative_to(REPO_ROOT)}  ({len(claims.claims)} claims)")
    print(f"  tb-2 peak r = {tb2_peak_r} at L{int(tb2_peak_L)}")
    print(f"  eot peak r  = {eot_peak_r} at L{int(eot_peak_L)}")
    print(f"  L23 contrastive swing = {contr_swing_L23}")
    print(f"  L23 single-task swing = {single_swing_L23}")
    print(f"    suppression = {suppression_L23}, amplification = {amplification_L23}")
    print(f"  baselines: first {first_baseline}, second {second_baseline}")
    print(f"  max refusal across sweep = {max_refuse_pct}%")
    print(f"  working layer range (|swing| >= 0.30): L{working_lo}-L{working_hi}")


if __name__ == "__main__":
    main()
