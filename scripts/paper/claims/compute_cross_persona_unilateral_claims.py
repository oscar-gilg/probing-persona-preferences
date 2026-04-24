"""Register claims from the cross_persona_unilateral experiment (§3.3).

Replaces the older differential cross-persona steering claims. For each of the
6 canonical personas, we register:

  - mean steering swing at |c|=0.05 (bias-adjusted; (first + second)/2)
  - first-span and second-span swings
  - no-steering baseline P(pick first-span) and P(pick second-span)
  - position bias (first − second) at zero steering

Plus a few aggregate claims (strongest/weakest persona, ratio, n personas).

Data sources (cherry-picked to main from the cross_persona_unilateral branch):

  experiments/cross_persona_unilateral/checkpoints/{persona}.parsed.jsonl
      4800 rows per persona from the GPU steering run.

  experiments/cross_persona_unilateral/checkpoints/{persona}_baseline.parsed.jsonl
      600 rows per persona from the API no-steering baseline.

Run:
  python scripts/paper/claims/compute_cross_persona_unilateral_claims.py
"""

from __future__ import annotations

import json
from pathlib import Path

from corroborate import ClaimSet


REPO_ROOT = Path(__file__).resolve().parents[3]
CK = REPO_ROOT / "experiments" / "cross_persona_unilateral" / "checkpoints"
PERSONAS = ["aura", "contrarian", "mathematician", "sadist", "slacker", "strategist"]


def _load(name: str) -> list[dict]:
    return [json.loads(l) for l in (CK / f"{name}.parsed.jsonl").read_text().splitlines() if l.strip()]


def _physical_task(span: str, ordering: int) -> str:
    if span == "first":
        return "a" if ordering == 0 else "b"
    return "b" if ordering == 0 else "a"


def _p_steered(rows: list[dict], cond: str, coef: float) -> float:
    span = "first" if cond == "unilateral_first" else "second"
    hits = n = 0
    for r in rows:
        if r["condition"] != cond:
            continue
        applied = r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1)
        if abs(applied - coef) > 1e-6:
            continue
        target = _physical_task(span, r["ordering"])
        hits += int(r["choice_original"] == target)
        n += 1
    return hits / n


def _baseline_p(rows: list[dict], span: str) -> float:
    hits = n = 0
    for r in rows:
        if r["choice_original"] not in ("a", "b"):
            continue
        target = _physical_task(span, r["ordering"])
        hits += int(r["choice_original"] == target)
        n += 1
    return hits / n


def main() -> None:
    claims = ClaimSet(source="scripts/paper/claims/compute_cross_persona_unilateral_claims.py")

    # Per-persona swings and baselines
    mean_swing: dict[str, float] = {}
    first_swing: dict[str, float] = {}
    second_swing: dict[str, float] = {}
    base_first: dict[str, float] = {}
    base_second: dict[str, float] = {}
    for p in PERSONAS:
        steer = _load(p)
        base = _load(f"{p}_baseline")
        fs = _p_steered(steer, "unilateral_first", 0.05) - _p_steered(steer, "unilateral_first", -0.05)
        ss = _p_steered(steer, "unilateral_second", 0.05) - _p_steered(steer, "unilateral_second", -0.05)
        first_swing[p] = round(fs, 3)
        second_swing[p] = round(ss, 3)
        mean_swing[p] = round((fs + ss) / 2, 3)
        base_first[p] = round(_baseline_p(base, "first"), 3)
        base_second[p] = round(_baseline_p(base, "second"), 3)

    data_paths_steer = [f"experiments/cross_persona_unilateral/checkpoints/{p}.parsed.jsonl" for p in PERSONAS]
    data_paths_base = [f"experiments/cross_persona_unilateral/checkpoints/{p}_baseline.parsed.jsonl" for p in PERSONAS]
    all_data_paths = data_paths_steer + data_paths_base

    claims.register(
        name="Cross persona unilateral mean swing at c 0.05",
        value=mean_swing,
        statement=(
            "Under each of the 6 canonical persona system prompts, unilateral steering along that "
            "persona's own ridge_L25 probe (tb-5 selector, injected at layer 25) produces a mean "
            "swing |P(+5%) − P(−5%)| in P(picked steered task), averaged across first-span and "
            "second-span injection (which removes position bias)."
        ),
        used_in=["sec:shared-steering", "fig:cross-persona-steering"],
        data_paths=data_paths_steer,
        derivation=(
            "For each persona, load {persona}.parsed.jsonl; compute P(picked steered task) per "
            "(condition, coef) where coef = signed_multiplier × (+1 if ordering==0 else −1); "
            "first-span target = a if ordering==0 else b; second-span target = the other; "
            "mean swing = [(first_{+0.05}−first_{−0.05}) + (second_{+0.05}−second_{−0.05})] / 2; "
            "round to 3dp."
        ),
    )

    claims.register(
        name="Cross persona unilateral first span swing at c 0.05",
        value=first_swing,
        statement=(
            "Per-persona first-span unilateral steering swing at |c|=0.05: "
            "P(picked first-span task | +5%) − P(picked first-span task | −5%). "
            "Reflects the mix of probe-driven effect and baseline position bias."
        ),
        used_in=["sec:shared-steering"],
        data_paths=data_paths_steer,
        derivation=(
            "For each persona, load {persona}.parsed.jsonl; compute P(picked first-span task) "
            "at coef=+0.05 and coef=−0.05 for the unilateral_first condition; swing = diff; 3dp."
        ),
    )

    claims.register(
        name="Cross persona unilateral second span swing at c 0.05",
        value=second_swing,
        statement=(
            "Per-persona second-span unilateral steering swing at |c|=0.05: "
            "P(picked second-span task | +5%) − P(picked second-span task | −5%). "
            "Reflects the mix of probe-driven effect and baseline position bias."
        ),
        used_in=["sec:shared-steering"],
        data_paths=data_paths_steer,
        derivation=(
            "For each persona, load {persona}.parsed.jsonl; compute P(picked second-span task) "
            "at coef=+0.05 and coef=−0.05 for the unilateral_second condition; swing = diff; 3dp."
        ),
    )

    claims.register(
        name="Cross persona no steering baseline first span",
        value=base_first,
        statement=(
            "Per-persona empirical baseline P(pick first-presented task) under no steering, "
            "measured via 600 OpenRouter API generations (100 pairs × 2 orderings × 3 trials) "
            "with each persona's system prompt and the completion_preference template."
        ),
        used_in=["sec:shared-steering", "fig:cross-persona-steering"],
        data_paths=data_paths_base,
        derivation=(
            "For each persona, load {persona}_baseline.parsed.jsonl; filter out refusals; "
            "target = first-span physical task (a if ordering==0 else b); "
            "P(first) = mean(choice_original == target); round to 3dp."
        ),
    )

    claims.register(
        name="Cross persona no steering baseline second span",
        value=base_second,
        statement=(
            "Per-persona empirical baseline P(pick second-presented task) under no steering, "
            "measured via 600 OpenRouter API generations per persona."
        ),
        used_in=["sec:shared-steering", "fig:cross-persona-steering"],
        data_paths=data_paths_base,
        derivation=(
            "For each persona, load {persona}_baseline.parsed.jsonl; filter out refusals; "
            "target = second-span physical task (b if ordering==0 else a); "
            "P(second) = mean(choice_original == target); round to 3dp."
        ),
    )

    # Aggregates for narrative
    strongest = max(mean_swing, key=mean_swing.get)
    weakest = min(mean_swing, key=mean_swing.get)
    ratio = round(mean_swing[strongest] / mean_swing[weakest], 2)

    claims.register(
        name="Cross persona unilateral strongest persona",
        value=strongest,
        statement=(
            "The canonical persona with the largest mean unilateral steering swing at |c|=0.05."
        ),
        used_in=["sec:shared-steering"],
        data_paths=data_paths_steer,
        derivation="argmax over mean_swing dict.",
    )

    claims.register(
        name="Cross persona unilateral weakest persona",
        value=weakest,
        statement=(
            "The canonical persona with the smallest (still non-zero) mean unilateral swing at |c|=0.05."
        ),
        used_in=["sec:shared-steering"],
        data_paths=data_paths_steer,
        derivation="argmin over mean_swing dict.",
    )

    claims.register(
        name="Cross persona unilateral swing ratio max over min",
        value=ratio,
        statement=(
            "Ratio of the strongest persona's mean unilateral swing to the weakest persona's mean "
            "unilateral swing at |c|=0.05."
        ),
        used_in=["sec:shared-steering"],
        data_paths=data_paths_steer,
        derivation="mean_swing[strongest] / mean_swing[weakest]; round to 2dp.",
    )

    # Position bias for the narrative around contrarian
    pos_bias_max_persona = max(
        ((p, round(abs(base_first[p] - base_second[p]), 3)) for p in PERSONAS),
        key=lambda kv: kv[1],
    )
    claims.register(
        name="Cross persona max position bias persona",
        value=pos_bias_max_persona[0],
        statement=(
            "The persona with the largest no-steering position bias (|P(first) − P(second)|)."
        ),
        used_in=["sec:shared-steering"],
        data_paths=data_paths_base,
        derivation="argmax |base_first[p] − base_second[p]|.",
    )

    claims.register(
        name="Cross persona max position bias magnitude",
        value=pos_bias_max_persona[1],
        statement=(
            "Maximum persona-level no-steering position bias: |P(first) − P(second)| under the "
            "contrarian prompt. Measured from 600 API generations with no steering."
        ),
        used_in=["sec:shared-steering"],
        data_paths=data_paths_base,
        derivation="max_p |base_first[p] − base_second[p]|; round to 3dp.",
    )

    # Core experimental parameters also worth registering
    claims.register(
        name="Cross persona unilateral n personas",
        value=len(PERSONAS),
        statement=(
            "Number of canonical personas tested in the cross-persona unilateral steering "
            "experiment (aura, contrarian, mathematician, sadist, slacker, strategist)."
        ),
        used_in=["sec:shared-steering"],
        data_paths=data_paths_steer,
        derivation="len(PERSONAS) where PERSONAS is the canonical 6-persona list from persona_sweep_final_six.",
    )

    claims.register(
        name="Cross persona unilateral injection layer",
        value=25,
        statement=(
            "Residual-stream layer at which the per-persona ridge probe direction is injected "
            "for the cross-persona unilateral steering experiment on Gemma-3-27B."
        ),
        used_in=["sec:shared-steering"],
        data_paths=data_paths_steer,
        derivation="Fixed hyperparameter: layer 25 (closest available per-persona probe to the L23 peak from the layer sweep).",
    )

    out_path = REPO_ROOT / "paper" / "claims" / "cross_persona_unilateral.json"
    claims.save(str(out_path))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
