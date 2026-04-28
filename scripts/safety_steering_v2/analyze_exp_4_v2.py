"""Compute disclosure-rate metrics for the exp_4_v2 sweep.

Joins `results.jsonl` (1,820 generations) with `results__judged.jsonl`
(disclosure-judge labels) on `(scenario_id, variant, steering_condition,
multiplier, trial)` and aggregates per (variant, bin_label, steering_condition,
multiplier).

Outputs:
- `aggregated.json` — nested dict keyed by (variant, bin, condition, multiplier)
  with rate_specific / rate_vague / rate_refused / rate_unaware / rate_none plus n.
- `per_scenario.json` — same metrics broken out per scenario_id, for the
  scenario-level bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

JOIN_KEYS = ("scenario_id", "variant", "steering_condition", "multiplier", "trial")
LEVELS = ("specific", "vague", "refused", "unaware", "none")


def _key(r: dict) -> tuple:
    return tuple(round(r[k], 4) if k == "multiplier" else r[k] for k in JOIN_KEYS)


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _rates(labels: list[str]) -> dict:
    n = len(labels)
    if n == 0:
        return {f"rate_{lvl}": 0.0 for lvl in LEVELS} | {"n": 0}
    c = Counter(labels)
    return {f"rate_{lvl}": c.get(lvl, 0) / n for lvl in LEVELS} | {"n": n}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path,
                        default=Path("experiments/safety_steering_v2/exp_4_v2/results.jsonl"),
                        nargs="?")
    parser.add_argument("--judged", type=Path, default=None,
                        help="Path to results__judged.jsonl (defaults to <results stem>__judged.jsonl)")
    args = parser.parse_args()

    results_path = args.results
    judged_path = args.judged or results_path.with_name(results_path.stem + "__judged.jsonl")

    results = _load_jsonl(results_path)
    judged = _load_jsonl(judged_path)

    by_key_results = {_key(r): r for r in results}
    judged_by_key = {_key(r): r for r in judged if "disclosure_level" in r}
    n_errors = sum(1 for r in judged if "judge_error" in r)
    print(f"results: {len(results)} rows · judged: {len(judged)} ({n_errors} errors)")

    missing = set(by_key_results) - set(judged_by_key)
    if missing:
        print(f"WARNING: {len(missing)} results rows have no judgment "
              f"(likely judge errors or empty responses); excluding from aggregation.")

    joined = []
    for k, r in by_key_results.items():
        if k not in judged_by_key:
            continue
        joined.append({**r, "disclosure_level": judged_by_key[k]["disclosure_level"]})

    # Aggregate per (variant, bin_label, steering_condition, multiplier)
    bucket_overall: dict[tuple, list[str]] = defaultdict(list)
    bucket_per_scen: dict[tuple, list[str]] = defaultdict(list)
    for r in joined:
        ko = (r["variant"], r["bin_label"], r["steering_condition"], round(r["multiplier"], 4))
        ks = ko + (r["scenario_id"],)
        bucket_overall[ko].append(r["disclosure_level"])
        bucket_per_scen[ks].append(r["disclosure_level"])

    aggregated = {}
    for (variant, bin_label, cond, mult), labels in bucket_overall.items():
        aggregated.setdefault(variant, {}).setdefault(bin_label, {}).setdefault(
            cond, {})[str(mult)] = _rates(labels)

    per_scenario = {}
    for (variant, bin_label, cond, mult, scen), labels in bucket_per_scen.items():
        per_scenario.setdefault(variant, {}).setdefault(bin_label, {}).setdefault(
            cond, {}).setdefault(str(mult), {})[scen] = _rates(labels)

    out_dir = results_path.parent
    (out_dir / "aggregated.json").write_text(json.dumps(aggregated, indent=2))
    (out_dir / "per_scenario.json").write_text(json.dumps(per_scenario, indent=2))
    print(f"Wrote {out_dir/'aggregated.json'} and {out_dir/'per_scenario.json'}")

    # Print headline numbers: ethical vs benign_twin disclosure_specific at
    # c=0 (no_steering) and c=+0.05 (critical_info_only), per bin.
    def _rate(variant, bin_label, cond, mult):
        try:
            return aggregated[variant][bin_label][cond][str(mult)]
        except KeyError:
            return None

    print("\n--- Headline contrast: disclosure_specific rate ---")
    print(f"{'bin':<12}{'cond':<32}{'c':<8}{'ethical':<10}{'benign':<10}{'gap':<8}")
    for bin_label in ("isolated", "distributed"):
        for cond, mult in [("no_steering", 0.0), ("critical_info_only", 0.05),
                           ("critical_info_only", 0.03), ("critical_info_only", -0.05),
                           ("generation_only", 0.05),
                           ("critical_info_plus_generation", 0.05)]:
            e = _rate("ethical", bin_label, cond, mult)
            b = _rate("benign_twin", bin_label, cond, mult)
            if e is None or b is None:
                continue
            es, bs = e["rate_specific"], b["rate_specific"]
            print(f"{bin_label:<12}{cond:<32}{mult:<8.2f}{es:<10.3f}{bs:<10.3f}{es-bs:<+8.3f}")


if __name__ == "__main__":
    main()
