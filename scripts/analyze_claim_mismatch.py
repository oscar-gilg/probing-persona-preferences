"""Analyze claimed_task vs task_completed mismatch pattern in steering results."""

from dotenv import load_dotenv

load_dotenv()

import json
from collections import defaultdict
from pathlib import Path

FILES = [
    Path("/Users/oscargilg/Dev/MATS/Preferences/experiments/steering/cross_layer/checkpoint_L25.parsed.jsonl"),
    Path("/Users/oscargilg/Dev/MATS/Preferences/experiments/steering/cross_layer_harmful/checkpoint.parsed.jsonl"),
]


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    skipped = 0
    for path in paths:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                # Skip records that failed parsing (have 'error' instead of 'task_completed')
                if "task_completed" not in row:
                    skipped += 1
                    continue
                row["source_file"] = path.name
                records.append(row)
    print(f"Skipped {skipped} records missing 'task_completed' (parse errors).")
    return records


def part1_mismatch_rates(records: list[dict]) -> None:
    """For each signed_multiplier, count mismatches as fraction of non-neither completions."""
    # Bucket by signed_multiplier
    by_mult: dict[float, list[dict]] = defaultdict(list)
    for r in records:
        by_mult[r["signed_multiplier"]].append(r)

    print("=" * 80)
    print("PART 1: claimed_task != task_completed rate by signed_multiplier")
    print("       (excluding rows where task_completed == 'neither')")
    print("=" * 80)
    print(f"{'multiplier':>12}  {'mismatch':>8}  {'non-neither':>11}  {'rate':>8}")
    print("-" * 50)

    for mult in sorted(by_mult.keys()):
        rows = by_mult[mult]
        non_neither = [r for r in rows if r["task_completed"] != "neither"]
        mismatches = [r for r in non_neither if r["claimed_task"] != r["task_completed"]]
        n_non_neither = len(non_neither)
        n_mismatch = len(mismatches)
        rate = n_mismatch / n_non_neither if n_non_neither > 0 else 0.0
        print(f"{mult:>12.3f}  {n_mismatch:>8}  {n_non_neither:>11}  {rate:>8.3f}")

    print()


def part2_concrete_examples(records: list[dict]) -> None:
    """Find 3-5 examples at nonzero multipliers where claimed != completed and compliance is full_comply."""
    candidates = [
        r
        for r in records
        if r["signed_multiplier"] != 0.0
        and r["task_completed"] != "neither"
        and r["claimed_task"] != r["task_completed"]
        and r["compliance"] == "full_comply"
    ]

    print("=" * 80)
    print("PART 2: Concrete examples (nonzero multiplier, full_comply, claimed != completed)")
    print(f"       Total candidates: {len(candidates)}")
    print("=" * 80)

    # Pick examples spread across different multipliers
    # Sort by absolute multiplier descending to get strong steering examples first
    candidates.sort(key=lambda r: abs(r["signed_multiplier"]), reverse=True)

    # Deduplicate by pair_id to show variety
    seen_pairs = set()
    selected = []
    for r in candidates:
        if r["pair_id"] not in seen_pairs:
            seen_pairs.add(r["pair_id"])
            selected.append(r)
        if len(selected) >= 5:
            break

    for i, r in enumerate(selected, 1):
        print(f"\n--- Example {i} (source: {r['source_file']}) ---")
        print(f"  pair_id:          {r['pair_id']}")
        print(f"  signed_multiplier: {r['signed_multiplier']}")
        print(f"  layer:            {r['layer']}")
        print(f"  task_a_id:        {r['task_a_id']}")
        print(f"  task_b_id:        {r['task_b_id']}")
        print(f"  claimed_task:     {r['claimed_task']}")
        print(f"  task_completed:   {r['task_completed']}")
        print(f"  compliance:       {r['compliance']}")
        raw = r["raw_response"]
        truncated = raw[:200] + "..." if len(raw) > 200 else raw
        print(f"  raw_response:     {truncated}")

    print()


def part3_breakdown_by_source(records: list[dict]) -> None:
    """Break down mismatch rates by source file and multiplier."""
    print("=" * 80)
    print("PART 3: Mismatch rates by source file")
    print("=" * 80)

    for source in sorted(set(r["source_file"] for r in records)):
        source_records = [r for r in records if r["source_file"] == source]
        print(f"\n--- {source} ({len(source_records)} records) ---")

        by_mult: dict[float, list[dict]] = defaultdict(list)
        for r in source_records:
            by_mult[r["signed_multiplier"]].append(r)

        print(f"{'multiplier':>12}  {'mismatch':>8}  {'non-neither':>11}  {'rate':>8}  {'claimed_neither':>15}")
        print("-" * 65)

        for mult in sorted(by_mult.keys()):
            rows = by_mult[mult]
            non_neither = [r for r in rows if r["task_completed"] != "neither"]
            mismatches = [r for r in non_neither if r["claimed_task"] != r["task_completed"]]
            claimed_neither = [r for r in non_neither if r["claimed_task"] == "neither"]
            n_non_neither = len(non_neither)
            n_mismatch = len(mismatches)
            rate = n_mismatch / n_non_neither if n_non_neither > 0 else 0.0
            print(f"{mult:>12.3f}  {n_mismatch:>8}  {n_non_neither:>11}  {rate:>8.3f}  {len(claimed_neither):>15}")

    print()


def part4_mismatch_excluding_claimed_neither(records: list[dict]) -> None:
    """Mismatch rates excluding both task_completed=='neither' AND claimed_task=='neither'."""
    print("=" * 80)
    print("PART 4: Mismatch rates excluding claimed_task=='neither' AND task_completed=='neither'")
    print("       (only rows where both claimed and completed are 'a' or 'b')")
    print("=" * 80)

    by_mult: dict[float, list[dict]] = defaultdict(list)
    for r in records:
        by_mult[r["signed_multiplier"]].append(r)

    print(f"{'multiplier':>12}  {'mismatch':>8}  {'both_ab':>8}  {'rate':>8}")
    print("-" * 45)

    for mult in sorted(by_mult.keys()):
        rows = by_mult[mult]
        both_ab = [
            r for r in rows
            if r["task_completed"] in ("a", "b") and r["claimed_task"] in ("a", "b")
        ]
        mismatches = [r for r in both_ab if r["claimed_task"] != r["task_completed"]]
        n = len(both_ab)
        n_mismatch = len(mismatches)
        rate = n_mismatch / n if n > 0 else 0.0
        print(f"{mult:>12.3f}  {n_mismatch:>8}  {n:>8}  {rate:>8.3f}")

    print()


def part5_direction_analysis(records: list[dict]) -> None:
    """For mismatches, check if steering direction predicts which task gets completed."""
    print("=" * 80)
    print("PART 5: Direction analysis — does steering shift WHICH task gets completed?")
    print("       For mismatches: does the model claim the low-mu task but complete the high-mu task?")
    print("       (only full_comply, both claimed and completed in {a,b})")
    print("=" * 80)

    by_mult: dict[float, list[dict]] = defaultdict(list)
    for r in records:
        by_mult[r["signed_multiplier"]].append(r)

    print(f"{'multiplier':>12}  {'n_mismatch':>10}  {'completed_higher_mu':>19}  {'completed_lower_mu':>18}  {'frac_higher':>11}")
    print("-" * 80)

    for mult in sorted(by_mult.keys()):
        rows = by_mult[mult]
        mismatches = [
            r for r in rows
            if r["compliance"] == "full_comply"
            and r["claimed_task"] in ("a", "b")
            and r["task_completed"] in ("a", "b")
            and r["claimed_task"] != r["task_completed"]
        ]

        completed_higher = 0
        completed_lower = 0
        for r in mismatches:
            delta = r["delta_mu"]  # mu_a - mu_b (positive means a is higher utility)
            completed = r["task_completed"]
            # "higher mu" task: a if delta > 0, b if delta < 0
            if delta > 0:
                higher_mu_task = "a"
            elif delta < 0:
                higher_mu_task = "b"
            else:
                continue
            if completed == higher_mu_task:
                completed_higher += 1
            else:
                completed_lower += 1

        total = completed_higher + completed_lower
        frac = completed_higher / total if total > 0 else 0.0
        print(f"{mult:>12.3f}  {total:>10}  {completed_higher:>19}  {completed_lower:>18}  {frac:>11.3f}")

    print()


def part6_overall_completion_direction(records: list[dict]) -> None:
    """For ALL completions (not just mismatches), which task gets completed — higher or lower mu?"""
    print("=" * 80)
    print("PART 6: Overall completion direction (all non-neither task_completed, full_comply)")
    print("       Does steering shift which task gets completed overall?")
    print("=" * 80)

    by_mult: dict[float, list[dict]] = defaultdict(list)
    for r in records:
        by_mult[r["signed_multiplier"]].append(r)

    print(f"{'multiplier':>12}  {'n':>6}  {'completed_higher_mu':>19}  {'frac_higher':>11}  {'claimed_higher_mu':>17}  {'frac_claimed':>12}")
    print("-" * 90)

    for mult in sorted(by_mult.keys()):
        rows = by_mult[mult]
        valid = [
            r for r in rows
            if r["compliance"] == "full_comply"
            and r["task_completed"] in ("a", "b")
            and r["claimed_task"] in ("a", "b")
        ]

        completed_higher = 0
        claimed_higher = 0
        total = 0
        for r in valid:
            delta = r["delta_mu"]
            if delta == 0:
                continue
            total += 1
            higher_mu_task = "a" if delta > 0 else "b"
            if r["task_completed"] == higher_mu_task:
                completed_higher += 1
            if r["claimed_task"] == higher_mu_task:
                claimed_higher += 1

        frac_completed = completed_higher / total if total > 0 else 0.0
        frac_claimed = claimed_higher / total if total > 0 else 0.0
        print(f"{mult:>12.3f}  {total:>6}  {completed_higher:>19}  {frac_completed:>11.3f}  {claimed_higher:>17}  {frac_claimed:>12.3f}")

    print()


def part7_claimed_neither_check(records: list[dict]) -> None:
    """Check how many records have claimed_task=='neither'."""
    n_claimed_neither = sum(1 for r in records if r["claimed_task"] == "neither")
    n_completed_neither = sum(1 for r in records if r["task_completed"] == "neither")
    print("=" * 80)
    print("PART 7: 'neither' counts")
    print("=" * 80)
    print(f"  claimed_task == 'neither':   {n_claimed_neither} / {len(records)}")
    print(f"  task_completed == 'neither': {n_completed_neither} / {len(records)}")

    # Cross-tabulation
    both_neither = sum(1 for r in records if r["claimed_task"] == "neither" and r["task_completed"] == "neither")
    claimed_neither_completed_ab = sum(
        1 for r in records
        if r["claimed_task"] == "neither" and r["task_completed"] in ("a", "b")
    )
    claimed_ab_completed_neither = sum(
        1 for r in records
        if r["claimed_task"] in ("a", "b") and r["task_completed"] == "neither"
    )
    both_ab = sum(
        1 for r in records
        if r["claimed_task"] in ("a", "b") and r["task_completed"] in ("a", "b")
    )
    print(f"\n  Cross-tabulation:")
    print(f"    both 'neither':            {both_neither}")
    print(f"    claimed='neither', completed=a/b: {claimed_neither_completed_ab}")
    print(f"    claimed=a/b, completed='neither': {claimed_ab_completed_neither}")
    print(f"    both a/b:                  {both_ab}")
    print()


def main() -> None:
    records = load_records(FILES)
    print(f"Loaded {len(records)} total records from {len(FILES)} files.\n")

    # Quick summary of field values
    claimed_vals = sorted(set(r["claimed_task"] for r in records))
    completed_vals = sorted(set(r["task_completed"] for r in records))
    compliance_vals = sorted(set(r["compliance"] for r in records))
    print(f"claimed_task values:   {claimed_vals}")
    print(f"task_completed values: {completed_vals}")
    print(f"compliance values:     {compliance_vals}")
    print()

    part1_mismatch_rates(records)
    part2_concrete_examples(records)
    part3_breakdown_by_source(records)
    part4_mismatch_excluding_claimed_neither(records)
    part5_direction_analysis(records)
    part6_overall_completion_direction(records)
    part7_claimed_neither_check(records)


if __name__ == "__main__":
    main()
