"""Compute summary statistics for one-sided steering decomposition."""

import json
from pathlib import Path

from src.steering.analysis import load_checkpoint, compute_p_steered


def analyze_dataset(name: str, parsed_path: Path):
    rows = load_checkpoint(parsed_path)
    print(f"\n{'='*60}")
    print(f"Dataset: {name} ({len(rows)} rows)")
    print(f"{'='*60}")

    # Compute p_steered using task_completed (judge content)
    results = compute_p_steered(rows, group_by=["condition"], choice_field="task_completed")

    # Print sigmoid table
    for cond in ["steer_first_L25", "steer_second_L25", "differential_L25"]:
        cond_rows = [r for r in results if r["condition"] == cond]
        print(f"\n{cond}:")
        for r in cond_rows:
            mult = r["signed_multiplier"]
            p = r["p_steered"]
            n = r["n_total"]
            neither = r["n_neither"]
            print(f"  mult={mult:+.3f}: P(steered)={p:.3f} (n={n}, neither={neither})")

    # Additivity test
    print(f"\nAdditivity test:")
    first_map = {r["signed_multiplier"]: r["p_steered"]
                 for r in results if r["condition"] == "steer_first_L25"}
    second_map = {r["signed_multiplier"]: r["p_steered"]
                  for r in results if r["condition"] == "steer_second_L25"}
    diff_map = {r["signed_multiplier"]: r["p_steered"]
                for r in results if r["condition"] == "differential_L25"}

    abs_devs = []
    for mult in sorted(first_map.keys()):
        if mult == 0:
            continue
        additive_pred = first_map[mult] + second_map[mult] - 0.5
        actual = diff_map[mult]
        dev = actual - additive_pred
        abs_devs.append(abs(dev))
        print(f"  mult={mult:+.3f}: first={first_map[mult]:.3f} + second={second_map[mult]:.3f} - 0.5 "
              f"= {additive_pred:.3f} vs diff={actual:.3f} (dev={dev:+.3f})")

    mad = sum(abs_devs) / len(abs_devs) if abs_devs else 0
    print(f"\n  MAD = {mad:.4f} ({'PASS' if mad < 0.05 else 'FAIL'} threshold=0.05)")

    # Also compute with choice_original for comparison
    results_regex = compute_p_steered(rows, group_by=["condition"], choice_field="choice_original")
    print(f"\nRegex label comparison (choice_original):")
    for cond in ["differential_L25"]:
        cond_rows = [r for r in results_regex if r["condition"] == cond]
        for r in cond_rows:
            mult = r["signed_multiplier"]
            p = r["p_steered"]
            p_judge = diff_map.get(mult, float("nan"))
            print(f"  mult={mult:+.3f}: regex={p:.3f} vs judge={p_judge:.3f}")

    # Save results as JSON for plotting
    output = {
        "name": name,
        "n_rows": len(rows),
        "results_judge": results,
        "results_regex": results_regex,
        "mad": mad,
    }
    out_path = parsed_path.parent / f"analysis_{name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")


base = Path("experiments/steering/one_sided")
analyze_dataset("harmful", base / "checkpoint_harmful.parsed.jsonl")
analyze_dataset("benign", base / "checkpoint_benign.parsed.jsonl")
