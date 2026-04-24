"""Compare unilateral vs differential at eot L32."""
import json
from collections import defaultdict
from pathlib import Path

paths = {
    "unilateral": "experiments/layer_sweep/checkpoints/eot_unilateral_L32.parsed.jsonl",
    # differential: we don't have eot L32 diagonal yet (pod 2 still running)
}

# Check if pod 2's eot_L32 diagonal is done yet
eot_diff = Path("experiments/layer_sweep/checkpoints/eot_probe_L32.parsed.jsonl")
if eot_diff.exists():
    paths["differential_eot_L32"] = str(eot_diff)

tb_diff = Path("experiments/layer_sweep/checkpoints/tb-2_probe_L32.parsed.jsonl")
if tb_diff.exists():
    paths["differential_tb2_L32"] = str(tb_diff)

for name, p in paths.items():
    rows = [json.loads(l) for l in Path(p).read_text().splitlines()]
    by_mult = defaultdict(list)
    for r in rows:
        if r["layer"] != 32:
            continue
        by_mult[r["signed_multiplier"]].append(r)

    print(f"\n=== {name} ({p}) ===")
    print(f"{'mult':>7} {'n':>4} {'P(+=a)':>8} {'P(-=b)':>8} {'P(refuse)':>10}")
    for mult in sorted(by_mult):
        group = by_mult[mult]
        # "+ steers toward a" convention: positive mult → expect choice_original=='a'
        intended = "a" if mult > 0 else "b"
        p_steered = sum(r["choice_original"] == intended for r in group) / len(group)
        # Also P(chose a) regardless of sign
        p_a = sum(r["choice_original"] == "a" for r in group) / len(group)
        p_b = sum(r["choice_original"] == "b" for r in group) / len(group)
        p_refuse = sum(r["choice_original"] not in ("a", "b") for r in group) / len(group)
        print(f"{mult:>+7.3f} {len(group):>4} {p_a:>8.3f} {p_b:>8.3f} {p_refuse:>10.3f}")
