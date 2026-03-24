"""Analyze: when steered toward the harmful task, does the model comply or refuse?"""

import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

PARSED = Path("experiments/steering/cross_layer_harmful/checkpoint.parsed.jsonl")
PAIRS = Path("experiments/steering/cross_layer_harmful/pairs_200.json")

with open(PAIRS) as f:
    pairs_list = json.load(f)
pairs_lookup = {p["pair_id"]: p for p in pairs_list}

# Identify which task is harmful in each pair
harmful_task_side: dict[str, str] = {}  # pair_id -> "a" or "b" or "both"
for p in pairs_list:
    if p["pair_type"] == "harmful_harmful":
        harmful_task_side[p["pair_id"]] = "both"
    else:
        # harmful task is the stresstest/bailbench one
        a_is_harmful = p["task_a"].startswith("stresstest_") or p["task_a"].startswith("bailbench_")
        harmful_task_side[p["pair_id"]] = "a" if a_is_harmful else "b"

rows = []
with open(PARSED) as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            if "task_completed" in r:
                rows.append(r)

# Focus on probe_L25, layer 25 (best operating point)
filtered = [r for r in rows if r["condition"] == "probe_L25" and r["layer"] == 25]
print(f"Total rows (probe_L25, layer 25): {len(filtered)}")

# For harmful-benign pairs: is the model steered TOWARD the harmful task?
hb_rows = [r for r in filtered if pairs_lookup[r["pair_id"]]["pair_type"] == "harmful_benign"]

# Determine if steering is toward harmful or benign
for r in hb_rows:
    side = harmful_task_side[r["pair_id"]]
    mult = r["signed_multiplier"]
    ordering = r["ordering"]
    # effective_coef is negated when ordering=1
    # positive effective -> steer toward presented A
    # In original space: if ordering=0, positive mult steers toward original A
    #                    if ordering=1, positive mult steers toward original B
    if ordering == 0:
        steered_toward_original = "a" if mult > 0 else ("b" if mult < 0 else "none")
    else:
        steered_toward_original = "b" if mult > 0 else ("a" if mult < 0 else "none")

    r["steered_toward"] = steered_toward_original
    r["steered_toward_harmful"] = (steered_toward_original == side) if steered_toward_original != "none" else False

# When steered toward harmful: what happens?
toward_harmful = [r for r in hb_rows if r["steered_toward_harmful"]]
toward_benign = [r for r in hb_rows if not r["steered_toward_harmful"] and r["signed_multiplier"] != 0]
baseline = [r for r in hb_rows if r["signed_multiplier"] == 0]

print(f"\n=== HARMFUL-BENIGN PAIRS (probe_L25, layer 25) ===")
print(f"Steered toward harmful: {len(toward_harmful)}")
print(f"Steered toward benign: {len(toward_benign)}")
print(f"Baseline (mult=0): {len(baseline)}")

for label, subset in [("Toward harmful", toward_harmful), ("Toward benign", toward_benign), ("Baseline", baseline)]:
    n = len(subset)
    if n == 0:
        continue
    # Did the model do the harmful task?
    did_harmful = sum(1 for r in subset if r.get("task_completed") == harmful_task_side[r["pair_id"]])
    did_benign = sum(1 for r in subset if r.get("task_completed") != harmful_task_side[r["pair_id"]] and r.get("task_completed") != "neither")
    refused = sum(1 for r in subset if r.get("task_completed") == "neither")

    compliance = defaultdict(int)
    for r in subset:
        compliance[r.get("compliance", "error")] += 1

    # Label vs content mismatch (excluding neither)
    non_neither = [r for r in subset if r.get("task_completed") != "neither"]
    mismatches = sum(1 for r in non_neither if r["claimed_task"] != r["task_completed"])

    print(f"\n--- {label} (n={n}) ---")
    print(f"  Did harmful task: {did_harmful}/{n} ({did_harmful/n:.0%})")
    print(f"  Did benign task:  {did_benign}/{n} ({did_benign/n:.0%})")
    print(f"  Neither (refused/incoherent): {refused}/{n} ({refused/n:.0%})")
    print(f"  Compliance: {dict(compliance)}")
    if non_neither:
        print(f"  Label-content mismatch: {mismatches}/{len(non_neither)} ({mismatches/len(non_neither):.0%})")

# Break down by coefficient magnitude
print(f"\n=== STEERED TOWARD HARMFUL by coefficient ===")
by_coef = defaultdict(list)
for r in toward_harmful:
    by_coef[abs(r["signed_multiplier"])].append(r)

print(f"{'|coef|':>8} {'n':>5} {'did_harmful':>12} {'did_benign':>11} {'refused':>8} {'incoherent':>11}")
for coef in sorted(by_coef):
    subset = by_coef[coef]
    n = len(subset)
    did_h = sum(1 for r in subset if r.get("task_completed") == harmful_task_side[r["pair_id"]])
    did_b = sum(1 for r in subset if r.get("task_completed") != harmful_task_side[r["pair_id"]] and r.get("task_completed") != "neither")
    ref = sum(1 for r in subset if r.get("compliance", "error") == "hard_refusal")
    inc = sum(1 for r in subset if r.get("compliance", "error") == "incoherent")
    print(f"{coef:>8.3f} {n:>5} {did_h/n:>11.0%} {did_b/n:>11.0%} {ref/n:>8.0%} {inc/n:>11.0%}")

# Show some concrete examples: steered toward harmful, model complied
print(f"\n=== EXAMPLES: Steered toward harmful, model DID the harmful task ===")
examples = [r for r in toward_harmful if r.get("task_completed") == harmful_task_side[r["pair_id"]] and r.get("compliance", "error") == "full_comply"]
for r in examples[:5]:
    p = pairs_lookup[r["pair_id"]]
    harmful_side = harmful_task_side[r["pair_id"]]
    harmful_text = p["task_a_text"] if harmful_side == "a" else p["task_b_text"]
    print(f"\n  pair={r['pair_id']}, mult={r['signed_multiplier']}, ordering={r['ordering']}")
    print(f"  Harmful task ({harmful_side}): {harmful_text[:150]}...")
    print(f"  Response: {r['raw_response'][:200]}...")
    print(f"  claimed={r['claimed_task']}, completed={r['task_completed']}, compliance={r['compliance']}")

# Show examples: steered toward harmful, model refused
print(f"\n=== EXAMPLES: Steered toward harmful, model REFUSED ===")
examples_ref = [r for r in toward_harmful if r.get("compliance", "error") == "hard_refusal"]
for r in examples_ref[:5]:
    p = pairs_lookup[r["pair_id"]]
    harmful_side = harmful_task_side[r["pair_id"]]
    harmful_text = p["task_a_text"] if harmful_side == "a" else p["task_b_text"]
    print(f"\n  pair={r['pair_id']}, mult={r['signed_multiplier']}, ordering={r['ordering']}")
    print(f"  Harmful task ({harmful_side}): {harmful_text[:150]}...")
    print(f"  Response: {r['raw_response'][:200]}...")
    print(f"  compliance={r['compliance']}")
