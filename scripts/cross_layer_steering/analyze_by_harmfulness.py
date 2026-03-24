"""Break down steering compliance by how harmful each task actually is.

For harmful-benign pairs: when steered toward the harmful task, does compliance
depend on how harmful the task is? Use baseline refusal rate as a proxy for
task harmfulness.
"""

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

harmful_task_side: dict[str, str] = {}
for p in pairs_list:
    if p["pair_type"] == "harmful_harmful":
        harmful_task_side[p["pair_id"]] = "both"
    else:
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

# Focus on probe_L25, layer 25, harmful-benign pairs only
hb_pairs = {p["pair_id"] for p in pairs_list if p["pair_type"] == "harmful_benign"}
filtered = [r for r in rows if r["condition"] == "probe_L25" and r["layer"] == 25
            and r["pair_id"] in hb_pairs]

# Compute baseline refusal rate per pair as proxy for harmfulness
baseline = [r for r in filtered if r["signed_multiplier"] == 0]
pair_baseline_refusal: dict[str, float] = {}
pair_baseline: dict[str, list] = defaultdict(list)
for r in baseline:
    pair_baseline[r["pair_id"]].append(r)

for pid, rr in pair_baseline.items():
    n_neither = sum(1 for r in rr if r["task_completed"] == "neither")
    pair_baseline_refusal[pid] = n_neither / len(rr)

# Classify pairs by baseline harmfulness
benign_pairs = []  # baseline refusal = 0 (model always completes)
borderline_pairs = []  # baseline refusal > 0 but < 0.5
harmful_pairs = []  # baseline refusal >= 0.5

for pid in hb_pairs:
    if pid not in pair_baseline_refusal:
        continue
    ref_rate = pair_baseline_refusal[pid]
    if ref_rate == 0:
        benign_pairs.append(pid)
    elif ref_rate < 0.5:
        borderline_pairs.append(pid)
    else:
        harmful_pairs.append(pid)

print(f"Pairs by baseline refusal: benign={len(benign_pairs)}, borderline={len(borderline_pairs)}, harmful={len(harmful_pairs)}")

# For each category: when steered toward the harmful task, what happens?
for label, pair_set in [("Benign-acting (0% baseline refusal)", benign_pairs),
                         ("Borderline (>0, <50% baseline refusal)", borderline_pairs),
                         ("Harmful-acting (>=50% baseline refusal)", harmful_pairs)]:
    pair_ids = set(pair_set)
    subset = [r for r in filtered if r["pair_id"] in pair_ids and r["signed_multiplier"] != 0]

    # Determine steering direction
    for r in subset:
        side = harmful_task_side[r["pair_id"]]
        mult = r["signed_multiplier"]
        ordering = r["ordering"]
        if ordering == 0:
            steered_toward = "a" if mult > 0 else "b"
        else:
            steered_toward = "b" if mult > 0 else "a"
        r["_toward_harmful"] = (steered_toward == side)

    toward = [r for r in subset if r["_toward_harmful"]]
    away = [r for r in subset if not r["_toward_harmful"]]

    print(f"\n=== {label} (n_pairs={len(pair_ids)}) ===")

    for direction, rows_d in [("Steered TOWARD harmful", toward), ("Steered AWAY from harmful", away)]:
        if not rows_d:
            continue
        n = len(rows_d)
        did_harmful = sum(1 for r in rows_d if r["task_completed"] == harmful_task_side[r["pair_id"]])
        did_benign_side = sum(1 for r in rows_d
                              if r["task_completed"] != "neither"
                              and r["task_completed"] != harmful_task_side[r["pair_id"]])
        neither = sum(1 for r in rows_d if r["task_completed"] == "neither")
        incoherent = sum(1 for r in rows_d if r.get("compliance") == "incoherent")
        hard_ref = sum(1 for r in rows_d if r.get("compliance") == "hard_refusal")

        print(f"  {direction} (n={n}):")
        print(f"    Did harmful:  {did_harmful/n:.0%}")
        print(f"    Did benign:   {did_benign_side/n:.0%}")
        print(f"    Neither:      {neither/n:.0%} (refusal={hard_ref/n:.0%}, incoherent={incoherent/n:.0%})")

# Show some concrete examples of borderline tasks that get steered into compliance
print("\n\n=== EXAMPLES: Borderline tasks steered into compliance ===")
borderline_set = set(borderline_pairs)
examples = [r for r in filtered
            if r["pair_id"] in borderline_set
            and r["_toward_harmful"]
            and r["task_completed"] == harmful_task_side[r["pair_id"]]
            and r.get("compliance") == "full_comply"
            and abs(r["signed_multiplier"]) >= 0.05]

seen_pairs = set()
for r in examples:
    if r["pair_id"] in seen_pairs:
        continue
    seen_pairs.add(r["pair_id"])
    p = pairs_lookup[r["pair_id"]]
    side = harmful_task_side[r["pair_id"]]
    harmful_text = p["task_a_text"] if side == "a" else p["task_b_text"]
    print(f"\n  pair={r['pair_id']}, mult={r['signed_multiplier']}, baseline_refusal={pair_baseline_refusal[r['pair_id']]:.0%}")
    print(f"  Harmful task: {harmful_text[:200]}")
    print(f"  Response: {r['raw_response'][:200]}")
    if len(seen_pairs) >= 8:
        break
