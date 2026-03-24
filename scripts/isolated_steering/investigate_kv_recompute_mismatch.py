"""Investigate the ~50% mismatch between Task A/B label prefix and choice_presented
in the KV recompute checkpoint.

For 20 random valid responses (seed=42, choice_original in "a","b") from the
kv_steering_recompute condition:
1. Print choice_presented, choice_original, ordering, signed_multiplier, first 100 chars
2. Check: does raw_response start with "Task A" or "Task B"? Compare with choice_presented.
3. Verify which task's content the response actually completes by matching to task texts.
"""

import json
import random
from pathlib import Path

# Load checkpoint
CHECKPOINT = Path("experiments/steering/isolated_steering/checkpoint_kv_recompute.jsonl")
PAIRS_FILE = Path("experiments/revealed_steering_v2/followup/pairs_500.json")

rows = []
with open(CHECKPOINT) as f:
    for line in f:
        rows.append(json.loads(line))

# Load pairs for task text lookup
with open(PAIRS_FILE) as f:
    pairs_list = json.load(f)
pair_lookup = {p["pair_id"]: p for p in pairs_list}

# Get unique conditions
conditions = set(r["condition"] for r in rows)
print(f"Total rows: {len(rows)}")
print(f"Conditions: {conditions}")

# Filter to kv_steering_recompute, valid choices only
recompute_rows = [
    r for r in rows
    if r["condition"] == "kv_steering_recompute"
    and r["choice_original"] in ("a", "b")
]
print(f"kv_steering_recompute rows with valid choice: {len(recompute_rows)}")

# Sample 20
random.seed(42)
sample = random.sample(recompute_rows, min(20, len(recompute_rows)))

# -----------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------

def detect_label_prefix(raw: str) -> str | None:
    """What label does the response literally start with?"""
    import re
    cleaned = re.sub(r"^[\s*#_`>]+", "", raw).lower()
    if cleaned.startswith("task a"):
        return "a"
    if cleaned.startswith("task b"):
        return "b"
    return None


def content_matches_task(raw_response: str, task_text: str) -> bool:
    """Heuristic: does the response contain significant words from the task text?"""
    # Take first 5 meaningful words (>3 chars) from task text
    words = [w.lower().strip(".,?!:;\"'()") for w in task_text.split()]
    meaningful = [w for w in words if len(w) > 3][:8]
    if not meaningful:
        return False
    raw_lower = raw_response.lower()
    matches = sum(1 for w in meaningful if w in raw_lower)
    return matches >= max(2, len(meaningful) // 2)


print("\n" + "=" * 120)
print("SAMPLE OF 20 RESPONSES")
print("=" * 120)

mismatch_label_vs_presented = 0
mismatch_content_vs_presented = 0

for i, r in enumerate(sample):
    pair_id = r["pair_id"]
    pair = pair_lookup[pair_id]

    # When ordering=0: presented A = task_a, presented B = task_b
    # When ordering=1: presented A = task_b, presented B = task_a
    ordering = r["ordering"]
    if ordering == 0:
        presented_a_text = pair["task_a_text"]
        presented_b_text = pair["task_b_text"]
    else:
        presented_a_text = pair["task_b_text"]
        presented_b_text = pair["task_a_text"]

    raw = r["raw_response"]
    label_prefix = detect_label_prefix(raw)
    choice_presented = r["choice_presented"]
    choice_original = r["choice_original"]

    # Content matching
    matches_a_content = content_matches_task(raw, presented_a_text)
    matches_b_content = content_matches_task(raw, presented_b_text)
    if matches_a_content and not matches_b_content:
        content_choice = "a"
    elif matches_b_content and not matches_a_content:
        content_choice = "b"
    elif matches_a_content and matches_b_content:
        content_choice = "both"
    else:
        content_choice = "neither"

    label_matches_presented = (label_prefix == choice_presented)
    content_matches_presented = (content_choice == choice_presented)

    if not label_matches_presented and label_prefix is not None:
        mismatch_label_vs_presented += 1
    if not content_matches_presented and content_choice in ("a", "b"):
        mismatch_content_vs_presented += 1

    print(f"\n--- Sample {i+1} ---")
    print(f"  pair_id={pair_id}  ordering={ordering}  signed_mult={r['signed_multiplier']}")
    print(f"  choice_presented={choice_presented}  choice_original={choice_original}")
    print(f"  label_prefix={label_prefix}  label==presented? {label_matches_presented}")
    print(f"  content_match: A={matches_a_content} B={matches_b_content} => content_choice={content_choice}  content==presented? {content_matches_presented}")
    print(f"  Presented A task: {presented_a_text[:80]}...")
    print(f"  Presented B task: {presented_b_text[:80]}...")
    print(f"  raw_response[:150]: {raw[:150]}")

print("\n" + "=" * 120)
print("SUMMARY")
print("=" * 120)
print(f"Label prefix != choice_presented: {mismatch_label_vs_presented}/20")
n_content_determined = sum(1 for r in sample if True)  # will recompute below
print(f"Content != choice_presented (where content determinable): {mismatch_content_vs_presented}")

# -----------------------------------------------------------------------
# Broader statistics on all kv_steering_recompute rows
# -----------------------------------------------------------------------

print("\n" + "=" * 120)
print("FULL DATASET STATISTICS (all kv_steering_recompute valid rows)")
print("=" * 120)

total = len(recompute_rows)
label_a_count = 0
label_b_count = 0
label_none_count = 0
label_matches_presented_count = 0
label_mismatches_presented_count = 0

content_matches_count = 0
content_mismatches_count = 0
content_ambiguous_count = 0

for r in recompute_rows:
    pair = pair_lookup[r["pair_id"]]
    ordering = r["ordering"]
    if ordering == 0:
        pres_a_text = pair["task_a_text"]
        pres_b_text = pair["task_b_text"]
    else:
        pres_a_text = pair["task_b_text"]
        pres_b_text = pair["task_a_text"]

    raw = r["raw_response"]
    lp = detect_label_prefix(raw)
    cp = r["choice_presented"]

    if lp == "a":
        label_a_count += 1
    elif lp == "b":
        label_b_count += 1
    else:
        label_none_count += 1

    if lp is not None:
        if lp == cp:
            label_matches_presented_count += 1
        else:
            label_mismatches_presented_count += 1

    # Content
    ma = content_matches_task(raw, pres_a_text)
    mb = content_matches_task(raw, pres_b_text)
    if ma and not mb:
        cc = "a"
    elif mb and not ma:
        cc = "b"
    else:
        cc = "ambiguous"

    if cc in ("a", "b"):
        if cc == cp:
            content_matches_count += 1
        else:
            content_mismatches_count += 1
    else:
        content_ambiguous_count += 1

print(f"Total valid rows: {total}")
print(f"\nLabel prefix distribution: A={label_a_count} B={label_b_count} None={label_none_count}")
labeled = label_a_count + label_b_count
print(f"Label prefix matches choice_presented: {label_matches_presented_count}/{labeled} ({100*label_matches_presented_count/labeled:.1f}%)")
print(f"Label prefix MISMATCHES choice_presented: {label_mismatches_presented_count}/{labeled} ({100*label_mismatches_presented_count/labeled:.1f}%)")

determinable = content_matches_count + content_mismatches_count
print(f"\nContent heuristic matches choice_presented: {content_matches_count}/{determinable} ({100*content_matches_count/determinable:.1f}%)")
print(f"Content heuristic MISMATCHES choice_presented: {content_mismatches_count}/{determinable} ({100*content_mismatches_count/determinable:.1f}%)")
print(f"Content ambiguous (both or neither): {content_ambiguous_count}")

# -----------------------------------------------------------------------
# Key question: does parse_sync use label prefix or content?
# -----------------------------------------------------------------------

print("\n" + "=" * 120)
print("PARSE LOGIC ANALYSIS")
print("=" * 120)
print("""
parse_sync() in CompletionChoiceFormat does:
  1. _exact_choice_match: response.strip() == "Task A" or "Task B" (almost never matches for completions)
  2. _extract_choice: checks if response STARTS WITH "Task A" or "Task B" (after stripping markdown)
  3. Returns "parse_fail" otherwise (which maps to "refusal" in the runner)

So choice_presented is ENTIRELY based on the label prefix.
The async semantic parser (parse_completion_choice_async) looks at content, but
parse_sync does NOT call it.

This means: when the model writes "Task A:" but completes Task B's content,
choice_presented = "a" even though the model actually chose Task B.
""")

# -----------------------------------------------------------------------
# Concrete examples of label vs content mismatch
# -----------------------------------------------------------------------

print("=" * 120)
print("EXAMPLES: label prefix says one thing, content says another")
print("=" * 120)

examples_found = 0
for r in recompute_rows:
    if examples_found >= 5:
        break
    pair = pair_lookup[r["pair_id"]]
    ordering = r["ordering"]
    if ordering == 0:
        pres_a_text = pair["task_a_text"]
        pres_b_text = pair["task_b_text"]
    else:
        pres_a_text = pair["task_b_text"]
        pres_b_text = pair["task_a_text"]

    raw = r["raw_response"]
    lp = detect_label_prefix(raw)
    ma = content_matches_task(raw, pres_a_text)
    mb = content_matches_task(raw, pres_b_text)

    if lp is not None and ma != mb:  # unambiguous content
        content = "a" if ma else "b"
        if content != lp:
            examples_found += 1
            print(f"\n--- Example {examples_found} ---")
            print(f"  pair_id={r['pair_id']}  ordering={ordering}")
            print(f"  Label prefix: Task {'A' if lp == 'a' else 'B'}")
            print(f"  Content matches: Task {'A' if content == 'a' else 'B'}")
            print(f"  choice_presented={r['choice_presented']}  choice_original={r['choice_original']}")
            print(f"  Task A (as presented): {pres_a_text[:100]}")
            print(f"  Task B (as presented): {pres_b_text[:100]}")
            print(f"  raw_response[:300]: {raw[:300]}")

if examples_found == 0:
    print("  No clear label-vs-content mismatches found with word heuristic.")
    print("  The 50% 'mismatch' may actually just be that choice_presented correctly")
    print("  reflects the label prefix AND the content in most cases.")

# -----------------------------------------------------------------------
# Investigate the "50% mismatch" claim directly
# -----------------------------------------------------------------------

print("\n" + "=" * 120)
print("INVESTIGATING THE '50% MISMATCH' CLAIM")
print("=" * 120)

# The claim is about mismatch between label prefix in raw_response and
# choice_original/choice_presented. Let's look at all the angles.

# 1. choice_presented vs choice_original discrepancy (due to ordering)
same = sum(1 for r in recompute_rows if r["choice_presented"] == r["choice_original"])
diff = sum(1 for r in recompute_rows if r["choice_presented"] != r["choice_original"])
print(f"\nchoice_presented == choice_original: {same}/{total} ({100*same/total:.1f}%)")
print(f"choice_presented != choice_original: {diff}/{total} ({100*diff/total:.1f}%)")
print("  (Difference is expected: _remap_choice flips when ordering=1)")

# 2. Breakdown by ordering
for ord_val in [0, 1]:
    ord_rows = [r for r in recompute_rows if r["ordering"] == ord_val]
    n = len(ord_rows)
    same_o = sum(1 for r in ord_rows if r["choice_presented"] == r["choice_original"])
    print(f"\n  ordering={ord_val}: {n} rows")
    print(f"    choice_presented == choice_original: {same_o}/{n} ({100*same_o/n:.1f}%)")
    print(f"    choice_presented != choice_original: {n - same_o}/{n} ({100*(n-same_o)/n:.1f}%)")

    # What label prefix does the response use?
    pref_a = sum(1 for r in ord_rows if detect_label_prefix(r["raw_response"]) == "a")
    pref_b = sum(1 for r in ord_rows if detect_label_prefix(r["raw_response"]) == "b")
    print(f"    Label prefix: A={pref_a} ({100*pref_a/n:.1f}%) B={pref_b} ({100*pref_b/n:.1f}%)")

# 3. The real question: when ordering=1, the model sees task_b as "Task A"
#    and task_a as "Task B". If it writes "Task A:", it chose what's labelled A
#    in its view (= original task_b). So choice_presented="a" and
#    choice_original = _remap("a", 1) = "b".
#    This is NOT a parsing error — the remap is correct!

print("\n" + "-" * 80)
print("KEY INSIGHT:")
print("-" * 80)
print("""
The '50% mismatch' between label prefix and choice_original is EXPECTED and CORRECT.

When ordering=1:
  - Original task_a is presented as "Task B", task_b as "Task A"
  - If model writes "Task A:", it chose the task labelled A in its view
  - That task is actually task_b in original ordering
  - parse_sync detects "Task A" -> choice_presented = "a"
  - _remap_choice("a", ordering=1) -> choice_original = "b"
  - So choice_original = "b" while the label says "Task A" — this is correct!

The remap correctly inverts back to original task ordering.

Half the data has ordering=0 (labels match originals) and half has ordering=1
(labels are swapped), so ~50% of the time the raw_response label won't match
choice_original. This is by design, not a bug.
""")

# 4. Verify: within each ordering, label prefix should ALWAYS equal choice_presented
for ord_val in [0, 1]:
    ord_rows = [r for r in recompute_rows if r["ordering"] == ord_val]
    n_match = sum(
        1 for r in ord_rows
        if detect_label_prefix(r["raw_response"]) == r["choice_presented"]
    )
    n_mismatch = sum(
        1 for r in ord_rows
        if detect_label_prefix(r["raw_response"]) is not None
        and detect_label_prefix(r["raw_response"]) != r["choice_presented"]
    )
    print(f"ordering={ord_val}: label_prefix==choice_presented in {n_match}/{n_match+n_mismatch} cases ({0 if n_match+n_mismatch==0 else 100*n_match/(n_match+n_mismatch):.1f}%)")

# 5. Check: label prefix in raw_response vs choice_original, by ordering
print("\nLabel prefix vs choice_original breakdown:")
for ord_val in [0, 1]:
    ord_rows = [r for r in recompute_rows if r["ordering"] == ord_val]
    lp_eq_co = sum(1 for r in ord_rows if detect_label_prefix(r["raw_response"]) == r["choice_original"])
    lp_ne_co = sum(1 for r in ord_rows if detect_label_prefix(r["raw_response"]) is not None and detect_label_prefix(r["raw_response"]) != r["choice_original"])
    total_with_lp = lp_eq_co + lp_ne_co
    print(f"  ordering={ord_val}: label_prefix==choice_original in {lp_eq_co}/{total_with_lp} ({0 if total_with_lp==0 else 100*lp_eq_co/total_with_lp:.1f}%)")

# 6. Content vs label: the 6 content mismatches are worth examining
print("\n" + "=" * 120)
print("DETAILED LOOK AT ALL 6 CONTENT vs LABEL MISMATCHES")
print("=" * 120)

mismatch_idx = 0
for r in recompute_rows:
    pair = pair_lookup[r["pair_id"]]
    ordering = r["ordering"]
    if ordering == 0:
        pres_a_text = pair["task_a_text"]
        pres_b_text = pair["task_b_text"]
    else:
        pres_a_text = pair["task_b_text"]
        pres_b_text = pair["task_a_text"]

    raw = r["raw_response"]
    lp = detect_label_prefix(raw)
    ma = content_matches_task(raw, pres_a_text)
    mb = content_matches_task(raw, pres_b_text)

    if lp is not None and ma != mb:
        content = "a" if ma else "b"
        if content != lp:
            mismatch_idx += 1
            print(f"\n--- Content mismatch {mismatch_idx} ---")
            print(f"  pair_id={r['pair_id']}  ordering={ordering}  signed_mult={r['signed_multiplier']}")
            print(f"  Label prefix: Task {'A' if lp == 'a' else 'B'}  Content matches: Task {'A' if content == 'a' else 'B'}")
            print(f"  choice_presented={r['choice_presented']}  choice_original={r['choice_original']}")
            print(f"  Pres A: {pres_a_text[:120]}")
            print(f"  Pres B: {pres_b_text[:120]}")
            print(f"  Response[:400]: {raw[:400]}")
