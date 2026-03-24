"""Print all 15 parser test examples in compact format for manual labeling."""

import json

path = "experiments/steering/isolated_steering/full_run/parser_test_examples.json"

with open(path) as f:
    data = json.load(f)

# Part 1: pair_0125 entries
print("=" * 80)
print("PART 1: All entries with pair_id 'pair_0125'")
print("=" * 80)

for category, examples in data.items():
    for ex in examples:
        if ex["pair_id"] == "pair_0125":
            print(f"\nCategory: {category}")
            print(f"task_a_text: {ex['task_a_text']}")
            print(f"task_b_text: {ex['task_b_text']}")
            print(f"choice_presented: {ex['choice_presented']}")
            print(f"raw_response: {ex['raw_response']}")

# Part 2: All 15 examples
print("\n\n" + "=" * 80)
print("PART 2: All 15 examples for manual labeling")
print("=" * 80)

n = 0
for category, examples in data.items():
    for ex in examples:
        n += 1
        print(f"\n--- Example {n} ({ex['pair_id']}, {category}) ---")
        print(f"Task A: {ex['task_a_text']}")
        print(f"Task B: {ex['task_b_text']}")
        print(f"Response: {ex['raw_response']}")
        print(f"choice_presented: {ex['choice_presented']}")

print(f"\nTotal examples: {n}")
