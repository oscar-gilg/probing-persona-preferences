"""Reclassify benign stresstest and bailbench tasks to stresstest_other."""

import json
from collections import Counter
from pathlib import Path

TOPICS_PATH = Path("data/topics/topics.json")

SENSITIVE_TOPICS = {
    "harmful_request",
    "value_conflict",
    "model_manipulation",
    "sensitive_creative",
    "security_legal",
}

with open(TOPICS_PATH) as f:
    topics = json.load(f)

# --- Step 1: Count topics for stresstest and bailbench tasks ---
def get_primary(entry: dict) -> str:
    model_key = next(iter(entry))
    return entry[model_key]["primary"]


stresstest_counts: Counter[str] = Counter()
bailbench_counts: Counter[str] = Counter()

for task_id, entry in topics.items():
    primary = get_primary(entry)
    if task_id.startswith("stresstest_"):
        stresstest_counts[primary] += 1
    elif task_id.startswith("bailbench_"):
        bailbench_counts[primary] += 1

print("=== Stresstest tasks by topic ===")
for topic, count in stresstest_counts.most_common():
    print(f"  {topic:30s} {count}")
print(f"  {'TOTAL':30s} {sum(stresstest_counts.values())}")

print()
print("=== Bailbench tasks by topic ===")
for topic, count in bailbench_counts.most_common():
    print(f"  {topic:30s} {count}")
print(f"  {'TOTAL':30s} {sum(bailbench_counts.values())}")

# --- Step 2 & 3: Reclassify benign topics to stresstest_other ---
reclassified_from: Counter[str] = Counter()
reclassified_count = 0

for task_id, entry in topics.items():
    if not (task_id.startswith("stresstest_") or task_id.startswith("bailbench_")):
        continue
    model_key = next(iter(entry))
    primary = entry[model_key]["primary"]
    if primary not in SENSITIVE_TOPICS:
        reclassified_from[primary] += 1
        reclassified_count += 1
        entry[model_key]["primary"] = "stresstest_other"

print()
print(f"=== Reclassified {reclassified_count} tasks to stresstest_other ===")
for topic, count in reclassified_from.most_common():
    print(f"  {topic:30s} -> stresstest_other  ({count} tasks)")

with open(TOPICS_PATH, "w") as f:
    json.dump(topics, f, sort_keys=False)

print()
print(f"Saved to {TOPICS_PATH}")
