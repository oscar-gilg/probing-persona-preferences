"""Inspect Qwen completions JSON schema to plan the missing-task extraction."""

import json
from pathlib import Path

path = Path("activations/qwen35_122b/pref_main/completions_with_activations.json")
with open(path) as f:
    completions = json.load(f)

print(f"{len(completions)} completions at {path}")
print("keys (first entry):", list(completions[0].keys())[:15])
has_prompt = "task_prompt" in completions[0]
print(f"has task_prompt: {has_prompt}")
if not has_prompt:
    print("sample entry:", {k: (v if not isinstance(v, str) or len(v) < 80 else v[:80] + '...') for k, v in completions[0].items()})

tids = {c["task_id"] for c in completions}
print(f"unique task_ids: {len(tids)}")
