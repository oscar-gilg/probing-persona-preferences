"""Check what Gemma produces for extreme-tier prompts at baseline (mult=0)."""
import json

lines = open("experiments/steering/open_ended_steering/safety_steering/results.jsonl").readlines()
records = [json.loads(l) for l in lines]

extreme_baseline = [
    r for r in records
    if r.get("tier") == "extreme"
    and r.get("experiment") == 1
    and abs(r.get("multiplier", 99)) < 0.001
]

print(f"Found {len(extreme_baseline)} extreme baseline records\n")
for r in extreme_baseline:
    print(f"=== {r['prompt_id']} trial={r['trial']} mult={r['multiplier']} ===")
    print(f"Prompt: {r['prompt_text'][:100]}...")
    resp = r.get("response", "NO RESPONSE")
    print(f"Response (first 300 chars):\n{resp[:300]}")
    print()
