import json

with open("experiments/steering/open_ended_steering/scored_results.jsonl") as f:
    rows = [json.loads(line) for line in f]

krebs = [r for r in rows if "krebs" in r.get("prompt_text", "").lower() or "krebs" in r.get("prompt_id", "").lower()]
if not krebs:
    krebs = [r for r in rows if "WIL" in r.get("prompt_id", "")]

for pid in sorted(set(r["prompt_id"] for r in krebs)):
    sub = [r for r in krebs if r["prompt_id"] == pid]
    print(f"\n{'='*60}")
    print(f"prompt_id: {pid}")
    print(f"prompt: {sub[0]['prompt_text'][:100]}")
    for m in [-0.05, 0.0, 0.05]:
        matches = [r for r in sub if r["multiplier"] == m and r["steering_mode"] == "all_tokens"]
        if matches:
            print(f"\n  mult={m}, engagement={matches[0]['engagement_score']}")
            print(f"  {matches[0]['response'][:200]}")
