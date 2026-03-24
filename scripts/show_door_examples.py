import json

with open("experiments/steering/open_ended_steering/scored_results.jsonl") as f:
    rows = [json.loads(line) for line in f]

door = [r for r in rows if r["prompt_id"] == "CRE_01" and r["steering_mode"] == "all_tokens"]

for m in [-0.05, -0.03, 0.0, 0.03, 0.05]:
    matches = [r for r in door if r["multiplier"] == m]
    print(f"\n{'='*60}")
    print(f"mult={m}, n={len(matches)}, engagement={[r['engagement_score'] for r in matches]}")
    if matches:
        print(f"Response (trial 0, first 300 chars):")
        print(matches[0]["response"][:300])
