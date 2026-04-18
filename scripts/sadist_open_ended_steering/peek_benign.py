"""Peek at a benign-tier sadist response at c=+0.03 to verify sadism isn't just anti-refusal."""
import json

with open("experiments/sadist_open_ended_steering/judged_sadist.jsonl") as f:
    rows = [json.loads(l) for l in f if l.strip()]

rows = [r for r in rows if r.get("tier") == "benign" and r.get("multiplier") == 0.03 and "judge_error" not in r]
rows.sort(key=lambda r: (-r["sadism_score"], -r["default_assistant_score"]))
for r in rows[:2]:
    print(f"\n--- {r['prompt_id']} | sadism={r['sadism_score']} default={r['default_assistant_score']} ---")
    print(f"PROMPT: {r['prompt_text']}")
    print(f"RESPONSE: {r['response'][:900]}")
