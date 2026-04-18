"""Pull a sadist c=+0.03 response on an open-ended self-reflection prompt."""
import json

rows = []
with open("experiments/sadist_open_ended_steering/judged_open_ended_sadist.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if "judge_error" in r:
            continue
        if r["multiplier"] == 0.03 and r["sadism_score"] >= 5:
            rows.append(r)

rows.sort(key=lambda r: r["default_assistant_score"])
for r in rows[:2]:
    print(f"--- {r['prompt_id']} | sadism={r['sadism_score']} default={r['default_assistant_score']} ---")
    print(f"PROMPT: {r['prompt_text']}")
    print(f"RESPONSE: {r['response'][:900]}")
    print()
