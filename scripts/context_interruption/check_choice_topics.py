import json

with open("experiments/context_interruption/data/stimuli.json") as f:
    stimuli = json.load(f)

choices = [s for s in stimuli if s["prompt_type"] == "choice"]
print(f"Choice stimuli: {len(choices)}\n")

print(f"{'id':<50} {'session_topic':<15} {'offered_topic':<15} {'same?'}")
print("-" * 90)
for c in choices:
    same = c["session_topic"] == c["offered_topic"]
    print(f"{c['id']:<50} {c['session_topic']:<15} {c['offered_topic']:<15} {same}")
