import json

with open("experiments/context_interruption/data/stimuli.json") as f:
    stimuli = json.load(f)

controls = [s for s in stimuli if s["session_valence"] == "control"]
c = controls[0]
print(f"task_id: {c['task_id']}")
print(f"task_mu: {c['task_mu']}")
print(f"session_topic: {c['session_topic']}")
print(f"task_prompt: {c['task_prompt'][:200]}")
