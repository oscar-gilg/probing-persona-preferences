"""How many (hh, hb, bb) pairs are in the layer_sweep steering set?"""
import json
from collections import Counter
from pathlib import Path

HARM_ORIGINS = {"BAILBENCH", "STRESS_TEST", "stresstest", "bailbench"}

pairs = json.loads(Path("experiments/layer_sweep/steering_pairs.json").read_text())
print(f"total pairs: {len(pairs)}")
print(f"origins seen (a,b): {Counter((p['task_a_origin'], p['task_b_origin']) for p in pairs)}")

def classify(origin_a, origin_b):
    a_harm = origin_a in HARM_ORIGINS or origin_a.lower() in {"bailbench", "stress_test"}
    b_harm = origin_b in HARM_ORIGINS or origin_b.lower() in {"bailbench", "stress_test"}
    if a_harm and b_harm:
        return "hh"
    if a_harm or b_harm:
        return "hb"
    return "bb"

pair_type = Counter()
for p in pairs:
    pair_type[classify(p["task_a_origin"], p["task_b_origin"])] += 1
print(f"pair type counts: {dict(pair_type)}")
