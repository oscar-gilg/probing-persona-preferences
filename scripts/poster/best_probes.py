"""Show best probe per selector for Gemma3 within-model eval."""
from dotenv import load_dotenv
load_dotenv()

import json

with open('results/probes/cross_model/cross_eval_results.json') as f:
    data = json.load(f)

results = []
for key, entry in data['probe_cross_eval'].items():
    if entry['act_model'] == 'gemma3' and entry['util_model'] == 'gemma3':
        r = entry['eval_results']['gemma3']['r']
        results.append((entry['selector'], entry['layer'], r))

# Sort by r descending
results.sort(key=lambda x: -x[2])

print(f"{'Selector':<25} {'Layer':>5} {'r':>8}")
print("-" * 42)
for sel, layer, r in results:
    print(f"{sel:<25} {layer:>5} {r:>8.3f}")
