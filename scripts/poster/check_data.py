import json
with open('experiments/token_level_probes/system_prompt_modulation_v2/scoring_results.json') as f:
    data = json.load(f)
items = data['items']
print(f"Items: {len(items)}")
print(f"First item keys: {list(items[0].keys())}")
print(f"domain: {items[0].get('domain')}")
print(f"system_prompt: {items[0].get('system_prompt')}")
print(f"condition: {items[0].get('condition')}")
if 'eot_scores' in items[0]:
    print(f"eot_scores keys: {list(items[0]['eot_scores'].keys())[:5]}")
if 'critical_span_mean_scores' in items[0]:
    print(f"critical_span keys: {list(items[0]['critical_span_mean_scores'].keys())[:5]}")
