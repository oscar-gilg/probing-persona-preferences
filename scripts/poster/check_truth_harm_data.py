import json
with open("experiments/token_level_probes/system_prompt_modulation_v2/scoring_results.json") as f:
    data = json.load(f)
print("Top-level keys:", list(data.keys())[:20])
if isinstance(data, list):
    print("It's a list, first item keys:", list(data[0].keys()) if data else "empty")
elif isinstance(data, dict):
    for k in list(data.keys())[:5]:
        v = data[k]
        print(f"  {k}: type={type(v).__name__}, len={len(v) if hasattr(v, '__len__') else 'N/A'}")
