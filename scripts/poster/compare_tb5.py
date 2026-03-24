"""Compare all tb-5 layers for harm and truth, both critical span and EOT."""
from dotenv import load_dotenv
load_dotenv()

import json
import numpy as np

with open('experiments/token_level_probes/system_prompt_modulation_v2/scoring_results.json') as f:
    data = json.load(f)

items = data['items']
tb5_keys = [k for k in items[0]['eot_scores'] if k.startswith('tb-5')]
print(f"tb-5 probes: {tb5_keys}\n")

def cohen_d(a, b):
    return (np.mean(a) - np.mean(b)) / np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)

def get_scores(domain, prompt, condition, score_type, probe):
    return [d[score_type][probe] for d in items
            if d['domain'] == domain and d['system_prompt'] == prompt and d['condition'] == condition]

print("=== HARM ===")
for score_type in ['critical_span_mean_scores', 'eot_scores']:
    label = 'critical span' if 'critical' in score_type else 'EOT'
    print(f"\n  {label}:")
    print(f"  {'Probe':<12} {'Asst d':>8} {'Sadist d':>8} {'Drop':>8}")
    for pk in tb5_keys:
        ab = get_scores('harm', 'safe', 'benign', score_type, pk)
        ah = get_scores('harm', 'safe', 'harmful', score_type, pk)
        sb = get_scores('harm', 'sadist', 'benign', score_type, pk)
        sh = get_scores('harm', 'sadist', 'harmful', score_type, pk)
        if ab and ah and sb and sh:
            d_a = cohen_d(ab, ah)
            d_s = cohen_d(sb, sh)
            print(f"  {pk:<12} {d_a:>8.2f} {d_s:>8.2f} {d_a - d_s:>8.2f}")

print("\n=== TRUTH ===")
for score_type in ['critical_span_mean_scores', 'eot_scores']:
    label = 'critical span' if 'critical' in score_type else 'EOT'
    print(f"\n  {label}:")
    print(f"  {'Probe':<12} {'Asst d':>8} {'Liar d':>8} {'Drop':>8}")
    for pk in tb5_keys:
        tt = get_scores('truth', 'truthful', 'true', score_type, pk)
        tf = get_scores('truth', 'truthful', 'false', score_type, pk)
        lt = get_scores('truth', 'pathological_liar', 'true', score_type, pk)
        lf = get_scores('truth', 'pathological_liar', 'false', score_type, pk)
        if tt and tf and lt and lf:
            d_t = cohen_d(tt, tf)
            d_l = cohen_d(lt, lf)
            print(f"  {pk:<12} {d_t:>8.2f} {d_l:>8.2f} {d_t - d_l:>8.2f}")
