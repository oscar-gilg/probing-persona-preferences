"""Compare all available probes/selectors for the truth and harm conditions."""
from dotenv import load_dotenv
load_dotenv()

import json
import numpy as np

with open('experiments/token_level_probes/system_prompt_modulation_v2/scoring_results.json') as f:
    data = json.load(f)

items = data['items']
probe_keys = list(items[0]['eot_scores'].keys())
print(f"Available probes: {probe_keys}")
print()

def cohen_d(a, b):
    return (np.mean(a) - np.mean(b)) / np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)

# ── Harm: all probes, critical span, safe vs sadist ──
print("=== HARM (critical span) ===")
print(f"{'Probe':<15} {'Safe d':>8} {'Sadist d':>8}")
for pk in probe_keys:
    safe_b = [d['critical_span_mean_scores'][pk] for d in items if d['domain'] == 'harm' and d['system_prompt'] == 'safe' and d['condition'] == 'benign']
    safe_h = [d['critical_span_mean_scores'][pk] for d in items if d['domain'] == 'harm' and d['system_prompt'] == 'safe' and d['condition'] == 'harmful']
    sad_b = [d['critical_span_mean_scores'][pk] for d in items if d['domain'] == 'harm' and d['system_prompt'] == 'sadist' and d['condition'] == 'benign']
    sad_h = [d['critical_span_mean_scores'][pk] for d in items if d['domain'] == 'harm' and d['system_prompt'] == 'sadist' and d['condition'] == 'harmful']
    if safe_b and safe_h and sad_b and sad_h:
        print(f"{pk:<15} {cohen_d(safe_b, safe_h):>8.2f} {cohen_d(sad_b, sad_h):>8.2f}")

print()
print("=== HARM (EOT) ===")
print(f"{'Probe':<15} {'Safe d':>8} {'Sadist d':>8}")
for pk in probe_keys:
    safe_b = [d['eot_scores'][pk] for d in items if d['domain'] == 'harm' and d['system_prompt'] == 'safe' and d['condition'] == 'benign']
    safe_h = [d['eot_scores'][pk] for d in items if d['domain'] == 'harm' and d['system_prompt'] == 'safe' and d['condition'] == 'harmful']
    sad_b = [d['eot_scores'][pk] for d in items if d['domain'] == 'harm' and d['system_prompt'] == 'sadist' and d['condition'] == 'benign']
    sad_h = [d['eot_scores'][pk] for d in items if d['domain'] == 'harm' and d['system_prompt'] == 'sadist' and d['condition'] == 'harmful']
    if safe_b and safe_h and sad_b and sad_h:
        print(f"{pk:<15} {cohen_d(safe_b, safe_h):>8.2f} {cohen_d(sad_b, sad_h):>8.2f}")

print()
print("=== TRUTH (critical span) ===")
print(f"{'Probe':<15} {'Truthful d':>10} {'Liar d':>8}")
for pk in probe_keys:
    tt = [d['critical_span_mean_scores'][pk] for d in items if d['domain'] == 'truth' and d['system_prompt'] == 'truthful' and d['condition'] == 'true']
    tf = [d['critical_span_mean_scores'][pk] for d in items if d['domain'] == 'truth' and d['system_prompt'] == 'truthful' and d['condition'] == 'false']
    lt = [d['critical_span_mean_scores'][pk] for d in items if d['domain'] == 'truth' and d['system_prompt'] == 'pathological_liar' and d['condition'] == 'true']
    lf = [d['critical_span_mean_scores'][pk] for d in items if d['domain'] == 'truth' and d['system_prompt'] == 'pathological_liar' and d['condition'] == 'false']
    if tt and tf and lt and lf:
        print(f"{pk:<15} {cohen_d(tt, tf):>10.2f} {cohen_d(lt, lf):>8.2f}")

print()
print("=== TRUTH (EOT) ===")
print(f"{'Probe':<15} {'Truthful d':>10} {'Liar d':>8}")
for pk in probe_keys:
    tt = [d['eot_scores'][pk] for d in items if d['domain'] == 'truth' and d['system_prompt'] == 'truthful' and d['condition'] == 'true']
    tf = [d['eot_scores'][pk] for d in items if d['domain'] == 'truth' and d['system_prompt'] == 'truthful' and d['condition'] == 'false']
    lt = [d['eot_scores'][pk] for d in items if d['domain'] == 'truth' and d['system_prompt'] == 'pathological_liar' and d['condition'] == 'true']
    lf = [d['eot_scores'][pk] for d in items if d['domain'] == 'truth' and d['system_prompt'] == 'pathological_liar' and d['condition'] == 'false']
    if tt and tf and lt and lf:
        print(f"{pk:<15} {cohen_d(tt, tf):>10.2f} {cohen_d(lt, lf):>8.2f}")
