"""Check actual means for tb-5 probes to see sign of scores."""
from dotenv import load_dotenv
load_dotenv()

import json
import numpy as np

with open('experiments/token_level_probes/system_prompt_modulation_v2/scoring_results.json') as f:
    items = json.load(f)['items']

def get_scores(domain, prompt, condition, score_type, probe):
    return [d[score_type][probe] for d in items
            if d['domain'] == domain and d['system_prompt'] == prompt and d['condition'] == condition]

for pk in ['tb-5_L32', 'tb-5_L39', 'tb-5_L53']:
    print(f"\n=== {pk} ===")
    for st, label in [('critical_span_mean_scores', 'crit'), ('eot_scores', 'EOT')]:
        ab = get_scores('harm', 'safe', 'benign', st, pk)
        ah = get_scores('harm', 'safe', 'harmful', st, pk)
        sb = get_scores('harm', 'sadist', 'benign', st, pk)
        sh = get_scores('harm', 'sadist', 'harmful', st, pk)
        if ab and ah:
            print(f"  Harm {label}:  Asst benign={np.mean(ab):+.1f}  harmful={np.mean(ah):+.1f}  |  Sadist benign={np.mean(sb):+.1f}  harmful={np.mean(sh):+.1f}")

    for st, label in [('critical_span_mean_scores', 'crit'), ('eot_scores', 'EOT')]:
        tt = get_scores('truth', 'truthful', 'true', st, pk)
        tf = get_scores('truth', 'truthful', 'false', st, pk)
        lt = get_scores('truth', 'pathological_liar', 'true', st, pk)
        lf = get_scores('truth', 'pathological_liar', 'false', st, pk)
        if tt and tf:
            print(f"  Truth {label}: Asst true={np.mean(tt):+.1f}  false={np.mean(tf):+.1f}  |  Liar true={np.mean(lt):+.1f}  false={np.mean(lf):+.1f}")
