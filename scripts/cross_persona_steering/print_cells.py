"""Print aggregated cells for quick sanity check."""
import json
from pathlib import Path

agg = json.load(open("experiments/cross_persona_steering/aggregated.json"))
for persona, data in agg["personas"].items():
    print(f"=== {persona} ===")
    cells = data["cells"]
    for c in sorted(cells.values(), key=lambda c: (c["condition"], c["coefficient"])):
        print(f"  {c['condition']:35s} c={c['coefficient']:+.2f}  "
              f"P_def={c['mean_default_pref']:.3f}  n={c['n_rows']}")
