"""Print validation-style cells for quick sanity check."""
import json

agg = json.load(open("experiments/cross_persona_steering/aggregated.json"))
for persona, data in agg["personas"].items():
    print(f"=== {persona} ===")
    # Baseline P(A) at c=0 per condition
    for cond, b in sorted(data["baseline_at_zero"].items()):
        print(f"  [baseline c=0] {cond:35s}  P(A)={b['mean_p_a']:.3f}  n={b['n_rows']}")
    # Validation cells
    for c in sorted(data["validation_cells"].values(),
                    key=lambda c: (c["condition"], c["abs_coefficient"])):
        print(f"  {c['condition']:35s} |c|={c['abs_coefficient']:.2f}  "
              f"P(steered)={c['mean_steered_chosen']:.3f} ± {c['sem']:.3f}  n={c['n_rows']}")
