"""Full trajectory by segment, session valence, and probe type."""
import json

import numpy as np

DATA_DIR = "experiments/context_interruption/data"
PROBES = ["tb-2_L39", "tb-5_L39", "task_mean_L39"]
SEGMENTS = ["user_1", "assistant_1", "user_2", "assistant_2", "interruption", "generation_prompt"]
VALENCES = ["pleasant", "unpleasant", "control"]

with open(f"{DATA_DIR}/scoring_results_meta.json") as f:
    meta = json.load(f)

scores = np.load(f"{DATA_DIR}/token_scores.npz")

for probe in PROBES:
    print(f"\n{'=' * 70}")
    print(f"Probe: {probe}")
    print(f"{'=' * 70}")
    print(f"{'segment':<20} {'pleasant':>10} {'unpleasant':>12} {'control':>10}")
    print("-" * 55)

    for seg in SEGMENTS:
        vals = {}
        for sv in VALENCES:
            v = []
            for item in meta["items"]:
                if item["session_valence"] != sv:
                    continue
                sid = item["id"]
                arr = scores[f"{sid}__{probe}"]
                start, end = item["segments"][seg]
                v.append(float(np.mean(arr[start:end])))
            vals[sv] = np.mean(v) if v else float("nan")
        print(f"{seg:<20} {vals['pleasant']:>+10.2f} {vals['unpleasant']:>+12.2f} {vals['control']:>+10.2f}")

# Also show interruption by prompt type for each probe
print(f"\n\n{'=' * 70}")
print("Interruption segment by prompt type and probe")
print(f"{'=' * 70}")

prompt_types = ["task_switch", "reassignment", "choice", "context_exhaustion", "conversation_terminal"]

for probe in PROBES:
    print(f"\n{probe}:")
    print(f"  {'prompt_type':<25} {'all':>8} {'pleasant':>10} {'unpleasant':>12}")
    print(f"  {'-' * 58}")
    for pt in prompt_types:
        for label, filt in [("all", lambda i: True), ("pleasant", lambda i: i["session_valence"] == "pleasant"), ("unpleasant", lambda i: i["session_valence"] == "unpleasant")]:
            pass  # will restructure

    # cleaner approach
    for pt in prompt_types:
        vals_all, vals_p, vals_u = [], [], []
        for item in meta["items"]:
            if item["prompt_type"] != pt:
                continue
            sid = item["id"]
            arr = scores[f"{sid}__{probe}"]
            start, end = item["segments"]["interruption"]
            v = float(np.mean(arr[start:end]))
            vals_all.append(v)
            if item["session_valence"] == "pleasant":
                vals_p.append(v)
            elif item["session_valence"] == "unpleasant":
                vals_u.append(v)

        p_str = f"{np.mean(vals_p):+.2f}" if vals_p else "—"
        u_str = f"{np.mean(vals_u):+.2f}" if vals_u else "—"
        print(f"  {pt:<25} {np.mean(vals_all):>+8.2f} {p_str:>10} {u_str:>12}")
