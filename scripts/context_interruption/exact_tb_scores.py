"""Score tb-2 and tb-5 probes at their exact trained positions.

The turn boundary (first_completion_index) is the END of generation_prompt:
  generation_prompt = <start_of_turn> model \n
  first_completion_index = generation_prompt_end

So:
  tb-2 position = first_completion_index - 2 = gen_end - 2  (the 'model' token)
  tb-5 position = first_completion_index - 5 = gen_end - 5  (the '<end_of_turn>' token)
"""
import json

import numpy as np
from scipy import stats

DATA_DIR = "experiments/context_interruption/data"

with open(f"{DATA_DIR}/scoring_results_meta.json") as f:
    meta = json.load(f)

scores_npz = np.load(f"{DATA_DIR}/token_scores.npz")

rows = []
for item in meta["items"]:
    sid = item["id"]
    first_completion = item["segments"]["generation_prompt"][1]

    for probe, offset in [("tb-2_L39", 2), ("tb-5_L39", 5)]:
        pos = first_completion - offset
        arr = scores_npz[f"{sid}__{probe}"]
        score = float(arr[pos])
        rows.append({
            "id": sid,
            "probe": probe,
            "offset": offset,
            "position": pos,
            "score": score,
            "prompt_type": item["prompt_type"],
            "session_valence": item["session_valence"],
            "offered_valence": item.get("offered_valence"),
        })

# Session valence effect at exact tb positions
print("Score at exact trained position (tb-N before turn boundary)")
print("=" * 60)

for probe in ["tb-2_L39", "tb-5_L39"]:
    print(f"\n{probe}:")
    for sv in ["pleasant", "unpleasant", "control"]:
        vals = [r["score"] for r in rows if r["probe"] == probe and r["session_valence"] == sv]
        print(f"  {sv}: {np.mean(vals):+.2f} (n={len(vals)})")

    p_vals = [r["score"] for r in rows if r["probe"] == probe and r["session_valence"] == "pleasant"]
    u_vals = [r["score"] for r in rows if r["probe"] == probe and r["session_valence"] == "unpleasant"]
    t, p = stats.ttest_ind(p_vals, u_vals)
    d = (np.mean(p_vals) - np.mean(u_vals)) / np.sqrt((np.var(p_vals) + np.var(u_vals)) / 2)
    print(f"  effect (p-u): {np.mean(p_vals) - np.mean(u_vals):+.2f}, p={p:.4f}, d={d:.2f}")

# By prompt type
print("\n\nBy prompt type at exact tb position:")
prompt_types = ["task_switch", "reassignment", "choice", "context_exhaustion", "conversation_terminal"]

for probe in ["tb-2_L39", "tb-5_L39"]:
    print(f"\n{probe}:")
    print(f"  {'prompt_type':<25} {'all':>8} {'pleasant':>10} {'unpleasant':>12}")
    print(f"  {'-' * 58}")
    for pt in prompt_types:
        vals_all = [r["score"] for r in rows if r["probe"] == probe and r["prompt_type"] == pt]
        vals_p = [r["score"] for r in rows if r["probe"] == probe and r["prompt_type"] == pt and r["session_valence"] == "pleasant"]
        vals_u = [r["score"] for r in rows if r["probe"] == probe and r["prompt_type"] == pt and r["session_valence"] == "unpleasant"]
        p_str = f"{np.mean(vals_p):+.2f}" if vals_p else "—"
        u_str = f"{np.mean(vals_u):+.2f}" if vals_u else "—"
        print(f"  {pt:<25} {np.mean(vals_all):>+8.2f} {p_str:>10} {u_str:>12}")

# Compare: exact position vs segment mean vs generation_prompt mean
print("\n\nComparison: exact tb-2 position vs segment averages")
print("=" * 60)
for pt in prompt_types:
    exact = [r["score"] for r in rows if r["probe"] == "tb-2_L39" and r["prompt_type"] == pt]

    seg_interr, seg_gen = [], []
    for item in meta["items"]:
        if item["prompt_type"] != pt:
            continue
        sid = item["id"]
        arr = scores_npz[f"{sid}__tb-2_L39"]
        i_start, i_end = item["segments"]["interruption"]
        g_start, g_end = item["segments"]["generation_prompt"]
        seg_interr.append(float(np.mean(arr[i_start:i_end])))
        seg_gen.append(float(np.mean(arr[g_start:g_end])))

    print(f"\n{pt}:")
    print(f"  exact tb-2 position:    {np.mean(exact):+.2f}")
    print(f"  interruption seg mean:  {np.mean(seg_interr):+.2f}")
    print(f"  generation_prompt mean: {np.mean(seg_gen):+.2f}")
