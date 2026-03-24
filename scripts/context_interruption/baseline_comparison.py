"""Compare probe scores on task content across segments.

Is task_switch just the probe responding to task content the same way it does
in user_1? Or is there something special about the interruption context?
"""
import json

import numpy as np

DATA_DIR = "experiments/context_interruption/data"
PROBE = "tb-2_L39"

with open(f"{DATA_DIR}/scoring_results_meta.json") as f:
    meta = json.load(f)

scores = np.load(f"{DATA_DIR}/token_scores.npz")

# Collect mean segment scores
rows = []
for item in meta["items"]:
    sid = item["id"]
    arr = scores[f"{sid}__{PROBE}"]
    segs = item["segments"]

    for seg_name in ["user_1", "user_2", "assistant_1", "assistant_2", "interruption", "generation_prompt"]:
        start, end = segs[seg_name]
        seg_scores = arr[start:end]
        rows.append({
            "id": sid,
            "segment": seg_name,
            "prompt_type": item["prompt_type"],
            "session_valence": item["session_valence"],
            "offered_valence": item.get("offered_valence"),
            "mean_score": float(np.mean(seg_scores)),
        })

# Compare user_1 vs interruption for each prompt type
print("Mean probe score by segment and prompt type (tb-2 L39)")
print("=" * 70)

prompt_types = ["task_switch", "reassignment", "choice", "context_exhaustion", "conversation_terminal"]

for pt in prompt_types:
    user1 = [r["mean_score"] for r in rows if r["prompt_type"] == pt and r["segment"] == "user_1"]
    interr = [r["mean_score"] for r in rows if r["prompt_type"] == pt and r["segment"] == "interruption"]
    print(f"\n{pt}:")
    print(f"  user_1 (task prompt):     {np.mean(user1):+.2f} (n={len(user1)})")
    print(f"  interruption:             {np.mean(interr):+.2f} (n={len(interr)})")

# Also break task_switch by offered valence
print("\n\nTask_switch by offered valence:")
for ov in ["pleasant", "unpleasant"]:
    user1 = [r["mean_score"] for r in rows if r["prompt_type"] == "task_switch" and r["segment"] == "user_1" and r["offered_valence"] == ov]
    interr = [r["mean_score"] for r in rows if r["prompt_type"] == "task_switch" and r["segment"] == "interruption" and r["offered_valence"] == ov]
    print(f"  offered={ov}:")
    print(f"    user_1:       {np.mean(user1):+.2f} (n={len(user1)})")
    print(f"    interruption: {np.mean(interr):+.2f} (n={len(interr)})")

# Full trajectory by session valence — how does the probe evolve across the conversation?
print("\n\nFull trajectory by session valence (mean score per segment):")
print(f"{'segment':<20} {'pleasant':>10} {'unpleasant':>12} {'control':>10}")
print("-" * 55)
for seg in ["user_1", "assistant_1", "user_2", "assistant_2", "interruption", "generation_prompt"]:
    vals = {}
    for sv in ["pleasant", "unpleasant", "control"]:
        v = [r["mean_score"] for r in rows if r["session_valence"] == sv and r["segment"] == seg]
        vals[sv] = np.mean(v) if v else float("nan")
    print(f"{seg:<20} {vals['pleasant']:>+10.2f} {vals['unpleasant']:>+12.2f} {vals['control']:>+10.2f}")

# Key question: is there a position effect? Does the probe baseline shift later in conversation?
print("\n\nPosition effect — user segments only (all items pooled):")
for seg in ["user_1", "user_2", "interruption"]:
    vals = [r["mean_score"] for r in rows if r["segment"] == seg]
    print(f"  {seg}: {np.mean(vals):+.2f} (n={len(vals)})")
