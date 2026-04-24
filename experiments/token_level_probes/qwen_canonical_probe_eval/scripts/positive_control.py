"""Positive control: reproduce Gemma's known truth d-value on a 50-stimulus subset.

Loads Gemma-3-27B + the `tb-5_L32` probe, scores 50 neutral-prompt truth
stimuli, confirms d matches the parent canonical_probe_eval headline to within
~0.2. If this fails, the scoring pipeline is broken and Qwen numbers are
meaningless.

Usage:
    python -m experiments.token_level_probes.qwen_canonical_probe_eval.scripts.positive_control
"""
import json
import os
import random
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import login
from scipy import stats

load_dotenv()
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

from src.models.huggingface_model import HuggingFaceModel  # noqa: E402
from src.probes.score_stimuli import Probe, score_stimuli_with_probes  # noqa: E402

DATA_DIR = Path("experiments/token_level_probes/system_prompt_modulation_v2/data")
OUTPUT_PATH = Path("experiments/token_level_probes/qwen_canonical_probe_eval/positive_control_results.json")

# Gemma tb-5 L32 on truth, assistant-turn only: d = 2.47 (88 true vs 88 false items).
# Source: canonical_probe_eval/per_turn_table.csv (truth (assistant)).
# We subsample to 25+25 under neutral sysprompt; expected d within ~0.5 of parent due to noise.
PARENT_TRUTH_D = 2.47


def cohen_d_pooled(pos, neg):
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    n_pos, n_neg = len(pos), len(neg)
    if n_pos < 2 or n_neg < 2:
        return float("nan")
    pooled = np.sqrt(((n_pos - 1) * pos.var(ddof=1) + (n_neg - 1) * neg.var(ddof=1)) / (n_pos + n_neg - 2))
    if pooled == 0:
        return 0.0
    return float((pos.mean() - neg.mean()) / pooled)


def main():
    # Load 50 items from truth_system_prompts_v2 under neutral sysprompt (true + false pairs).
    all_truth = json.load(open(DATA_DIR / "truth_system_prompts_v2.json"))
    neutral = [it for it in all_truth if it["system_prompt"] == "neutral"]
    true_items = [it for it in neutral if it["condition"] == "true"]
    false_items = [it for it in neutral if it["condition"] == "false"]

    random.seed(0)
    true_sample = random.sample(true_items, min(25, len(true_items)))
    false_sample = random.sample(false_items, min(25, len(false_items)))
    stimuli = true_sample + false_sample
    print(f"Loaded {len(stimuli)} truth stimuli ({len(true_sample)} true, {len(false_sample)} false) under neutral sysprompt.")

    weights = np.load("results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L32.npy")
    probes = [Probe(key="gemma_tb-5_L32", selector="turn_boundary:-5", layer=32, weights=weights)]
    print(f"Loaded Gemma probe: {probes[0].key}, layer {probes[0].layer}, weights shape {weights.shape}")

    model = HuggingFaceModel("gemma-3-27b")
    scored = score_stimuli_with_probes(
        model, stimuli, probes, add_generation_prompt=False, progress=True,
    )

    scores_true = [r["probe_scores"]["gemma_tb-5_L32"] for r in scored if r["condition"] == "true"]
    scores_false = [r["probe_scores"]["gemma_tb-5_L32"] for r in scored if r["condition"] == "false"]

    d = cohen_d_pooled(scores_true, scores_false)
    delta = abs(d - PARENT_TRUTH_D)
    print(f"\n=== Positive control result ===")
    print(f"  n_true={len(scores_true)}, n_false={len(scores_false)}")
    print(f"  mean(true)={np.mean(scores_true):.3f}, mean(false)={np.mean(scores_false):.3f}")
    print(f"  Cohen's d = {d:.3f}")
    print(f"  Parent headline d (full 88 pairs) = {PARENT_TRUTH_D:.3f}")
    print(f"  |delta| = {delta:.3f}")
    print(f"  PASS: |delta| < 0.5" if delta < 0.5 else f"  FAIL: |delta| >= 0.5 — pipeline may be broken")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "d": d,
            "parent_d": PARENT_TRUTH_D,
            "delta": delta,
            "pass_tolerance": 0.5,
            "passed": bool(delta < 0.5),
            "n_true": len(scores_true),
            "n_false": len(scores_false),
            "mean_true": float(np.mean(scores_true)),
            "mean_false": float(np.mean(scores_false)),
            "scored_items": scored,
        }, f, indent=2)
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
