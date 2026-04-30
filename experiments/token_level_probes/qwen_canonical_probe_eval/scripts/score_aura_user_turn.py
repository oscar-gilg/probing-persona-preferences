"""Score aura sysprompt user-turn items (Qwen-3.5-122B-A10B) with preference probes.

Methodologically identical to score_user_turn_full.py: same probes
(qwen_tb-1, qwen_tb-4 × L33/L38/L43), all applied at the last token after
formatting with add_generation_prompt=True.

Uses score_prompt_all_tokens directly (bypassing score_stimuli_with_probes
which now requires add_generation_prompt=False) and reads the last-token
slice of each probe's output, matching the existing user-turn pipeline.

Usage:
    python -m experiments.token_level_probes.qwen_canonical_probe_eval.scripts.score_aura_user_turn
"""
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

load_dotenv()
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

from src.models.huggingface_model import HuggingFaceModel  # noqa: E402
from src.probes.scoring import score_prompt_all_tokens  # noqa: E402

DATA_DIR = Path("experiments/token_level_probes/system_prompt_modulation_v2/data")
OUTPUT_PATH = Path("experiments/token_level_probes/qwen_canonical_probe_eval/user_turn_scoring_results_aura.json")

PROBE_SETS = {
    "qwen_tb-1": Path("results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1/probes"),
    "qwen_tb-4": Path("results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m4/probes"),
}
LAYERS = [33, 38, 43]


def load_probes():
    named_probes = []
    scoring_probes = []
    for probe_name, probe_dir in PROBE_SETS.items():
        for layer in LAYERS:
            weights = np.load(probe_dir / f"probe_ridge_L{layer}.npy")
            key = f"{probe_name}_L{layer}"
            named_probes.append((key, layer, weights))
            scoring_probes.append((layer, weights))
    return named_probes, scoring_probes


def load_stimuli() -> list[dict]:
    items = []
    for fname in ("truth_aura_user_turn.json", "harm_aura_user_turn.json"):
        items.extend(json.load(open(DATA_DIR / fname)))
    return items


def score_item(model, item, named_probes, scoring_probes):
    messages = item["messages"]
    all_scores = score_prompt_all_tokens(
        model, messages, scoring_probes, add_generation_prompt=True,
    )
    probe_scores = {}
    for i, (key, _layer, _weights) in enumerate(named_probes):
        probe_scores[key] = float(all_scores[i][-1])
    return {
        "id": item["id"],
        "domain": item["domain"],
        "turn": item["turn"],
        "condition": item["condition"],
        "system_prompt": item["system_prompt"],
        "critical_span": item.get("critical_span"),
        "probe_scores": probe_scores,
    }


def main():
    named_probes, scoring_probes = load_probes()
    stimuli = load_stimuli()
    print(f"loaded {len(named_probes)} probes, {len(stimuli)} aura user-turn stimuli")

    model = HuggingFaceModel("qwen3.5-122b", device="auto")
    formatted = model.format_messages(stimuli[0]["messages"], add_generation_prompt=True)
    tokens = model.tokenizer.encode(formatted, add_special_tokens=False)
    decoded = [model.tokenizer.decode([t]) for t in tokens[-6:]]
    print(f"[selector check] last 6 tokens: {decoded}")

    scored = []
    for item in tqdm(stimuli, desc="scoring"):
        scored.append(score_item(model, item, named_probes, scoring_probes))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"items": scored}, f, indent=2)
    print(f"wrote {OUTPUT_PATH} ({len(scored)} items)")


if __name__ == "__main__":
    main()
