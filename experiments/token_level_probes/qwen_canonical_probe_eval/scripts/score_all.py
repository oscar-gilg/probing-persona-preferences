"""Score Qwen-3.5-122B on truth + harm stimuli for §4.1 replication.

Applies 6 probes (tb-1 / tb-4 × L33/L38/L43) to each stimulus at its matched
selector in one forward pass per stimulus. All items are assistant-turn,
add_generation_prompt=False.

Usage:
    python -m experiments.token_level_probes.qwen_canonical_probe_eval.scripts.score_all
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

load_dotenv()
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

from src.models.huggingface_model import HuggingFaceModel  # noqa: E402
from src.probes.score_stimuli import Probe, load_probes_from_manifest, score_stimuli_with_probes  # noqa: E402

DATA_DIR = Path("experiments/token_level_probes/system_prompt_modulation_v2/data")
OUTPUT_PATH = Path("experiments/token_level_probes/qwen_canonical_probe_eval/scoring_results.json")

PROBE_SETS = {
    # All probes APPLIED at the last token (turn_boundary:-1); the selector in the
    # dict key encodes where the probe was TRAINED. Matches the Gemma parent
    # convention (system_prompt_modulation_v2/scripts/score_all.py applies every
    # probe at scores_arr[-1]).
    "qwen_tb-1": ("turn_boundary:-1", "results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1/probes"),
    "qwen_tb-4": ("turn_boundary:-1", "results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m4/probes"),
}
LAYERS = [33, 38, 43]


def load_stimuli() -> list[dict]:
    items = []
    for filename in ["truth_system_prompts_v2.json", "harm_system_prompts_v2.json"]:
        items.extend(json.load(open(DATA_DIR / filename)))
    return items


def main():
    probes = load_probes_from_manifest(PROBE_SETS, LAYERS)
    print(f"loaded {len(probes)} probes: {[p.key for p in probes]}")

    stimuli = load_stimuli()
    print(f"loaded {len(stimuli)} stimuli")

    # Selector sanity check on one stimulus
    # device="auto" enables device_map="auto" for multi-GPU sharding (Qwen-122B needs ~244GB, so 3+ GPUs).
    model = HuggingFaceModel("qwen3.5-122b", device="auto")
    formatted = model.format_messages(stimuli[0]["messages"], add_generation_prompt=False)
    tokens = model.tokenizer.encode(formatted, add_special_tokens=False)
    decoded = [model.tokenizer.decode([t]) for t in tokens[-6:]]
    print(f"[selector check] last 6 tokens of first stimulus: {decoded}")
    print(f"                 turn_boundary:-1 = tokens[-1] = {decoded[-1]!r}")
    print(f"                 turn_boundary:-4 = tokens[-4] = {decoded[-3]!r}")

    scored = score_stimuli_with_probes(
        model, stimuli, probes, add_generation_prompt=False, progress=True,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"items": scored}, f, indent=2)
    print(f"wrote {OUTPUT_PATH} ({len(scored)} items)")


if __name__ == "__main__":
    main()
