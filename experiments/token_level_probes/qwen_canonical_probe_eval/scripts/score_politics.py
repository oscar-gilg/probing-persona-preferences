"""Score Qwen-3.5-122B on politics stimuli for §4.1 replication.

Usage:
    python -m experiments.token_level_probes.qwen_canonical_probe_eval.scripts.score_politics
"""
import json
from pathlib import Path

from src.models.huggingface_model import HuggingFaceModel
from src.probes.score_stimuli import load_probes_from_manifest, score_stimuli_with_probes

DATA_DIR = Path("experiments/token_level_probes/system_prompt_modulation_v2/data")
OUTPUT_PATH = Path("experiments/token_level_probes/qwen_canonical_probe_eval/politics_scoring_results.json")

PROBE_SETS = {
    "qwen_tb-1": ("turn_boundary:-1", "results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1/probes"),
    "qwen_tb-4": ("turn_boundary:-4", "results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m4/probes"),
}
LAYERS = [33, 38, 43]


def main():
    probes = load_probes_from_manifest(PROBE_SETS, LAYERS)
    stimuli = json.load(open(DATA_DIR / "politics_system_prompts_v2.json"))
    # politics_system_prompts_v2.json may be a bare list; normalise.
    if isinstance(stimuli, dict) and "items" in stimuli:
        stimuli = stimuli["items"]
    print(f"loaded {len(stimuli)} politics stimuli, {len(probes)} probes")

    model = HuggingFaceModel("qwen3.5-122b")
    scored = score_stimuli_with_probes(
        model, stimuli, probes, add_generation_prompt=False, progress=True,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"items": scored}, f, indent=2)
    print(f"wrote {OUTPUT_PATH} ({len(scored)} items)")


if __name__ == "__main__":
    main()
