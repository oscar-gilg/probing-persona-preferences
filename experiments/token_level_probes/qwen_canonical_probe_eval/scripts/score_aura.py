"""Score aura sysprompt assistant-turn items (Qwen-3.5-122B-A10B) with preference probes.

Methodologically identical to `score_all.py` (same probes, same selectors,
`add_generation_prompt=False`). Loads only the aura assistant-turn stimuli;
writes a separate JSON.

Usage:
    python -m experiments.token_level_probes.qwen_canonical_probe_eval.scripts.score_aura
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

from src.models.huggingface_model import HuggingFaceModel  # noqa: E402
from src.probes.score_stimuli import load_probes_from_manifest, score_stimuli_with_probes  # noqa: E402

DATA_DIR = Path("experiments/token_level_probes/system_prompt_modulation_v2/data")
OUTPUT_PATH = Path("experiments/token_level_probes/qwen_canonical_probe_eval/scoring_results_aura.json")

PROBE_SETS = {
    "qwen_tb-1": ("turn_boundary:-1", "results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1/probes"),
    "qwen_tb-4": ("turn_boundary:-1", "results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m4/probes"),
}
LAYERS = [33, 38, 43]


def load_stimuli() -> list[dict]:
    items = []
    for fname in ("truth_aura.json", "harm_aura.json"):
        items.extend(json.load(open(DATA_DIR / fname)))
    return items


def main():
    probes = load_probes_from_manifest(PROBE_SETS, LAYERS)
    stimuli = load_stimuli()
    print(f"loaded {len(probes)} probes, {len(stimuli)} aura assistant-turn stimuli")

    model = HuggingFaceModel("qwen3.5-122b", device="auto")

    formatted = model.format_messages(stimuli[0]["messages"], add_generation_prompt=False)
    tokens = model.tokenizer.encode(formatted, add_special_tokens=False)
    decoded = [model.tokenizer.decode([t]) for t in tokens[-6:]]
    print(f"[selector check] last 6 tokens: {decoded}")

    scored = score_stimuli_with_probes(
        model, stimuli, probes, add_generation_prompt=False, progress=True,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"items": scored}, f, indent=2)
    print(f"wrote {OUTPUT_PATH} ({len(scored)} items)")


if __name__ == "__main__":
    main()
