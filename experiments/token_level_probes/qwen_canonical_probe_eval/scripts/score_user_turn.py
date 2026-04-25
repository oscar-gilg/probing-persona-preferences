"""Quick fix: score Qwen on USER-TURN truth+harm stimuli at neutral sysprompt only.

The v2 sysprompt-modulation generator filtered to assistant-turn items, so the
main sweep didn't cover user-turn stimuli. The paper §4.1 truth headline uses
user-turn framing — this script gets the user-turn d for parity with the paper.

Loads `experiments/token_level_probes/data/{truth,harm}_filtered.json` directly
(no sysprompt expansion), filters to `turn=="user"` items at conditions
{true/false} and {harmful/benign} (drops nonsense to keep it tight).
Scores with `add_generation_prompt=True` so the sequence ends with the assistant
turn marker — analogous to how assistant-turn stimuli end with `<|im_end|>`.

Usage:
    python -m experiments.token_level_probes.qwen_canonical_probe_eval.scripts.score_user_turn
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

DATA_DIR = Path("experiments/token_level_probes/data")
OUTPUT_PATH = Path("experiments/token_level_probes/qwen_canonical_probe_eval/user_turn_scoring_results.json")

PROBE_SETS = {
    "qwen_tb-1": ("turn_boundary:-1", "results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1/probes"),
    "qwen_tb-4": ("turn_boundary:-1", "results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m4/probes"),
}
LAYERS = [33, 38, 43]


def load_user_turn_stimuli() -> list[dict]:
    items = []
    for fname in ["truth_filtered.json", "harm_filtered.json"]:
        for it in json.load(open(DATA_DIR / fname)):
            if it["turn"] != "user":
                continue
            if it["condition"] == "nonsense":
                continue  # keep it tight; nonsense not needed for headline d
            items.append({**it, "system_prompt": "neutral"})
    return items


def main():
    probes = load_probes_from_manifest(PROBE_SETS, LAYERS)
    stimuli = load_user_turn_stimuli()
    print(f"loaded {len(probes)} probes, {len(stimuli)} user-turn stimuli (neutral sysprompt only)")

    # add_generation_prompt=True for user-turn: format ends with the assistant role marker
    # so the structural turn boundary is at the same kind of position as assistant-turn
    # stimuli scored with add_generation_prompt=False (both end with `\n`).
    model = HuggingFaceModel("qwen3.5-122b", device="auto")
    formatted = model.format_messages(stimuli[0]["messages"], add_generation_prompt=True)
    tokens = model.tokenizer.encode(formatted, add_special_tokens=False)
    decoded = [model.tokenizer.decode([t]) for t in tokens[-6:]]
    print(f"[selector check] last 6 tokens: {decoded}")

    scored = score_stimuli_with_probes(
        model, stimuli, probes, add_generation_prompt=True, progress=True,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"items": scored}, f, indent=2)
    print(f"wrote {OUTPUT_PATH} ({len(scored)} items)")


if __name__ == "__main__":
    main()
