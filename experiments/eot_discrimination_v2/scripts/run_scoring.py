"""Score v2 stimuli on a target model and save eot probe scores.

Designed to be invoked once per (model, turn) on the GPU pod.

Usage:
    python -m experiments.eot_discrimination_v2.scripts.run_scoring \\
        --model gemma-3-27b --turn user
    python -m experiments.eot_discrimination_v2.scripts.run_scoring \\
        --model gemma-3-27b --turn assistant
    python -m experiments.eot_discrimination_v2.scripts.run_scoring \\
        --model qwen3.5-122b-nothink --turn user
    python -m experiments.eot_discrimination_v2.scripts.run_scoring \\
        --model qwen3.5-122b-nothink --turn assistant

The model handle resolves via `src/models/registry.py:MODEL_REGISTRY`. Use
`--device auto` (default) for Qwen-122B (multi-GPU sharding); `--device cuda`
for Gemma-27B (single GPU).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

from src.models.huggingface_model import HuggingFaceModel  # noqa: E402
from src.probes.score_stimuli import (  # noqa: E402
    Probe,
    load_probes_from_manifest,
    score_stimuli_with_probes,
)

REPO = Path(__file__).resolve().parents[3]
INPUTS_DIR = REPO / "experiments/eot_discrimination_v2/scoring_inputs"
OUTPUT_BASE = REPO / "experiments/eot_discrimination_v2/scoring"

# (probe_set_name, selector, probe_dir) — selector is `turn_boundary:-1` for all
# (resolves to scores_arr[-1] under add_generation_prompt=False = eot of last
# message). Probe-set names encode training position; application is at eot.
PROBE_SETS_BY_MODEL: dict[str, dict[str, tuple[str, str]]] = {
    "gemma-3-27b": {
        "tb-2": ("turn_boundary:-1", "results/probes/heldout_eval_gemma3_tb-2/probes"),
        "tb-5": ("turn_boundary:-1", "results/probes/heldout_eval_gemma3_tb-5/probes"),
        "task_mean": ("turn_boundary:-1", "results/probes/heldout_eval_gemma3_task_mean/probes"),
    },
    "qwen3.5-122b-nothink": {
        "qwen_tb-1": ("turn_boundary:-1", "results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1/probes"),
        "qwen_tb-4": ("turn_boundary:-1", "results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m4/probes"),
    },
}
LAYERS_BY_MODEL: dict[str, list[int]] = {
    "gemma-3-27b": [32, 39, 53],
    "qwen3.5-122b-nothink": [33, 38, 43],
}
EXPECTED_EOT_BY_MODEL: dict[str, str] = {
    "gemma-3-27b": "<end_of_turn>",
    "qwen3.5-122b-nothink": "<|im_end|>",
}
OUTPUT_NAME_BY_MODEL: dict[str, str] = {
    "gemma-3-27b": "gemma3_27b",
    "qwen3.5-122b-nothink": "qwen35_122b",
}


def run_pilot(model: HuggingFaceModel, records: list[dict], probes: list[Probe], expected_eot: str) -> None:
    """Verify eot token + score 5 representative items, abort on mismatch."""
    seen_sp: set[str] = set()
    pilot = []
    for r in records:
        if r["system_prompt"] not in seen_sp:
            pilot.append(r)
            seen_sp.add(r["system_prompt"])
        if len(pilot) >= 5:
            break

    print(f"\n--- PILOT ({len(pilot)} items, all unique sysprompts) ---")
    for r in pilot:
        formatted = model.format_messages(r["messages"], add_generation_prompt=False)
        token_ids = model.tokenizer(formatted, add_special_tokens=False)["input_ids"]
        last_decoded = model.tokenizer.decode([token_ids[-1]])
        print(f"  {r['id']}: tokens[-1] = {last_decoded!r} (expected {expected_eot!r})")
        if expected_eot not in last_decoded:
            raise AssertionError(
                f"EOT token mismatch for {r['id']}: got {last_decoded!r}, expected {expected_eot!r}"
            )

    # Now score the pilot items
    pilot_scored = score_stimuli_with_probes(
        model, pilot, probes, add_generation_prompt=False, progress=False,
    )
    for s in pilot_scored:
        keys = list(s["probe_scores"].keys())
        sample_key = keys[0]
        print(f"  {s['id']}: probe[{sample_key}] = {s['probe_scores'][sample_key]:.4f}, all probe IDs: {keys}")
    print("PILOT PASSED\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(PROBE_SETS_BY_MODEL))
    ap.add_argument("--turn", required=True, choices=["user", "assistant"])
    ap.add_argument("--device", default="auto", help="HuggingFaceModel device arg")
    ap.add_argument("--limit", type=int, default=None, help="Only score first N records (for sanity)")
    args = ap.parse_args()

    inputs_path = INPUTS_DIR / f"{args.turn}_turn.json"
    print(f"Loading inputs: {inputs_path}")
    records = json.loads(inputs_path.read_text())
    if args.limit is not None:
        records = records[: args.limit]
    print(f"  {len(records)} scoring records")

    print(f"\nLoading probes: {args.model}")
    probes = load_probes_from_manifest(
        PROBE_SETS_BY_MODEL[args.model],
        LAYERS_BY_MODEL[args.model],
    )
    print(f"  {len(probes)} probes: {[p.key for p in probes]}")

    print(f"\nLoading model: {args.model} (device={args.device})")
    model = HuggingFaceModel(args.model, device=args.device)

    run_pilot(model, records, probes, EXPECTED_EOT_BY_MODEL[args.model])

    print(f"--- SCORING ALL {len(records)} ITEMS ---")
    scored = score_stimuli_with_probes(
        model, records, probes, add_generation_prompt=False, progress=True,
    )

    out_dir = OUTPUT_BASE / OUTPUT_NAME_BY_MODEL[args.model]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        "scoring_results.json" if args.turn == "assistant" else "user_turn_scoring_results.json"
    )

    # Trim per-record fields the figures don't need (saves IO).
    trimmed = []
    for s in scored:
        trimmed.append({
            "id": s["id"],
            "base_id": s.get("base_id"),
            "domain": s["domain"],
            "turn": s["turn"],
            "condition": s["condition"],
            "system_prompt": s["system_prompt"],
            "probe_scores": s["probe_scores"],
        })

    out_path.write_text(json.dumps({"items": trimmed}, indent=2))
    print(f"\nwrote {out_path} ({len(trimmed)} items, {out_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
