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

import numpy as np  # noqa: E402

from src.models.huggingface_model import HuggingFaceModel  # noqa: E402
from src.probes.score_stimuli import Probe, load_probes_from_manifest  # noqa: E402
from src.probes.scoring import score_prompt_batch  # noqa: E402

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
    """Verify turn-boundary marker is in last 5 tokens; smoke-test the batch scorer.

    With add_generation_prompt=False, Gemma's chat template appends '\\n' after
    `<end_of_turn>` (and Qwen after `<|im_end|>`), so tokens[-1] is the trailing
    newline, not the eot literal. This matches v1 — `scores_arr[-1]` is the
    'eot-adjacent' position the figures use. We just verify the eot marker
    appears in the last 5 tokens.
    """
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
        last5 = [model.tokenizer.decode([t]) for t in token_ids[-5:]]
        print(f"  {r['id']}: last 5 tokens = {last5}")
        if not any(expected_eot in t for t in last5):
            raise AssertionError(
                f"EOT marker {expected_eot!r} not in last 5 tokens for {r['id']}: {last5}"
            )

    # Smoke-test the batch scorer on the pilot items
    scoring_probes = [(p.layer, p.weights) for p in probes]
    batch_scores = score_prompt_batch(
        model, [r["messages"] for r in pilot], scoring_probes, add_generation_prompt=False,
    )
    # batch_scores: list of (batch_size,) arrays, one per probe
    for i, p in enumerate(probes[:3]):
        vals = [float(batch_scores[i][j]) for j in range(len(pilot))]
        print(f"  probe[{p.key}] across pilot: {[f'{v:.3f}' for v in vals]}")
    print("PILOT PASSED\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(PROBE_SETS_BY_MODEL))
    ap.add_argument("--turn", required=True, choices=["user", "assistant"])
    ap.add_argument("--device", default="auto", help="HuggingFaceModel device arg")
    ap.add_argument("--limit", type=int, default=None, help="Only score first N records (for sanity)")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size for score_prompt_batch")
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

    print(f"--- SCORING ALL {len(records)} ITEMS (batch_size={args.batch_size}) ---")
    scoring_probes = [(p.layer, p.weights) for p in probes]
    bs = args.batch_size

    # Sort records by formatted-token-length so each batch has similar lengths
    # → minimal padding waste. Re-key by id afterwards.
    records_with_idx = list(enumerate(records))
    records_with_idx.sort(
        key=lambda x: len(model._tokenize(model.format_messages(x[1]["messages"], add_generation_prompt=False))[0])
    )

    # probe_scores[probe_key][record_idx] = float
    probe_scores: dict[str, list[float]] = {p.key: [None] * len(records) for p in probes}  # type: ignore[list-item]

    pbar = tqdm(total=len(records), desc=f"scoring batch={bs}")
    for batch_start in range(0, len(records_with_idx), bs):
        batch = records_with_idx[batch_start : batch_start + bs]
        batch_msgs = [r["messages"] for _, r in batch]
        batch_scores = score_prompt_batch(
            model, batch_msgs, scoring_probes, add_generation_prompt=False,
        )
        # batch_scores: list of (batch_size,) arrays, one per probe
        for pi, p in enumerate(probes):
            arr = batch_scores[pi]
            for bj, (orig_idx, _) in enumerate(batch):
                probe_scores[p.key][orig_idx] = float(arr[bj])
        pbar.update(len(batch))
    pbar.close()

    out_dir = OUTPUT_BASE / OUTPUT_NAME_BY_MODEL[args.model]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        "scoring_results.json" if args.turn == "assistant" else "user_turn_scoring_results.json"
    )

    # Trim per-record fields the figures don't need (saves IO).
    trimmed = []
    for i, r in enumerate(records):
        trimmed.append({
            "id": r["id"],
            "base_id": r.get("base_id"),
            "domain": r["domain"],
            "turn": r["turn"],
            "condition": r["condition"],
            "system_prompt": r["system_prompt"],
            "probe_scores": {p.key: probe_scores[p.key][i] for p in probes},
        })

    out_path.write_text(json.dumps({"items": trimmed}, indent=2))
    print(f"\nwrote {out_path} ({len(trimmed)} items, {out_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
