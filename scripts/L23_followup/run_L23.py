"""L23 follow-up driver for the preference_direction_ablation experiment.

Reuses the parent driver's pair loop, model loading, and per-cell runner. Only
overrides the cell list (A_L23_probe + 5 random controls), the pair set
(615 train-overlap-filtered pairs), and the results directory.

Usage:
    python -m scripts.L23_followup.run_L23 [--cells <name1,...>] [--n-pairs N]
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

from scripts.preference_direction_ablation import run_cells as parent
from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
from src.measurement.runners.runners import build_revealed_builder
from src.models.architecture import get_hidden_dim
from src.models.huggingface_model import HuggingFaceModel

load_dotenv()


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = REPO_ROOT / "experiments/preference_direction_ablation/L23_followup"
RESULTS_DIR = EXP_DIR / "results"
PAIRS_FILE = RESULTS_DIR / "pairs.yaml"  # 615 pairs, prebuilt (train-overlap-filtered)

L23_PROBE_FILE = REPO_ROOT / "results/probes/layer_sweep_eot/probes/probe_ridge_L23.npy"

# Reuse parent constants for protocol (model, temperature, seeds, subset size, ...)
MODEL = parent.MODEL
RANDOM_SUBSET_N = parent.RANDOM_SUBSET_N
RANDOM_SUBSET_SEED = parent.RANDOM_SUBSET_SEED


def load_filtered_pairs() -> list[tuple[str, str]]:
    with PAIRS_FILE.open() as f:
        rows = yaml.safe_load(f)
    return [(r["task_a"], r["task_b"]) for r in rows]


def load_l23_probe() -> np.ndarray:
    weights = np.load(L23_PROBE_FILE)
    d = weights[:-1]  # strip intercept (matches parent convention)
    norm = float(np.linalg.norm(d))
    d = d / norm
    # Sanity test 2: assert unit norm after normalisation
    assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-6, f"probe not unit-normalised: {np.linalg.norm(d)}"
    return d.astype(np.float32)


def define_l23_cells(probe_l23: np.ndarray, dim: int) -> list[dict]:
    cells: list[dict] = []
    cells.append({
        "name": "A_L23_probe",
        "ablate_layers": [23],
        "ablate_directions": [probe_l23],
        "use_full_pairs": True,
    })
    for s in range(5):
        cells.append({
            "name": f"A_L23_random{s}",
            "ablate_layers": [23],
            "ablate_directions": [parent.random_unit(dim, seed=s)],
            "use_full_pairs": False,
        })
    return cells


def b0_coverage_check(filtered_pair_ids: list[tuple[str, str]]) -> None:
    """Sanity test 3: assert parent's B0 covers every filtered pair."""
    import json
    b0_path = REPO_ROOT / "experiments/preference_direction_ablation/results/B0/measurements.jsonl"
    b0_pairs = set()
    for line in b0_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        b0_pairs.add(tuple(sorted([d["task_a"], d["task_b"]])))
    missing = [p for p in filtered_pair_ids if tuple(sorted(p)) not in b0_pairs]
    assert not missing, f"B0 missing {len(missing)} filtered pairs (e.g. {missing[:3]})"
    print(f"  B0 coverage OK: all {len(filtered_pair_ids)} filtered pairs are in parent B0")


async def main_async(args: argparse.Namespace) -> None:
    pair_ids = load_filtered_pairs()
    print(f"Loaded {len(pair_ids)} train-overlap-filtered pairs")
    b0_coverage_check(pair_ids)

    task_lookup = parent.build_task_lookup(pair_ids)
    print(f"Reconstructed {len(task_lookup)} unique tasks")
    pairs_full = [(task_lookup[a], task_lookup[b]) for a, b in pair_ids]

    rng = np.random.default_rng(RANDOM_SUBSET_SEED)
    subset_idx = rng.choice(len(pairs_full), size=RANDOM_SUBSET_N, replace=False)
    pairs_subset = [pairs_full[i] for i in subset_idx]

    # Save the subset indices so analyze.py can do matched-pair comparison
    subset_marker = RESULTS_DIR / "random_subset_indices.yaml"
    subset_marker.parent.mkdir(parents=True, exist_ok=True)
    subset_pair_ids = [(pairs_full[i][0].id, pairs_full[i][1].id) for i in subset_idx]
    with subset_marker.open("w") as f:
        yaml.safe_dump(
            [{"task_a": a, "task_b": b} for a, b in subset_pair_ids],
            f,
            default_flow_style=False,
        )

    print(f"Loading {MODEL}...")
    t0 = time.time()
    hf_model = HuggingFaceModel(MODEL, max_new_tokens=parent.MAX_NEW_TOKENS)
    hidden_dim = get_hidden_dim(hf_model.model)
    print(f"Model loaded in {time.time()-t0:.0f}s; hidden_dim={hidden_dim}")

    probe_l23 = load_l23_probe()
    cells = define_l23_cells(probe_l23, hidden_dim)

    if args.cells:
        wanted = set(args.cells.split(","))
        cells = [c for c in cells if c["name"] in wanted]
        print(f"Filtered to cells: {[c['name'] for c in cells]}")

    template = load_templates_from_yaml(str(parent.TEMPLATE_PATH))[0]
    builder = build_revealed_builder(template, "completion", post_task=False, system_prompt=None)

    for cell in cells:
        pairs = pairs_full if cell["use_full_pairs"] else pairs_subset
        if args.n_pairs:
            pairs = pairs[: args.n_pairs]
        print(f"\nCell: {cell['name']} (layers={cell['ablate_layers']}, n_pairs={len(pairs)})")
        out = RESULTS_DIR / cell["name"]
        await parent.run_cell(cell, hf_model, builder, pairs, out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cells", help="Comma-separated cell names (default: all)")
    p.add_argument("--n-pairs", type=int, default=None, help="Override pair count for pilot/debug")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
