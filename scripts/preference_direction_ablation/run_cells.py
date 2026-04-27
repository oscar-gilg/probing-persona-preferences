"""Driver for the preference_direction_ablation experiment.

For each ablation cell, generate 3 canonical-order seeds + 1 swapped-order seed
per pair at T=0.7, parse via the standard CompletionChoiceFormat judge, and save
to per-cell JSONL.

Cells:
  - B0: no hooks (baseline, full 723 pairs)
  - A_L25_probe / A_L32_probe: rank-1 ablation at one layer (full 723)
  - B_two_probe: layer-matched ablation at L25 and L32 (full 723)
  - C_band_probe: L25 probe applied at every layer in [L25..L34] (full 723)
  - {cond}_random{0..4}: 5 isotropic unit-vector controls per ablation condition
    on a fixed 100-pair subset.

Reuses: build_revealed_builder, RevealedPreferenceMeasurer, CompletionChoiceFormat
parser. Only the per-pair loop is custom (the standard runner builds C(N,2) pairs
from a task list, not arbitrary pair lists).

Usage:
    python -m scripts.preference_direction_ablation.run_cells [--cells <name1,name2,...>]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
from src.measurement.runners.runners import build_revealed_builder
from src.models.architecture import get_hidden_dim
from src.models.huggingface_model import HuggingFaceModel
from src.steering.client import SteeredHFClient
from src.task_data import load_filtered_tasks, parse_origins
from src.types import BinaryPreferenceMeasurement

load_dotenv()


# --- paths and constants ---
REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = REPO_ROOT / "experiments/preference_direction_ablation"
RESULTS_DIR = EXP_DIR / "results"
PROBE_DIR = REPO_ROOT / "results/probes/heldout_eval_gemma3_tb-1/probes"
PAIRS_FILE = (
    REPO_ROOT
    / "results/experiments/uniform_eval_gemma3_27b_v3"
    / "pre_task_revealed"
    / "completion_preference_gemma-3-27b_completion_canonical_seed0_uniform_eval"
    / "measurements.yaml"
)
TEMPLATE_PATH = REPO_ROOT / "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml"

MODEL = "google/gemma-3-27b-it"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
N_CANONICAL_SEEDS = 3
RANDOM_SUBSET_N = 100
RANDOM_SUBSET_SEED = 42
ORIGINS = ["wildchat", "alpaca", "math", "bailbench", "stress_test"]


# --- pair / task loading ---

def load_unique_pair_ids() -> list[tuple[str, str]]:
    with PAIRS_FILE.open() as f:
        rows = yaml.safe_load(f)
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for row in rows:
        pair = (row["task_a"], row["task_b"])
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def build_task_lookup(pair_ids: list[tuple[str, str]]) -> dict:
    needed: set[str] = set()
    for a, b in pair_ids:
        needed.add(a)
        needed.add(b)
    tasks = load_filtered_tasks(
        n=len(needed),
        origins=parse_origins(ORIGINS),
        task_ids=needed,
        seed=None,
    )
    lookup = {t.id: t for t in tasks}
    missing = needed - set(lookup)
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} task IDs from datasets (sample: {sorted(missing)[:5]})"
        )
    return lookup


# --- probe and random direction utilities ---

def load_probes() -> dict[int, np.ndarray]:
    probes: dict[int, np.ndarray] = {}
    for layer in (25, 32):
        weights = np.load(PROBE_DIR / f"probe_ridge_L{layer}.npy")
        d = weights[:-1]  # strip intercept
        d = d / np.linalg.norm(d)
        probes[layer] = d.astype(np.float32)
    return probes


def random_unit(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


# --- cell definitions ---

def define_cells(probes: dict[int, np.ndarray], dim: int) -> list[dict]:
    cells: list[dict] = []
    cells.append({"name": "B0", "ablate_layers": [], "ablate_directions": [], "use_full_pairs": True})
    cells.append({"name": "A_L25_probe", "ablate_layers": [25], "ablate_directions": [probes[25]], "use_full_pairs": True})
    cells.append({"name": "A_L32_probe", "ablate_layers": [32], "ablate_directions": [probes[32]], "use_full_pairs": True})
    cells.append({"name": "B_two_probe", "ablate_layers": [25, 32], "ablate_directions": [probes[25], probes[32]], "use_full_pairs": True})
    cells.append({"name": "C_band_probe", "ablate_layers": list(range(25, 35)), "ablate_directions": [probes[25]] * 10, "use_full_pairs": True})

    for cond_name, layers in (("A_L25", [25]), ("A_L32", [32]), ("B_two", [25, 32]), ("C_band", list(range(25, 35)))):
        for s in range(5):
            if cond_name in ("A_L25", "A_L32"):
                dirs = [random_unit(dim, seed=s)]
            elif cond_name == "B_two":
                dirs = [random_unit(dim, seed=100 * s + i) for i in range(2)]
            else:  # C_band — same single random direction at all 10 layers (mirror probe variant)
                d = random_unit(dim, seed=s)
                dirs = [d] * len(layers)
            cells.append({
                "name": f"{cond_name}_random{s}",
                "ablate_layers": layers,
                "ablate_directions": dirs,
                "use_full_pairs": False,
            })
    return cells


# --- per-pair generation + parse ---

async def measure_pair(client, builder, task_a, task_b) -> list[dict]:
    rows: list[dict] = []
    for order, ta, tb in (("canonical", task_a, task_b), ("swapped", task_b, task_a)):
        n = N_CANONICAL_SEEDS if order == "canonical" else 1
        prompt = builder.build(ta, tb)
        responses = client.generate_n(
            prompt.messages, n=n, temperature=TEMPERATURE,
            task_prompts=[t.prompt for t in prompt.tasks],
        )
        parse_results = await asyncio.gather(
            *[prompt.measurer.parse(r, prompt) for r in responses],
            return_exceptions=True,
        )
        for i, (resp, parsed) in enumerate(zip(responses, parse_results)):
            if isinstance(parsed, Exception):
                choice_presented = "parse_error"
            elif isinstance(parsed.result, BinaryPreferenceMeasurement):
                choice_presented = parsed.result.choice
            else:
                choice_presented = "refusal"
            if order == "swapped" and choice_presented in ("a", "b"):
                choice_canonical = "b" if choice_presented == "a" else "a"
            else:
                choice_canonical = choice_presented
            rows.append({
                "task_a": task_a.id,
                "task_b": task_b.id,
                "order": order,
                "sample_idx": i,
                "choice_presented": choice_presented,
                "choice_canonical": choice_canonical,
                "raw_response": resp,
            })
    return rows


# --- per-cell driver ---

async def run_cell(cell: dict, hf_model: HuggingFaceModel, builder, pairs: list, output_dir: Path) -> None:
    client = SteeredHFClient(
        hf_model=hf_model,
        ablate_layers=cell["ablate_layers"],
        ablate_directions=cell["ablate_directions"],
    )
    output = output_dir / "measurements.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    done_pairs: set[tuple[str, str]] = set()
    if output.exists():
        for line in output.read_text().splitlines():
            d = json.loads(line)
            done_pairs.add((d["task_a"], d["task_b"]))

    todo = [(a, b) for a, b in pairs if (a.id, b.id) not in done_pairs]
    print(f"  [{cell['name']}] {len(todo)} pairs to generate ({len(done_pairs)} already done)")

    if not todo:
        return

    t0 = time.time()
    with output.open("a") as f:
        for i, (a, b) in enumerate(todo):
            try:
                rows = await measure_pair(client, builder, a, b)
            except Exception as exc:
                print(f"  [{cell['name']}] pair ({a.id},{b.id}) failed: {type(exc).__name__}: {exc}")
                continue
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if (i + 1) % 25 == 0 or (i + 1) == len(todo):
                rate = (i + 1) / (time.time() - t0)
                eta = (len(todo) - (i + 1)) / rate / 60
                print(f"  [{cell['name']}] {i+1}/{len(todo)} pairs · {rate:.2f} pair/s · eta {eta:.1f}m")
    print(f"  [{cell['name']}] DONE in {(time.time()-t0)/60:.1f}m")


# --- main ---

async def main_async(args: argparse.Namespace) -> None:
    pair_ids = load_unique_pair_ids()
    print(f"Loaded {len(pair_ids)} unique pairs")
    task_lookup = build_task_lookup(pair_ids)
    print(f"Reconstructed {len(task_lookup)} unique tasks")
    pairs_full = [(task_lookup[a], task_lookup[b]) for a, b in pair_ids]

    rng = np.random.default_rng(RANDOM_SUBSET_SEED)
    subset_idx = rng.choice(len(pairs_full), size=RANDOM_SUBSET_N, replace=False)
    pairs_subset = [pairs_full[i] for i in subset_idx]

    print(f"Loading {MODEL}...")
    t0 = time.time()
    hf_model = HuggingFaceModel(MODEL, max_new_tokens=MAX_NEW_TOKENS)
    hidden_dim = get_hidden_dim(hf_model.model)
    print(f"Model loaded in {time.time()-t0:.0f}s; hidden_dim={hidden_dim}")

    probes = load_probes()
    cells = define_cells(probes, hidden_dim)

    if args.cells:
        wanted = set(args.cells.split(","))
        cells = [c for c in cells if c["name"] in wanted]
        print(f"Filtered to cells: {[c['name'] for c in cells]}")

    template = load_templates_from_yaml(str(TEMPLATE_PATH))[0]
    builder = build_revealed_builder(template, "completion", post_task=False, system_prompt=None)

    for cell in cells:
        pairs = pairs_full if cell["use_full_pairs"] else pairs_subset
        if args.n_pairs:
            pairs = pairs[: args.n_pairs]
        print(f"\nCell: {cell['name']} (layers={cell['ablate_layers']}, n_pairs={len(pairs)})")
        out = RESULTS_DIR / cell["name"]
        await run_cell(cell, hf_model, builder, pairs, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", help="Comma-separated cell names (default: all)")
    parser.add_argument("--n-pairs", type=int, default=None, help="Override pair count for pilot/debug")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
