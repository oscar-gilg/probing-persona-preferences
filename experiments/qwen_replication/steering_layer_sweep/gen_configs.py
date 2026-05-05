"""Generate steering YAML configs for the Qwen layer-sweep experiment.

CLI subcommands:
  phase-a              Emit 12 diagonal YAMLs across {tb1, tb4} x {12, 24, 28, 33, 38, 43}.
  phase-b --selector S --peak-layer L
                       Emit 2 harm-breakdown YAMLs (contrastive + single-task) at
                       the chosen (selector, layer).

Norms are computed inline via src.steering.calibration.per_layer_norms against the
matching activation NPZ for each selector and embedded in the YAML.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.steering.calibration import per_layer_norms


load_dotenv()

PROBE_LAYERS = [12, 24, 28, 33, 38, 43]
MULTIPLIERS = [-0.05, -0.03, 0.03, 0.05]

TEMPLATE_PATH = "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml"
PHASE_A_PAIRS = "experiments/qwen_replication/steering_layer_sweep/steering_pairs_50.json"
PHASE_B_PAIRS = "experiments/qwen_replication/steering_layer_sweep/steering_pairs_150.json"
CHECKPOINTS_DIR = "experiments/qwen_replication/steering_layer_sweep/checkpoints"
CONFIG_DIR = Path("configs/steering/qwen_layer_sweep")

# Selector slug used in filenames -> (probe-manifest dir, activation NPZ).
SELECTORS = {
    "tb1": {
        "manifest": "results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1/",
        "activations": Path("activations/qwen35_122b/pref_main/activations_turn_boundary:-1.npz"),
    },
    "tb4": {
        "manifest": "results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m4/",
        "activations": Path("activations/qwen35_122b/pref_main/activations_turn_boundary:-4.npz"),
    },
}


def _dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def _phase_a_config(selector: str, layer: int, mean_norm: float) -> dict:
    return {
        "model": "qwen3.5-122b-nothink",
        "max_new_tokens": 64,
        "pairs_path": PHASE_A_PAIRS,
        "probe_manifest": SELECTORS[selector]["manifest"],
        "checkpoint_path": f"{CHECKPOINTS_DIR}/phase_a_{selector}_L{layer:02d}.jsonl",
        "mean_norm": {int(layer): float(mean_norm)},
        "n_trials": 3,
        "temperature": 1.0,
        "seed": 42,
        "n_pairs": None,
        "template_path": TEMPLATE_PATH,
        "conditions": [
            {
                "name": f"phase_a_{selector}_L{layer:02d}",
                "cache_injection": "differential",
                "probe": f"ridge_L{layer}",
                "layers": [int(layer)],
                "multipliers": list(MULTIPLIERS),
                "spans": {"first": 1, "second": -1},
            }
        ],
    }


def _phase_b_contrastive_config(selector: str, layer: int, mean_norm: float) -> dict:
    return {
        "model": "qwen3.5-122b-nothink",
        "max_new_tokens": 64,
        "pairs_path": PHASE_B_PAIRS,
        "probe_manifest": SELECTORS[selector]["manifest"],
        "checkpoint_path": f"{CHECKPOINTS_DIR}/phase_b_{selector}_contrastive_L{layer:02d}_150.jsonl",
        "mean_norm": {int(layer): float(mean_norm)},
        "n_trials": 3,
        "temperature": 1.0,
        "seed": 42,
        "n_pairs": None,
        "template_path": TEMPLATE_PATH,
        "conditions": [
            {
                "name": f"contrastive_{selector}_L{layer:02d}",
                "cache_injection": "differential",
                "probe": f"ridge_L{layer}",
                "layers": [int(layer)],
                "multipliers": list(MULTIPLIERS),
                "spans": {"first": 1, "second": -1},
            }
        ],
    }


def _phase_b_single_task_config(selector: str, layer: int, mean_norm: float) -> dict:
    base_condition = {
        "cache_injection": "differential",
        "probe": f"ridge_L{layer}",
        "layers": [int(layer)],
        "multipliers": list(MULTIPLIERS),
    }
    return {
        "model": "qwen3.5-122b-nothink",
        "max_new_tokens": 64,
        "pairs_path": PHASE_B_PAIRS,
        "probe_manifest": SELECTORS[selector]["manifest"],
        "checkpoint_path": f"{CHECKPOINTS_DIR}/phase_b_{selector}_single_task_L{layer:02d}_150.jsonl",
        "mean_norm": {int(layer): float(mean_norm)},
        "n_trials": 3,
        "temperature": 1.0,
        "seed": 42,
        "n_pairs": None,
        "template_path": TEMPLATE_PATH,
        "conditions": [
            {
                "name": f"unilateral_first_{selector}_L{layer:02d}",
                **base_condition,
                "spans": {"first": 1},
            },
            {
                "name": f"unilateral_second_{selector}_L{layer:02d}",
                **base_condition,
                "spans": {"second": 1},
            },
        ],
    }


def _emit_phase_a() -> None:
    written: list[Path] = []
    for selector, meta in SELECTORS.items():
        print(f"Computing per-layer norms for selector={selector} from {meta['activations']}")
        norms = per_layer_norms(meta["activations"], layers=PROBE_LAYERS)
        for layer in PROBE_LAYERS:
            cfg = _phase_a_config(selector, layer, norms[layer])
            path = CONFIG_DIR / f"phase_a_{selector}_L{layer:02d}.yaml"
            _dump_yaml(path, cfg)
            written.append(path)
    print(f"\nWrote {len(written)} Phase A configs:")
    for p in written:
        print(f"  {p}")


def _emit_phase_b(selector: str, layer: int) -> None:
    if selector not in SELECTORS:
        raise ValueError(f"Unknown selector {selector}; choose from {list(SELECTORS)}")
    if layer not in PROBE_LAYERS:
        raise ValueError(f"Unknown layer {layer}; choose from {PROBE_LAYERS}")

    activations = SELECTORS[selector]["activations"]
    print(f"Computing norm at L{layer} from {activations}")
    norms = per_layer_norms(activations, layers=[layer])

    written: list[Path] = []
    for builder in (_phase_b_contrastive_config, _phase_b_single_task_config):
        cfg = builder(selector, layer, norms[layer])
        suffix = "contrastive" if builder is _phase_b_contrastive_config else "single_task"
        path = CONFIG_DIR / f"phase_b_{selector}_{suffix}_L{layer:02d}_150.yaml"
        _dump_yaml(path, cfg)
        written.append(path)
    print(f"\nWrote {len(written)} Phase B configs at (selector={selector}, layer={layer}):")
    for p in written:
        print(f"  {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    sub.add_parser("phase-a", help="Emit the 12 diagonal Phase A configs.")
    pb = sub.add_parser("phase-b", help="Emit the 2 harm-breakdown configs at the peak.")
    pb.add_argument("--selector", required=True, choices=list(SELECTORS.keys()))
    pb.add_argument("--peak-layer", type=int, required=True, choices=PROBE_LAYERS)
    args = parser.parse_args()

    if args.phase == "phase-a":
        _emit_phase_a()
    else:
        _emit_phase_b(args.selector, args.peak_layer)


if __name__ == "__main__":
    main()
