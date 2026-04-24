"""Generate steering YAML configs for the layer sweep.

One config per (selector, probe_layer). Diagonal configs only inject at the
probe's own layer; spine configs inject across all layers <= 35. Each config's
`mean_norm:` is a dict {injection_layer: norm} populated via
src.steering.calibration.per_layer_norms.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.steering.calibration import per_layer_norms


load_dotenv()

ALL_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59]
SPINE_INJECT_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
MULTIPLIERS = [-0.05, -0.03, 0.03, 0.05]
TEMPLATE_PATH = "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml"
PAIRS_PATH = "experiments/layer_sweep/steering_pairs.json"
CHECKPOINTS_DIR = "experiments/layer_sweep/checkpoints"
CONFIG_DIR = Path("configs/steering/layer_sweep")

SELECTOR_ACTIVATIONS = {
    "tb-2": Path("activations/gemma-3-27b_it/pref_layer_sweep/activations_turn_boundary:-2.npz"),
    "eot": Path("activations/gemma-3-27b_it/pref_layer_sweep/activations_eot.npz"),
}


def _n_trials(probe_layer: int) -> int:
    return 3 if probe_layer <= 35 else 1


def _config_for_probe(
    selector: str,
    probe_layer: int,
    layers: list[int],
    mean_norm_dict: dict[int, float],
    n_trials: int,
) -> dict:
    probe_id = f"ridge_L{probe_layer:02d}"
    return {
        "model": "gemma-3-27b",
        "max_new_tokens": 64,
        "pairs_path": PAIRS_PATH,
        "probe_manifest": f"results/probes/layer_sweep/{selector}/",
        "checkpoint_path": f"{CHECKPOINTS_DIR}/{selector}_probe_L{probe_layer:02d}.jsonl",
        "mean_norm": {int(L): float(mean_norm_dict[L]) for L in layers},
        "n_trials": n_trials,
        "temperature": 1.0,
        "seed": 42,
        "n_pairs": None,
        "template_path": TEMPLATE_PATH,
        "conditions": [
            {
                "name": f"probe_L{probe_layer:02d}",
                "cache_injection": "differential",
                "probe": probe_id,
                "layers": list(layers),
                "multipliers": list(MULTIPLIERS),
            }
        ],
    }


def _dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector", required=True, choices=list(SELECTOR_ACTIVATIONS.keys()))
    parser.add_argument(
        "--spine-layers",
        required=True,
        help="Comma-separated probe layers to use as the spine (e.g. '11,23,32,44,53')",
    )
    args = parser.parse_args()

    spine_layers = [int(x.strip()) for x in args.spine_layers.split(",") if x.strip()]
    unknown = [L for L in spine_layers if L not in ALL_LAYERS]
    if unknown:
        raise ValueError(f"Spine layers {unknown} not in the sampled layer set {ALL_LAYERS}")

    activations_path = SELECTOR_ACTIVATIONS[args.selector]
    needed_layers = sorted(set(ALL_LAYERS) | set(SPINE_INJECT_LAYERS))
    print(f"Computing per-layer norms for {len(needed_layers)} layers in {activations_path}")
    norms = per_layer_norms(activations_path, layers=needed_layers)

    written: list[Path] = []
    for probe_layer in ALL_LAYERS:
        if probe_layer in spine_layers:
            layers = list(SPINE_INJECT_LAYERS)
            n_trials = 3
        else:
            layers = [probe_layer]
            n_trials = _n_trials(probe_layer)

        cfg = _config_for_probe(
            selector=args.selector,
            probe_layer=probe_layer,
            layers=layers,
            mean_norm_dict=norms,
            n_trials=n_trials,
        )
        path = CONFIG_DIR / f"{args.selector}_probe_L{probe_layer:02d}.yaml"
        _dump_yaml(path, cfg)
        written.append(path)

    print(f"\nWrote {len(written)} configs under {CONFIG_DIR.resolve()}:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
