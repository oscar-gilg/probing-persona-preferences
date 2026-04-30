"""Persona-vector computation: mean-difference of filtered response activations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

LAYER_KEY_PREFIX = "layer_"


@dataclass(frozen=True)
class PersonaVectorMetadata:
    persona: str
    model: str
    selector: str
    layers: list[int]
    n_pos_total: int
    n_neg_total: int
    n_pos_kept: int
    n_neg_kept: int
    pos_keep_threshold: float
    neg_keep_threshold: float
    coefficient_multipliers: list[float]


def compute_persona_vector(
    pos_acts: dict[int, np.ndarray],
    neg_acts: dict[int, np.ndarray],
    keep_pos_idx: list[int],
    keep_neg_idx: list[int],
    *,
    unit_normalize: bool = True,
) -> dict[int, np.ndarray]:
    """Mean-diff persona vector per layer.

    Args:
        pos_acts: layer -> (N_pos, D) activations under positive prompts.
        neg_acts: layer -> (N_neg, D) activations under negative prompts.
        keep_pos_idx, keep_neg_idx: indices to retain (filtered by trait score,
            refusal, coherence by the orchestrator).
        unit_normalize: if True (default), L2-normalize the per-layer mean-diff so
            downstream `coefficient = multiplier × mean_norm` directly equals the
            perturbation magnitude. The raw mean-diff norm depends on the kept set
            size and trait separability, which varies across personas/models —
            unit-normalizing makes coefficient calibration portable.

    Returns:
        layer -> (D,) mean-diff direction (unit-norm if unit_normalize=True).
    """
    if not keep_pos_idx or not keep_neg_idx:
        raise ValueError("Cannot compute persona vector with empty keep set.")
    if pos_acts.keys() != neg_acts.keys():
        raise ValueError(f"Layer mismatch: pos {pos_acts.keys()} vs neg {neg_acts.keys()}")

    out: dict[int, np.ndarray] = {}
    for layer in sorted(pos_acts.keys()):
        pos = pos_acts[layer][keep_pos_idx]
        neg = neg_acts[layer][keep_neg_idx]
        diff = pos.mean(axis=0) - neg.mean(axis=0)
        if unit_normalize:
            norm = np.linalg.norm(diff)
            if norm > 0:
                diff = diff / norm
        out[layer] = diff
    return out


def per_layer_mean_norm(
    pos_acts: dict[int, np.ndarray],
    neg_acts: dict[int, np.ndarray],
    keep_pos_idx: list[int],
    keep_neg_idx: list[int],
) -> dict[int, float]:
    """Mean L2 norm of the kept activations, per layer. Used to convert
    fractional steering coefficients (e.g. 0.05 × mean_norm) to raw α.
    """
    out: dict[int, float] = {}
    for layer in sorted(pos_acts.keys()):
        kept = np.concatenate(
            [pos_acts[layer][keep_pos_idx], neg_acts[layer][keep_neg_idx]],
            axis=0,
        )
        out[layer] = float(np.linalg.norm(kept, axis=1).mean())
    return out


def save_persona_vector(
    out_path: Path,
    vectors: dict[int, np.ndarray],
    mean_norms: dict[int, float],
    metadata: PersonaVectorMetadata,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"{LAYER_KEY_PREFIX}{layer}": vec for layer, vec in vectors.items()}
    arrays["mean_norms_layers"] = np.array(sorted(mean_norms.keys()), dtype=np.int32)
    arrays["mean_norms_values"] = np.array(
        [mean_norms[layer] for layer in sorted(mean_norms.keys())], dtype=np.float64
    )
    np.savez(out_path, **arrays)
    metadata_path = out_path.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2))


def load_persona_vector(path: Path) -> tuple[dict[int, np.ndarray], dict[int, float], dict]:
    data = np.load(path)
    vectors: dict[int, np.ndarray] = {}
    for key in data.files:
        if key.startswith(LAYER_KEY_PREFIX):
            layer = int(key[len(LAYER_KEY_PREFIX):])
            vectors[layer] = data[key]
    layers_arr = data["mean_norms_layers"]
    values_arr = data["mean_norms_values"]
    mean_norms = {int(layer): float(value) for layer, value in zip(layers_arr, values_arr, strict=True)}
    metadata_path = path.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return vectors, mean_norms, metadata
