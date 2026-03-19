"""KV cache manipulation utilities for isolated steering."""

from __future__ import annotations

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from src.models.architecture import get_v_proj_weight, get_num_kv_heads, get_head_dim


def project_to_v_space(model, layer_idx: int, direction: np.ndarray) -> torch.Tensor:
    """Project a residual-stream direction through W_v to get a V-space direction.

    Returns shape (num_kv_heads, head_dim) on model device.
    """
    w_v = get_v_proj_weight(model, layer_idx)  # (num_kv_heads * head_dim, hidden_size)
    d_vec = torch.tensor(direction, dtype=w_v.dtype, device=w_v.device)
    projected = w_v @ d_vec  # (num_kv_heads * head_dim,)
    n_heads = get_num_kv_heads(model)
    d_head = get_head_dim(model)
    return projected.reshape(n_heads, d_head)


def modify_cache_v_at_positions(
    cache: DynamicCache,
    layer_idx: int,
    start: int,
    end: int,
    v_direction: torch.Tensor,
    coefficient: float,
) -> None:
    """Add coefficient * v_direction to V cache entries at [start, end). In-place.

    v_direction: shape (num_kv_heads, head_dim)
    V cache shape: (batch, n_kv_heads, seq_len, d_head)
    """
    layer = cache.layers[layer_idx]
    # Clone to escape inference tensor restriction (cache created under inference_mode)
    new_values = layer.values.clone()
    new_values[:, :, start:end, :] += coefficient * v_direction.unsqueeze(1)
    layer.values = new_values


def combine_caches(
    base: DynamicCache,
    overlays: list[tuple[DynamicCache, int, int]],
) -> DynamicCache:
    """Clone base cache, then overwrite [start, end) from each overlay's source cache.

    overlays: list of (source_cache, start, end) — applied in order.
    """
    combined = DynamicCache()
    for layer_idx in range(len(base)):
        k = base.layers[layer_idx].keys.clone()
        v = base.layers[layer_idx].values.clone()
        for source, start, end in overlays:
            k[:, :, start:end, :] = source.layers[layer_idx].keys[:, :, start:end, :]
            v[:, :, start:end, :] = source.layers[layer_idx].values[:, :, start:end, :]
        combined.update(k, v, layer_idx)
    return combined
