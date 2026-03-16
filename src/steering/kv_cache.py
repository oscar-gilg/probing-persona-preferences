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
    cache_a: DynamicCache,
    cache_b: DynamicCache,
    b_start: int,
    b_end: int,
) -> DynamicCache:
    """Clone cache_a, overwrite [b_start, b_end) positions from cache_b at every layer.

    Both K and V are overwritten because the residual stream modification during
    the steered forward pass propagates through the full forward pass.
    """
    combined = DynamicCache()
    for layer_idx in range(len(cache_a)):
        k_a = cache_a.layers[layer_idx].keys
        v_a = cache_a.layers[layer_idx].values
        k_b = cache_b.layers[layer_idx].keys
        v_b = cache_b.layers[layer_idx].values
        k = k_a.clone()
        v = v_a.clone()
        k[:, :, b_start:b_end, :] = k_b[:, :, b_start:b_end, :]
        v[:, :, b_start:b_end, :] = v_b[:, :, b_start:b_end, :]
        combined.update(k, v, layer_idx)
    return combined
