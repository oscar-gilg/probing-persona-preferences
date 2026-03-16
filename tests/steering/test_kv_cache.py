"""Unit tests for KV cache manipulation utilities (no GPU required)."""

import numpy as np
import torch
import pytest
from unittest.mock import MagicMock
from transformers.cache_utils import DynamicCache

from src.steering.kv_cache import (
    combine_caches,
    modify_cache_v_at_positions,
    project_to_v_space,
)


def _make_cache(n_layers: int, batch: int, n_heads: int, seq_len: int, d_head: int) -> DynamicCache:
    cache = DynamicCache()
    for layer_idx in range(n_layers):
        k = torch.randn(batch, n_heads, seq_len, d_head)
        v = torch.randn(batch, n_heads, seq_len, d_head)
        cache.update(k, v, layer_idx)
    return cache


class TestModifyCacheVAtPositions:
    def test_modifies_target_positions_only(self):
        cache = _make_cache(n_layers=2, batch=1, n_heads=4, seq_len=10, d_head=8)
        original_v = cache.layers[0].values.clone()
        v_dir = torch.ones(4, 8)

        modify_cache_v_at_positions(cache, layer_idx=0, start=3, end=6, v_direction=v_dir, coefficient=1.0)

        # Positions 3-5 should be modified
        assert not torch.equal(cache.layers[0].values[:, :, 3:6, :], original_v[:, :, 3:6, :])
        # Positions outside 3-6 should be unchanged
        assert torch.equal(cache.layers[0].values[:, :, :3, :], original_v[:, :, :3, :])
        assert torch.equal(cache.layers[0].values[:, :, 6:, :], original_v[:, :, 6:, :])

    def test_coefficient_scaling(self):
        cache = _make_cache(n_layers=1, batch=1, n_heads=2, seq_len=5, d_head=4)
        original_v = cache.layers[0].values.clone()
        v_dir = torch.ones(2, 4)

        modify_cache_v_at_positions(cache, layer_idx=0, start=1, end=3, v_direction=v_dir, coefficient=2.5)

        diff = cache.layers[0].values[:, :, 1:3, :] - original_v[:, :, 1:3, :]
        expected = 2.5 * v_dir
        assert torch.allclose(diff[0], expected.unsqueeze(1).expand_as(diff[0]))

    def test_negative_coefficient(self):
        cache = _make_cache(n_layers=1, batch=1, n_heads=2, seq_len=5, d_head=4)
        original_v = cache.layers[0].values.clone()
        v_dir = torch.ones(2, 4)

        modify_cache_v_at_positions(cache, layer_idx=0, start=0, end=2, v_direction=v_dir, coefficient=-1.0)

        diff = cache.layers[0].values[:, :, 0:2, :] - original_v[:, :, 0:2, :]
        expected = -1.0 * v_dir
        assert torch.allclose(diff[0], expected.unsqueeze(1).expand_as(diff[0]))


class TestCombineCaches:
    def test_combines_correct_positions(self):
        cache_a = _make_cache(n_layers=2, batch=1, n_heads=4, seq_len=10, d_head=8)
        cache_b = _make_cache(n_layers=2, batch=1, n_heads=4, seq_len=10, d_head=8)

        combined = combine_caches(cache_a, cache_b, b_start=3, b_end=6)

        for layer_idx in range(2):
            k_a = cache_a.layers[layer_idx].keys
            v_a = cache_a.layers[layer_idx].values
            k_b = cache_b.layers[layer_idx].keys
            v_b = cache_b.layers[layer_idx].values
            k_c = combined.layers[layer_idx].keys
            v_c = combined.layers[layer_idx].values

            # Positions outside [3, 6) should come from cache_a
            assert torch.equal(k_c[:, :, :3, :], k_a[:, :, :3, :])
            assert torch.equal(k_c[:, :, 6:, :], k_a[:, :, 6:, :])
            assert torch.equal(v_c[:, :, :3, :], v_a[:, :, :3, :])
            assert torch.equal(v_c[:, :, 6:, :], v_a[:, :, 6:, :])
            # Positions [3, 6) should come from cache_b
            assert torch.equal(k_c[:, :, 3:6, :], k_b[:, :, 3:6, :])
            assert torch.equal(v_c[:, :, 3:6, :], v_b[:, :, 3:6, :])

    def test_does_not_modify_originals(self):
        cache_a = _make_cache(n_layers=1, batch=1, n_heads=2, seq_len=5, d_head=4)
        cache_b = _make_cache(n_layers=1, batch=1, n_heads=2, seq_len=5, d_head=4)
        a_k_orig = cache_a.layers[0].keys.clone()
        b_k_orig = cache_b.layers[0].keys.clone()

        combine_caches(cache_a, cache_b, b_start=1, b_end=3)

        assert torch.equal(cache_a.layers[0].keys, a_k_orig)
        assert torch.equal(cache_b.layers[0].keys, b_k_orig)

    def test_combined_is_independent_of_originals(self):
        cache_a = _make_cache(n_layers=1, batch=1, n_heads=2, seq_len=5, d_head=4)
        cache_b = _make_cache(n_layers=1, batch=1, n_heads=2, seq_len=5, d_head=4)

        combined = combine_caches(cache_a, cache_b, b_start=1, b_end=3)
        combined_k_orig = combined.layers[0].keys.clone()

        cache_a.layers[0].keys += 999.0
        assert torch.equal(combined.layers[0].keys, combined_k_orig)


class TestProjectToVSpace:
    def test_shape_and_device(self):
        mock_model = MagicMock()
        mock_model.config.model_type = "llama"
        mock_model.config.num_key_value_heads = 4
        mock_model.config.head_dim = 8
        del mock_model.config.text_config

        hidden_size = 16
        w_v = torch.randn(32, hidden_size)
        mock_layer = MagicMock()
        mock_layer.self_attn.v_proj.weight = w_v
        mock_model.model.layers.__getitem__ = MagicMock(return_value=mock_layer)

        direction = np.random.randn(hidden_size).astype(np.float32)
        result = project_to_v_space(mock_model, layer_idx=0, direction=direction)

        assert result.shape == (4, 8)

    def test_projection_correctness(self):
        mock_model = MagicMock()
        mock_model.config.model_type = "llama"
        mock_model.config.num_key_value_heads = 2
        mock_model.config.head_dim = 3
        del mock_model.config.text_config

        w_v = torch.eye(6, 6)
        mock_layer = MagicMock()
        mock_layer.self_attn.v_proj.weight = w_v
        mock_model.model.layers.__getitem__ = MagicMock(return_value=mock_layer)

        direction = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        result = project_to_v_space(mock_model, layer_idx=0, direction=direction)

        expected = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert torch.allclose(result, expected)
