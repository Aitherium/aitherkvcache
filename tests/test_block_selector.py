"""
Tests for BlockSelector (sparse block attention filtering).
"""

import pytest
import torch

from aither_kvcache.block_selector import BlockRepresentativeCache, BlockSelector


class TestBlockRepresentativeCache:
    def test_init(self):
        centroids = torch.randn(16)
        cache = BlockRepresentativeCache(
            max_blocks=100, num_kv_heads=8, half_d=64,
            device=torch.device("cpu"), centroids=centroids,
        )
        assert cache.max_blocks == 100
        assert cache.half_d == 64

    def test_shapes(self):
        centroids = torch.randn(16)
        cache = BlockRepresentativeCache(
            max_blocks=50, num_kv_heads=4, half_d=64,
            device=torch.device("cpu"), centroids=centroids,
        )
        assert cache._acc_even.shape == (50, 4, 64)
        assert cache._acc_odd.shape == (50, 4, 64)
        assert cache._counts.shape == (50, 4)


class TestBlockSelector:
    def test_disabled_by_default(self):
        """select_ratio=1.0 means no filtering."""
        centroids = torch.randn(16)
        selector = BlockSelector(
            max_blocks=100, num_kv_heads=8, half_d=64,
            device=torch.device("cpu"), centroids=centroids,
            select_ratio=1.0,
        )
        assert not selector.enabled

    def test_enabled_with_ratio(self):
        centroids = torch.randn(16)
        selector = BlockSelector(
            max_blocks=100, num_kv_heads=8, half_d=64,
            device=torch.device("cpu"), centroids=centroids,
            select_ratio=0.5,
        )
        assert selector.enabled
        assert selector.max_selected > 0

    def test_select_passthrough_when_disabled(self):
        """When disabled, select() returns inputs unchanged."""
        centroids = torch.randn(16)
        selector = BlockSelector(
            max_blocks=100, num_kv_heads=8, half_d=64,
            device=torch.device("cpu"), centroids=centroids,
            select_ratio=1.0,
        )
        bt = torch.arange(4).unsqueeze(0)
        cl = torch.tensor([64])
        q_even = torch.randn(1, 8, 64)
        q_odd = torch.randn(1, 8, 64)

        new_bt, new_cl = selector.select(
            q_even, q_odd, bt, cl, gqa_ratio=1, block_size=16)
        torch.testing.assert_close(new_bt, bt)
        torch.testing.assert_close(new_cl, cl)
