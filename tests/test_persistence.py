"""
Tests for TQKV binary persistence format.
"""

import os
import tempfile
import pytest
import torch
import numpy as np

from turboquant import TurboQuant
from turboquant.packing import packed_size
from aither_kvcache.persistence import (
    save_tqkv, load_tqkv, mmap_tqkv,
    _pack_header, _unpack_header,
    MAGIC, FORMAT_VERSION, HEADER_SIZE,
)


class TestHeader:
    def test_header_roundtrip(self):
        header = _pack_header(
            bits=4, head_dim=128, num_heads=8,
            num_layers=32, block_size=16, num_blocks=100,
        )
        assert len(header) == HEADER_SIZE
        meta = _unpack_header(header)
        assert meta["bits"] == 4
        assert meta["head_dim"] == 128
        assert meta["num_heads"] == 8
        assert meta["num_layers"] == 32
        assert meta["block_size"] == 16
        assert meta["num_blocks"] == 100
        assert meta["has_correction"] is False

    def test_header_with_correction_flag(self):
        header = _pack_header(
            bits=3, head_dim=64, num_heads=4,
            num_layers=16, block_size=8, num_blocks=50,
            has_correction=True,
        )
        meta = _unpack_header(header)
        assert meta["has_correction"] is True
        assert meta["bits"] == 3

    def test_invalid_magic(self):
        bad = b"XXXX" + b"\x00" * 28
        with pytest.raises(ValueError, match="Invalid magic"):
            _unpack_header(bad)

    def test_short_header(self):
        with pytest.raises(ValueError, match="Header too short"):
            _unpack_header(b"TQ")


class TestSaveLoad:
    @pytest.fixture
    def tqkv_data(self):
        num_blocks = 4
        block_size = 16
        num_heads = 8
        head_dim = 128
        bits = 4
        pd = packed_size(head_dim, bits)

        k_packed = torch.randint(0, 256, (num_blocks, block_size, num_heads, pd),
                                 dtype=torch.uint8)
        k_norms = torch.randn(num_blocks, block_size, num_heads).float()
        v_packed = torch.randint(0, 256, (num_blocks, block_size, num_heads, pd),
                                 dtype=torch.uint8)
        v_norms = torch.randn(num_blocks, block_size, num_heads).float()
        return k_packed, k_norms, v_packed, v_norms, bits, head_dim

    def test_save_load_roundtrip(self, tqkv_data):
        k_packed, k_norms, v_packed, v_norms, bits, head_dim = tqkv_data
        with tempfile.NamedTemporaryFile(suffix=".tqkv", delete=False) as f:
            path = f.name

        try:
            save_tqkv(path, k_packed, k_norms, v_packed, v_norms,
                       bits=bits, head_dim=head_dim)

            k2, kn2, v2, vn2, meta = load_tqkv(path)
            assert meta["bits"] == bits
            assert meta["head_dim"] == head_dim
            assert meta["num_blocks"] == k_packed.shape[0]
            assert meta["block_size"] == k_packed.shape[1]
            assert meta["num_heads"] == k_packed.shape[2]

            torch.testing.assert_close(k2, k_packed)
            torch.testing.assert_close(kn2, k_norms)
            torch.testing.assert_close(v2, v_packed)
            torch.testing.assert_close(vn2, v_norms)
        finally:
            os.unlink(path)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_all_bit_widths(self, bits):
        num_blocks, block_size, num_heads, head_dim = 2, 8, 4, 64
        pd = packed_size(head_dim, bits)

        k_packed = torch.randint(0, 256, (num_blocks, block_size, num_heads, pd),
                                 dtype=torch.uint8)
        k_norms = torch.randn(num_blocks, block_size, num_heads).float()
        v_packed = torch.randint(0, 256, (num_blocks, block_size, num_heads, pd),
                                 dtype=torch.uint8)
        v_norms = torch.randn(num_blocks, block_size, num_heads).float()

        with tempfile.NamedTemporaryFile(suffix=".tqkv", delete=False) as f:
            path = f.name

        try:
            save_tqkv(path, k_packed, k_norms, v_packed, v_norms,
                       bits=bits, head_dim=head_dim)
            k2, kn2, v2, vn2, meta = load_tqkv(path)
            assert meta["bits"] == bits
            torch.testing.assert_close(k2, k_packed)
        finally:
            os.unlink(path)

    def test_mmap(self, tqkv_data):
        k_packed, k_norms, v_packed, v_norms, bits, head_dim = tqkv_data
        with tempfile.NamedTemporaryFile(suffix=".tqkv", delete=False) as f:
            path = f.name

        try:
            save_tqkv(path, k_packed, k_norms, v_packed, v_norms,
                       bits=bits, head_dim=head_dim)

            k_mm, kn_mm, v_mm, vn_mm, meta = mmap_tqkv(path)
            assert meta["bits"] == bits
            np.testing.assert_array_equal(k_mm, k_packed.numpy())
            np.testing.assert_array_almost_equal(kn_mm, k_norms.numpy(), decimal=5)
            # Release mmap references before cleanup (Windows file locking)
            del k_mm, kn_mm, v_mm, vn_mm
        finally:
            try:
                os.unlink(path)
            except PermissionError:
                pass  # Windows mmap may hold the file briefly

    def test_truncated_file(self, tqkv_data):
        k_packed, k_norms, v_packed, v_norms, bits, head_dim = tqkv_data
        with tempfile.NamedTemporaryFile(suffix=".tqkv", delete=False) as f:
            path = f.name

        try:
            save_tqkv(path, k_packed, k_norms, v_packed, v_norms,
                       bits=bits, head_dim=head_dim)

            # Truncate the file
            with open(path, "r+b") as f:
                f.truncate(HEADER_SIZE + 10)

            with pytest.raises(ValueError, match="File too short"):
                load_tqkv(path)
        finally:
            os.unlink(path)
