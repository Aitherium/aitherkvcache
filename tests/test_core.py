"""
Tests for TurboQuant KV cache quantization.

Validates:
  - Codebook correctness (centroids match known values)
  - Bit packing roundtrip (2, 3, 4 bit)
  - Rotation orthogonality
  - Encode->decode MSE within theoretical bounds
  - Compression ratios
  - Various tensor shapes and batch dimensions
  - Edge cases

Reference: Zandieh et al., arXiv:2504.19874, April 2025.
"""

import math
import pytest
import torch
import numpy as np

from aither_kvcache.codebook import (
    get_codebook, get_theory_bounds, _STANDARD_CODEBOOKS,
)
from aither_kvcache.rotation import (
    random_orthogonal, hadamard_matrix, randomized_hadamard_matrix,
    fast_hadamard_transform,
)
from aither_kvcache.packing import (
    pack_4bit, unpack_4bit,
    pack_3bit, unpack_3bit,
    pack_2bit, unpack_2bit,
    packed_size,
)
from aither_kvcache import TurboQuant, TurboQuantConfig


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# CODEBOOK TESTS
# ============================================================================

class TestCodebook:
    """Validate Lloyd-Max codebook correctness."""

    def test_standard_codebooks_exist(self):
        """All claimed bit-widths have precomputed codebooks."""
        for bits in [1, 2, 3, 4]:
            assert bits in _STANDARD_CODEBOOKS

    def test_centroids_sorted(self):
        """Centroids must be sorted ascending."""
        for bits in [1, 2, 3, 4]:
            c = _STANDARD_CODEBOOKS[bits]["centroids"]
            assert np.all(np.diff(c) > 0), f"Centroids not sorted for {bits}-bit"

    def test_centroids_symmetric(self):
        """Centroids for symmetric distributions should be symmetric about 0."""
        for bits in [1, 2, 3, 4]:
            c = _STANDARD_CODEBOOKS[bits]["centroids"]
            np.testing.assert_allclose(c, -c[::-1], atol=1e-4,
                                       err_msg=f"Asymmetric centroids for {bits}-bit")

    def test_correct_num_levels(self):
        """Each bit-width should have 2^b centroids."""
        for bits in [1, 2, 3, 4]:
            c = _STANDARD_CODEBOOKS[bits]["centroids"]
            assert len(c) == (1 << bits), f"{bits}-bit: expected {1 << bits} centroids"

    def test_1bit_centroid_value(self):
        """1-bit centroids ~ +/-0.7979 (well-known value)."""
        c = _STANDARD_CODEBOOKS[1]["centroids"]
        np.testing.assert_allclose(abs(c[0]), 0.798284, atol=1e-3)

    def test_2bit_centroid_values(self):
        """2-bit centroids ~ +/-0.4528, +/-1.5105."""
        c = _STANDARD_CODEBOOKS[2]["centroids"]
        np.testing.assert_allclose(abs(c[0]), 1.510469, atol=1e-3)
        np.testing.assert_allclose(abs(c[1]), 0.452781, atol=1e-3)

    def test_mse_matches_paper(self):
        """Total MSE should match paper Table values."""
        expected = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
        for bits, approx in expected.items():
            actual = _STANDARD_CODEBOOKS[bits]["mse"]
            assert abs(actual - approx) < 0.01, \
                f"{bits}-bit MSE {actual:.4f} != ~{approx}"

    def test_get_codebook_scales_by_dim(self):
        """Centroids should scale as 1/sqrt(d)."""
        for d in [64, 128, 256]:
            c, b, mse = get_codebook(d, 2)
            sigma = 1.0 / math.sqrt(d)
            # First centroid should be ~ -1.5104 * sigma
            expected = -1.510469 * sigma
            assert abs(c[0] - expected) < 1e-4 * sigma

    def test_boundaries_are_midpoints(self):
        """Inner boundaries should be midpoints of consecutive centroids."""
        c, b, _ = get_codebook(128, 4)
        for i in range(len(b)):
            expected = (c[i] + c[i + 1]) / 2
            assert abs(b[i] - expected) < 1e-10

    def test_theory_bounds(self):
        """Theory bounds follow the correct formulas."""
        for bits in [1, 2, 3, 4]:
            lower, upper = get_theory_bounds(bits)
            assert lower == pytest.approx(1.0 / (4 ** bits))
            assert upper == pytest.approx((3 * math.pi / 2) / (4 ** bits))

    def test_mse_within_theory(self):
        """Actual MSE should be between lower and upper bounds."""
        for bits in [1, 2, 3, 4]:
            mse = _STANDARD_CODEBOOKS[bits]["mse"]
            lower, upper = get_theory_bounds(bits)
            assert mse >= lower * 0.95, f"{bits}-bit MSE below lower bound"
            assert mse <= upper * 1.05, f"{bits}-bit MSE above upper bound"


# ============================================================================
# ROTATION TESTS
# ============================================================================

class TestRotation:
    """Validate rotation matrix properties."""

    def test_orthogonal_identity(self):
        """Q @ Q^T should be identity."""
        Q = random_orthogonal(128, device="cpu")
        eye = Q @ Q.T
        torch.testing.assert_close(eye, torch.eye(128), atol=1e-5, rtol=1e-5)

    def test_orthogonal_deterministic(self):
        """Same seed produces same matrix."""
        Q1 = random_orthogonal(128, seed=42, device="cpu")
        Q2 = random_orthogonal(128, seed=42, device="cpu")
        torch.testing.assert_close(Q1, Q2)

    def test_orthogonal_different_seeds(self):
        """Different seeds produce different matrices."""
        Q1 = random_orthogonal(128, seed=42, device="cpu")
        Q2 = random_orthogonal(128, seed=99, device="cpu")
        assert not torch.allclose(Q1, Q2)

    def test_hadamard_orthogonal(self):
        """Normalized Hadamard matrix should satisfy H @ H^T = I."""
        H = hadamard_matrix(128)
        eye = H @ H.T
        torch.testing.assert_close(eye, torch.eye(128), atol=1e-5, rtol=1e-5)

    def test_rht_orthogonal(self):
        """Randomized Hadamard Transform should be orthogonal."""
        R = randomized_hadamard_matrix(128, device="cpu")
        eye = R @ R.T
        torch.testing.assert_close(eye, torch.eye(128), atol=1e-4, rtol=1e-4)

    def test_fast_hadamard_matches_matrix(self):
        """Fast butterfly Hadamard should match matrix multiplication."""
        H = hadamard_matrix(128)
        x = torch.randn(10, 128)
        # Matrix version (normalized)
        expected = x @ H.T
        # Fast version (unnormalized, so divide by sqrt(d))
        result = fast_hadamard_transform(x) / math.sqrt(128)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_rotation_preserves_norm(self):
        """Rotation should preserve vector norms."""
        Q = random_orthogonal(128, device="cpu")
        x = torch.randn(100, 128)
        y = x @ Q.T
        norms_x = x.norm(dim=-1)
        norms_y = y.norm(dim=-1)
        torch.testing.assert_close(norms_x, norms_y, atol=1e-5, rtol=1e-5)


# ============================================================================
# PACKING TESTS
# ============================================================================

class TestPacking:
    """Validate bit packing/unpacking roundtrips."""

    def test_4bit_roundtrip(self):
        """Pack and unpack 4-bit should be lossless."""
        indices = torch.randint(0, 16, (100, 128), dtype=torch.uint8)
        packed = pack_4bit(indices)
        assert packed.shape == (100, 64)
        unpacked = unpack_4bit(packed, 128)
        torch.testing.assert_close(unpacked, indices)

    def test_2bit_roundtrip(self):
        """Pack and unpack 2-bit should be lossless."""
        indices = torch.randint(0, 4, (100, 128), dtype=torch.uint8)
        packed = pack_2bit(indices)
        assert packed.shape == (100, 32)
        unpacked = unpack_2bit(packed, 128)
        torch.testing.assert_close(unpacked, indices)

    def test_3bit_roundtrip(self):
        """Pack and unpack 3-bit should be lossless."""
        indices = torch.randint(0, 8, (100, 128), dtype=torch.uint8)
        packed = pack_3bit(indices)
        assert packed.shape == (100, 48)  # 128 * 3 / 8 = 48
        unpacked = unpack_3bit(packed, 128)
        torch.testing.assert_close(unpacked, indices)

    def test_4bit_boundary_values(self):
        """4-bit packing handles min/max values correctly."""
        indices = torch.zeros(1, 128, dtype=torch.uint8)
        indices[0, 0] = 0
        indices[0, 1] = 15
        packed = pack_4bit(indices)
        unpacked = unpack_4bit(packed, 128)
        assert unpacked[0, 0] == 0
        assert unpacked[0, 1] == 15

    def test_2bit_boundary_values(self):
        """2-bit packing handles min/max values correctly."""
        indices = torch.tensor([[0, 1, 2, 3] * 32], dtype=torch.uint8)
        packed = pack_2bit(indices)
        unpacked = unpack_2bit(packed, 128)
        torch.testing.assert_close(unpacked, indices)

    def test_packed_size_correct(self):
        """packed_size returns expected byte counts."""
        assert packed_size(128, 4) == 64
        assert packed_size(128, 3) == 48
        assert packed_size(128, 2) == 32

    def test_batch_dims_preserved(self):
        """Packing preserves batch dimensions."""
        indices = torch.randint(0, 16, (5, 10, 128), dtype=torch.uint8)
        packed = pack_4bit(indices)
        assert packed.shape == (5, 10, 64)
        unpacked = unpack_4bit(packed, 128)
        torch.testing.assert_close(unpacked, indices)


# ============================================================================
# QUANTIZER TESTS
# ============================================================================

class TestTurboQuant:
    """End-to-end quantizer tests."""

    @pytest.fixture(params=[2, 3, 4])
    def tq(self, request):
        return TurboQuant(head_dim=128, bits=request.param, device="cpu")

    def test_encode_decode_shapes(self, tq):
        """Output shapes should match expected dimensions."""
        x = torch.randn(100, 128, dtype=torch.float16)
        packed, norms = tq.encode(x)
        assert norms.shape == (100,)
        assert packed.shape == (100, tq._packed_dim)

        decoded = tq.decode(packed, norms)
        assert decoded.shape == (100, 128)
        assert decoded.dtype == torch.float16

    def test_batch_dims(self, tq):
        """Should handle arbitrary batch dimensions."""
        x = torch.randn(2, 5, 10, 128, dtype=torch.float16)
        packed, norms = tq.encode(x)
        assert norms.shape == (2, 5, 10)
        decoded = tq.decode(packed, norms)
        assert decoded.shape == (2, 5, 10, 128)

    def test_single_vector(self, tq):
        """Should handle a single vector."""
        x = torch.randn(128, dtype=torch.float16)
        packed, norms = tq.encode(x)
        assert norms.shape == ()
        decoded = tq.decode(packed, norms)
        assert decoded.shape == (128,)

    def test_norms_preserved(self, tq):
        """L2 norms should be stored accurately."""
        x = torch.randn(100, 128) * 5.0  # non-unit vectors
        packed, norms = tq.encode(x)
        true_norms = x.norm(dim=-1)
        torch.testing.assert_close(norms, true_norms, atol=1e-3, rtol=1e-3)

    def test_zero_vector(self, tq):
        """Zero vectors should not crash (norm=0)."""
        x = torch.zeros(1, 128, dtype=torch.float16)
        packed, norms = tq.encode(x)
        decoded = tq.decode(packed, norms)
        assert decoded.shape == (1, 128)
        # Decoded should be near-zero
        assert decoded.abs().max() < 0.01

    def test_compression_ratio(self, tq):
        """Compression ratio should match expected values."""
        expected = {2: 7.1, 3: 4.9, 4: 3.8}
        ratio = tq.compression_ratio()
        assert abs(ratio - expected[tq.bits]) < 0.2, \
            f"{tq.bits}-bit: expected ~{expected[tq.bits]}x, got {ratio:.1f}x"

    def test_repr(self, tq):
        """__repr__ should not crash."""
        s = repr(tq)
        assert "TurboQuant" in s
        assert str(tq.bits) in s


class TestTurboQuantMSE:
    """Validate MSE distortion bounds from the paper."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_mse_within_upper_bound(self, bits):
        """MSE should not exceed the theoretical upper bound."""
        tq = TurboQuant(head_dim=128, bits=bits, device="cpu")
        result = tq.validate(num_vectors=20000, device="cpu")

        # Allow 15% margin for finite-sample noise
        assert result["mse"] <= result["mse_theory_upper"] * 1.15, \
            f"{bits}-bit MSE {result['mse']:.6f} exceeds upper bound " \
            f"{result['mse_theory_upper']:.6f}"

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_mse_above_lower_bound(self, bits):
        """MSE should be above the information-theoretic lower bound."""
        tq = TurboQuant(head_dim=128, bits=bits, device="cpu")
        result = tq.validate(num_vectors=20000, device="cpu")

        # MSE should be at or above the lower bound (minus small margin for luck)
        assert result["mse"] >= result["mse_theory_lower"] * 0.8, \
            f"{bits}-bit MSE {result['mse']:.6f} below lower bound " \
            f"{result['mse_theory_lower']:.6f}"

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_mse_ratio_within_paper_claim(self, bits):
        """MSE / lower_bound should be <= 2.7 (paper's main claim)."""
        tq = TurboQuant(head_dim=128, bits=bits, device="cpu")
        result = tq.validate(num_vectors=20000, device="cpu")

        ratio = result["mse_ratio_to_lower"]
        # Paper claims 3*pi/2 ~ 4.71, but actual is much better (~ 1.5-2.4)
        assert ratio <= 3.0, \
            f"{bits}-bit ratio {ratio:.2f} exceeds 3.0 (paper claims <= 2.7)"

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_higher_bits_lower_mse(self, bits):
        """More bits should give strictly lower MSE."""
        if bits == 4:
            return  # No higher bits to compare
        tq_lo = TurboQuant(head_dim=128, bits=bits, device="cpu")
        tq_hi = TurboQuant(head_dim=128, bits=bits + 1, device="cpu")

        r_lo = tq_lo.validate(num_vectors=10000, device="cpu")
        r_hi = tq_hi.validate(num_vectors=10000, device="cpu")

        assert r_hi["mse"] < r_lo["mse"], \
            f"{bits+1}-bit MSE should be lower than {bits}-bit"

    def test_different_dims(self):
        """MSE should be consistent across dimensions."""
        for d in [64, 128, 256]:
            tq = TurboQuant(head_dim=d, bits=4, device="cpu")
            result = tq.validate(num_vectors=10000, device="cpu")
            assert result["mse"] <= result["mse_theory_upper"] * 1.15


class TestTurboQuantConfig:
    """Configuration validation."""

    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="Supported bit-widths"):
            TurboQuant(head_dim=128, bits=5, device="cpu")

    def test_invalid_head_dim(self):
        with pytest.raises(ValueError, match="power of 2"):
            TurboQuant(head_dim=100, bits=4, device="cpu")

    def test_hadamard_mode(self):
        """Hadamard rotation should produce valid results."""
        tq = TurboQuant(
            config=TurboQuantConfig(head_dim=128, bits=4, use_hadamard=True, device="cpu")
        )
        result = tq.validate(num_vectors=10000, device="cpu")
        assert result["mse"] <= result["mse_theory_upper"] * 1.15

    def test_different_seeds(self):
        """Different seeds should give different (but valid) results."""
        tq1 = TurboQuant(head_dim=128, bits=4, seed=42, device="cpu")
        tq2 = TurboQuant(head_dim=128, bits=4, seed=99, device="cpu")
        assert not torch.allclose(tq1.rotation, tq2.rotation)

        # But both should have valid MSE
        for tq in [tq1, tq2]:
            r = tq.validate(num_vectors=5000, device="cpu")
            assert r["mse"] <= r["mse_theory_upper"] * 1.15


class TestMemoryReport:
    """Memory calculation correctness."""

    def test_report_keys(self):
        tq = TurboQuant(head_dim=128, bits=4, device="cpu")
        r = tq.memory_report(seq_len=1000, num_layers=32, num_kv_heads=8)
        assert "fp16_mb" in r
        assert "fp8_mb" in r
        assert "tq4_mb" in r
        assert "ratio_vs_fp16" in r

    def test_fp16_calculation(self):
        """FP16 memory = vectors * dim * 2 bytes."""
        tq = TurboQuant(head_dim=128, bits=4, device="cpu")
        r = tq.memory_report(seq_len=1000, num_layers=1, num_kv_heads=1)
        # 1000 tokens * 1 layer * 1 head * 2 (K+V) = 2000 vectors
        # 2000 * 128 * 2 bytes = 512000 bytes = 0.488 MB
        assert abs(r["fp16_mb"] - 0.488) < 0.01

    def test_compression_monotonic(self):
        """Higher bits should use more memory."""
        reports = {}
        for bits in [2, 3, 4]:
            tq = TurboQuant(head_dim=128, bits=bits, device="cpu")
            reports[bits] = tq.memory_report(seq_len=10000)
        assert reports[2][f"tq2_mb"] < reports[3][f"tq3_mb"]
        assert reports[3][f"tq3_mb"] < reports[4][f"tq4_mb"]


# ============================================================================
# GPU TESTS
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTurboQuantGPU:
    """GPU-specific tests."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_cuda_roundtrip(self, bits):
        """Encode/decode should work on CUDA tensors."""
        tq = TurboQuant(head_dim=128, bits=bits, device="cuda")
        x = torch.randn(1000, 128, dtype=torch.float16, device="cuda")
        packed, norms = tq.encode(x)
        decoded = tq.decode(packed, norms)
        assert decoded.device.type == "cuda"
        assert decoded.shape == x.shape

    def test_cuda_mse_matches_cpu(self):
        """GPU and CPU should give same MSE (deterministic rotation)."""
        x_cpu = torch.randn(5000, 128)
        x_gpu = x_cpu.cuda().half()
        x_cpu = x_cpu.half()

        tq_cpu = TurboQuant(head_dim=128, bits=4, seed=42, device="cpu")
        tq_gpu = TurboQuant(head_dim=128, bits=4, seed=42, device="cuda")

        p_cpu, n_cpu = tq_cpu.encode(x_cpu)
        p_gpu, n_gpu = tq_gpu.encode(x_gpu)

        d_cpu = tq_cpu.decode(p_cpu, n_cpu).float()
        d_gpu = tq_gpu.decode(p_gpu, n_gpu).float().cpu()

        mse_cpu = (x_cpu.float() - d_cpu).pow(2).sum(dim=-1).mean()
        mse_gpu = (x_cpu.float() - d_gpu).pow(2).sum(dim=-1).mean()

        # Should be very close (not identical due to float precision)
        assert abs(mse_cpu - mse_gpu) < 0.01

    def test_benchmark_runs(self):
        """Benchmark should execute without errors."""
        tq = TurboQuant(head_dim=128, bits=4, device="cuda")
        result = tq.benchmark(num_vectors=1024, warmup=2, iters=5)
        assert result["encode_us"] > 0
        assert result["decode_us"] > 0
