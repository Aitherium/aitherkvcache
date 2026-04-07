"""
TriAttention test suite for the aither-kvcache package.

Tests spectral primitives, encoder roundtrip, trig series scoring,
full attention pipeline, compression ratios, and Qwen3.5 calibration.

Run: pytest tests/test_triattention.py -v
"""

import math
import pytest
import torch

from aither_kvcache.triattention.config import TriAttentionConfig
from aither_kvcache.triattention.spectral import (
    rope_frequencies,
    pair_energies,
    topk_pairs,
    spectral_concentration,
    pair_energy_profile,
    trig_series_coefficients,
    dct_matrix,
    dct,
    idct,
)
from aither_kvcache.triattention.encoder import SpectralKVEncoder, SpectralEncoding
from aither_kvcache.triattention.scorer import TrigSeriesScorer
from aither_kvcache.triattention.cache import SpectralKVCache
from aither_kvcache.triattention.attention import TriAttention, _apply_rope
from aither_kvcache.triattention.calibration import (
    QWEN3_5_PROFILES,
    get_profile,
    get_config_for_model,
)


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def default_config():
    return TriAttentionConfig(
        head_dim=128, num_freqs=12, coeff_bits=4,
        num_kv_heads=8, num_query_heads=32, rope_base=1_000_000.0,
    )


@pytest.fixture
def small_config():
    return TriAttentionConfig(
        head_dim=32, num_freqs=4, coeff_bits=4,
        num_kv_heads=2, num_query_heads=4, rope_base=10000.0,
    )


# ── Config tests ──────────────────────────────────────────────────

class TestConfig:
    def test_compression_ratio_int4(self, default_config):
        c = default_config
        assert c.bytes_per_kv_token == 26
        assert c.fp16_bytes_per_kv_token == 256
        assert c.compression_ratio > 9.0

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="head_dim must be even"):
            TriAttentionConfig(head_dim=127)
        with pytest.raises(ValueError, match="num_freqs.*cannot exceed"):
            TriAttentionConfig(head_dim=32, num_freqs=20)
        with pytest.raises(ValueError, match="coeff_bits"):
            TriAttentionConfig(coeff_bits=6)

    def test_freq_budget_schedule(self):
        c = TriAttentionConfig(
            num_freqs=10,
            layer_freq_schedule=[16, 14, 12, 10, 10, 10, 12, 14],
        )
        assert c.freq_budget(0) == 16
        assert c.freq_budget(3) == 10
        assert c.freq_budget(100) == 10


# ── Spectral primitives ──────────────────────────────────────────

class TestSpectral:
    def test_rope_frequencies_monotonic(self):
        theta = rope_frequencies(128, base=1e6)
        assert theta.shape == (64,)
        assert (theta[:-1] > theta[1:]).all()

    def test_pair_energies_sum_equals_norm_sq(self):
        x = torch.randn(4, 128)
        E = pair_energies(x)
        assert E.sum(dim=-1) == pytest.approx(
            (x ** 2).sum(dim=-1), rel=1e-5
        )

    def test_topk_pairs_sorted_uint8(self):
        x = torch.randn(4, 128)
        idx, vals = topk_pairs(x, 12)
        assert idx.dtype == torch.uint8
        assert idx.shape == (4, 12)
        for i in range(4):
            assert (idx[i, :-1] <= idx[i, 1:]).all()

    def test_spectral_concentration_full(self):
        x = torch.randn(10, 128)
        ratio = spectral_concentration(x, 64)
        assert ratio.mean() == pytest.approx(1.0, abs=1e-5)

    def test_trig_series_coefficients(self):
        q = torch.randn(4, 128)
        k = torch.randn(4, 128)
        c, s = trig_series_coefficients(q, k)
        expected_c = q[:, 0::2] * k[:, 0::2] + q[:, 1::2] * k[:, 1::2]
        assert c == pytest.approx(expected_c, rel=1e-5)

    def test_dct_orthonormal(self):
        D = dct_matrix(32)
        assert (D @ D.T) == pytest.approx(torch.eye(32), abs=1e-5)

    def test_dct_roundtrip(self):
        x = torch.randn(4, 64)
        D = dct_matrix(64)
        assert idct(dct(x, D), D) == pytest.approx(x, abs=1e-4)


# ── Encoder tests ─────────────────────────────────────────────────

class TestEncoder:
    def test_encode_decode_fp16(self):
        config = TriAttentionConfig(head_dim=128, num_freqs=12, coeff_bits=16)
        enc = SpectralKVEncoder(config)
        x = torch.randn(4, 8, 128)
        encoded = enc.encode(x)
        decoded = enc.decode(encoded)
        assert decoded.shape == x.shape

    def test_encode_decode_int4(self, default_config):
        enc = SpectralKVEncoder(default_config)
        x = torch.randn(4, 8, 128)
        encoded = enc.encode(x)
        decoded = enc.decode(encoded)
        assert decoded.shape == x.shape
        cos_sim = torch.nn.functional.cosine_similarity(
            x.reshape(-1, 128).float(), decoded.reshape(-1, 128).float(), dim=-1
        )
        assert cos_sim.mean() > 0.5

    def test_encode_shapes_int4(self, default_config):
        enc = SpectralKVEncoder(default_config)
        x = torch.randn(2, 8, 128)
        encoded = enc.encode(x)
        assert encoded.indices.shape == (2, 8, 12)
        assert encoded.indices.dtype == torch.uint8
        assert encoded.packed.dtype == torch.uint8
        assert encoded.scales.dtype == torch.float16

    def test_compression_stats(self, default_config):
        enc = SpectralKVEncoder(default_config)
        stats = enc.compression_stats(torch.randn(100, 128))
        assert stats["compression_ratio"] > 9.0
        assert "mse" in stats
        assert "cosine_similarity" in stats


# ── Scorer tests ──────────────────────────────────────────────────

class TestScorer:
    def test_score_shape(self, small_config):
        scorer = TrigSeriesScorer(small_config)
        encoder = SpectralKVEncoder(small_config)
        B, S, QH, KVH, D = 2, 16, 4, 2, 32
        query = torch.randn(B, QH, D)
        keys = torch.randn(B, S, KVH, D)
        k_enc = encoder.encode(keys)
        scores = scorer.score(query, k_enc)
        assert scores.shape == (B, QH, S)

    def test_trig_score_exact_all_freqs(self):
        """All frequency pairs + fp16 = exact trig series scoring."""
        D = 32
        config = TriAttentionConfig(
            head_dim=D, num_freqs=D // 2, coeff_bits=16,
            num_kv_heads=1, num_query_heads=1, rope_base=10000.0,
        )
        scorer = TrigSeriesScorer(config)
        encoder = SpectralKVEncoder(config)
        B, S = 1, 8
        query = torch.randn(B, 1, D)
        keys = torch.randn(B, S, 1, D)
        q_pos = torch.tensor([S])
        k_pos = torch.arange(S).unsqueeze(0)

        k_enc = encoder.encode(keys)
        scores_tri = scorer.score(query, k_enc, q_pos, k_pos)

        theta = rope_frequencies(D, 10000.0)
        q_rope = _apply_rope(query, q_pos.unsqueeze(-1), theta)
        scores_ref = torch.zeros(B, 1, S)
        for s in range(S):
            k_rope = _apply_rope(keys[:, s, :, :], k_pos[:, s:s+1].unsqueeze(-1), theta)
            scores_ref[:, :, s] = (q_rope[:, :, :] * k_rope).sum(dim=-1) / math.sqrt(D)

        assert scores_tri == pytest.approx(scores_ref, abs=1e-3)

    def test_position_sensitivity(self, small_config):
        scorer = TrigSeriesScorer(small_config)
        encoder = SpectralKVEncoder(small_config)
        query = torch.randn(1, 4, 32)
        keys = torch.randn(1, 16, 2, 32)
        k_pos = torch.arange(16).unsqueeze(0)
        k_enc = encoder.encode(keys)
        s0 = scorer.score(query, k_enc, torch.tensor([0]), k_pos)
        s100 = scorer.score(query, k_enc, torch.tensor([100]), k_pos)
        assert not torch.allclose(s0, s100, atol=1e-3)


# ── Cache tests ───────────────────────────────────────────────────

class TestCache:
    def test_store_and_fetch(self, default_config):
        cache = SpectralKVCache(default_config, max_blocks=16)
        enc = SpectralKVEncoder(default_config)
        keys = torch.randn(default_config.block_size, default_config.num_kv_heads, 128)
        k_enc = enc.encode(keys)
        cache.store_block(0, k_enc, k_enc)
        k_out, _ = cache.fetch_sequence(torch.tensor([0]), default_config.block_size)
        assert torch.equal(k_out.indices, k_enc.indices)

    def test_copy_and_clear(self, default_config):
        cache = SpectralKVCache(default_config, max_blocks=16)
        cache.k_indices[0].fill_(42)
        cache._block_fill[0] = 8
        cache.copy_block(0, 5)
        assert torch.equal(cache.k_indices[5], cache.k_indices[0])
        cache.clear_block(5)
        assert (cache.k_indices[5] == 0).all()

    def test_memory_stats(self, default_config):
        stats = SpectralKVCache(default_config, max_blocks=64).memory_stats()
        assert stats["compression_ratio"] > 9.0


# ── Full attention pipeline ───────────────────────────────────────

class TestTriAttention:
    def test_forward_shape(self, small_config):
        tri = TriAttention(small_config)
        output = tri.forward(
            torch.randn(2, 4, 32),
            torch.randn(2, 8, 2, 32),
            torch.randn(2, 8, 2, 32),
        )
        assert output.shape == (2, 4, 32)

    def test_decode_step(self, small_config):
        tri = TriAttention(small_config)
        key = torch.randn(1, 16, 2, 32)
        val = torch.randn(1, 16, 2, 32)
        k_enc, v_enc = tri.encode_kv(key, val)
        output = tri.decode_step(
            torch.randn(1, 4, 32), k_enc, v_enc,
            torch.tensor([16]), torch.arange(16).unsqueeze(0),
        )
        assert output.shape == (1, 4, 32)
        assert torch.isfinite(output).all()

    def test_gqa(self):
        config = TriAttentionConfig(
            head_dim=32, num_freqs=8, coeff_bits=16,
            num_kv_heads=2, num_query_heads=8,
        )
        output = TriAttention(config).forward(
            torch.randn(1, 8, 32),
            torch.randn(1, 4, 2, 32),
            torch.randn(1, 4, 2, 32),
        )
        assert output.shape == (1, 8, 32)
        assert torch.isfinite(output).all()

    def test_all_bit_widths(self):
        for bits in [4, 8, 16]:
            config = TriAttentionConfig(
                head_dim=32, num_freqs=4, coeff_bits=bits,
                num_kv_heads=1, num_query_heads=1,
            )
            out = TriAttention(config).forward(
                torch.randn(1, 1, 32),
                torch.randn(1, 4, 1, 32),
                torch.randn(1, 4, 1, 32),
            )
            assert torch.isfinite(out).all(), f"NaN/Inf for bits={bits}"


# ── Math properties ───────────────────────────────────────────────

class TestMathProperties:
    def test_trig_series_exact(self):
        """Full trig series = exact RoPE attention score."""
        D, B = 64, 4
        q, k = torch.randn(B, D), torch.randn(B, D)
        theta = rope_frequencies(D, 10000.0)
        m, n = 5, 12
        delta = n - m

        q_e, q_o = q[:, 0::2], q[:, 1::2]
        k_e, k_o = k[:, 0::2], k[:, 1::2]
        a_m, a_n = m * theta, n * theta

        q_re = q_e * torch.cos(a_m) - q_o * torch.sin(a_m)
        q_ro = q_e * torch.sin(a_m) + q_o * torch.cos(a_m)
        k_re = k_e * torch.cos(a_n) - k_o * torch.sin(a_n)
        k_ro = k_e * torch.sin(a_n) + k_o * torch.cos(a_n)
        score_rope = (q_re * k_re + q_ro * k_ro).sum(dim=-1)

        c = q_e * k_e + q_o * k_o
        s = q_o * k_e - q_e * k_o
        score_trig = (
            c * torch.cos(torch.tensor(float(delta)) * theta)
            + s * torch.sin(torch.tensor(float(delta)) * theta)
        ).sum(dim=-1)

        assert score_trig == pytest.approx(score_rope, rel=1e-4)

    def test_truncation_error_bound(self):
        D, F, B = 128, 12, 100
        q, k = torch.randn(B, D), torch.randn(B, D)
        c_full, _ = trig_series_coefficients(q, k)
        score_full = c_full.sum(dim=-1)

        idx, _ = topk_pairs(k, F)
        c_trunc, _ = trig_series_coefficients(q, k, freq_indices=idx)
        score_trunc = c_trunc.sum(dim=-1)

        error = (score_full - score_trunc).abs()
        energies = pair_energies(k)
        top_E, _ = energies.topk(F, dim=-1)
        residual = energies.sum(dim=-1) - top_E.sum(dim=-1)
        bound = q.norm(dim=-1) * residual.sqrt()
        assert (error <= bound * 1.01 + 1e-4).all()

    def test_10x_compression(self, default_config):
        assert default_config.kv_compression_ratio > 9.5


# ── RoPE helper ───────────────────────────────────────────────────

class TestRoPE:
    def test_preserves_norm(self):
        x = torch.randn(4, 8, 32)
        theta = rope_frequencies(32)
        x_r = _apply_rope(x, torch.arange(4).unsqueeze(-1), theta)
        assert (x_r ** 2).sum(-1) == pytest.approx((x ** 2).sum(-1), rel=1e-4)


# ── Calibration ───────────────────────────────────────────────────

class TestCalibration:
    def test_qwen35_profiles(self):
        assert len(QWEN3_5_PROFILES) >= 6
        for size in ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]:
            assert get_profile(size) is not None

    def test_profile_to_config(self):
        config = get_config_for_model("Qwen3.5-8B")
        assert config.head_dim == 128
        assert config.model_family == "qwen3.5"
        assert config.layer_freq_schedule is not None

    def test_early_layers_more_freqs(self):
        p = get_profile("8B")
        assert p.layer_profiles[0].recommended_freqs >= p.layer_profiles[18].recommended_freqs


# ── Integration ───────────────────────────────────────────────────

class TestIntegration:
    def test_end_to_end_pipeline(self):
        config = TriAttentionConfig(
            head_dim=32, num_freqs=4, coeff_bits=4,
            num_kv_heads=2, num_query_heads=4, block_size=8,
        )
        encoder = SpectralKVEncoder(config)
        cache = SpectralKVCache(config, max_blocks=4)
        tri = TriAttention(config)

        S, KVH, D = 16, 2, 32
        keys = torch.randn(S, KVH, D)
        values = torch.randn(S, KVH, D)

        for blk in range(2):
            s, e = blk * 8, (blk + 1) * 8
            k_enc = encoder.encode(keys[s:e])
            v_enc = encoder.encode(values[s:e])
            cache.store_block(blk, k_enc, v_enc, num_tokens=8)

        k_enc, v_enc = cache.fetch_sequence(torch.tensor([0, 1]), S)
        k_enc = SpectralEncoding(k_enc.indices.unsqueeze(0), k_enc.packed.unsqueeze(0), k_enc.scales.unsqueeze(0))
        v_enc = SpectralEncoding(v_enc.indices.unsqueeze(0), v_enc.packed.unsqueeze(0), v_enc.scales.unsqueeze(0))

        output = tri.decode_step(
            torch.randn(1, 4, 32), k_enc, v_enc,
            torch.tensor([S]), torch.arange(S).unsqueeze(0),
        )
        assert output.shape == (1, 4, 32)
        assert torch.isfinite(output).all()
