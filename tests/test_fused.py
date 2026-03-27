"""
Tests for the fused TQ paged attention kernel.

Validates the core mathematical insight: attention can be computed in the
rotated domain without ever materializing the decompressed KV cache.

  q^T @ k = ||k|| * (Pi @ q)^T @ y_hat_k
  output = Pi^T @ (sum_i w_i * ||v_i|| * y_hat_v_i)
"""

import math
import pytest
import torch

from turboquant.quantizer import TurboQuant, TurboQuantConfig
from turboquant.rotation import random_orthogonal
from turboquant.codebook import get_codebook
from turboquant.packing import pack_4bit, unpack_4bit
from turboquant.fused_attention import TQPagedAttentionRef, TQPagedAttention

DEVICE = "cpu"
HEAD_DIM = 128
BLOCK_SIZE = 16
NUM_KV_HEADS = 4
NUM_QUERY_HEADS = 4  # no GQA for simplicity


# ============================================================================
# MATH VALIDATION
# ============================================================================

class TestRotatedDomainMath:
    """Verify the core rotated-domain equivalence."""

    def test_rotated_dot_product(self):
        """(Pi @ q)^T @ y_hat == q^T @ (Pi^T @ y_hat) for random vectors."""
        Pi = random_orthogonal(HEAD_DIM, device=DEVICE)
        q = torch.randn(HEAD_DIM)
        y_hat = torch.randn(HEAD_DIM)

        # Standard: q^T @ (Pi^T @ y_hat)
        standard = torch.dot(q, Pi.T @ y_hat)

        # Rotated: (Pi @ q)^T @ y_hat
        rotated = torch.dot(Pi @ q, y_hat)

        torch.testing.assert_close(standard, rotated, atol=1e-5, rtol=1e-5)

    def test_score_with_norm(self):
        """score = ||k|| * (Pi @ q)^T @ y_hat matches q^T @ k."""
        Pi = random_orthogonal(HEAD_DIM, device=DEVICE)
        q = torch.randn(HEAD_DIM)
        k = torch.randn(HEAD_DIM) * 3.0  # non-unit

        # TQ encode k
        norm_k = k.norm()
        k_unit = k / norm_k
        y = Pi @ k_unit
        # Quantize y (just use y directly for math validation)
        y_hat = y  # no quantization error for this test

        # Standard score
        standard = torch.dot(q, k)

        # Rotated score
        q_rot = Pi @ q
        rotated = norm_k * torch.dot(q_rot, y_hat)

        torch.testing.assert_close(standard, rotated, atol=1e-4, rtol=1e-4)

    def test_even_odd_split_dot(self):
        """Split dot product matches full dot product."""
        a = torch.randn(HEAD_DIM)
        b = torch.randn(HEAD_DIM)

        full_dot = torch.dot(a, b)
        split_dot = torch.dot(a[0::2], b[0::2]) + torch.dot(a[1::2], b[1::2])

        torch.testing.assert_close(full_dot, split_dot, atol=1e-6, rtol=1e-6)

    def test_v_accumulation_in_rotated_domain(self):
        """Pi^T @ sum(w_i * ||v_i|| * y_hat_i) == sum(w_i * v_hat_i)."""
        Pi = random_orthogonal(HEAD_DIM, device=DEVICE)
        n = 10
        weights = torch.softmax(torch.randn(n), dim=0)

        vs = torch.randn(n, HEAD_DIM)
        norms = vs.norm(dim=-1)
        vs_unit = vs / norms.unsqueeze(-1)
        y_hats = (vs_unit @ Pi.T)  # y_hat = Pi @ v_unit -> v_unit @ Pi^T in batch

        # Standard: sum(w_i * v_i)
        standard = (weights.unsqueeze(-1) * vs).sum(dim=0)

        # Rotated: Pi^T @ sum(w_i * norm_i * y_hat_i)
        acc_rotated = (weights.unsqueeze(-1) * norms.unsqueeze(-1) * y_hats).sum(dim=0)
        rotated = Pi.T @ acc_rotated

        torch.testing.assert_close(standard, rotated, atol=1e-4, rtol=1e-4)

    def test_online_softmax(self):
        """Online softmax matches torch.softmax."""
        scores = torch.randn(50)

        # Standard
        expected = torch.softmax(scores, dim=0)

        # Online
        m = torch.tensor(float("-inf"))
        l = torch.tensor(0.0)
        for s in scores:
            m_new = torch.maximum(m, s)
            alpha = torch.exp(m - m_new)
            beta = torch.exp(s - m_new)
            l = alpha * l + beta
            m = m_new

        # Recompute weights
        online_weights = torch.exp(scores - m) / l

        torch.testing.assert_close(online_weights, expected, atol=1e-5, rtol=1e-5)


# ============================================================================
# HELPER: Lightweight KV cache for standalone testing
# ============================================================================

class _SimpleKVCache:
    """
    Minimal KV cache for testing fused attention without the vLLM backend.
    Stores TQ-compressed keys and values in paged blocks.
    """

    def __init__(self, num_blocks, block_size, num_kv_heads, head_dim, bits=4):
        self.tq = TurboQuant(head_dim=head_dim, bits=bits, device=DEVICE)
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        packed_dim = self.tq._packed_dim

        # [num_blocks, block_size, num_kv_heads, packed_dim]
        self.k_packed = torch.zeros(num_blocks, block_size, num_kv_heads, packed_dim, dtype=torch.uint8)
        self.k_norms = torch.zeros(num_blocks, block_size, num_kv_heads)
        self.v_packed = torch.zeros(num_blocks, block_size, num_kv_heads, packed_dim, dtype=torch.uint8)
        self.v_norms = torch.zeros(num_blocks, block_size, num_kv_heads)

    def store(self, keys, values, slots):
        """Store keys/values at given slot positions."""
        for i, slot in enumerate(slots):
            if slot < 0:
                continue
            block_idx = slot // self.block_size
            offset = slot % self.block_size
            for h in range(self.num_kv_heads):
                k_packed, k_norm = self.tq.encode(keys[i, h].unsqueeze(0))
                v_packed, v_norm = self.tq.encode(values[i, h].unsqueeze(0))
                self.k_packed[block_idx, offset, h] = k_packed.squeeze(0)
                self.k_norms[block_idx, offset, h] = k_norm.squeeze(0)
                self.v_packed[block_idx, offset, h] = v_packed.squeeze(0)
                self.v_norms[block_idx, offset, h] = v_norm.squeeze(0)

    def decompress(self):
        """Decompress all keys and values for reference comparison."""
        nb, bs, nh, pd = self.k_packed.shape
        k_buf = torch.zeros(nb, bs, nh, self.head_dim, dtype=torch.float16)
        v_buf = torch.zeros(nb, bs, nh, self.head_dim, dtype=torch.float16)
        for b in range(nb):
            for s in range(bs):
                for h in range(nh):
                    if self.k_norms[b, s, h].abs() > 0:
                        k_buf[b, s, h] = self.tq.decode(
                            self.k_packed[b, s, h].unsqueeze(0),
                            self.k_norms[b, s, h].unsqueeze(0),
                        ).squeeze(0)
                        v_buf[b, s, h] = self.tq.decode(
                            self.v_packed[b, s, h].unsqueeze(0),
                            self.v_norms[b, s, h].unsqueeze(0),
                        ).squeeze(0)
        return k_buf, v_buf


def _make_cache(num_blocks, num_tokens, bits=4):
    """Create a simple TQ cache with random data for testing."""
    cache = _SimpleKVCache(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, bits=bits)
    keys = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)
    values = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)
    slots = torch.arange(num_tokens, dtype=torch.int64)
    cache.store(keys, values, slots)
    return cache, keys, values


def _standard_attention(query, keys, values, num_tokens):
    """Standard dense attention for reference."""
    scale = 1.0 / math.sqrt(HEAD_DIM)
    q = query[0].float()
    k = keys[:num_tokens].float()
    v = values[:num_tokens].float()

    output = torch.zeros_like(q)
    for qh in range(q.shape[0]):
        kv_head = qh % NUM_KV_HEADS
        scores = (q[qh] @ k[:, kv_head].T) * scale
        weights = torch.softmax(scores, dim=0)
        out = weights @ v[:, kv_head]
        output[qh] = out

    return output.unsqueeze(0).to(query.dtype)


# ============================================================================
# FUSED ATTENTION TESTS
# ============================================================================

class TestFusedAttention:
    """Test fused TQ attention against standard attention."""

    def test_single_block(self):
        """Single block, single sequence -- fused vs decompress+standard."""
        num_tokens = 8
        num_blocks = 1
        cache, keys, values = _make_cache(num_blocks, num_tokens)

        query = torch.randn(1, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float16)
        block_tables = torch.tensor([[0]], dtype=torch.int64)
        context_lens = torch.tensor([num_tokens], dtype=torch.int64)

        # Path 1: Fused TQ attention (rotated domain)
        attn = TQPagedAttention(cache.tq, NUM_QUERY_HEADS)
        fused_out = attn.forward(
            query, cache.k_packed, cache.k_norms,
            cache.v_packed, cache.v_norms,
            block_tables, context_lens,
            block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS,
        )

        # Path 2: Decompress TQ -> standard attention
        k_buf, v_buf = cache.decompress()
        slots = torch.arange(num_tokens, dtype=torch.int64)
        k_dec = k_buf[slots // BLOCK_SIZE, slots % BLOCK_SIZE]
        v_dec = v_buf[slots // BLOCK_SIZE, slots % BLOCK_SIZE]
        std_out = _standard_attention(query, k_dec, v_dec, num_tokens)

        cosine_sim = torch.nn.functional.cosine_similarity(
            fused_out.float().flatten(), std_out.float().flatten(), dim=0
        )
        assert cosine_sim > 0.99, f"Cosine similarity {cosine_sim:.4f} too low"

    def test_multi_block(self):
        """Multiple blocks, single sequence."""
        num_tokens = 48  # 3 blocks
        num_blocks = 3
        cache, keys, values = _make_cache(num_blocks, num_tokens)

        query = torch.randn(1, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float16)
        block_tables = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        context_lens = torch.tensor([num_tokens], dtype=torch.int64)

        attn = TQPagedAttention(cache.tq, NUM_QUERY_HEADS)
        fused_out = attn.forward(
            query, cache.k_packed, cache.k_norms,
            cache.v_packed, cache.v_norms,
            block_tables, context_lens,
            block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS,
        )

        k_buf, v_buf = cache.decompress()
        slots = torch.arange(num_tokens, dtype=torch.int64)
        k_dec = k_buf[slots // BLOCK_SIZE, slots % BLOCK_SIZE]
        v_dec = v_buf[slots // BLOCK_SIZE, slots % BLOCK_SIZE]
        std_out = _standard_attention(query, k_dec, v_dec, num_tokens)

        cosine_sim = torch.nn.functional.cosine_similarity(
            fused_out.float().flatten(), std_out.float().flatten(), dim=0
        )
        assert cosine_sim > 0.99, f"Cosine sim {cosine_sim:.4f} too low for multi-block"

    def test_partial_last_block(self):
        """Last block partially filled."""
        num_tokens = 20  # 1 full block + 4 tokens
        num_blocks = 2
        cache, keys, values = _make_cache(num_blocks, num_tokens)

        query = torch.randn(1, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float16)
        block_tables = torch.tensor([[0, 1]], dtype=torch.int64)
        context_lens = torch.tensor([num_tokens], dtype=torch.int64)

        attn = TQPagedAttention(cache.tq, NUM_QUERY_HEADS)
        fused_out = attn.forward(
            query, cache.k_packed, cache.k_norms,
            cache.v_packed, cache.v_norms,
            block_tables, context_lens,
            block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS,
        )

        assert fused_out.shape == (1, NUM_QUERY_HEADS, HEAD_DIM)
        assert not torch.isnan(fused_out).any()
        assert not torch.isinf(fused_out).any()

    def test_zero_context(self):
        """Empty context should produce zero output."""
        cache = _SimpleKVCache(4, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)

        query = torch.randn(1, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float16)
        block_tables = torch.zeros(1, 4, dtype=torch.int64)
        context_lens = torch.tensor([0], dtype=torch.int64)

        attn = TQPagedAttention(cache.tq, NUM_QUERY_HEADS)
        out = attn.forward(
            query, cache.k_packed, cache.k_norms,
            cache.v_packed, cache.v_norms,
            block_tables, context_lens,
            block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS,
        )

        assert out.abs().sum() == 0

    def test_output_shape(self):
        """Output should match [num_seqs, num_query_heads, head_dim]."""
        cache, _, _ = _make_cache(4, 32)

        query = torch.randn(1, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float16)
        block_tables = torch.tensor([[0, 1]], dtype=torch.int64)
        context_lens = torch.tensor([32], dtype=torch.int64)

        attn = TQPagedAttention(cache.tq, NUM_QUERY_HEADS)
        out = attn.forward(
            query, cache.k_packed, cache.k_norms,
            cache.v_packed, cache.v_norms,
            block_tables, context_lens,
            block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS,
        )

        assert out.shape == (1, NUM_QUERY_HEADS, HEAD_DIM)

    def test_fused_vs_decompress_path(self):
        """Fused attention should match decompress->standard attention path."""
        num_tokens = 32
        cache, keys, values = _make_cache(2, num_tokens)

        query = torch.randn(1, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float16)
        block_tables = torch.tensor([[0, 1]], dtype=torch.int64)
        context_lens = torch.tensor([num_tokens], dtype=torch.int64)

        attn = TQPagedAttention(cache.tq, NUM_QUERY_HEADS)
        fused_out = attn.forward(
            query, cache.k_packed, cache.k_norms,
            cache.v_packed, cache.v_norms,
            block_tables, context_lens,
            block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS,
        )

        k_buf, v_buf = cache.decompress()
        slots = torch.arange(num_tokens, dtype=torch.int64)
        k_decompressed = k_buf[slots // BLOCK_SIZE, slots % BLOCK_SIZE]
        v_decompressed = v_buf[slots // BLOCK_SIZE, slots % BLOCK_SIZE]
        std_out = _standard_attention(query, k_decompressed, v_decompressed, num_tokens)

        diff = (fused_out.float() - std_out.float()).abs().max().item()
        assert diff < 0.05, f"Fused vs decompress max diff {diff:.4f} too large"


class TestFusedAttentionNumerics:
    """Numerical stability and edge cases."""

    def test_large_norms(self):
        """Should handle keys/values with large norms."""
        cache = _SimpleKVCache(1, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
        keys = torch.randn(8, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16) * 100
        values = torch.randn(8, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16) * 100
        cache.store(keys, values, torch.arange(8, dtype=torch.int64))

        query = torch.randn(1, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float16)
        attn = TQPagedAttention(cache.tq, NUM_QUERY_HEADS)
        out = attn.forward(
            query, cache.k_packed, cache.k_norms,
            cache.v_packed, cache.v_norms,
            torch.tensor([[0]], dtype=torch.int64),
            torch.tensor([8], dtype=torch.int64),
            block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS,
        )
        assert not torch.isnan(out).any(), "NaN in output with large norms"
        assert not torch.isinf(out).any(), "Inf in output with large norms"

    def test_single_token_context(self):
        """Single token in KV cache."""
        cache, keys, values = _make_cache(1, 1)

        query = torch.randn(1, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float16)
        attn = TQPagedAttention(cache.tq, NUM_QUERY_HEADS)
        out = attn.forward(
            query, cache.k_packed, cache.k_norms,
            cache.v_packed, cache.v_norms,
            torch.tensor([[0]], dtype=torch.int64),
            torch.tensor([1], dtype=torch.int64),
            block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS,
        )
        # With single token, output should be the value (softmax weight = 1.0)
        assert out.shape == (1, NUM_QUERY_HEADS, HEAD_DIM)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_different_bitwidths(self, bits):
        """Should work at all supported bit-widths."""
        cache = _SimpleKVCache(2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, bits=bits)
        keys = torch.randn(16, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)
        values = torch.randn_like(keys)
        cache.store(keys, values, torch.arange(16, dtype=torch.int64))

        query = torch.randn(1, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float16)
        attn = TQPagedAttention(cache.tq, NUM_QUERY_HEADS)
        out = attn.forward(
            query, cache.k_packed, cache.k_norms,
            cache.v_packed, cache.v_norms,
            torch.tensor([[0, 1]], dtype=torch.int64),
            torch.tensor([16], dtype=torch.int64),
            block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS,
        )
        assert not torch.isnan(out).any()
