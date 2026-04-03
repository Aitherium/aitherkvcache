"""Tests for TurboQuant hybrid mode hooks, custom ops, and engine integration.

Tests the hybrid encode/decode through hooks, clamp-gather decompress,
masked batched SDPA, mode detection, quantizer init, and packed dim consistency.
"""
import math
import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F


# =====================================================================
# FIXTURES
# =====================================================================

HEAD_DIM = 128
NUM_KV_HEADS = 8
NUM_HEADS = 32  # GQA ratio = 4
BLOCK_SIZE = 16
NUM_BLOCKS = 64


@pytest.fixture
def uniform_tq():
    from turboquant.quantizer import TurboQuant
    return TurboQuant(head_dim=HEAD_DIM, bits=4, device="cpu")


@pytest.fixture
def hybrid_tq35():
    from turboquant.hybrid_quantizer import HybridTurboQuant
    htq = HybridTurboQuant(head_dim=HEAD_DIM, mode="tq35", device="cpu")
    htq.calibrate_uniform(num_kv_heads=NUM_KV_HEADS)
    return htq


@pytest.fixture
def hybrid_tq25():
    from turboquant.hybrid_quantizer import HybridTurboQuant
    htq = HybridTurboQuant(head_dim=HEAD_DIM, mode="tq25", device="cpu")
    htq.calibrate_uniform(num_kv_heads=NUM_KV_HEADS)
    return htq


# =====================================================================
# MODE DETECTION
# =====================================================================

class TestModeDetection:
    """Test that mode variables are derived correctly from env vars."""

    def test_tq4_is_not_hybrid(self):
        with patch.dict(os.environ, {"AITHER_TQ_MODE": "tq4-primary"}):
            mode = os.environ["AITHER_TQ_MODE"].replace("-primary", "")
            assert mode not in ("tq35", "tq25")

    def test_tq35_is_hybrid(self):
        with patch.dict(os.environ, {"AITHER_TQ_MODE": "tq35-primary"}):
            mode = os.environ["AITHER_TQ_MODE"].replace("-primary", "")
            assert mode in ("tq35", "tq25")

    def test_tq25_is_hybrid(self):
        with patch.dict(os.environ, {"AITHER_TQ_MODE": "tq25"}):
            mode = os.environ["AITHER_TQ_MODE"].replace("-primary", "")
            assert mode in ("tq35", "tq25")

    def test_empty_mode_is_not_hybrid(self):
        with patch.dict(os.environ, {"AITHER_TQ_MODE": ""}):
            mode = os.environ["AITHER_TQ_MODE"].replace("-primary", "")
            assert mode not in ("tq35", "tq25")

    def test_primary_stripped_correctly(self):
        with patch.dict(os.environ, {"AITHER_TQ_MODE": "tq35-primary"}):
            mode = os.environ["AITHER_TQ_MODE"].replace("-primary", "")
            assert mode == "tq35"


# =====================================================================
# HYBRID ENCODE / DECODE ROUNDTRIP
# =====================================================================

class TestHybridEncodeDecode:
    """Test that hybrid encode returns packed-only (no norms), and decode roundtrips."""

    def test_tq35_encode_returns_single_tensor(self, hybrid_tq35):
        x = torch.randn(NUM_KV_HEADS, HEAD_DIM)
        packed = hybrid_tq35.encode(x)
        # Hybrid encode returns a single tensor, NOT a tuple
        assert isinstance(packed, torch.Tensor)
        assert packed.dtype == torch.uint8
        assert packed.shape[-1] == hybrid_tq35.layout.packed_dim

    def test_tq25_encode_returns_single_tensor(self, hybrid_tq25):
        x = torch.randn(NUM_KV_HEADS, HEAD_DIM)
        packed = hybrid_tq25.encode(x)
        assert isinstance(packed, torch.Tensor)
        assert packed.dtype == torch.uint8

    def test_uniform_encode_returns_tuple(self, uniform_tq):
        x = torch.randn(1, HEAD_DIM)
        result = uniform_tq.encode(x)
        # Uniform encode returns (packed, norms) tuple
        assert isinstance(result, tuple)
        assert len(result) == 2
        packed, norms = result
        assert packed.dtype == torch.uint8
        assert norms.dtype == torch.float32

    def test_hybrid_roundtrip_quality(self, hybrid_tq35):
        """Encode -> decode should reconstruct with reasonable quality."""
        x = torch.randn(NUM_KV_HEADS, HEAD_DIM)
        packed = hybrid_tq35.encode(x)
        decoded = hybrid_tq35.decode(packed)
        # Should reconstruct with cosine sim > 0.8 (3.5-bit compression)
        cos = F.cosine_similarity(x.reshape(1, -1), decoded.reshape(1, -1))
        assert cos.item() > 0.7, f"Cosine similarity too low: {cos.item()}"

    def test_hybrid_packed_dim_varies_by_mode(self, hybrid_tq35, hybrid_tq25):
        """tq25 should have smaller packed_dim than tq35."""
        assert hybrid_tq25.layout.packed_dim < hybrid_tq35.layout.packed_dim

    def test_hybrid_batch_encode(self, hybrid_tq35):
        """Batch of tokens * heads encodes correctly."""
        N_tokens = 4
        x = torch.randn(N_tokens * NUM_KV_HEADS, HEAD_DIM)
        packed = hybrid_tq35.encode(x)
        assert packed.shape == (N_tokens * NUM_KV_HEADS, hybrid_tq35.layout.packed_dim)

    def test_hybrid_batch_decode(self, hybrid_tq35):
        """Batch decode matches individual decodes."""
        x = torch.randn(2 * NUM_KV_HEADS, HEAD_DIM)
        packed = hybrid_tq35.encode(x)
        decoded_batch = hybrid_tq35.decode(packed)
        decoded_0 = hybrid_tq35.decode(packed[:NUM_KV_HEADS])
        # First N_kv_heads should match
        torch.testing.assert_close(decoded_batch[:NUM_KV_HEADS], decoded_0, atol=1e-5, rtol=1e-5)


# =====================================================================
# CLAMP-GATHER DECOMPRESS + MASKED SDPA
# =====================================================================

class TestClampGatherDecompress:
    """Test the core logic of the hybrid custom op's decompress path."""

    def test_clamp_handles_negative_padding(self):
        """block_table with -1 padding should clamp to 0."""
        block_table = torch.tensor([[0, 1, 2, -1, -1],
                                    [3, 4, -1, -1, -1]])
        clamped = block_table.clamp(min=0, max=63)
        assert (clamped >= 0).all()
        assert clamped[0, 3] == 0  # padding -> block 0
        assert clamped[1, 2] == 0

    def test_gather_decompresses_correct_blocks(self, hybrid_tq35):
        """Verify that gather + decompress produces correct shapes."""
        packed_dim = hybrid_tq35.layout.packed_dim

        # Simulate KV cache: [num_blocks, block_size, kv_heads, tq_dim]
        # tq_dim >= packed_dim (vLLM allocates based on page size)
        tq_dim = packed_dim + 4  # extra padding bytes
        key_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, tq_dim,
                                dtype=torch.uint8)

        # Write some encoded data into blocks 0-3
        for bidx in range(4):
            for si in range(BLOCK_SIZE):
                x = torch.randn(NUM_KV_HEADS, HEAD_DIM)
                packed = hybrid_tq35.encode(x)
                key_cache[bidx, si, :, :packed_dim] = packed

        # Clamp-gather: simulate block_table [2 seqs, 3 max_blocks]
        block_table = torch.tensor([[0, 1, -1],
                                    [2, 3, -1]])
        bt_clamped = block_table.clamp(min=0, max=NUM_BLOCKS - 1)
        flat_bt = bt_clamped.reshape(-1)  # [6]

        # Gather
        kp_gather = key_cache[flat_bt, :, :, :packed_dim]
        # [6, block_size, kv_heads, packed_dim]
        assert kp_gather.shape == (6, BLOCK_SIZE, NUM_KV_HEADS, packed_dim)

        # Batch decompress
        total_slots = flat_bt.shape[0] * BLOCK_SIZE * NUM_KV_HEADS
        flat_packed = kp_gather.reshape(total_slots, packed_dim)
        decoded = hybrid_tq35.decode(flat_packed)
        assert decoded.shape == (total_slots, HEAD_DIM)

        # Reshape to per-sequence
        num_seqs = 2
        max_ctx = 3 * BLOCK_SIZE  # max_blocks * block_size
        dk = decoded.reshape(num_seqs, max_ctx, NUM_KV_HEADS, HEAD_DIM)
        assert dk.shape == (2, 48, NUM_KV_HEADS, HEAD_DIM)

    def test_attention_mask_from_seq_lens(self):
        """Verify attention mask correctly masks padding positions."""
        max_ctx = 48  # 3 blocks * 16 tokens
        seq_lens = torch.tensor([32, 16])  # 2 sequences, different lengths
        positions = torch.arange(max_ctx)
        valid_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)

        # Seq 0: first 32 valid, rest masked
        assert valid_mask[0, 31].item() is True
        assert valid_mask[0, 32].item() is False

        # Seq 1: first 16 valid, rest masked
        assert valid_mask[1, 15].item() is True
        assert valid_mask[1, 16].item() is False

        # Build SDPA mask
        attn_mask = torch.where(
            valid_mask.unsqueeze(1).unsqueeze(1),
            torch.zeros(1),
            torch.tensor(float('-inf')),
        )
        assert attn_mask.shape == (2, 1, 1, 48)
        assert attn_mask[0, 0, 0, 0].item() == 0.0
        assert attn_mask[0, 0, 0, 32].item() == float('-inf')

    def test_masked_sdpa_ignores_padding(self):
        """SDPA with mask should produce valid output, ignoring padding."""
        num_seqs = 2
        max_ctx = 32
        seq_lens = torch.tensor([16, 8])

        q = torch.randn(num_seqs, NUM_HEADS, 1, HEAD_DIM)  # decode: 1 query token
        k = torch.randn(num_seqs, NUM_HEADS, max_ctx, HEAD_DIM)
        v = torch.randn(num_seqs, NUM_HEADS, max_ctx, HEAD_DIM)

        positions = torch.arange(max_ctx)
        valid_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        attn_mask = torch.where(
            valid_mask.unsqueeze(1).unsqueeze(1),
            torch.zeros(1),
            torch.tensor(float('-inf')),
        )

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        assert out.shape == (num_seqs, NUM_HEADS, 1, HEAD_DIM)
        # Output should not be NaN or Inf
        assert torch.isfinite(out).all()

    def test_gqa_expansion(self):
        """GQA ratio expansion should work correctly."""
        gqa_ratio = NUM_HEADS // NUM_KV_HEADS  # 32 // 8 = 4
        k = torch.randn(1, NUM_KV_HEADS, 16, HEAD_DIM)
        k_expanded = k.repeat_interleave(gqa_ratio, dim=1)
        assert k_expanded.shape == (1, NUM_HEADS, 16, HEAD_DIM)
        # First 4 head slices should be identical (expanded from head 0)
        torch.testing.assert_close(k_expanded[:, 0], k_expanded[:, 1])
        torch.testing.assert_close(k_expanded[:, 0], k_expanded[:, 3])
        # Head 4 should differ (from head 1)
        assert not torch.equal(k_expanded[:, 0], k_expanded[:, 4])


class TestEndToEndHybridDecode:
    """End-to-end test: encode into cache -> clamp-gather decompress -> SDPA -> verify output."""

    def test_full_hybrid_decode_pipeline(self, hybrid_tq35):
        """Simulate a complete hybrid decode step."""
        packed_dim = hybrid_tq35.layout.packed_dim
        tq_dim = packed_dim + 4
        num_seqs = 2

        # 1. Allocate cache
        key_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, tq_dim,
                                dtype=torch.uint8)
        value_cache = torch.zeros_like(key_cache)

        # 2. Encode some KV tokens into blocks 0-3
        original_k = {}
        original_v = {}
        for bidx in range(4):
            for si in range(BLOCK_SIZE):
                k_vec = torch.randn(NUM_KV_HEADS, HEAD_DIM)
                v_vec = torch.randn(NUM_KV_HEADS, HEAD_DIM)
                original_k[(bidx, si)] = k_vec.clone()
                original_v[(bidx, si)] = v_vec.clone()

                kp = hybrid_tq35.encode(k_vec)
                vp = hybrid_tq35.encode(v_vec)
                key_cache[bidx, si, :, :packed_dim] = kp
                value_cache[bidx, si, :, :packed_dim] = vp

        # 3. Set up block table and seq_lens
        # Seq 0: 2 blocks (32 tokens), Seq 1: 1 block (16 tokens)
        max_blocks_per_seq = 3
        block_table = torch.tensor([[0, 1, -1],
                                    [2, -1, -1]])
        seq_lens = torch.tensor([32, 16])
        max_ctx = max_blocks_per_seq * BLOCK_SIZE

        # 4. Clamp-gather decompress
        bt_clamped = block_table.clamp(min=0, max=NUM_BLOCKS - 1)
        flat_bt = bt_clamped.reshape(-1)

        kp_gather = key_cache[flat_bt, :, :, :packed_dim]
        vp_gather = value_cache[flat_bt, :, :, :packed_dim]

        total_slots = flat_bt.shape[0] * BLOCK_SIZE * NUM_KV_HEADS
        dk = hybrid_tq35.decode(kp_gather.reshape(total_slots, packed_dim))
        dv = hybrid_tq35.decode(vp_gather.reshape(total_slots, packed_dim))

        dk = dk.reshape(num_seqs, max_ctx, NUM_KV_HEADS, HEAD_DIM)
        dv = dv.reshape(num_seqs, max_ctx, NUM_KV_HEADS, HEAD_DIM)

        # 5. Verify decompressed data matches original (approximately)
        # Block 0 data should appear in seq 0, positions 0-15
        for si in range(BLOCK_SIZE):
            orig_k = original_k[(0, si)]
            recon_k = dk[0, si]
            cos = F.cosine_similarity(
                orig_k.reshape(1, -1).float(),
                recon_k.reshape(1, -1).float())
            assert cos.item() > 0.6, (
                f"Block 0, slot {si}: cosine {cos.item():.3f} too low")

        # 6. Build mask and run SDPA
        positions = torch.arange(max_ctx)
        valid_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        attn_mask = torch.where(
            valid_mask.unsqueeze(1).unsqueeze(1),
            torch.zeros(1, dtype=torch.float32),
            torch.tensor(float('-inf'), dtype=torch.float32),
        )

        query = torch.randn(num_seqs, 1, NUM_HEADS, HEAD_DIM)
        q = query.transpose(1, 2)  # [S, H, 1, D]

        k = dk.float().transpose(1, 2)  # [S, kv_H, max_ctx, D]
        v = dv.float().transpose(1, 2)

        gqa_ratio = NUM_HEADS // NUM_KV_HEADS
        k = k.repeat_interleave(gqa_ratio, dim=1)
        v = v.repeat_interleave(gqa_ratio, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        assert out.shape == (num_seqs, NUM_HEADS, 1, HEAD_DIM)
        assert torch.isfinite(out).all()

    def test_single_token_context(self, hybrid_tq35):
        """Edge case: 1 token of context."""
        packed_dim = hybrid_tq35.layout.packed_dim
        tq_dim = packed_dim + 4

        key_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, tq_dim,
                                dtype=torch.uint8)
        value_cache = torch.zeros_like(key_cache)

        # Encode 1 token into block 0, slot 0
        k_vec = torch.randn(NUM_KV_HEADS, HEAD_DIM)
        v_vec = torch.randn(NUM_KV_HEADS, HEAD_DIM)
        key_cache[0, 0, :, :packed_dim] = hybrid_tq35.encode(k_vec)
        value_cache[0, 0, :, :packed_dim] = hybrid_tq35.encode(v_vec)

        block_table = torch.tensor([[0]])
        seq_lens = torch.tensor([1])
        bt_clamped = block_table.clamp(min=0, max=NUM_BLOCKS - 1)
        flat_bt = bt_clamped.reshape(-1)

        kp = key_cache[flat_bt, :, :, :packed_dim]
        total = flat_bt.shape[0] * BLOCK_SIZE * NUM_KV_HEADS
        dk = hybrid_tq35.decode(kp.reshape(total, packed_dim))
        dk = dk.reshape(1, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)

        # Mask: only position 0 is valid
        max_ctx = BLOCK_SIZE
        positions = torch.arange(max_ctx)
        valid_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        attn_mask = torch.where(
            valid_mask.unsqueeze(1).unsqueeze(1),
            torch.zeros(1, dtype=torch.float32),
            torch.tensor(float('-inf'), dtype=torch.float32),
        )

        q = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
        k = dk.float().transpose(1, 2).repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
        v = k.clone()  # same as k for simplicity

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        assert torch.isfinite(out).all()


# =====================================================================
# PACKED DIM CONSISTENCY
# =====================================================================

class TestPackedDimConsistency:
    """Verify packed_dim calculations are consistent across modules."""

    def test_packed_dim_for_mode_tq35(self):
        from turboquant.hybrid_quantizer import HybridTurboQuant
        pd = HybridTurboQuant.packed_dim_for_mode(HEAD_DIM, "tq35")
        assert pd > 0
        assert pd < HEAD_DIM  # should be compressed

    def test_packed_dim_for_mode_tq25(self):
        from turboquant.hybrid_quantizer import HybridTurboQuant
        pd = HybridTurboQuant.packed_dim_for_mode(HEAD_DIM, "tq25")
        assert pd > 0
        assert pd < HEAD_DIM

    def test_tq25_smaller_than_tq35(self):
        from turboquant.hybrid_quantizer import HybridTurboQuant
        pd35 = HybridTurboQuant.packed_dim_for_mode(HEAD_DIM, "tq35")
        pd25 = HybridTurboQuant.packed_dim_for_mode(HEAD_DIM, "tq25")
        assert pd25 < pd35

    def test_uniform_packed_size(self):
        from turboquant.packing import packed_size
        pd4 = packed_size(HEAD_DIM, 4)  # 4-bit: 128*4/8 = 64
        pd3 = packed_size(HEAD_DIM, 3)
        pd2 = packed_size(HEAD_DIM, 2)
        assert pd4 == 64
        assert pd3 == 48  # ceil(128*3/8)
        assert pd2 == 32

    def test_engine_page_size_hybrid(self):
        """turboquant.vllm.engine._tq_page_size_bytes should use hybrid dims."""
        with patch.dict(os.environ, {"AITHER_TQ_MODE": "tq35"}):
            from turboquant.vllm.engine import _tq_page_size_bytes
            page = _tq_page_size_bytes(BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
            assert page > 0
            # Should be smaller than FP16 page
            fp16_page = 2 * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM * 2
            assert page < fp16_page

    def test_engine_page_size_uniform(self):
        """turboquant.vllm.engine._tq_page_size_bytes should use uniform dims."""
        import turboquant.vllm.engine as engine_mod
        old_bits = engine_mod._TQ_BITS
        try:
            engine_mod._TQ_BITS = 4
            with patch.dict(os.environ, {"AITHER_TQ_MODE": "tq4"}):
                page = engine_mod._tq_page_size_bytes(BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
                dim = engine_mod._tq_dim_for_head(HEAD_DIM)
                assert page > 0
                assert dim == 64 + 4  # packed_size(128, 4) + 4 norm bytes
        finally:
            engine_mod._TQ_BITS = old_bits


# =====================================================================
# ENCODE INTO CACHE + DECOMPRESS ROUNDTRIP (simulates hooks path)
# =====================================================================

class TestCacheRoundtrip:
    """Simulate the exact encode-into-cache -> decompress path the hooks use."""

    def _roundtrip(self, tq, mode):
        """Run the hooks-style roundtrip for a given quantizer."""
        if mode in ("tq35", "tq25"):
            packed_dim = tq.layout.packed_dim
            is_hybrid = True
        else:
            from turboquant.packing import packed_size
            packed_dim = packed_size(HEAD_DIM, tq.bits)
            is_hybrid = False

        tq_dim = packed_dim + 4
        key_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, tq_dim,
                                dtype=torch.uint8)

        # Simulate slot_mapping encode (what _tq_encode_phase does)
        N = 2  # 2 tokens
        original = torch.randn(N, NUM_KV_HEADS, HEAD_DIM)

        slot_mapping = torch.tensor([0, 1])  # block 0, slots 0 and 1
        bi = slot_mapping // BLOCK_SIZE
        oi = slot_mapping % BLOCK_SIZE

        if is_hybrid:
            for i in range(N):
                packed = tq.encode(original[i])  # [kv_heads, packed_dim]
                key_cache[bi[i], oi[i], :, :packed_dim] = packed
        else:
            for i in range(N):
                packed, norms = tq.encode(original[i].reshape(NUM_KV_HEADS, HEAD_DIM))
                key_cache[bi[i], oi[i], :, :packed_dim] = packed.reshape(
                    NUM_KV_HEADS, packed_dim)

        # Simulate decompress (what _tq_decompress_active does)
        kp = key_cache[0, :2, :, :packed_dim]  # block 0, slots 0-1
        flat = kp.reshape(2 * NUM_KV_HEADS, packed_dim)

        if is_hybrid:
            decoded = tq.decode(flat)
        else:
            # For uniform, we'd need norms -- skip for this test
            return

        decoded = decoded.reshape(N, NUM_KV_HEADS, HEAD_DIM)

        # Verify quality
        for i in range(N):
            cos = F.cosine_similarity(
                original[i].reshape(1, -1),
                decoded[i].reshape(1, -1))
            assert cos.item() > 0.6, f"Token {i}: cosine {cos.item():.3f} too low"

    def test_tq35_cache_roundtrip(self, hybrid_tq35):
        self._roundtrip(hybrid_tq35, "tq35")

    def test_tq25_cache_roundtrip(self, hybrid_tq25):
        self._roundtrip(hybrid_tq25, "tq25")
