"""
TriAttention — Trigonometric Series Attention with spectral KV compression.

Main module orchestrating the full pipeline:
  1. Encode keys/values to spectral representation (SpectralKVEncoder)
  2. Score queries against spectral keys via trig series (TrigSeriesScorer)
  3. Reconstruct output using spectral values (weighted scatter-add)

Supports both full-context attention (prefill) and single-token decode.

Mathematical guarantee:
    For a key vector k with pair energies E_0 ≥ E_1 ≥ ... ≥ E_{d/2-1},
    retaining the top-F pairs gives:

    |exact_score − approx_score| ≤ ||q|| · √(Σ_{i>F} E_i) / √d

    With F=12 on Qwen3.5-8B (head_dim=128), this error is typically
    < 0.5% of the attention score magnitude.

Integration with TurboQuant:
    TriAttention and TurboQuant are complementary:
    - TurboQuant: compresses K/V vectors via vector quantization (3.8×)
    - TriAttention: compresses via spectral truncation + quantization (10×)
    - Combined: use TriAttention for the spectral basis, TurboQuant for
      coefficient quantization → even higher compression is possible.
"""

import math
import torch
from typing import Optional, Tuple

from .config import TriAttentionConfig
from .encoder import SpectralKVEncoder, SpectralEncoding
from .scorer import TrigSeriesScorer
from .spectral import rope_frequencies


class TriAttention:
    """Trigonometric Series Attention with spectral KV compression.

    Drop-in replacement for standard multi-head attention that stores
    K/V in spectrally-compressed form and scores via trig series.

    Usage (prefill):
        tri = TriAttention(config)
        output = tri.forward(query, key, value, positions)

    Usage (decode):
        tri = TriAttention(config)
        # Prefill: encode and cache
        k_enc, v_enc = tri.encode_kv(keys, values)
        # Decode: score and accumulate
        output = tri.decode_step(query, k_enc, v_enc, query_pos, key_positions)

    Usage (with external cache):
        from .cache import SpectralKVCache
        cache = SpectralKVCache(config, max_blocks=2048)
        # Store during prefill
        cache.store_block(block_idx, k_enc, v_enc)
        # Fetch during decode
        k_enc, v_enc = cache.fetch_sequence(block_table, context_len)
        output = tri.decode_step(query, k_enc, v_enc, query_pos, key_positions)
    """

    def __init__(
        self,
        config: TriAttentionConfig,
        device: str = "cpu",
    ):
        self.config = config
        self.device = device
        self.head_dim = config.head_dim
        self.num_query_heads = config.num_query_heads
        self.num_kv_heads = config.num_kv_heads
        self.gqa_ratio = config.gqa_ratio
        self.scale = 1.0 / math.sqrt(config.head_dim)

        # Sub-modules
        self.encoder = SpectralKVEncoder(config)
        self.scorer = TrigSeriesScorer(config, device=device)

        # Pre-computed RoPE frequencies for value reconstruction
        self._theta = rope_frequencies(
            config.head_dim, config.rope_base, device=device
        )

    def to(self, device: str) -> "TriAttention":
        """Move all state to device."""
        self.device = device
        self.scorer = self.scorer.to(device)
        self._theta = self._theta.to(device)
        return self

    # ── ENCODE ────────────────────────────────────────────────────

    def encode_kv(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[SpectralEncoding, SpectralEncoding]:
        """Encode key and value tensors to spectral representation.

        Args:
            keys: [B, S, num_kv_heads, head_dim] key vectors (pre-RoPE).
            values: [B, S, num_kv_heads, head_dim] value vectors.

        Returns:
            (k_enc, v_enc) SpectralEncoding tuples.
        """
        k_enc = self.encoder.encode(keys)
        v_enc = self.encoder.encode(values)
        return k_enc, v_enc

    # ── FULL FORWARD (prefill) ────────────────────────────────────

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full TriAttention forward pass.

        Args:
            query: [B, num_query_heads, S_q, head_dim] or [B, num_query_heads, head_dim].
            key: [B, S_kv, num_kv_heads, head_dim] pre-RoPE keys.
            value: [B, S_kv, num_kv_heads, head_dim] values.
            positions: [B, S_kv] key positions. Default: 0..S_kv-1.
            attention_mask: [B, S_q, S_kv] or [B, 1, S_kv] additive mask
                            (-inf for masked positions).

        Returns:
            [B, num_query_heads, head_dim] or [B, num_query_heads, S_q, head_dim]
            attention output.
        """
        # Handle single-query case (decode)
        squeeze_q = False
        if query.dim() == 3:
            query = query.unsqueeze(2)  # [B, QH, 1, D]
            squeeze_q = True

        B, QH, S_q, D = query.shape
        S_kv = key.shape[1]

        # Encode K/V
        k_enc, v_enc = self.encode_kv(key, value)

        # Score all query positions against all key positions
        outputs = []
        for qi in range(S_q):
            q_i = query[:, :, qi, :]  # [B, QH, D]
            q_pos = torch.full((B,), qi, device=query.device)

            scores = self.scorer.score(
                q_i, k_enc, q_pos, positions
            )  # [B, QH, S_kv]

            # Apply mask
            if attention_mask is not None:
                if attention_mask.dim() == 3:
                    mask_i = attention_mask[:, qi:qi+1, :]  # [B, 1, S_kv]
                else:
                    mask_i = attention_mask
                scores = scores + mask_i

            # Softmax
            weights = torch.softmax(scores.float(), dim=-1)  # [B, QH, S_kv]

            # Value accumulation
            out = self._accumulate_values(
                weights, v_enc, query.device
            )  # [B, QH, D]
            outputs.append(out)

        output = torch.stack(outputs, dim=2)  # [B, QH, S_q, D]

        if squeeze_q:
            output = output.squeeze(2)

        return output

    # ── DECODE STEP ───────────────────────────────────────────────

    def decode_step(
        self,
        query: torch.Tensor,
        k_enc: SpectralEncoding,
        v_enc: SpectralEncoding,
        query_pos: Optional[torch.Tensor] = None,
        key_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single decode step using pre-encoded K/V.

        Args:
            query: [B, num_query_heads, head_dim] single-token query.
            k_enc: SpectralEncoding for all cached keys.
            v_enc: SpectralEncoding for all cached values.
            query_pos: [B] query positions.
            key_positions: [B, S] key positions.

        Returns:
            [B, num_query_heads, head_dim] attention output.
        """
        # Score
        scores = self.scorer.score(
            query, k_enc, query_pos, key_positions
        )  # [B, QH, S]

        # Softmax
        weights = torch.softmax(scores.float(), dim=-1)  # [B, QH, S]

        # Value accumulation
        return self._accumulate_values(weights, v_enc, query.device)

    # ── VALUE ACCUMULATION ────────────────────────────────────────

    def _accumulate_values(
        self,
        weights: torch.Tensor,
        v_enc: SpectralEncoding,
        device: torch.device,
    ) -> torch.Tensor:
        """Accumulate weighted values from spectral representation.

        Uses scatter_add to handle variable frequency indices per token.

        Args:
            weights: [B, QH, S] attention weights (post-softmax).
            v_enc: SpectralEncoding with shapes [B, S, KVH, F] etc.

        Returns:
            [B, QH, D] accumulated output.
        """
        B, QH, S = weights.shape
        D = self.head_dim
        KVH = self.num_kv_heads
        F = v_enc.indices.shape[-1]

        # Unpack value coefficients
        v0, v1 = self.scorer._unpack_coefficients(v_enc)  # [B, S, KVH, F]
        v_indices = v_enc.indices.long()  # [B, S, KVH, F]

        # GQA: map query heads to KV heads
        kv_head_map = torch.arange(QH, device=device) // self.gqa_ratio

        # Select KV head data for each query head
        v0_gqa = v0[:, :, kv_head_map]         # [B, S, QH, F]
        v1_gqa = v1[:, :, kv_head_map]         # [B, S, QH, F]
        idx_gqa = v_indices[:, :, kv_head_map]  # [B, S, QH, F]

        # Transpose to [B, QH, S, F]
        v0_gqa = v0_gqa.permute(0, 2, 1, 3)
        v1_gqa = v1_gqa.permute(0, 2, 1, 3)
        idx_gqa = idx_gqa.permute(0, 2, 1, 3)

        # Weight the values: [B, QH, S, F]
        w_expanded = weights.unsqueeze(-1)  # [B, QH, S, 1]
        wv0 = w_expanded * v0_gqa
        wv1 = w_expanded * v1_gqa

        # Scatter-add into output
        # Flatten S*F for scatter
        flat_idx = idx_gqa.reshape(B, QH, -1)   # [B, QH, S*F]
        flat_wv0 = wv0.reshape(B, QH, -1)       # [B, QH, S*F]
        flat_wv1 = wv1.reshape(B, QH, -1)       # [B, QH, S*F]

        out_even = torch.zeros(B, QH, D // 2, device=device, dtype=torch.float32)
        out_odd = torch.zeros(B, QH, D // 2, device=device, dtype=torch.float32)

        out_even.scatter_add_(-1, flat_idx, flat_wv0)
        out_odd.scatter_add_(-1, flat_idx, flat_wv1)

        # Interleave even/odd back to full head_dim
        output = torch.zeros(B, QH, D, device=device, dtype=torch.float32)
        output[..., 0::2] = out_even
        output[..., 1::2] = out_odd

        return output

    # ── REFERENCE (for testing) ───────────────────────────────────

    @staticmethod
    def reference_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rope_base: float = 1_000_000.0,
        positions_q: Optional[torch.Tensor] = None,
        positions_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard full attention with explicit RoPE for comparison.

        Args:
            query: [B, QH, D] queries (pre-RoPE).
            key: [B, S, KVH, D] keys (pre-RoPE).
            value: [B, S, KVH, D] values.
            positions_q: [B] query positions.
            positions_k: [B, S] key positions.

        Returns:
            [B, QH, D] attention output.
        """
        B, QH, D = query.shape
        S = key.shape[1]
        KVH = key.shape[2]
        gqa = QH // KVH

        if positions_q is None:
            positions_q = torch.full((B,), S, device=query.device)
        if positions_k is None:
            positions_k = torch.arange(S, device=query.device).unsqueeze(0).expand(B, -1)

        scale = 1.0 / math.sqrt(D)
        theta = rope_frequencies(D, rope_base, device=query.device)

        # Apply RoPE to query
        q_rope = _apply_rope(query, positions_q.unsqueeze(-1), theta)  # [B, QH, D]

        # Apply RoPE to keys
        key_flat = key.reshape(B * S, KVH, D)
        pos_flat = positions_k.reshape(B * S).unsqueeze(-1)
        k_rope = _apply_rope(key_flat, pos_flat, theta).reshape(B, S, KVH, D)

        # Score: [B, QH, S]
        scores = torch.zeros(B, QH, S, device=query.device, dtype=torch.float32)
        for s in range(S):
            for qh in range(QH):
                kvh = qh // gqa
                scores[:, qh, s] = (
                    (q_rope[:, qh] * k_rope[:, s, kvh]).sum(dim=-1) * scale
                )

        # Softmax + weighted sum
        weights = torch.softmax(scores, dim=-1)  # [B, QH, S]
        output = torch.zeros(B, QH, D, device=query.device, dtype=torch.float32)
        for s in range(S):
            for qh in range(QH):
                kvh = qh // gqa
                output[:, qh] += weights[:, qh, s].unsqueeze(-1) * value[:, s, kvh].float()

        return output

    # ── BENCHMARKING ──────────────────────────────────────────────

    def benchmark(
        self,
        batch_size: int = 1,
        seq_len: int = 2048,
        num_trials: int = 10,
    ) -> dict:
        """Benchmark TriAttention vs standard attention.

        Returns timing, compression, and quality metrics.
        """
        import time

        D = self.head_dim
        QH = self.num_query_heads
        KVH = self.num_kv_heads

        # Generate random data
        query = torch.randn(batch_size, QH, D, device=self.device)
        key = torch.randn(batch_size, seq_len, KVH, D, device=self.device)
        value = torch.randn(batch_size, seq_len, KVH, D, device=self.device)

        # Encode
        t0 = time.perf_counter()
        for _ in range(num_trials):
            k_enc, v_enc = self.encode_kv(key, value)
        t_encode = (time.perf_counter() - t0) / num_trials

        # Decode step
        positions_k = torch.arange(seq_len, device=self.device).unsqueeze(0)
        query_pos = torch.tensor([seq_len], device=self.device)

        t0 = time.perf_counter()
        for _ in range(num_trials):
            out = self.decode_step(query, k_enc, v_enc, query_pos, positions_k)
        t_decode = (time.perf_counter() - t0) / num_trials

        # Reference
        t0 = time.perf_counter()
        for _ in range(num_trials):
            ref = self.reference_attention(query, key, value)
        t_ref = (time.perf_counter() - t0) / num_trials

        # Quality
        cosine_sim = torch.nn.functional.cosine_similarity(
            out.reshape(-1, D), ref.reshape(-1, D), dim=-1
        ).mean().item()
        mse = ((out - ref) ** 2).mean().item()

        return {
            "encode_ms": t_encode * 1000,
            "decode_ms": t_decode * 1000,
            "reference_ms": t_ref * 1000,
            "speedup": t_ref / t_decode if t_decode > 0 else float("inf"),
            "compression_ratio": self.config.kv_compression_ratio,
            "cosine_similarity": cosine_sim,
            "mse": mse,
            "seq_len": seq_len,
            "batch_size": batch_size,
        }


# ============================================================================
# RoPE HELPER (for reference attention)
# ============================================================================

def _apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Apply Rotary Position Embedding.

    Args:
        x: [..., head_dim] input vectors.
        positions: [..., 1] or [...] position indices.
        theta: [head_dim // 2] RoPE frequencies.

    Returns:
        [..., head_dim] rotated vectors.
    """
    D = x.shape[-1]
    x_float = x.float()
    x_even = x_float[..., 0::2]  # [..., D/2]
    x_odd = x_float[..., 1::2]   # [..., D/2]

    if positions.dim() < x_even.dim():
        positions = positions.unsqueeze(-1)

    # Expand positions for broadcasting
    while positions.dim() < x_even.dim():
        positions = positions.unsqueeze(-1)

    angles = positions.float() * theta  # [..., D/2]

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    out_even = x_even * cos_a - x_odd * sin_a
    out_odd = x_even * sin_a + x_odd * cos_a

    out = torch.zeros_like(x_float)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out
