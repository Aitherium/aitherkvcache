"""
Trigonometric Series Scorer — attention scoring from spectral KV representations.

The core of TriAttention: compute attention scores WITHOUT materializing
full key vectors, using the RoPE trig series decomposition.

Mathematical foundation:

    With RoPE, the attention score between query at position m and key at
    position n decomposes exactly as a trigonometric series in Δ = n − m:

    score(q, k, m, n) = (1/√d) Σ_{i=0}^{d/2-1} [c_i cos(Δθ_i) + s_i sin(Δθ_i)]

    where:
        c_i = q_{2i} k_{2i} + q_{2i+1} k_{2i+1}   (pair dot product)
        s_i = q_{2i+1} k_{2i} - q_{2i} k_{2i+1}   (pair cross product)
        θ_i = base^{-2i/d}                          (RoPE frequency)
        Δ = n - m                                     (position difference)

    By retaining only the top-F frequency pairs (by key energy), the
    truncated series gives a bounded approximation:

    |exact_score - approx_score| ≤ ||q|| · ||k_residual|| / √d

    where k_residual are the discarded frequency pair components.

Scoring modes:
    1. Pre-RoPE (default): Keys stored before RoPE; position handled via
       analytic cos/sin phases. Best compression, exact position handling.
    2. Post-RoPE: Keys stored after RoPE; simple sparse dot product.
       No phase computation needed, but worse spectral concentration.
"""

import math
import torch
from typing import Optional, Tuple

from .config import TriAttentionConfig
from .encoder import SpectralEncoding
from .spectral import rope_frequencies


class TrigSeriesScorer:
    """Compute attention scores from spectral KV representations.

    Supports both pre-RoPE (trig series with analytic phases) and
    post-RoPE (sparse dot product) scoring.

    Usage:
        scorer = TrigSeriesScorer(config)
        scores = scorer.score(query, k_enc, query_pos, key_positions)
    """

    def __init__(
        self,
        config: TriAttentionConfig,
        device: str = "cpu",
    ):
        self.config = config
        self.head_dim = config.head_dim
        self.num_pairs = config.num_pairs
        self.num_freqs = config.num_freqs
        self.coeff_bits = config.coeff_bits
        self.pre_rope = config.pre_rope
        self.scale = 1.0 / math.sqrt(config.head_dim)
        self.gqa_ratio = config.gqa_ratio

        # Precompute RoPE frequencies
        self._theta = rope_frequencies(
            config.head_dim, config.rope_base, device=device
        )

        # Quantization parameters (mirror encoder)
        if self.coeff_bits == 4:
            self._half_code = 7.5
        elif self.coeff_bits == 8:
            self._half_code = 127.5

    def to(self, device: str) -> "TrigSeriesScorer":
        """Move scorer state to device."""
        self._theta = self._theta.to(device)
        return self

    def _unpack_coefficients(
        self,
        enc: SpectralEncoding,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unpack spectral encoding to even/odd coefficient tensors.

        Returns:
            k0: [..., F] even-dimension values (k_{2i})
            k1: [..., F] odd-dimension values (k_{2i+1})
        """
        if self.coeff_bits == 16:
            return enc.packed[..., 0].float(), enc.packed[..., 1].float()

        scales = enc.scales.float()

        if self.coeff_bits == 4:
            v0 = (enc.packed >> 4).float()
            v1 = (enc.packed & 0x0F).float()
            k0 = (v0 / self._half_code - 1.0) * scales.unsqueeze(-1)
            k1 = (v1 / self._half_code - 1.0) * scales.unsqueeze(-1)
            return k0, k1

        elif self.coeff_bits == 8:
            F = enc.indices.shape[-1]
            raw = enc.packed.float().reshape(*enc.packed.shape[:-1], F, 2)
            k0 = (raw[..., 0] / self._half_code - 1.0) * scales.unsqueeze(-1)
            k1 = (raw[..., 1] / self._half_code - 1.0) * scales.unsqueeze(-1)
            return k0, k1

        raise ValueError(f"Unsupported coeff_bits: {self.coeff_bits}")

    def score(
        self,
        query: torch.Tensor,
        k_enc: SpectralEncoding,
        query_pos: Optional[torch.Tensor] = None,
        key_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention scores via trigonometric series.

        Args:
            query: [B, num_query_heads, head_dim] query vectors (pre-RoPE).
            k_enc: SpectralEncoding with shapes:
                   indices: [B, S, num_kv_heads, F]
                   packed:  [B, S, num_kv_heads, F] (or [..., F, 2])
                   scales:  [B, S, num_kv_heads]
            query_pos: [B] query positions (required for pre-RoPE mode).
            key_positions: [B, S] key positions (default: 0, 1, ..., S-1).

        Returns:
            [B, num_query_heads, S] attention scores (pre-softmax logits).
        """
        B, QH, D = query.shape
        S = k_enc.indices.shape[1]
        KVH = k_enc.indices.shape[2]
        F = k_enc.indices.shape[3]
        device = query.device

        # Default positions: 0, 1, 2, ..., S-1
        if key_positions is None:
            key_positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        if query_pos is None:
            query_pos = torch.full((B,), S, device=device, dtype=key_positions.dtype)

        # Unpack key coefficients
        k0, k1 = self._unpack_coefficients(k_enc)  # [B, S, KVH, F]
        k_indices = k_enc.indices.long()  # [B, S, KVH, F]

        # GQA: map query heads to KV heads
        kv_head_map = torch.arange(QH, device=device) // self.gqa_ratio

        # Select KV head data for each query head
        k_indices_gqa = k_indices[:, :, kv_head_map]  # [B, S, QH, F]
        k0_gqa = k0[:, :, kv_head_map]                # [B, S, QH, F]
        k1_gqa = k1[:, :, kv_head_map]                # [B, S, QH, F]

        # Transpose to [B, QH, S, F]
        k_indices_gqa = k_indices_gqa.permute(0, 2, 1, 3)
        k0_gqa = k0_gqa.permute(0, 2, 1, 3)
        k1_gqa = k1_gqa.permute(0, 2, 1, 3)

        # Split query into even/odd pair components
        q_even = query.float()[..., 0::2]  # [B, QH, D//2]
        q_odd = query.float()[..., 1::2]   # [B, QH, D//2]

        # Gather query components at retained frequency indices
        q_e = torch.gather(
            q_even.unsqueeze(2).expand(-1, -1, S, -1),
            -1, k_indices_gqa,
        )  # [B, QH, S, F]
        q_o = torch.gather(
            q_odd.unsqueeze(2).expand(-1, -1, S, -1),
            -1, k_indices_gqa,
        )  # [B, QH, S, F]

        # Pair products: the trig series coefficients
        c = q_e * k0_gqa + q_o * k1_gqa    # [B, QH, S, F] cosine coeff
        s = q_o * k0_gqa - q_e * k1_gqa    # [B, QH, S, F] sine coeff

        if self.pre_rope:
            # Trig series scoring with RoPE phases
            theta = self._theta.to(device)
            theta_at_idx = theta[k_indices_gqa]  # [B, QH, S, F]

            # Position difference: [B, 1, S, 1] for broadcasting
            # key_positions: [B, S], query_pos: [B]
            delta = (
                key_positions.float().view(B, 1, S, 1)       # [B, 1, S, 1]
                - query_pos.float().view(B, 1, 1, 1)          # [B, 1, 1, 1]
            )  # [B, 1, S, 1]

            phases = delta * theta_at_idx  # [B, QH, S, F]
            cos_ph = torch.cos(phases)
            sin_ph = torch.sin(phases)

            # Trigonometric series: score = Σ_f [c_f cos(Δθ_f) + s_f sin(Δθ_f)]
            raw_scores = (c * cos_ph + s * sin_ph).sum(dim=-1)  # [B, QH, S]
        else:
            # Post-RoPE: simple sparse dot product (no phase needed)
            raw_scores = c.sum(dim=-1)  # [B, QH, S]

        # Scale by 1/√d
        scores = raw_scores * self.scale

        return scores

    def score_single(
        self,
        query: torch.Tensor,
        k_enc: SpectralEncoding,
        query_pos: int,
        key_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score a single query against all keys (decode-optimized).

        Args:
            query: [1, num_query_heads, head_dim] single query.
            k_enc: Full KV cache spectral encoding.
            query_pos: Integer query position.
            key_positions: [1, S] key positions.

        Returns:
            [1, num_query_heads, S] attention scores.
        """
        pos_tensor = torch.tensor([query_pos], device=query.device)
        return self.score(query, k_enc, pos_tensor, key_positions)
