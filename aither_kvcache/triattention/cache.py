"""
Spectral KV Cache — paged storage for TriAttention spectral representations.

Stores spectrally-compressed keys and values in a paged layout compatible
with vLLM's block-based KV cache management. Each page holds `block_size`
tokens worth of spectral coefficients.

Memory layout per physical block:
    k_indices:  [block_size, num_kv_heads, num_freqs]    uint8
    k_packed:   [block_size, num_kv_heads, num_freqs]    uint8  (int4 mode)
    k_scales:   [block_size, num_kv_heads]               float16
    v_indices:  [block_size, num_kv_heads, num_freqs]    uint8
    v_packed:   [block_size, num_kv_heads, num_freqs]    uint8  (int4 mode)
    v_scales:   [block_size, num_kv_heads]               float16

Total bytes per block (int4, F=12, KVH=8, BS=16):
    K: 16 × 8 × (12 + 12 + 2) = 16 × 8 × 26 = 3,328 bytes
    V: same = 3,328 bytes
    Total: 6,656 bytes per block

vs FP16 paged cache:
    K: 16 × 8 × 256 = 32,768 bytes
    V: same = 32,768 bytes
    Total: 65,536 bytes per block

Compression: 65,536 / 6,656 = 9.85× ≈ 10×
"""

import math
import torch
from typing import Optional, Tuple

from .config import TriAttentionConfig
from .encoder import SpectralKVEncoder, SpectralEncoding


class SpectralKVCache:
    """Paged KV cache storing spectral representations.

    Manages physical blocks of spectrally-compressed key/value tokens.
    Compatible with vLLM's block table indirection scheme.

    Usage:
        cache = SpectralKVCache(config, max_blocks=1024)
        cache.store(block_idx=0, position=5, head_idx=2,
                    k_enc=key_encoding, v_enc=val_encoding)
        k_enc, v_enc = cache.fetch_block(block_idx=0)
    """

    def __init__(
        self,
        config: TriAttentionConfig,
        max_blocks: int = 1024,
        device: str = "cpu",
    ):
        self.config = config
        self.max_blocks = max_blocks
        self.block_size = config.block_size
        self.num_kv_heads = config.num_kv_heads
        self.num_freqs = config.num_freqs
        self.device = device

        F = config.num_freqs
        KVH = config.num_kv_heads
        BS = config.block_size

        # Allocate storage tensors
        self.k_indices = torch.zeros(
            max_blocks, BS, KVH, F, dtype=torch.uint8, device=device
        )
        self.k_packed = torch.zeros(
            max_blocks, BS, KVH, F, dtype=torch.uint8, device=device
        )
        self.k_scales = torch.zeros(
            max_blocks, BS, KVH, dtype=torch.float16, device=device
        )
        self.v_indices = torch.zeros(
            max_blocks, BS, KVH, F, dtype=torch.uint8, device=device
        )
        self.v_packed = torch.zeros(
            max_blocks, BS, KVH, F, dtype=torch.uint8, device=device
        )
        self.v_scales = torch.zeros(
            max_blocks, BS, KVH, dtype=torch.float16, device=device
        )

        # Block metadata
        self._block_fill = torch.zeros(
            max_blocks, dtype=torch.int32, device=device
        )

    @property
    def bytes_allocated(self) -> int:
        """Total bytes allocated for the cache."""
        per_block = (
            self.k_indices.element_size() * self.k_indices[0].numel()
            + self.k_packed.element_size() * self.k_packed[0].numel()
            + self.k_scales.element_size() * self.k_scales[0].numel()
        ) * 2  # K and V
        return per_block * self.max_blocks

    @property
    def bytes_per_token(self) -> int:
        """Bytes per K+V token pair."""
        return self.config.bytes_per_kv_token * 2

    def store_token(
        self,
        block_idx: int,
        slot_idx: int,
        k_enc: SpectralEncoding,
        v_enc: SpectralEncoding,
    ):
        """Store a single token's spectral K/V into a cache block.

        Args:
            block_idx: Physical block index.
            slot_idx: Position within the block (0 to block_size-1).
            k_enc: Encoded key with shapes [num_kv_heads, F] etc.
            v_enc: Encoded value with same shapes.
        """
        self.k_indices[block_idx, slot_idx] = k_enc.indices
        self.k_packed[block_idx, slot_idx] = k_enc.packed
        self.k_scales[block_idx, slot_idx] = k_enc.scales

        self.v_indices[block_idx, slot_idx] = v_enc.indices
        self.v_packed[block_idx, slot_idx] = v_enc.packed
        self.v_scales[block_idx, slot_idx] = v_enc.scales

        self._block_fill[block_idx] = max(
            self._block_fill[block_idx].item(), slot_idx + 1
        )

    def store_block(
        self,
        block_idx: int,
        k_enc: SpectralEncoding,
        v_enc: SpectralEncoding,
        num_tokens: Optional[int] = None,
    ):
        """Store a full block of tokens.

        Args:
            block_idx: Physical block index.
            k_enc: Encoded keys with shapes [num_tokens, num_kv_heads, F] etc.
            v_enc: Encoded values with same shapes.
            num_tokens: Actual number of valid tokens (<= block_size).
        """
        if num_tokens is None:
            num_tokens = k_enc.indices.shape[0]
        n = min(num_tokens, self.block_size)

        self.k_indices[block_idx, :n] = k_enc.indices[:n]
        self.k_packed[block_idx, :n] = k_enc.packed[:n]
        self.k_scales[block_idx, :n] = k_enc.scales[:n]

        self.v_indices[block_idx, :n] = v_enc.indices[:n]
        self.v_packed[block_idx, :n] = v_enc.packed[:n]
        self.v_scales[block_idx, :n] = v_enc.scales[:n]

        self._block_fill[block_idx] = n

    def fetch_sequence(
        self,
        block_table: torch.Tensor,
        context_len: int,
    ) -> Tuple[SpectralEncoding, SpectralEncoding]:
        """Fetch all K/V tokens for a sequence using its block table.

        Args:
            block_table: [max_blocks_per_seq] physical block indices.
            context_len: Number of valid tokens.

        Returns:
            (k_enc, v_enc) with shapes [context_len, num_kv_heads, F] etc.
        """
        num_blocks = math.ceil(context_len / self.block_size)
        phys_blocks = block_table[:num_blocks].long()

        # Gather from physical blocks
        all_k_idx = self.k_indices[phys_blocks]  # [num_blocks, BS, KVH, F]
        all_k_pak = self.k_packed[phys_blocks]
        all_k_scl = self.k_scales[phys_blocks]
        all_v_idx = self.v_indices[phys_blocks]
        all_v_pak = self.v_packed[phys_blocks]
        all_v_scl = self.v_scales[phys_blocks]

        # Flatten blocks: [num_blocks * BS, KVH, F] then trim
        all_k_idx = all_k_idx.reshape(-1, *all_k_idx.shape[2:])[:context_len]
        all_k_pak = all_k_pak.reshape(-1, *all_k_pak.shape[2:])[:context_len]
        all_k_scl = all_k_scl.reshape(-1, *all_k_scl.shape[2:])[:context_len]
        all_v_idx = all_v_idx.reshape(-1, *all_v_idx.shape[2:])[:context_len]
        all_v_pak = all_v_pak.reshape(-1, *all_v_pak.shape[2:])[:context_len]
        all_v_scl = all_v_scl.reshape(-1, *all_v_scl.shape[2:])[:context_len]

        k_enc = SpectralEncoding(all_k_idx, all_k_pak, all_k_scl)
        v_enc = SpectralEncoding(all_v_idx, all_v_pak, all_v_scl)
        return k_enc, v_enc

    def copy_block(self, src_block: int, dst_block: int):
        """Copy a physical block (for copy-on-write / fork)."""
        self.k_indices[dst_block] = self.k_indices[src_block]
        self.k_packed[dst_block] = self.k_packed[src_block]
        self.k_scales[dst_block] = self.k_scales[src_block]
        self.v_indices[dst_block] = self.v_indices[src_block]
        self.v_packed[dst_block] = self.v_packed[src_block]
        self.v_scales[dst_block] = self.v_scales[src_block]
        self._block_fill[dst_block] = self._block_fill[src_block]

    def clear_block(self, block_idx: int):
        """Zero out a physical block."""
        self.k_indices[block_idx].zero_()
        self.k_packed[block_idx].zero_()
        self.k_scales[block_idx].zero_()
        self.v_indices[block_idx].zero_()
        self.v_packed[block_idx].zero_()
        self.v_scales[block_idx].zero_()
        self._block_fill[block_idx] = 0

    def memory_stats(self) -> dict:
        """Cache memory statistics."""
        used_blocks = (self._block_fill > 0).sum().item()
        total_tokens = self._block_fill.sum().item()
        return {
            "max_blocks": self.max_blocks,
            "used_blocks": used_blocks,
            "total_tokens": total_tokens,
            "bytes_allocated": self.bytes_allocated,
            "bytes_per_token": self.bytes_per_token,
            "compression_ratio": self.config.kv_compression_ratio,
            "equivalent_fp16_bytes": total_tokens * self.config.head_dim * 2 * 2,
        }
