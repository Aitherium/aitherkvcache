"""
TQ Block Selector — semantic-aware block selection for fused attention.

Scores KV cache blocks by relevance to the current query and selects
top-k blocks for attention, reducing kernel iterations from O(all_blocks)
to O(selected_blocks). Inspired by DynSplit-KV (arXiv:2602.03184).

The fused Triton kernel is unchanged — this module only filters the
block_tables passed to it. At 280K tokens (17,500 blocks), selecting
10% reduces kernel work by ~10x.

Integration:
    selector = BlockSelector(max_blocks, num_kv_heads, half_d, device,
                             centroids, select_ratio=0.1)
    # During encode:
    selector.update_reps(block_indices, positions, k_packed, k_norms)
    # During decode (inside _triton_forward):
    filtered_bt, filtered_cl = selector.select(
        q_even, q_odd, block_tables, context_lens, gqa_ratio, block_size)
"""

import math
import os
import torch
from typing import Optional, Tuple


class BlockRepresentativeCache:
    """Per-block key representative vectors for fast block scoring.

    Stores the norm-weighted mean of rotated-domain key vectors per block
    per KV-head, in even/odd split format (matching the fused kernel layout).
    Computed entirely from compressed data — no decompression needed.

    Memory: 2 * max_blocks * num_kv_heads * half_d * 4 bytes.
    At 17,500 blocks, 8 heads, 64 half_d: ~71.7 MB (0.2% of 32 GB VRAM).
    """

    def __init__(
        self,
        max_blocks: int,
        num_kv_heads: int,
        half_d: int,
        device: torch.device,
        centroids: torch.Tensor,
    ):
        self.max_blocks = max_blocks
        self.num_kv_heads = num_kv_heads
        self.half_d = half_d
        self.centroids = centroids  # [16] for 4-bit

        # Accumulated (not yet divided by count)
        self._acc_even = torch.zeros(
            max_blocks, num_kv_heads, half_d,
            dtype=torch.float32, device=device)
        self._acc_odd = torch.zeros_like(self._acc_even)
        self._counts = torch.zeros(
            max_blocks, num_kv_heads,
            dtype=torch.int16, device=device)

        # Finalized representatives (divided by count)
        self.rep_even = torch.zeros_like(self._acc_even)
        self.rep_odd = torch.zeros_like(self._acc_odd)

    def update(
        self,
        block_indices: torch.Tensor,   # [N] int64 — physical block indices
        positions: torch.Tensor,       # [N] int64 — positions within block
        k_packed: torch.Tensor,        # [N, num_kv_heads, packed_dim] uint8
        k_norms: torch.Tensor,         # [N, num_kv_heads] float32
    ):
        """Incrementally update representatives for newly written tokens.

        Uses the already-computed packed indices and norms from _tq_write_primary.
        Unpacks nibbles via codebook lookup to get rotated-domain key vectors.
        """
        N, H, PD = k_packed.shape
        centroids = self.centroids

        # Unpack 4-bit nibbles -> codebook gather -> even/odd key vectors
        kp = k_packed.to(torch.int32)          # [N, H, PD]
        k_even = centroids[(kp >> 4) & 0xF]    # [N, H, PD] float32
        k_odd = centroids[kp & 0xF]             # [N, H, PD] float32

        # Weight by norms: [N, H, 1] * [N, H, PD]
        kn = k_norms.unsqueeze(-1)              # [N, H, 1]
        weighted_even = kn * k_even              # [N, H, PD]
        weighted_odd = kn * k_odd

        # Scatter-add into accumulators
        bi = block_indices                       # [N]
        # Expand block_indices for scatter: [N, H, PD]
        bi_exp = bi.view(N, 1, 1).expand(N, H, PD)

        self._acc_even.scatter_add_(0, bi_exp, weighted_even)
        self._acc_odd.scatter_add_(0, bi_exp, weighted_odd)

        # Update counts: [N, H] -> scatter_add into [max_blocks, H]
        bi_h = bi.view(N, 1).expand(N, H)
        ones = torch.ones(N, H, dtype=torch.int16, device=k_packed.device)
        self._counts.scatter_add_(0, bi_h, ones)

    def finalize(self, block_indices: Optional[torch.Tensor] = None):
        """Divide accumulated reps by count to produce final representatives.

        Call periodically (e.g., after prefill, or when blocks are full).
        If block_indices is None, finalizes all blocks with count > 0.
        """
        if block_indices is not None:
            mask = self._counts[block_indices] > 0
            safe_count = self._counts[block_indices].clamp(min=1).unsqueeze(-1).float()
            self.rep_even[block_indices] = self._acc_even[block_indices] / safe_count
            self.rep_odd[block_indices] = self._acc_odd[block_indices] / safe_count
        else:
            safe_count = self._counts.clamp(min=1).unsqueeze(-1).float()
            self.rep_even.copy_(self._acc_even / safe_count)
            self.rep_odd.copy_(self._acc_odd / safe_count)

    def clear_blocks(self, block_indices: torch.Tensor):
        """Zero out reps and counts for freed blocks."""
        self._acc_even[block_indices] = 0
        self._acc_odd[block_indices] = 0
        self._counts[block_indices] = 0
        self.rep_even[block_indices] = 0
        self.rep_odd[block_indices] = 0


def score_blocks(
    q_even: torch.Tensor,       # [num_seqs, num_q_heads, HALF_D]
    q_odd: torch.Tensor,        # same
    rep_even: torch.Tensor,     # [max_blocks, num_kv_heads, HALF_D]
    rep_odd: torch.Tensor,      # same
    block_tables: torch.Tensor, # [num_seqs, max_blocks_per_seq]
    context_lens: torch.Tensor, # [num_seqs]
    gqa_ratio: int,
    block_size: int,
) -> torch.Tensor:
    """Score each block's relevance to the current query.

    Returns [num_seqs, max_blocks_per_seq] float32 scores.

    Uses GQA-reduced mean query (num_q_heads -> num_kv_heads) for efficiency.
    """
    num_seqs, num_q_heads, half_d = q_even.shape
    num_kv_heads = num_q_heads // gqa_ratio
    max_bps = block_tables.shape[1]

    # GQA reduction: average query heads within each group
    # [S, Q, HD] -> [S, KV, gqa, HD] -> [S, KV, HD]
    q_e = q_even.reshape(num_seqs, num_kv_heads, gqa_ratio, half_d).mean(dim=2)
    q_o = q_odd.reshape(num_seqs, num_kv_heads, gqa_ratio, half_d).mean(dim=2)

    # Gather representatives for blocks in block_tables
    # block_tables: [S, B] -> clamp for safe gather
    bt_flat = block_tables.reshape(-1).clamp(min=0)
    # [S*B, KV, HD]
    gathered_even = rep_even[bt_flat].reshape(num_seqs, max_bps, num_kv_heads, half_d)
    gathered_odd = rep_odd[bt_flat].reshape(num_seqs, max_bps, num_kv_heads, half_d)

    # Score: sum over KV heads and half_d
    # q_e: [S, 1, KV, HD] * gathered: [S, B, KV, HD] -> sum over KV, HD -> [S, B]
    scores = (
        (q_e.unsqueeze(1) * gathered_even).sum(dim=(-1, -2)) +
        (q_o.unsqueeze(1) * gathered_odd).sum(dim=(-1, -2))
    )

    # Mask blocks beyond context length
    num_blocks_per_seq = (context_lens + block_size - 1) // block_size  # [S]
    block_indices = torch.arange(max_bps, device=scores.device).unsqueeze(0)  # [1, B]
    invalid = block_indices >= num_blocks_per_seq.unsqueeze(1)  # [S, B]
    scores.masked_fill_(invalid, float("-inf"))

    return scores


def select_blocks(
    scores: torch.Tensor,           # [num_seqs, max_blocks_per_seq]
    block_tables: torch.Tensor,     # [num_seqs, max_blocks_per_seq]
    context_lens: torch.Tensor,     # [num_seqs]
    block_size: int,
    max_selected: int,
    num_sink_blocks: int = 2,
    num_recent_blocks: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select top-k blocks and return compacted (filtered_block_tables, filtered_context_lens).

    Output shape is always [num_seqs, max_selected] — CUDA graph safe.
    Unselected slots are filled with block 0 and masked by adjusted context_lens.
    """
    num_seqs, max_bps = scores.shape
    device = scores.device

    num_blocks_per_seq = (context_lens + block_size - 1) // block_size  # [S]
    block_idx = torch.arange(max_bps, device=device).unsqueeze(0)  # [1, B]

    # Force-include sink blocks (first N) and recent blocks (last N)
    scores_sel = scores.clone()
    sink_mask = block_idx < num_sink_blocks
    recent_mask = block_idx >= (num_blocks_per_seq.unsqueeze(1) - num_recent_blocks)
    valid_mask = block_idx < num_blocks_per_seq.unsqueeze(1)
    mandatory = (sink_mask | recent_mask) & valid_mask
    scores_sel[mandatory] = float("inf")
    scores_sel[~valid_mask] = float("-inf")

    # Select top-k (k = max_selected, fixed shape)
    k = min(max_selected, max_bps)
    _, topk_idx = scores_sel.topk(k, dim=1, sorted=False)  # [S, k]

    # Sort selected indices to preserve block order (important for kernel's
    # sequential online softmax — blocks should be in logical order)
    topk_idx, _ = topk_idx.sort(dim=1)

    # Gather selected physical blocks
    filtered_bt = block_tables.gather(1, topk_idx)  # [S, k]

    # Pad to max_selected if k < max_selected
    if k < max_selected:
        pad = torch.zeros(num_seqs, max_selected - k, dtype=filtered_bt.dtype, device=device)
        filtered_bt = torch.cat([filtered_bt, pad], dim=1)

    # Adjusted context lens: num selected blocks * block_size, capped by original
    num_selected = k  # fixed, same for all seqs
    filtered_cl = torch.minimum(
        context_lens,
        torch.full_like(context_lens, num_selected * block_size),
    )

    return filtered_bt, filtered_cl


class BlockSelector:
    """Orchestrates block scoring and selection for TQ fused attention.

    When select_ratio < 1.0, scores blocks and passes only the most relevant
    ones to the fused kernel. When select_ratio >= 1.0 (default), does nothing.
    """

    def __init__(
        self,
        max_blocks: int,
        num_kv_heads: int,
        half_d: int,
        device: torch.device,
        centroids: torch.Tensor,
        select_ratio: float = 1.0,
        num_sink_blocks: int = 2,
        num_recent_blocks: int = 2,
        min_blocks_for_selection: int = 64,
    ):
        self.select_ratio = select_ratio
        self.num_sink_blocks = num_sink_blocks
        self.num_recent_blocks = num_recent_blocks
        self.min_blocks = min_blocks_for_selection
        self.enabled = select_ratio < 1.0

        if self.enabled:
            self._rep_cache = BlockRepresentativeCache(
                max_blocks, num_kv_heads, half_d, device, centroids)
            # Pre-compute max_selected for CUDA graph compatibility
            self._max_selected = max(
                num_sink_blocks + num_recent_blocks + 1,
                math.ceil(max_blocks * select_ratio),
            )
        else:
            self._rep_cache = None
            self._max_selected = 0

        # Pre-allocated output buffers (set on first select call)
        self._filtered_bt_buf = None
        self._filtered_cl_buf = None

    @property
    def max_selected(self) -> int:
        return self._max_selected

    def update_reps(
        self,
        block_indices: torch.Tensor,
        positions: torch.Tensor,
        k_packed: torch.Tensor,
        k_norms: torch.Tensor,
    ):
        """Update block representatives after encoding new tokens."""
        if self._rep_cache is not None:
            self._rep_cache.update(block_indices, positions, k_packed, k_norms)
            # Finalize blocks that just got updated (cheap for decode: 1 block)
            unique_blocks = block_indices.unique()
            self._rep_cache.finalize(unique_blocks)

    def clear_blocks(self, block_indices: torch.Tensor):
        """Clear representatives for freed/evicted blocks."""
        if self._rep_cache is not None:
            self._rep_cache.clear_blocks(block_indices)

    def select(
        self,
        q_even: torch.Tensor,          # [num_seqs, num_q_heads, HALF_D]
        q_odd: torch.Tensor,
        block_tables: torch.Tensor,     # [num_seqs, max_blocks_per_seq]
        context_lens: torch.Tensor,     # [num_seqs]
        gqa_ratio: int,
        block_size: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score blocks and return filtered (block_tables, context_lens).

        Returns original inputs unchanged if selection is disabled or
        context is too short to benefit.
        """
        if not self.enabled:
            return block_tables, context_lens

        max_bps = block_tables.shape[1]
        if max_bps < self.min_blocks:
            return block_tables, context_lens

        # Score all blocks
        scores = score_blocks(
            q_even, q_odd,
            self._rep_cache.rep_even, self._rep_cache.rep_odd,
            block_tables, context_lens, gqa_ratio, block_size,
        )

        # Select top-k
        filtered_bt, filtered_cl = select_blocks(
            scores, block_tables, context_lens, block_size,
            max_selected=self._max_selected,
            num_sink_blocks=self.num_sink_blocks,
            num_recent_blocks=self.num_recent_blocks,
        )

        return filtered_bt, filtered_cl
