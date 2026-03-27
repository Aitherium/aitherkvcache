"""
Fused TQ Paged Attention -- reads directly from TurboQuant-compressed KV cache.

Key optimization: compute attention in the ROTATED domain.
  - Rotate query once:  q_rot = Pi @ q
  - Score:              score_i = norm_K_i * (q_rot^T @ y_hat_K_i) / sqrt(d)
  - Accumulate V:       acc += w_i * norm_V_i * y_hat_V_i    (rotated domain)
  - Rotate back once:   output = Pi^T @ (acc / sum_w)

No decompression buffer needed. Reads packed uint8 + float32 norms directly.

4-bit unpacking optimization: split dot product into even/odd halves to avoid
interleaving after nibble extraction:
  score = q_rot_even @ codebook[high_nibble] + q_rot_odd @ codebook[low_nibble]

Architecture (Triton path):
  Pre-processing (PyTorch):
    1. q_rot = query @ rotation.T            (one matmul per batch)
    2. q_even = q_rot[..., 0::2]             (stride slice)
    3. q_odd  = q_rot[..., 1::2]

  Kernel (Triton):
    4. For each (seq, head): online softmax over all KV tokens
       - Unpack K: uint8 nibbles -> codebook gather -> k_even, k_odd
       - Score: k_norm * (dot(q_even, k_even) + dot(q_odd, k_odd)) * scale
       - Unpack V, accumulate in even/odd accumulators

  Post-processing (PyTorch):
    5. Interleave acc_even, acc_odd -> out_rot
    6. output = out_rot @ rotation            (inverse rotation)

This completely avoids Triton's register-indexing limitations -- no computed
indices, no stride slicing, no in-kernel matmul. All arrays are [HALF_D].
"""

import math
import torch
from typing import Optional

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ============================================================================
# PYTORCH REFERENCE IMPLEMENTATION (CPU + GPU, no Triton)
# ============================================================================

class TQPagedAttentionRef:
    """
    Reference implementation of fused TQ paged attention in pure PyTorch.
    Used for testing and as CPU fallback.
    """

    def __init__(
        self,
        rotation: torch.Tensor,
        centroids: torch.Tensor,
        scale: float,
        num_kv_heads: int,
        num_query_heads: int,
        head_dim: int,
        block_size: int,
        bits: int = 4,
    ):
        self.rotation = rotation          # [head_dim, head_dim]
        self.centroids = centroids        # [num_levels]
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.num_query_heads = num_query_heads
        self.gqa_ratio = num_query_heads // num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.bits = bits

        # Import unpack functions for all bit widths
        from .packing import unpack_4bit, unpack_3bit, unpack_2bit
        self._unpack_fns = {4: unpack_4bit, 3: unpack_3bit, 2: unpack_2bit}

    def _unpack_to_full(self, packed_vec: torch.Tensor) -> torch.Tensor:
        """Unpack a single packed vector to full head_dim via codebook lookup."""
        # packed_vec: [packed_dim] uint8
        unpack_fn = self._unpack_fns.get(self.bits, self._unpack_fns[4])
        indices = unpack_fn(packed_vec.unsqueeze(0), self.head_dim).squeeze(0)
        return self.centroids[indices.long()]

    def forward(
        self,
        query: torch.Tensor,            # [num_seqs, num_query_heads, head_dim]
        k_packed: torch.Tensor,          # [num_blocks, block_size, num_kv_heads, packed_dim]
        k_norms: torch.Tensor,           # [num_blocks, block_size, num_kv_heads]
        v_packed: torch.Tensor,
        v_norms: torch.Tensor,
        block_tables: torch.Tensor,      # [num_seqs, max_blocks_per_seq]
        context_lens: torch.Tensor,      # [num_seqs]
    ) -> torch.Tensor:
        """Run fused TQ attention. Returns [num_seqs, num_query_heads, head_dim]."""
        num_seqs = query.shape[0]
        device = query.device
        dtype = query.dtype

        output = torch.zeros_like(query)

        for seq_idx in range(num_seqs):
            ctx_len = context_lens[seq_idx].item()
            if ctx_len == 0:
                continue

            num_blocks = math.ceil(ctx_len / self.block_size)

            for qh in range(self.num_query_heads):
                kv_head = qh // self.gqa_ratio

                # Rotate query once: q_rot = Pi @ q
                q = query[seq_idx, qh].float()                  # [D]
                q_rot = self.rotation @ q                        # [D]

                # Split into even/odd for unpacking optimization
                q_rot_even = q_rot[0::2]                         # [D/2]
                q_rot_odd = q_rot[1::2]                          # [D/2]

                # Online softmax accumulators
                m = torch.tensor(float("-inf"), device=device)
                l = torch.tensor(0.0, device=device)
                acc_even = torch.zeros_like(q_rot_even)
                acc_odd = torch.zeros_like(q_rot_odd)

                for blk_idx in range(num_blocks):
                    phys_block = block_tables[seq_idx, blk_idx].item()
                    start = blk_idx * self.block_size
                    end = min(start + self.block_size, ctx_len)
                    n_valid = end - start

                    for pos in range(n_valid):
                        # Unpack K -> full rotated vector via codebook
                        packed_k = k_packed[phys_block, pos, kv_head]
                        k_full = self._unpack_to_full(packed_k)
                        k_even = k_full[0::2]
                        k_odd = k_full[1::2]

                        # Score in rotated domain
                        k_norm = k_norms[phys_block, pos, kv_head].float()
                        score = (
                            torch.dot(q_rot_even, k_even)
                            + torch.dot(q_rot_odd, k_odd)
                        ) * k_norm * self.scale

                        # Online softmax update
                        m_new = torch.maximum(m, score)
                        alpha = torch.exp(m - m_new)
                        beta = torch.exp(score - m_new)
                        l = alpha * l + beta
                        acc_even = alpha * acc_even
                        acc_odd = alpha * acc_odd
                        m = m_new

                        # Unpack V and accumulate in rotated domain
                        packed_v = v_packed[phys_block, pos, kv_head]
                        v_full = self._unpack_to_full(packed_v)
                        v_even = v_full[0::2]
                        v_odd = v_full[1::2]
                        v_norm = v_norms[phys_block, pos, kv_head].float()

                        acc_even = acc_even + beta * v_norm * v_even
                        acc_odd = acc_odd + beta * v_norm * v_odd

                # Normalize
                if l > 0:
                    acc_even = acc_even / l
                    acc_odd = acc_odd / l

                # Interleave even/odd back to full head_dim
                out_rot = torch.zeros(self.head_dim, device=device, dtype=torch.float32)
                out_rot[0::2] = acc_even
                out_rot[1::2] = acc_odd

                # Rotate back: output = Pi^T @ out_rot
                out = self.rotation.T @ out_rot
                output[seq_idx, qh] = out.to(dtype)

        return output


# ============================================================================
# TRITON FUSED KERNEL (GPU, 4-bit)
# ============================================================================
#
# The key insight that makes this work: rotation and even/odd splitting happen
# OUTSIDE the kernel in PyTorch. The Triton kernel receives pre-split q_even
# and q_odd as separate [HALF_D] arrays, operates entirely on [HALF_D]-shaped
# registers, and outputs acc_even and acc_odd. No computed indexing, no stride
# slicing, no in-kernel matmul.
#
# Pre:  q_rot = query @ rotation.T; q_even = q_rot[..., 0::2]; q_odd = q_rot[..., 1::2]
# Post: out_rot[..., 0::2] = acc_even; out_rot[..., 1::2] = acc_odd; out = out_rot @ rotation
# ============================================================================

if HAS_TRITON:

    @triton.jit
    def _tq_paged_attn_fused_4bit(
        # Pre-rotated, pre-split query
        q_even_ptr,           # [num_seqs, num_q_heads, HALF_D] float32
        q_odd_ptr,            # [num_seqs, num_q_heads, HALF_D] float32
        # Compressed KV cache (same layout for K and V)
        k_packed_ptr,         # [max_phys_blocks, BLOCK_SIZE, num_kv_heads, HALF_D] uint8
        k_norms_ptr,          # [max_phys_blocks, BLOCK_SIZE, num_kv_heads] float32
        v_packed_ptr,         # same layout as k_packed
        v_norms_ptr,          # same layout as k_norms
        # Paged block indirection
        block_tables_ptr,     # [num_seqs, max_blocks_per_seq] int32/int64
        context_lens_ptr,     # [num_seqs] int32/int64
        # Codebook
        centroids_ptr,        # [16] float32
        # Output accumulators (rotated domain, even/odd split)
        out_even_ptr,         # [num_seqs, num_q_heads, HALF_D] float32
        out_odd_ptr,          # [num_seqs, num_q_heads, HALF_D] float32
        # Strides for q_even / q_odd
        stride_qe_seq,
        stride_qe_head,
        # Strides for packed KV: [blocks, positions, heads, packed_dim]
        stride_kp_block,
        stride_kp_pos,
        stride_kp_head,
        # Strides for norms: [blocks, positions, heads]
        stride_kn_block,
        stride_kn_pos,
        # Stride for block_tables: [seqs, blocks]
        stride_bt_seq,
        # Strides for output even/odd
        stride_oe_seq,
        stride_oe_head,
        # Scalars
        scale,                # 1/sqrt(head_dim)
        gqa_ratio,            # num_q_heads // num_kv_heads
        max_blocks_per_seq,   # padded block table width
        # Compile-time constants
        BLOCK_SIZE: tl.constexpr,   # tokens per block (typically 16)
        HALF_D: tl.constexpr,       # head_dim // 2 (= packed_dim for 4-bit)
    ):
        """
        Fused TQ paged attention for 4-bit compressed KV cache.

        Grid: (num_q_heads, num_seqs)
        Each program computes attention for one query head over one sequence's
        entire compressed KV context using online softmax.

        All register arrays are [HALF_D] -- no computed indexing needed.
        """
        qh = tl.program_id(0)
        seq_idx = tl.program_id(1)

        ctx_len = tl.load(context_lens_ptr + seq_idx)
        half_offs = tl.arange(0, HALF_D)

        if ctx_len == 0:
            # Zero output for empty context
            oe_base = seq_idx * stride_oe_seq + qh * stride_oe_head
            tl.store(out_even_ptr + oe_base + half_offs,
                     tl.zeros([HALF_D], dtype=tl.float32))
            tl.store(out_odd_ptr + oe_base + half_offs,
                     tl.zeros([HALF_D], dtype=tl.float32))
            return

        kv_head = qh // gqa_ratio
        num_blocks = tl.cdiv(ctx_len, BLOCK_SIZE)

        # ── Load pre-rotated, pre-split query ─────────────────────
        qe_base = seq_idx * stride_qe_seq + qh * stride_qe_head
        q_even = tl.load(q_even_ptr + qe_base + half_offs)
        q_odd = tl.load(q_odd_ptr + qe_base + half_offs)

        # ── Online softmax state ──────────────────────────────────
        m_i = tl.full([], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([], dtype=tl.float32)
        acc_even = tl.zeros([HALF_D], dtype=tl.float32)
        acc_odd = tl.zeros([HALF_D], dtype=tl.float32)

        # ── Main attention loop ───────────────────────────────────
        # max_blocks_per_seq is uniform across all threads (kernel arg),
        # allowing Triton to generate optimized loop code. The per-thread
        # guard (blk < num_blocks) is a cheap comparison.
        for blk in range(max_blocks_per_seq):
            if blk < num_blocks:
                phys_block = tl.load(
                    block_tables_ptr + seq_idx * stride_bt_seq + blk)

                for pos in tl.static_range(BLOCK_SIZE):
                    token_idx = blk * BLOCK_SIZE + pos
                    if token_idx < ctx_len:
                        # ── Unpack K: nibbles -> codebook gather ──
                        kp_base = (phys_block * stride_kp_block
                                   + pos * stride_kp_pos
                                   + kv_head * stride_kp_head)
                        kp = tl.load(
                            k_packed_ptr + kp_base + half_offs
                        ).to(tl.int32)

                        k_even = tl.load(centroids_ptr + ((kp >> 4) & 0xF))
                        k_odd = tl.load(centroids_ptr + (kp & 0xF))

                        # ── Score ──
                        k_norm = tl.load(
                            k_norms_ptr
                            + phys_block * stride_kn_block
                            + pos * stride_kn_pos
                            + kv_head
                        )
                        score = (k_norm
                                 * (tl.sum(q_even * k_even)
                                    + tl.sum(q_odd * k_odd))
                                 * scale)

                        # ── Online softmax update ──
                        m_new = tl.maximum(m_i, score)
                        alpha = tl.exp(m_i - m_new)
                        beta = tl.exp(score - m_new)
                        l_i = alpha * l_i + beta
                        acc_even = alpha * acc_even
                        acc_odd = alpha * acc_odd
                        m_i = m_new

                        # ── Unpack V + accumulate ──
                        vp_base = (phys_block * stride_kp_block
                                   + pos * stride_kp_pos
                                   + kv_head * stride_kp_head)
                        vp = tl.load(
                            v_packed_ptr + vp_base + half_offs
                        ).to(tl.int32)

                        v_even = tl.load(centroids_ptr + ((vp >> 4) & 0xF))
                        v_odd = tl.load(centroids_ptr + (vp & 0xF))
                        v_norm = tl.load(
                            v_norms_ptr
                            + phys_block * stride_kn_block
                            + pos * stride_kn_pos
                            + kv_head
                        )

                        bvn = beta * v_norm
                        acc_even += bvn * v_even
                        acc_odd += bvn * v_odd

        # ── Normalize ─────────────────────────────────────────────
        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        acc_even = acc_even / safe_l
        acc_odd = acc_odd / safe_l

        # ── Store output (rotated domain, even/odd) ───────────────
        oe_base = seq_idx * stride_oe_seq + qh * stride_oe_head
        tl.store(out_even_ptr + oe_base + half_offs, acc_even)
        tl.store(out_odd_ptr + oe_base + half_offs, acc_odd)

    # ================================================================
    # SPLIT-K KERNELS: parallel block processing + log-sum-exp reduce
    # ================================================================
    # Grid: (num_q_heads, num_seqs, NUM_SPLITS) for phase 1
    #        (num_q_heads, num_seqs) for phase 2 (reduction)
    # Each split processes ceil(num_blocks / NUM_SPLITS) blocks, writes
    # partial (acc_even, acc_odd, m, l) to intermediate storage.
    # Reduction combines via online softmax merge.
    # ================================================================

    @triton.jit
    def _tq_splitk_phase1(
        q_even_ptr, q_odd_ptr,
        k_packed_ptr, k_norms_ptr,
        v_packed_ptr, v_norms_ptr,
        block_tables_ptr, context_lens_ptr,
        centroids_ptr,
        # Partial outputs: [num_q_heads, num_seqs, NUM_SPLITS, HALF_D]
        part_even_ptr, part_odd_ptr,
        # Partial softmax state: [num_q_heads, num_seqs, NUM_SPLITS]
        part_m_ptr, part_l_ptr,
        # Strides
        stride_qe_seq, stride_qe_head,
        stride_kp_block, stride_kp_pos, stride_kp_head,
        stride_kn_block, stride_kn_pos,
        stride_bt_seq,
        stride_pe_head, stride_pe_seq, stride_pe_split,
        stride_pm_head, stride_pm_seq,
        # Scalars
        scale, gqa_ratio, max_blocks_per_seq,
        BLOCK_SIZE: tl.constexpr,
        HALF_D: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        """Phase 1: each split processes a subset of KV blocks."""
        qh = tl.program_id(0)
        seq_idx = tl.program_id(1)
        split_idx = tl.program_id(2)

        ctx_len = tl.load(context_lens_ptr + seq_idx)
        half_offs = tl.arange(0, HALF_D)

        num_blocks = tl.cdiv(ctx_len, BLOCK_SIZE)
        # Block range for this split
        blocks_per_split = tl.cdiv(num_blocks, NUM_SPLITS)
        blk_start = split_idx * blocks_per_split
        blk_end = tl.minimum(blk_start + blocks_per_split, num_blocks)

        kv_head = qh // gqa_ratio

        # Load query
        qe_base = seq_idx * stride_qe_seq + qh * stride_qe_head
        q_even = tl.load(q_even_ptr + qe_base + half_offs)
        q_odd = tl.load(q_odd_ptr + qe_base + half_offs)

        # Online softmax state for this split
        m_i = tl.full([], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([], dtype=tl.float32)
        acc_even = tl.zeros([HALF_D], dtype=tl.float32)
        acc_odd = tl.zeros([HALF_D], dtype=tl.float32)

        for blk in range(max_blocks_per_seq):
            if blk >= blk_start and blk < blk_end:
                phys_block = tl.load(
                    block_tables_ptr + seq_idx * stride_bt_seq + blk)

                for pos in tl.static_range(BLOCK_SIZE):
                    token_idx = blk * BLOCK_SIZE + pos
                    if token_idx < ctx_len:
                        kp_base = (phys_block * stride_kp_block
                                   + pos * stride_kp_pos
                                   + kv_head * stride_kp_head)
                        kp = tl.load(
                            k_packed_ptr + kp_base + half_offs
                        ).to(tl.int32)

                        k_even = tl.load(centroids_ptr + ((kp >> 4) & 0xF))
                        k_odd = tl.load(centroids_ptr + (kp & 0xF))

                        k_norm = tl.load(
                            k_norms_ptr + phys_block * stride_kn_block
                            + pos * stride_kn_pos + kv_head)
                        score = (k_norm
                                 * (tl.sum(q_even * k_even)
                                    + tl.sum(q_odd * k_odd))
                                 * scale)

                        m_new = tl.maximum(m_i, score)
                        alpha = tl.exp(m_i - m_new)
                        beta = tl.exp(score - m_new)
                        l_i = alpha * l_i + beta
                        acc_even = alpha * acc_even
                        acc_odd = alpha * acc_odd
                        m_i = m_new

                        vp_base = (phys_block * stride_kp_block
                                   + pos * stride_kp_pos
                                   + kv_head * stride_kp_head)
                        vp = tl.load(
                            v_packed_ptr + vp_base + half_offs
                        ).to(tl.int32)

                        v_even = tl.load(centroids_ptr + ((vp >> 4) & 0xF))
                        v_odd = tl.load(centroids_ptr + (vp & 0xF))
                        v_norm = tl.load(
                            v_norms_ptr + phys_block * stride_kn_block
                            + pos * stride_kn_pos + kv_head)

                        bvn = beta * v_norm
                        acc_even += bvn * v_even
                        acc_odd += bvn * v_odd

        # Store partial results
        pe_base = qh * stride_pe_head + seq_idx * stride_pe_seq + split_idx * stride_pe_split
        tl.store(part_even_ptr + pe_base + half_offs, acc_even)
        tl.store(part_odd_ptr + pe_base + half_offs, acc_odd)

        pm_base = qh * stride_pm_head + seq_idx * stride_pm_seq + split_idx
        tl.store(part_m_ptr + pm_base, m_i)
        tl.store(part_l_ptr + pm_base, l_i)

    @triton.jit
    def _tq_splitk_reduce(
        part_even_ptr, part_odd_ptr,
        part_m_ptr, part_l_ptr,
        out_even_ptr, out_odd_ptr,
        stride_pe_head, stride_pe_seq, stride_pe_split,
        stride_pm_head, stride_pm_seq,
        stride_oe_seq, stride_oe_head,
        HALF_D: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        """Phase 2: merge partial softmax results across splits."""
        qh = tl.program_id(0)
        seq_idx = tl.program_id(1)

        half_offs = tl.arange(0, HALF_D)

        # Load first split as initial state
        pe_base = qh * stride_pe_head + seq_idx * stride_pe_seq
        pm_base = qh * stride_pm_head + seq_idx * stride_pm_seq

        acc_even = tl.load(part_even_ptr + pe_base + half_offs)
        acc_odd = tl.load(part_odd_ptr + pe_base + half_offs)
        m_acc = tl.load(part_m_ptr + pm_base)
        l_acc = tl.load(part_l_ptr + pm_base)

        # Merge remaining splits via online softmax combination
        for s in tl.static_range(1, NUM_SPLITS):
            s_base = pe_base + s * stride_pe_split
            s_even = tl.load(part_even_ptr + s_base + half_offs)
            s_odd = tl.load(part_odd_ptr + s_base + half_offs)
            m_s = tl.load(part_m_ptr + pm_base + s)
            l_s = tl.load(part_l_ptr + pm_base + s)

            # Log-sum-exp merge
            m_new = tl.maximum(m_acc, m_s)
            alpha = tl.exp(m_acc - m_new)
            beta = tl.exp(m_s - m_new)
            l_new = alpha * l_acc + beta * l_s

            acc_even = alpha * acc_even + beta * s_even
            acc_odd = alpha * acc_odd + beta * s_odd
            m_acc = m_new
            l_acc = l_new

        # Normalize
        safe_l = tl.where(l_acc > 0.0, l_acc, 1.0)
        acc_even = acc_even / safe_l
        acc_odd = acc_odd / safe_l

        oe_base = seq_idx * stride_oe_seq + qh * stride_oe_head
        tl.store(out_even_ptr + oe_base + half_offs, acc_even)
        tl.store(out_odd_ptr + oe_base + half_offs, acc_odd)


# ============================================================================
# PUBLIC API
# ============================================================================

class TQPagedAttention:
    """
    Fused paged attention that reads directly from TQ-compressed KV cache.
    No decompression buffer. Uses rotated-domain dot products.

    Dispatches to:
      - Triton kernel: 4-bit, CUDA, Triton available
      - PyTorch reference: all other cases (CPU, 2/3-bit, no Triton)

    Usage:
        attn = TQPagedAttention(tq_quantizer, num_query_heads=32)
        output = attn.forward(query, k_packed, k_norms, v_packed, v_norms,
                              block_tables, context_lens)
    """

    def __init__(self, tq, num_query_heads: int):
        """
        Args:
            tq: TurboQuant instance (has rotation, centroids, head_dim, etc.)
            num_query_heads: total query heads (for GQA mapping)
        """
        self.rotation = tq.rotation
        self.centroids = tq.centroids
        self.scale = 1.0 / math.sqrt(tq.head_dim)
        self.head_dim = tq.head_dim
        self.num_levels = tq.num_levels
        self.bits = tq.bits
        self.packed_dim = tq._packed_dim
        self.num_query_heads = num_query_heads

        # Reference implementation (always available)
        self._ref = TQPagedAttentionRef(
            rotation=tq.rotation,
            centroids=tq.centroids,
            scale=self.scale,
            num_kv_heads=8,  # will be overridden per call
            num_query_heads=num_query_heads,
            head_dim=tq.head_dim,
            block_size=16,
            bits=tq.bits,
        )

        self._use_triton = HAS_TRITON and torch.cuda.is_available() and tq.bits == 4

    def forward(
        self,
        query: torch.Tensor,
        k_packed: torch.Tensor,
        k_norms: torch.Tensor,
        v_packed: torch.Tensor,
        v_norms: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int = 16,
        num_kv_heads: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Run fused TQ paged attention.

        Returns: [num_seqs, num_query_heads, head_dim] attention output.
        """
        if num_kv_heads is None:
            num_kv_heads = k_packed.shape[2]

        if self._use_triton and query.is_cuda and self.bits == 4:
            return self._triton_forward(
                query, k_packed, k_norms, v_packed, v_norms,
                block_tables, context_lens, block_size, num_kv_heads,
            )

        # Reference path (CPU, non-4-bit, or Triton unavailable)
        self._ref.num_kv_heads = num_kv_heads
        self._ref.gqa_ratio = self.num_query_heads // num_kv_heads
        self._ref.block_size = block_size
        return self._ref.forward(
            query, k_packed, k_norms, v_packed, v_norms,
            block_tables, context_lens,
        )

    # Split-k threshold: use split-k for contexts with more blocks than this.
    # Below this, the single-program kernel has less overhead.
    SPLITK_THRESHOLD = 128  # 128 blocks = 2048 tokens at block_size=16
    SPLITK_NUM_SPLITS = 8  # 8 splits -> 8x more GPU parallelism

    def _triton_forward(
        self,
        query: torch.Tensor,
        k_packed: torch.Tensor,
        k_norms: torch.Tensor,
        v_packed: torch.Tensor,
        v_norms: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        """
        Triton-accelerated fused attention for 4-bit TQ cache.

        Short context (< 512 tokens): single-program kernel
        Long context (>= 512 tokens): split-k with 8-way parallelism
        """
        num_seqs, num_q_heads, head_dim = query.shape
        half_d = head_dim // 2
        gqa_ratio = num_q_heads // num_kv_heads
        max_blocks_per_seq = block_tables.shape[1]

        # ── Step 1: Pre-rotate query, split even/odd ──────────────
        q_rot = torch.matmul(query.float(), self.rotation.T)
        q_even = q_rot[..., 0::2].contiguous()
        q_odd = q_rot[..., 1::2].contiguous()

        out_even = torch.empty_like(q_even)
        out_odd = torch.empty_like(q_odd)

        # ── Step 2: Choose kernel based on context length ─────────
        use_splitk = max_blocks_per_seq >= self.SPLITK_THRESHOLD

        if use_splitk:
            NS = self.SPLITK_NUM_SPLITS
            # Intermediate storage: [num_q_heads, num_seqs, NUM_SPLITS, HALF_D]
            part_even = torch.empty(
                num_q_heads, num_seqs, NS, half_d,
                dtype=torch.float32, device=query.device)
            part_odd = torch.empty_like(part_even)
            # Softmax state: [num_q_heads, num_seqs, NUM_SPLITS]
            part_m = torch.empty(
                num_q_heads, num_seqs, NS,
                dtype=torch.float32, device=query.device)
            part_l = torch.empty_like(part_m)

            # Phase 1: parallel block processing
            _tq_splitk_phase1[(num_q_heads, num_seqs, NS)](
                q_even, q_odd,
                k_packed, k_norms, v_packed, v_norms,
                block_tables, context_lens,
                self.centroids,
                part_even, part_odd, part_m, part_l,
                q_even.stride(0), q_even.stride(1),
                k_packed.stride(0), k_packed.stride(1), k_packed.stride(2),
                k_norms.stride(0), k_norms.stride(1),
                block_tables.stride(0),
                part_even.stride(0), part_even.stride(1), part_even.stride(2),
                part_m.stride(0), part_m.stride(1),
                self.scale, gqa_ratio, max_blocks_per_seq,
                BLOCK_SIZE=block_size, HALF_D=half_d, NUM_SPLITS=NS,
            )

            # Phase 2: reduce partial results
            _tq_splitk_reduce[(num_q_heads, num_seqs)](
                part_even, part_odd, part_m, part_l,
                out_even, out_odd,
                part_even.stride(0), part_even.stride(1), part_even.stride(2),
                part_m.stride(0), part_m.stride(1),
                out_even.stride(0), out_even.stride(1),
                HALF_D=half_d, NUM_SPLITS=NS,
            )
        else:
            # Single-program kernel for short contexts
            _tq_paged_attn_fused_4bit[(num_q_heads, num_seqs)](
                q_even, q_odd,
                k_packed, k_norms, v_packed, v_norms,
                block_tables, context_lens,
                self.centroids,
                out_even, out_odd,
                q_even.stride(0), q_even.stride(1),
                k_packed.stride(0), k_packed.stride(1), k_packed.stride(2),
                k_norms.stride(0), k_norms.stride(1),
                block_tables.stride(0),
                out_even.stride(0), out_even.stride(1),
                self.scale, gqa_ratio, max_blocks_per_seq,
                BLOCK_SIZE=block_size, HALF_D=half_d,
            )

        # ── Step 3: Interleave + inverse rotation ─────────────────
        out_rot = torch.empty(
            num_seqs, num_q_heads, head_dim,
            device=query.device, dtype=torch.float32,
        )
        out_rot[..., 0::2] = out_even
        out_rot[..., 1::2] = out_odd

        output = torch.matmul(out_rot, self.rotation)
        return output.to(query.dtype)
