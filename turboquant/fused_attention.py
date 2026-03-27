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
# TRITON FUSED KERNEL
# ============================================================================

if HAS_TRITON:

    @triton.jit
    def _tq_paged_attn_decode_kernel(
        output_ptr,
        query_ptr,
        k_packed_ptr,
        k_norms_ptr,
        v_packed_ptr,
        v_norms_ptr,
        block_tables_ptr,
        context_lens_ptr,
        rotation_ptr,
        centroids_ptr,
        scale,
        stride_q_seq,
        stride_q_head,
        stride_kp_block,
        stride_kp_pos,
        stride_kp_head,
        stride_kn_block,
        stride_kn_pos,
        stride_bt_seq,
        stride_o_seq,
        stride_o_head,
        stride_rot_row,
        gqa_ratio,
        max_blocks_per_seq,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HALF_DIM: tl.constexpr,
        PACKED_DIM: tl.constexpr,
        NUM_LEVELS: tl.constexpr,
    ):
        # Grid: (num_kv_heads, num_seqs)
        kv_head = tl.program_id(0)
        seq_idx = tl.program_id(1)

        ctx_len = tl.load(context_lens_ptr + seq_idx)
        if ctx_len == 0:
            return

        # -- Load query and rotate --
        qh = kv_head * gqa_ratio
        q_offs = tl.arange(0, HEAD_DIM)
        q = tl.load(query_ptr + seq_idx * stride_q_seq + qh * stride_q_head + q_offs).to(tl.float32)

        # q_rot = Pi @ q  (matrix-vector multiply)
        q_rot = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for row in tl.static_range(HEAD_DIM):
            rot_row = tl.load(rotation_ptr + row * stride_rot_row + tl.arange(0, HEAD_DIM))
            q_rot_val = tl.sum(rot_row.to(tl.float32) * q)
            q_rot += tl.where(q_offs == row, q_rot_val, 0.0)

        even_mask = (q_offs % 2) == 0
        odd_mask = (q_offs % 2) == 1

        # -- Load codebook --
        c_offs = tl.arange(0, NUM_LEVELS)
        centroids = tl.load(centroids_ptr + c_offs)

        # -- Online softmax loop over blocks --
        m_i = tl.full([], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([], dtype=tl.float32)
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        num_blocks = tl.cdiv(ctx_len, BLOCK_SIZE)

        for blk in range(num_blocks):
            phys = tl.load(block_tables_ptr + seq_idx * stride_bt_seq + blk)

            for pos in tl.static_range(BLOCK_SIZE):
                token_idx = blk * BLOCK_SIZE + pos
                if token_idx >= ctx_len:
                    break

                # Load K packed: [PACKED_DIM] uint8
                kp_base = phys * stride_kp_block + pos * stride_kp_pos + kv_head * stride_kp_head
                kp = tl.load(k_packed_ptr + kp_base + tl.arange(0, PACKED_DIM)).to(tl.int32)

                # Unpack 4-bit -> indices -> codebook lookup
                even_idx = (kp >> 4) & 0xF
                odd_idx = kp & 0xF
                k_even = tl.load(centroids_ptr + even_idx)
                k_odd = tl.load(centroids_ptr + odd_idx)

                # Dot product score
                k_norm = tl.load(k_norms_ptr + phys * stride_kn_block + pos * stride_kn_pos + kv_head)
                score_even = tl.sum(k_even * q_rot[0::2])
                score_odd = tl.sum(k_odd * q_rot[1::2])
                score = (score_even + score_odd) * k_norm * scale

                # Online softmax
                m_new = tl.maximum(m_i, score)
                alpha = tl.exp(m_i - m_new)
                beta = tl.exp(score - m_new)
                l_i = alpha * l_i + beta
                acc = alpha * acc

                # V accumulation
                vp_base = phys * stride_kp_block + pos * stride_kp_pos + kv_head * stride_kp_head
                vp = tl.load(v_packed_ptr + vp_base + tl.arange(0, PACKED_DIM)).to(tl.int32)
                v_even_idx = (vp >> 4) & 0xF
                v_odd_idx = vp & 0xF
                v_even = tl.load(centroids_ptr + v_even_idx)
                v_odd = tl.load(centroids_ptr + v_odd_idx)
                v_norm = tl.load(v_norms_ptr + phys * stride_kn_block + pos * stride_kn_pos + kv_head)

                bvn = beta * v_norm
                acc = acc + tl.where(even_mask, bvn * v_even, bvn * v_odd)

                m_i = m_new

        # -- Normalize --
        acc = acc / l_i

        # -- Inverse rotation: output = Pi^T @ acc --
        out = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for col in tl.static_range(HEAD_DIM):
            rot_col = tl.load(rotation_ptr + tl.arange(0, HEAD_DIM) * stride_rot_row + col)
            out_val = tl.sum(rot_col.to(tl.float32) * acc)
            out += tl.where(q_offs == col, out_val, 0.0)

        # -- Store output --
        o_base = seq_idx * stride_o_seq + qh * stride_o_head
        tl.store(output_ptr + o_base + q_offs, out.to(tl.float16))


# ============================================================================
# PUBLIC API
# ============================================================================

class TQPagedAttention:
    """
    Fused paged attention that reads directly from TQ-compressed KV cache.
    No decompression buffer. Uses rotated-domain dot products.

    Usage:
        attn = TQPagedAttention(tq_quantizer)
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

        # Use reference implementation (correct, works on CPU and GPU)
        self._ref.num_kv_heads = num_kv_heads
        self._ref.gqa_ratio = self.num_query_heads // num_kv_heads
        self._ref.block_size = block_size
        return self._ref.forward(
            query, k_packed, k_norms, v_packed, v_norms,
            block_tables, context_lens,
        )
