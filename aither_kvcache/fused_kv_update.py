"""
Fused Triton kernel for TQ4 KV cache update.

Replaces the 3-step encode path (rotate -> quantize+pack -> scatter) with a
single kernel launch. Grid: (num_tokens, num_kv_heads).

Each program reads one raw KV vector, rotates it, quantizes to 4-bit indices,
packs nibble pairs, and writes directly to TQGPUCache storage.

Performance gains:
  - 3+ kernel launches collapsed to 1
  - No intermediate tensor allocation (rotation output, quantized indices)
  - Scatter-write directly to 5D cache storage from within the kernel
  - Rotation matmul tiled in-register (128x128 @ TILE=32 = 4 tiles)

Compatibility:
  - SM_80 (Ampere) through SM_120 (Blackwell excluded by default)
  - Any head_dim that is a power of 2 (64, 128, 256)
  - 4-bit uniform quantization only (16 levels, 15 boundaries)
  - PyTorch fallback when Triton is unavailable
"""

import logging

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

logger = logging.getLogger("aither.turboquant.fused_kv_update")

__all__ = [
    "fused_tq4_kv_update",
    "fused_encode_and_store",
    "HAS_TRITON",
]


# ============================================================================
# TRITON KERNEL
# ============================================================================

if HAS_TRITON:

    @triton.jit
    def _fused_tq4_kv_update_kernel(
        # Input: raw KV vectors
        x_ptr,              # [num_tokens, num_kv_heads, head_dim] float16/bf16/float32
        # Slot mapping
        slot_mapping_ptr,   # [num_tokens] int64
        # Output: TQGPUCache packed storage (one layer slice)
        out_packed_ptr,     # [max_blocks, block_size, num_kv_heads, packed_dim] uint8
        out_norms_ptr,      # [max_blocks, block_size, num_kv_heads] float32
        # Rotation matrix
        rotation_ptr,       # [head_dim, head_dim] float32
        # Quantization boundaries
        boundaries_ptr,     # [15] float32
        # Strides for x: [tokens, heads, dims]
        stride_x_token,
        stride_x_head,
        # Strides for out_packed: [blocks, positions, heads, packed_dim]
        stride_op_block,
        stride_op_pos,
        stride_op_head,
        # Strides for out_norms: [blocks, positions, heads]
        stride_on_block,
        stride_on_pos,
        # Scalar parameters
        block_size,         # vLLM block size (typically 16)
        max_blocks,         # TQ cache capacity in blocks
        num_kv_heads,       # number of KV heads
        # Compile-time constants
        HEAD_DIM: tl.constexpr,    # full head dimension (64, 128, 256)
        HALF_D: tl.constexpr,      # head_dim // 2 = packed_dim for 4-bit
        TILE_R: tl.constexpr,      # rotation tile size (32 or 64)
    ):
        """
        Fused TQ4 encode-and-store for one (token, head) pair.

        Grid: (num_tokens, num_kv_heads)

        Per program:
          1. Read slot_mapping; early-exit if slot < 0 or block out of range
          2. Load raw x[token, head, :] and compute L2 norm
          3. Normalize to unit sphere
          4. Tile-multiply by rotation matrix (in float32)
          5. Searchsorted quantize to 4-bit indices
          6. Pack pairs into uint8 nibbles
          7. Scatter-write packed bytes and norm to cache
        """
        token_id = tl.program_id(0)
        head_id = tl.program_id(1)

        # --- Step 1: slot mapping & bounds check ---
        slot = tl.load(slot_mapping_ptr + token_id)
        if slot < 0:
            return

        block_idx = slot // block_size
        block_pos = slot % block_size

        if block_idx >= max_blocks:
            return

        # --- Step 2: load raw vector x[token, head, :] ---
        x_base = token_id * stride_x_token + head_id * stride_x_head
        dim_offs = tl.arange(0, HEAD_DIM)
        x_vec = tl.load(x_ptr + x_base + dim_offs).to(tl.float32)

        # --- Step 3: compute L2 norm and normalize ---
        norm_sq = tl.sum(x_vec * x_vec)
        norm = tl.sqrt(norm_sq + 1e-20)
        x_vec / norm

        # --- Step 4: tiled rotation y = rotation @ x_unit ---
        # rotation is [HEAD_DIM, HEAD_DIM] stored row-major.
        # y[i] = dot(rotation[i, :], x_unit)
        # We tile over the output dimension in chunks of TILE_R and accumulate
        # the inner product over chunks of input dimension, also TILE_R.
        #
        # Since we need all HEAD_DIM outputs for quantization, we compute them
        # in TILE_R-wide chunks and store into a register array.
        # Because Triton cannot dynamically index a register array, we use a
        # different approach: for each output tile, we accumulate the full dot
        # product and immediately quantize that tile's outputs. This avoids
        # needing to store all HEAD_DIM rotated values simultaneously.
        #
        # However, the even/odd nibble packing requires pairs from indices
        # (2k, 2k+1), which are adjacent in the output. So we process output
        # dimensions in pairs and quantize+pack within each TILE_R chunk.

        # Strategy: process HALF_D pairs. For each pair (2k, 2k+1):
        #   y_even[k] = dot(rotation[2k, :], x_unit)
        #   y_odd[k]  = dot(rotation[2k+1, :], x_unit)
        #   quantize both, pack into one uint8.
        #
        # We batch this: process TILE_R//2 pairs at a time (TILE_R output dims).

        pair_offs = tl.arange(0, HALF_D)  # [0, 1, ..., HALF_D-1]

        # Compute y_even = rotation[2k, :] @ x_unit for all k
        # and y_odd = rotation[2k+1, :] @ x_unit for all k
        # using tiled accumulation over the inner dimension.
        y_even = tl.zeros([HALF_D], dtype=tl.float32)
        y_odd = tl.zeros([HALF_D], dtype=tl.float32)

        # Tile the inner-dimension dot product. For each tile of TILE_R input
        # dims, load the corresponding rotation matrix block and multiply by
        # the x_unit slice. We reload x and normalize per-tile instead of
        # trying to dynamically slice the x_unit register array -- Triton
        # static_range guarantees the tile offset is a compile-time constant
        # within each unrolled iteration, but the loaded x_unit[HEAD_DIM] is a
        # single register block that cannot be dynamically sliced. Reloading
        # from L1/L2 (already cached from step 2) is near-free.
        #
        # rotation[row, col] is at rotation_ptr + row * HEAD_DIM + col
        even_row_offsets = pair_offs * 2        # [HALF_D]
        odd_row_offsets = pair_offs * 2 + 1     # [HALF_D]

        for t in tl.static_range(0, HEAD_DIM, TILE_R):
            tile_offs = tl.arange(0, TILE_R)

            # Reload x slice and normalize (data is in L1 cache from step 2)
            x_tile = tl.load(x_ptr + x_base + t + tile_offs).to(tl.float32) / norm

            # Load rotation submatrices: [HALF_D, TILE_R]
            # even rows: rotation[2k, t:t+TILE_R]
            # odd rows:  rotation[2k+1, t:t+TILE_R]
            rot_even = tl.load(
                rotation_ptr
                + even_row_offsets[:, None] * HEAD_DIM
                + (t + tile_offs)[None, :],
            )  # [HALF_D, TILE_R]

            rot_odd = tl.load(
                rotation_ptr
                + odd_row_offsets[:, None] * HEAD_DIM
                + (t + tile_offs)[None, :],
            )  # [HALF_D, TILE_R]

            # Accumulate: [HALF_D, TILE_R] * [1, TILE_R] -> sum(axis=1) -> [HALF_D]
            y_even += tl.sum(rot_even * x_tile[None, :], axis=1)
            y_odd += tl.sum(rot_odd * x_tile[None, :], axis=1)

        # --- Step 5: searchsorted quantize ---
        # Count how many of the 15 boundaries each value exceeds.
        idx_even = tl.zeros([HALF_D], dtype=tl.int32)
        idx_odd = tl.zeros([HALF_D], dtype=tl.int32)

        for b in tl.static_range(15):
            boundary = tl.load(boundaries_ptr + b)
            idx_even += (y_even >= boundary).to(tl.int32)
            idx_odd += (y_odd >= boundary).to(tl.int32)

        # --- Step 6: pack into uint8 nibbles ---
        packed = ((idx_even & 0xF) << 4) | (idx_odd & 0xF)

        # --- Step 7: scatter-write to cache ---
        # out_packed[block_idx, block_pos, head_id, :]
        op_base = (block_idx * stride_op_block
                   + block_pos * stride_op_pos
                   + head_id * stride_op_head)
        tl.store(out_packed_ptr + op_base + pair_offs, packed.to(tl.uint8))

        # out_norms[block_idx, block_pos, head_id]
        on_offset = (block_idx * stride_on_block
                     + block_pos * stride_on_pos
                     + head_id)
        tl.store(out_norms_ptr + on_offset, norm)


# ============================================================================
# PYTORCH FALLBACK
# ============================================================================

def _pytorch_tq4_kv_update(
    x: torch.Tensor,
    slot_mapping: torch.Tensor,
    out_packed: torch.Tensor,
    out_norms: torch.Tensor,
    rotation: torch.Tensor,
    boundaries: torch.Tensor,
    block_size: int,
    max_blocks: int,
) -> None:
    """
    PyTorch fallback for fused TQ4 KV update.

    Equivalent to the Triton kernel but using standard PyTorch ops. Used when
    Triton is unavailable (CPU, non-CUDA devices, import failure).

    Args:
        x: [num_tokens, num_kv_heads, head_dim] raw KV vectors.
        slot_mapping: [num_tokens] int64 slot indices (-1 = padding).
        out_packed: [max_blocks, block_size, num_kv_heads, packed_dim] uint8 output.
        out_norms: [max_blocks, block_size, num_kv_heads] float32 output.
        rotation: [head_dim, head_dim] float32 rotation matrix.
        boundaries: [15] float32 sorted quantization boundaries.
        block_size: Tokens per block.
        max_blocks: Maximum block count in cache.
    """
    # Filter valid slots
    valid = slot_mapping >= 0
    if not valid.any():
        return

    valid_slots = slot_mapping[valid]
    block_idx = valid_slots // block_size

    # Filter blocks within cache capacity
    in_range = block_idx < max_blocks
    if not in_range.any():
        return

    valid_slots = valid_slots[in_range]
    block_idx = block_idx[in_range]
    block_pos = valid_slots % block_size

    x_valid = x[valid][in_range].contiguous()  # [N, H, D]
    N, H, D = x_valid.shape
    half_d = D // 2

    # Flatten to [N*H, D] for batch processing
    x_flat = x_valid.reshape(N * H, D).float()

    # 1) Compute norms
    norms = x_flat.norm(dim=-1)  # [N*H]

    # 2) Normalize
    x_unit = x_flat / (norms.unsqueeze(-1) + 1e-10)

    # 3) Rotate: y = x_unit @ rotation^T
    y = torch.matmul(x_unit, rotation.T).contiguous()  # [N*H, D]

    # 4) Searchsorted quantize
    indices = torch.searchsorted(boundaries, y)
    indices = indices.clamp(0, 15)

    # 5) Pack 4-bit pairs into uint8
    idx_even = indices[..., 0::2].to(torch.uint8)  # [N*H, D//2]
    idx_odd = indices[..., 1::2].to(torch.uint8)
    packed = (idx_even << 4) | idx_odd  # [N*H, D//2]

    # 6) Reshape and scatter-write
    packed = packed.reshape(N, H, half_d)
    norms = norms.reshape(N, H)

    out_packed[block_idx, block_pos] = packed
    out_norms[block_idx, block_pos] = norms


# ============================================================================
# PUBLIC API: fused_tq4_kv_update
# ============================================================================

def fused_tq4_kv_update(
    x: torch.Tensor,
    slot_mapping: torch.Tensor,
    out_packed: torch.Tensor,
    out_norms: torch.Tensor,
    rotation: torch.Tensor,
    boundaries: torch.Tensor,
    block_size: int = 16,
    max_blocks: int = 0,
) -> None:
    """
    Fused TQ4 quantize-and-store for one cache tensor (K or V).

    Reads raw float16/bf16 vectors, applies rotation, quantizes to 4-bit,
    packs into nibble pairs, and scatter-writes directly to TQGPUCache storage
    -- all in a single kernel launch (Triton path) or equivalent PyTorch ops.

    Args:
        x: [num_tokens, num_kv_heads, head_dim] raw KV vectors (float16/bf16/float32).
        slot_mapping: [num_tokens] int64 slot indices. Negative values indicate
            padding tokens and are skipped.
        out_packed: [max_blocks, block_size, num_kv_heads, packed_dim] uint8
            destination for packed 4-bit indices.
        out_norms: [max_blocks, block_size, num_kv_heads] float32
            destination for L2 norms.
        rotation: [head_dim, head_dim] float32 orthogonal rotation matrix.
        boundaries: [15] float32 sorted quantization boundaries for 16 levels.
        block_size: Tokens per vLLM block (default 16).
        max_blocks: Maximum number of blocks in the TQ cache. If 0, inferred
            from out_packed.shape[0].

    Returns:
        None. Writes directly to out_packed and out_norms in-place.

    Raises:
        ValueError: If x dimensions are inconsistent or head_dim is not a
            supported power of 2.
    """
    if max_blocks == 0:
        max_blocks = out_packed.shape[0]

    num_tokens, num_kv_heads, head_dim = x.shape

    if num_tokens == 0:
        return

    # Validate head_dim is power of 2 and supported
    if head_dim <= 0 or (head_dim & (head_dim - 1)) != 0:
        raise ValueError(f"head_dim must be a power of 2, got {head_dim}")

    half_d = head_dim // 2

    # Dispatch: Triton on CUDA (all SMs including Blackwell SM_100+)
    use_triton = HAS_TRITON and x.is_cuda

    if use_triton:
        # Ensure contiguous layout for Triton pointer arithmetic
        x_contig = x.contiguous()
        slot_contig = slot_mapping.contiguous()

        # Select rotation tile size: use 64 for head_dim >= 128, else 32.
        # The tile must evenly divide head_dim.
        if head_dim >= 128:
            tile_r = 64
        else:
            tile_r = 32
        # Ensure tile divides head_dim
        while head_dim % tile_r != 0 and tile_r > 1:
            tile_r //= 2

        grid = (num_tokens, num_kv_heads)

        _fused_tq4_kv_update_kernel[grid](
            # Input
            x_contig,
            slot_contig,
            # Output
            out_packed,
            out_norms,
            # Rotation + boundaries
            rotation,
            boundaries,
            # Strides for x: [tokens, heads, dims] -- dim stride is 1 (contiguous)
            x_contig.stride(0),
            x_contig.stride(1),
            # Strides for out_packed: [blocks, positions, heads, packed_dim]
            out_packed.stride(0),
            out_packed.stride(1),
            out_packed.stride(2),
            # Strides for out_norms: [blocks, positions, heads]
            out_norms.stride(0),
            out_norms.stride(1),
            # Scalar parameters
            block_size,
            max_blocks,
            num_kv_heads,
            # Compile-time constants
            HEAD_DIM=head_dim,
            HALF_D=half_d,
            TILE_R=tile_r,
        )
    else:
        _pytorch_tq4_kv_update(
            x, slot_mapping, out_packed, out_norms,
            rotation, boundaries, block_size, max_blocks,
        )


# ============================================================================
# INTEGRATION HELPER: fused_encode_and_store
# ============================================================================

def fused_encode_and_store(
    tq_cache,
    layer_idx: int,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """
    Fused TQ4 encode+store for both K and V in a single layer.

    Drop-in replacement for ``TQGPUCache.encode_and_store()``. Extracts the
    rotation matrix and quantization boundaries from the cache's TurboQuant
    instance and calls :func:`fused_tq4_kv_update` twice (once for K, once
    for V).

    Args:
        tq_cache: TQGPUCache instance (from vllm_custom_backend.py). Must have
            attributes: ``tq`` (TurboQuant), ``k_packed``, ``k_norms``,
            ``v_packed``, ``v_norms``, ``block_size``, ``max_blocks``.
        layer_idx: Transformer layer index.
        key: [num_tokens, num_kv_heads, head_dim] raw key vectors.
        value: [num_tokens, num_kv_heads, head_dim] raw value vectors.
        slot_mapping: [num_tokens] int64 slot indices from vLLM attention metadata.

    Returns:
        None. Writes directly to tq_cache's packed/norms tensors for the given layer.
    """
    tq = tq_cache.tq
    rotation = tq.rotation       # [head_dim, head_dim] float32
    boundaries = tq.boundaries_inner  # [15] float32

    block_size = tq_cache.block_size
    max_blocks = tq_cache.max_blocks

    # K cache: write to tq_cache.k_packed[layer_idx], tq_cache.k_norms[layer_idx]
    fused_tq4_kv_update(
        x=key,
        slot_mapping=slot_mapping,
        out_packed=tq_cache.k_packed[layer_idx],
        out_norms=tq_cache.k_norms[layer_idx],
        rotation=rotation,
        boundaries=boundaries,
        block_size=block_size,
        max_blocks=max_blocks,
    )

    # V cache: write to tq_cache.v_packed[layer_idx], tq_cache.v_norms[layer_idx]
    fused_tq4_kv_update(
        x=value,
        slot_mapping=slot_mapping,
        out_packed=tq_cache.v_packed[layer_idx],
        out_norms=tq_cache.v_norms[layer_idx],
        rotation=rotation,
        boundaries=boundaries,
        block_size=block_size,
        max_blocks=max_blocks,
    )
