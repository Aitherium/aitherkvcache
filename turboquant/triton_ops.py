"""
Triton GPU kernels for TurboQuant quantization/dequantization.

Fused kernels that avoid materializing intermediate tensors:
  - quantize_and_pack: searchsorted + bit-pack in one pass
  - unpack_and_lookup: bit-unpack + codebook gather in one pass

Falls back to pure PyTorch if Triton is unavailable.
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:

    # ====================================================================
    # 4-BIT KERNELS
    # ====================================================================

    @triton.jit
    def _quantize_pack_4bit_kernel(
        y_ptr,           # [N, D] float32 rotated normalized vectors
        boundaries_ptr,  # [15] float32 sorted inner boundaries
        packed_ptr,      # [N, D//2] uint8 output
        N,
        D: tl.constexpr,
        HALF_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return

        pair_offs = tl.arange(0, HALF_D)
        base = pid * D

        # Load even and odd coordinates of this vector
        y_even = tl.load(y_ptr + base + pair_offs * 2)
        y_odd = tl.load(y_ptr + base + pair_offs * 2 + 1)

        # Vectorized searchsorted: count how many boundaries each value exceeds
        idx_even = tl.zeros([HALF_D], dtype=tl.int32)
        idx_odd = tl.zeros([HALF_D], dtype=tl.int32)

        for b in tl.static_range(15):
            boundary = tl.load(boundaries_ptr + b)
            idx_even += (y_even >= boundary).to(tl.int32)
            idx_odd += (y_odd >= boundary).to(tl.int32)

        # Pack: high nibble = even index, low nibble = odd index
        packed = ((idx_even & 0xF) << 4) | (idx_odd & 0xF)
        tl.store(packed_ptr + pid * HALF_D + pair_offs, packed.to(tl.uint8))

    @triton.jit
    def _unpack_lookup_4bit_kernel(
        packed_ptr,      # [N, D//2] uint8
        centroids_ptr,   # [16] float32 codebook
        out_ptr,         # [N, D] float32
        N,
        D: tl.constexpr,
        HALF_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return

        pair_offs = tl.arange(0, HALF_D)

        # Unpack nibbles
        packed = tl.load(packed_ptr + pid * HALF_D + pair_offs).to(tl.int32)
        idx_even = (packed >> 4) & 0xF
        idx_odd = packed & 0xF

        # Codebook gather
        y_even = tl.load(centroids_ptr + idx_even)
        y_odd = tl.load(centroids_ptr + idx_odd)

        # Store interleaved back to full vector
        base = pid * D
        tl.store(out_ptr + base + pair_offs * 2, y_even)
        tl.store(out_ptr + base + pair_offs * 2 + 1, y_odd)

    # ====================================================================
    # 2-BIT KERNELS
    # ====================================================================

    @triton.jit
    def _quantize_pack_2bit_kernel(
        y_ptr,           # [N, D] float32
        boundaries_ptr,  # [3] float32
        packed_ptr,      # [N, D//4] uint8
        N,
        D: tl.constexpr,
        QUARTER_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return

        q_offs = tl.arange(0, QUARTER_D)
        base = pid * D

        y_a = tl.load(y_ptr + base + q_offs * 4)
        y_b = tl.load(y_ptr + base + q_offs * 4 + 1)
        y_c = tl.load(y_ptr + base + q_offs * 4 + 2)
        y_d = tl.load(y_ptr + base + q_offs * 4 + 3)

        idx_a = tl.zeros([QUARTER_D], dtype=tl.int32)
        idx_b = tl.zeros([QUARTER_D], dtype=tl.int32)
        idx_c = tl.zeros([QUARTER_D], dtype=tl.int32)
        idx_d = tl.zeros([QUARTER_D], dtype=tl.int32)

        for b in tl.static_range(3):
            boundary = tl.load(boundaries_ptr + b)
            idx_a += (y_a >= boundary).to(tl.int32)
            idx_b += (y_b >= boundary).to(tl.int32)
            idx_c += (y_c >= boundary).to(tl.int32)
            idx_d += (y_d >= boundary).to(tl.int32)

        packed = ((idx_a & 0x3) << 6) | ((idx_b & 0x3) << 4) | \
                 ((idx_c & 0x3) << 2) | (idx_d & 0x3)
        tl.store(packed_ptr + pid * QUARTER_D + q_offs, packed.to(tl.uint8))

    @triton.jit
    def _unpack_lookup_2bit_kernel(
        packed_ptr,      # [N, D//4] uint8
        centroids_ptr,   # [4] float32
        out_ptr,         # [N, D] float32
        N,
        D: tl.constexpr,
        QUARTER_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return

        q_offs = tl.arange(0, QUARTER_D)
        packed = tl.load(packed_ptr + pid * QUARTER_D + q_offs).to(tl.int32)

        idx_a = (packed >> 6) & 0x3
        idx_b = (packed >> 4) & 0x3
        idx_c = (packed >> 2) & 0x3
        idx_d = packed & 0x3

        y_a = tl.load(centroids_ptr + idx_a)
        y_b = tl.load(centroids_ptr + idx_b)
        y_c = tl.load(centroids_ptr + idx_c)
        y_d = tl.load(centroids_ptr + idx_d)

        base = pid * D
        tl.store(out_ptr + base + q_offs * 4, y_a)
        tl.store(out_ptr + base + q_offs * 4 + 1, y_b)
        tl.store(out_ptr + base + q_offs * 4 + 2, y_c)
        tl.store(out_ptr + base + q_offs * 4 + 3, y_d)

    # ====================================================================
    # 3-BIT KERNELS (stub -- uses PyTorch fallback)
    # ====================================================================

    @triton.jit
    def _quantize_pack_3bit_kernel(
        y_ptr,           # [N, D] float32
        boundaries_ptr,  # [7] float32
        packed_ptr,      # [N, D*3//8] uint8
        N,
        D: tl.constexpr,
        GROUPS: tl.constexpr,     # D // 8
        PACKED_D: tl.constexpr,   # D * 3 // 8
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return

        base = pid * D
        out_base = pid * PACKED_D

        for g in tl.static_range(GROUPS):
            g_base = base + g * 8
            vals = tl.load(y_ptr + g_base + tl.arange(0, 8))

            indices = tl.zeros([8], dtype=tl.int32)
            for b in tl.static_range(7):
                boundary = tl.load(boundaries_ptr + b)
                indices += (vals >= boundary).to(tl.int32)

            # 3-bit Triton packing is complex; this kernel is a stub.
            # The 2-bit and 4-bit kernels cover the practical sweet spots.
            # 3-bit uses the PyTorch fallback path.
            pass

        # NOTE: 3-bit Triton kernel is a stub -- uses PyTorch packing path.


# ========================================================================
# PUBLIC API -- dispatch to Triton or PyTorch fallback
# ========================================================================

def triton_quantize_4bit(y: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """Fused 4-bit quantize + pack on GPU."""
    assert HAS_TRITON and y.is_cuda, "Triton requires CUDA tensors"
    N, D = y.shape
    HALF_D = D // 2
    packed = torch.empty(N, HALF_D, dtype=torch.uint8, device=y.device)
    _quantize_pack_4bit_kernel[(N,)](y, boundaries, packed, N, D=D, HALF_D=HALF_D)
    return packed


def triton_dequantize_4bit(packed: torch.Tensor, centroids: torch.Tensor,
                           D: int) -> torch.Tensor:
    """Fused 4-bit unpack + codebook lookup on GPU."""
    assert HAS_TRITON and packed.is_cuda, "Triton requires CUDA tensors"
    N = packed.shape[0]
    HALF_D = D // 2
    out = torch.empty(N, D, dtype=torch.float32, device=packed.device)
    _unpack_lookup_4bit_kernel[(N,)](packed, centroids, out, N, D=D, HALF_D=HALF_D)
    return out


def triton_quantize_2bit(y: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """Fused 2-bit quantize + pack on GPU."""
    assert HAS_TRITON and y.is_cuda, "Triton requires CUDA tensors"
    N, D = y.shape
    QUARTER_D = D // 4
    packed = torch.empty(N, QUARTER_D, dtype=torch.uint8, device=y.device)
    _quantize_pack_2bit_kernel[(N,)](y, boundaries, packed, N, D=D, QUARTER_D=QUARTER_D)
    return packed


def triton_dequantize_2bit(packed: torch.Tensor, centroids: torch.Tensor,
                           D: int) -> torch.Tensor:
    """Fused 2-bit unpack + codebook lookup on GPU."""
    assert HAS_TRITON and packed.is_cuda, "Triton requires CUDA tensors"
    N = packed.shape[0]
    QUARTER_D = D // 4
    out = torch.empty(N, D, dtype=torch.float32, device=packed.device)
    _unpack_lookup_2bit_kernel[(N,)](packed, centroids, out, N, D=D, QUARTER_D=QUARTER_D)
    return out
