"""
Hybrid Bit-Width TurboQuant Quantizer for KV cache compression.

Implements a two-group quantization scheme inspired by wizzense/vllm-turboquant,
adapted to the aither-kvcache architecture.  Unlike the uniform-bit TurboQuant
(quantizer.py), dimensions are split into an *outlier group* (high-variance
dims, more bits) and a *regular group* (low-variance dims, fewer bits).  Each
group then receives:

    1. L2 normalization
    2. Structured Hadamard rotation (fast Walsh-Hadamard with random sign flips)
    3. MSE-optimal Lloyd-Max quantization (dimension-aware Beta codebook)
    4. QJL residual encoding (1-bit sign sketch of the quantization residual)

Decoding reverses all steps and scatters the two groups back to the full
head_dim using pre-computed index maps.

Modes
-----
tq35 (3.5 avg bits):
    group0 = 50% dims @ 3-bit MSE + 1-bit QJL  (outlier)
    group1 = 50% dims @ 2-bit MSE + 1-bit QJL  (regular)

tq25 (2.5 avg bits):
    group0 = 25% dims @ 2-bit MSE + 1-bit QJL  (outlier)
    group1 = 75% dims @ 1-bit MSE + 1-bit QJL  (regular)

Group indices are determined by ``calibrate()`` (variance-based ranking on a
sample batch) or ``calibrate_uniform()`` (data-oblivious split).

Reference: Zandieh et al., "TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate", arXiv:2504.19874, April 2025.
"""

from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

__all__ = ["HybridTurboQuant", "HybridLayout", "GroupLayout"]


# ============================================================================
# MODE CONFIGURATIONS
# ============================================================================

MODE_CONFIGS: Dict[str, dict] = {
    "tq35": {
        "outlier_ratio": 0.50,   # 50% of dims are outliers
        "group_bits": (4, 3),    # (outlier total bits, regular total bits)
        # outlier: 3-bit MSE + 1-bit QJL = 4 effective bits
        # regular: 2-bit MSE + 1-bit QJL = 3 effective bits
    },
    "tq25": {
        "outlier_ratio": 0.25,
        "group_bits": (3, 2),
        # outlier: 2-bit MSE + 1-bit QJL = 3 effective bits
        # regular: 1-bit MSE + 1-bit QJL = 2 effective bits
    },
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class GroupLayout:
    """Byte layout for one quantization group within a packed vector."""
    dim: int                      # number of dimensions in this group
    mse_bits: int                 # bits for MSE codebook indices
    mse_packed_bytes: int         # ceil(dim * mse_bits / 8)
    qjl_packed_bytes: int         # ceil(dim / 8) -- 1 bit per dim
    norm_bytes: int               # 2 (fp16 vector norm)
    residual_norm_bytes: int      # 2 (fp16 residual norm)
    total_bytes: int              # sum of above


@dataclass(frozen=True)
class HybridLayout:
    """Combined byte layout for one full packed vector (group0 + group1)."""
    group0: GroupLayout
    group1: GroupLayout
    packed_dim: int               # total bytes per vector


def _make_group_layout(dim: int, mse_bits: int) -> GroupLayout:
    """Compute byte layout for a single group."""
    mse_packed = math.ceil(dim * mse_bits / 8)
    qjl_packed = math.ceil(dim / 8)
    norm = 2          # fp16 vector norm
    res_norm = 2      # fp16 residual norm
    total = mse_packed + qjl_packed + norm + res_norm
    return GroupLayout(
        dim=dim,
        mse_bits=mse_bits,
        mse_packed_bytes=mse_packed,
        qjl_packed_bytes=qjl_packed,
        norm_bytes=norm,
        residual_norm_bytes=res_norm,
        total_bytes=total,
    )


def _make_hybrid_layout(head_dim: int, mode: str) -> HybridLayout:
    """Build the full HybridLayout from head_dim and mode string."""
    cfg = MODE_CONFIGS[mode]
    outlier_ratio = cfg["outlier_ratio"]
    g0_total_bits, g1_total_bits = cfg["group_bits"]
    g0_mse_bits = g0_total_bits - 1   # 1 bit reserved for QJL sign
    g1_mse_bits = g1_total_bits - 1

    g0_dim = round(head_dim * outlier_ratio)
    g1_dim = head_dim - g0_dim

    g0 = _make_group_layout(g0_dim, g0_mse_bits)
    g1 = _make_group_layout(g1_dim, g1_mse_bits)
    return HybridLayout(group0=g0, group1=g1, packed_dim=g0.total_bytes + g1.total_bytes)


# ============================================================================
# FAST WALSH-HADAMARD TRANSFORM
# ============================================================================

def fwht_pow2(x: torch.Tensor) -> torch.Tensor:
    """In-place fast Walsh-Hadamard transform for power-of-2 last dimension.

    Operates on ``x`` in-place via butterfly operations.  O(N log N).

    Args:
        x: Tensor whose last dimension is a power of 2.

    Returns:
        The same tensor ``x`` (modified in-place).
    """
    size = x.shape[-1]
    h = 1
    while h < size:
        # Reshape so pairs of (h) elements are adjacent on the last axis
        x_view = x.reshape(*x.shape[:-1], -1, h * 2)
        left = x_view[..., :h].clone()
        right = x_view[..., h:].clone()
        x_view[..., :h] = left + right
        x_view[..., h:] = left - right
        h *= 2
    return x


def _decompose_pow2(n: int) -> list[int]:
    """Decompose *n* into descending powers of 2.

    Example: 96 -> [64, 32].  Used to handle non-power-of-2 dimensions by
    applying FWHT independently to each block.
    """
    parts: list[int] = []
    while n > 0:
        p = 1 << (n.bit_length() - 1)
        parts.append(p)
        n -= p
    return parts


def fwht_general(x: torch.Tensor) -> torch.Tensor:
    """Walsh-Hadamard transform supporting arbitrary last-dimension sizes.

    For power-of-2 sizes, delegates directly to :func:`fwht_pow2`.
    For non-power-of-2, splits into power-of-2 blocks and transforms each
    independently (block-diagonal Hadamard).

    Args:
        x: Tensor of arbitrary shape. Last dimension will be transformed.

    Returns:
        Transformed tensor (new allocation; ``x`` is not modified).
    """
    size = x.shape[-1]
    if size > 0 and (size & (size - 1)) == 0:
        return fwht_pow2(x.clone())

    parts = _decompose_pow2(size)
    out = x.clone()
    offset = 0
    for p in parts:
        block = out[..., offset:offset + p].contiguous()
        fwht_pow2(block)
        out[..., offset:offset + p] = block
        offset += p
    return out


# ============================================================================
# STRUCTURED HADAMARD TRANSFORM
# ============================================================================

def structured_hadamard(
    x: torch.Tensor,
    signs: torch.Tensor,
    normalized: bool = True,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply ``H * diag(signs) * x`` with optional normalization.

    ``H`` is the Walsh-Hadamard matrix (applied via FWHT), and ``signs`` is a
    random +/-1 vector that de-correlates the input.

    Forward:  multiply by signs, then FWHT.
    Inverse:  FWHT (self-inverse), then multiply by signs.

    Normalization divides by ``sqrt(dim)`` so that the transform is unitary.

    Args:
        x:          [..., dim] tensor.
        signs:      [dim] tensor of +1/-1 values.
        normalized: If True, divide by sqrt(dim).
        inverse:    If True, apply the inverse transform.

    Returns:
        Transformed tensor of the same shape.
    """
    dim = x.shape[-1]
    if inverse:
        y = fwht_general(x)
        y = y * signs
    else:
        y = x * signs
        y = fwht_general(y)
    if normalized:
        y = y / math.sqrt(dim)
    return y


# ============================================================================
# DIMENSION-AWARE LLOYD-MAX CODEBOOK (Beta distribution)
# ============================================================================

@functools.cache
def dimension_aware_codebook(dim: int, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Lloyd-Max centroids from the exact Beta(1/2, (d-1)/2) distribution.

    The marginal distribution of a single coordinate on S^{d-1} has PDF
    proportional to ``(1 - x^2)^((d-3)/2)`` on (-1, 1).  We run the iterative
    Lloyd algorithm on a dense grid weighted by this PDF.

    Args:
        dim: The ambient dimension *d* (number of coordinates in the group).
        bits: Number of quantization bits (1, 2, or 3).

    Returns:
        centroids:        [2^bits] float32 tensor of optimal reconstruction values.
        boundaries_inner: [2^bits - 1] float32 tensor of decision boundaries.
    """
    num_levels = 1 << bits
    grid_size = 10000
    max_iter = 200
    tol = 1e-10

    # Dense grid on (-1, 1)
    grid = torch.linspace(-0.999, 0.999, grid_size, dtype=torch.float64)

    # Beta PDF weight: (1 - x^2)^((d-3)/2), unnormalized
    exponent = (dim - 3) / 2.0
    if exponent >= 0:
        weights = (1.0 - grid * grid).pow(exponent)
    else:
        # d=1 or d=2: exponent < 0 -- clamp to avoid division by zero
        weights = (1.0 - grid * grid).clamp(min=1e-30).pow(exponent)
    weights = weights / weights.sum()

    # Initialize centroids uniformly
    centroids = torch.linspace(
        grid[0].item(), grid[-1].item(), num_levels, dtype=torch.float64,
    )

    for _ in range(max_iter):
        # Decision boundaries = midpoints
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # Assign each grid point to nearest centroid
        assignments = torch.searchsorted(boundaries, grid)
        assignments = assignments.clamp(0, num_levels - 1)

        # Update centroids = weighted conditional means
        new_centroids = torch.zeros_like(centroids)
        for k in range(num_levels):
            mask = assignments == k
            w = weights[mask]
            if w.sum() > 1e-30:
                new_centroids[k] = (grid[mask] * w).sum() / w.sum()
            else:
                new_centroids[k] = centroids[k]

        if (centroids - new_centroids).abs().max() < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    boundaries_inner = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids.float(), boundaries_inner.float()


# ============================================================================
# GENERIC BIT PACKING / UNPACKING
# ============================================================================

def pack_multibit(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack integer indices (0..2^bits-1) into uint8 bytes.

    The last dimension is packed.  Indices are written MSB-first into bytes.

    Args:
        indices: [..., dim] tensor with values in [0, 2^bits - 1].
        bits:    Bit width per index (1, 2, 3, or 4).

    Returns:
        [..., ceil(dim * bits / 8)] uint8 tensor.
    """
    dim = indices.shape[-1]
    total_bits = dim * bits
    out_bytes = math.ceil(total_bits / 8)

    # Flatten to [N, dim]
    batch_shape = indices.shape[:-1]
    flat = indices.reshape(-1, dim).to(torch.int32)
    N = flat.shape[0]

    packed = torch.zeros(N, out_bytes, dtype=torch.uint8, device=indices.device)

    for i in range(dim):
        bit_pos = i * bits               # global bit offset
        byte_idx = bit_pos // 8
        bit_off = bit_pos % 8             # offset within the byte (from MSB)
        val = flat[:, i] & ((1 << bits) - 1)

        remaining = bits
        cur_byte = byte_idx
        cur_off = bit_off
        while remaining > 0:
            space = 8 - cur_off            # bits left in current byte
            write = min(space, remaining)
            shift = space - write          # align to the available slot
            mask = ((1 << write) - 1)
            chunk = (val >> (remaining - write)) & mask
            packed[:, cur_byte] |= (chunk << shift).to(torch.uint8)
            remaining -= write
            cur_byte += 1
            cur_off = 0

    return packed.reshape(*batch_shape, out_bytes)


def unpack_multibit(packed: torch.Tensor, dim: int, bits: int) -> torch.Tensor:
    """Unpack uint8 bytes back to integer indices.

    Args:
        packed: [..., ceil(dim * bits / 8)] uint8 tensor.
        dim:    Number of original values per vector.
        bits:   Bit width per index (1, 2, 3, or 4).

    Returns:
        [..., dim] uint8 tensor with values in [0, 2^bits - 1].
    """
    batch_shape = packed.shape[:-1]
    num_bytes = packed.shape[-1]
    flat = packed.reshape(-1, num_bytes).to(torch.int32)
    N = flat.shape[0]

    out = torch.zeros(N, dim, dtype=torch.uint8, device=packed.device)

    for i in range(dim):
        bit_pos = i * bits
        byte_idx = bit_pos // 8
        bit_off = bit_pos % 8

        val = torch.zeros(N, dtype=torch.int32, device=packed.device)
        remaining = bits
        cur_byte = byte_idx
        cur_off = bit_off
        while remaining > 0:
            space = 8 - cur_off
            read = min(space, remaining)
            shift = space - read
            mask = (1 << read) - 1
            chunk = (flat[:, cur_byte] >> shift) & mask
            val = val | (chunk << (remaining - read))
            remaining -= read
            cur_byte += 1
            cur_off = 0

        out[:, i] = val.to(torch.uint8)

    return out.reshape(*batch_shape, dim)


def _pack_sign_bits(x: torch.Tensor) -> torch.Tensor:
    """Pack sign bits of ``x`` into uint8 bytes (positive=1, negative=0).

    Args:
        x: [..., dim] tensor.

    Returns:
        [..., ceil(dim / 8)] uint8 tensor.
    """
    dim = x.shape[-1]
    signs = (x >= 0).to(torch.uint8)     # 1 for positive, 0 for negative
    return pack_multibit(signs, bits=1)


def _unpack_sign_bits(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Unpack sign-bit bytes into +/-1 float tensor.

    Args:
        packed: [..., ceil(dim / 8)] uint8 tensor.
        dim:    Original dimension.

    Returns:
        [..., dim] float32 tensor of +1 / -1 values.
    """
    bits = unpack_multibit(packed, dim, bits=1)
    return (2.0 * bits.float() - 1.0)


# ============================================================================
# FP16 NORM HELPERS (stored as 2 x uint8)
# ============================================================================

def _fp16_to_uint8_pair(x: torch.Tensor) -> torch.Tensor:
    """Convert a scalar (or batch of scalars) to 2 uint8 bytes via fp16 view.

    Args:
        x: [...] float tensor.

    Returns:
        [..., 2] uint8 tensor.
    """
    fp16 = x.float().to(torch.float16)
    return fp16.view(torch.uint8).reshape(*x.shape, 2)


def _uint8_pair_to_fp16(packed: torch.Tensor) -> torch.Tensor:
    """Recover float values from 2-byte uint8 pairs.

    Args:
        packed: [..., 2] uint8 tensor.

    Returns:
        [...] float32 tensor.
    """
    # Reshape so last dim becomes the 2-byte view
    shape = packed.shape[:-1]
    flat = packed.reshape(-1, 2).contiguous()
    fp16 = flat.view(torch.float16).reshape(-1)
    return fp16.float().reshape(shape)


# ============================================================================
# HYBRID TURBOQUANT
# ============================================================================

class HybridTurboQuant:
    """Hybrid bit-width KV cache quantizer with QJL residual encoding.

    Splits each vector's dimensions into two groups (outlier / regular),
    quantizes each group with a different bit budget, and adds a 1-bit QJL
    residual sketch on top.

    Usage::

        htq = HybridTurboQuant(head_dim=128, mode="tq35")
        htq.calibrate(sample_kv_vectors)       # or htq.calibrate_uniform()
        packed = htq.encode(kv_vectors)
        decoded = htq.decode(packed)
        print(htq.validate())

    Args:
        head_dim: Dimension per attention head (e.g. 128).
        mode:     ``"tq35"`` (3.5 avg bits) or ``"tq25"`` (2.5 avg bits).
        seed:     RNG seed for deterministic Hadamard sign vectors.
        device:   ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        head_dim: int = 128,
        mode: str = "tq35",
        seed: int = 42,
        device: str = "cuda",
    ) -> None:
        if mode not in MODE_CONFIGS:
            raise ValueError(
                f"Unsupported mode {mode!r}. Choose from {list(MODE_CONFIGS.keys())}."
            )

        self.head_dim = head_dim
        self.mode = mode
        self.seed = seed
        self.device = device

        # Build layout
        self.layout: HybridLayout = _make_hybrid_layout(head_dim, mode)

        cfg = MODE_CONFIGS[mode]
        g0_total, g1_total = cfg["group_bits"]
        self._g0_mse_bits: int = g0_total - 1
        self._g1_mse_bits: int = g1_total - 1
        self._g0_dim: int = self.layout.group0.dim
        self._g1_dim: int = self.layout.group1.dim

        # Precompute Hadamard sign vectors (one per group, plus separate QJL signs)
        gen = torch.Generator(device="cpu").manual_seed(seed)

        def _rand_signs(n: int) -> torch.Tensor:
            bits = torch.randint(0, 2, (n,), generator=gen, dtype=torch.float32)
            return (2 * bits - 1).to(device)

        self._g0_had_signs: torch.Tensor = _rand_signs(self._g0_dim)
        self._g1_had_signs: torch.Tensor = _rand_signs(self._g1_dim)
        self._g0_qjl_signs: torch.Tensor = _rand_signs(self._g0_dim)
        self._g1_qjl_signs: torch.Tensor = _rand_signs(self._g1_dim)

        # Precompute codebooks (cached globally)
        self._g0_centroids, self._g0_boundaries = dimension_aware_codebook(
            self._g0_dim, self._g0_mse_bits,
        )
        self._g0_centroids = self._g0_centroids.to(device)
        self._g0_boundaries = self._g0_boundaries.to(device)

        self._g1_centroids, self._g1_boundaries = dimension_aware_codebook(
            self._g1_dim, self._g1_mse_bits,
        )
        self._g1_centroids = self._g1_centroids.to(device)
        self._g1_boundaries = self._g1_boundaries.to(device)

        # QJL scale factors: sqrt(pi/2) / group_dim
        self._g0_qjl_scale: float = math.sqrt(math.pi / 2.0) / self._g0_dim
        self._g1_qjl_scale: float = math.sqrt(math.pi / 2.0) / self._g1_dim

        # Group indices -- must be set by calibrate() or calibrate_uniform()
        # Shape: [num_kv_heads, g0_dim] and [num_kv_heads, g1_dim] respectively
        self._group0_indices: Optional[torch.Tensor] = None
        self._group1_indices: Optional[torch.Tensor] = None
        self._num_kv_heads: int = 0
        self._calibrated: bool = False

    # ================================================================
    # CALIBRATION
    # ================================================================

    def calibrate(self, sample_vectors: torch.Tensor) -> None:
        """Compute per-head dimension importance from sample data.

        Ranks dimensions by their variance across the sample and assigns the
        top-P% to the outlier group (group0).

        Args:
            sample_vectors: [num_samples, num_kv_heads, head_dim] tensor of
                            representative KV vectors.
        """
        if sample_vectors.dim() != 3:
            raise ValueError(
                f"Expected [num_samples, num_kv_heads, head_dim], "
                f"got shape {list(sample_vectors.shape)}"
            )
        num_samples, num_kv_heads, hdim = sample_vectors.shape
        if hdim != self.head_dim:
            raise ValueError(
                f"head_dim mismatch: expected {self.head_dim}, got {hdim}"
            )

        self._num_kv_heads = num_kv_heads

        # Per-dimension variance score: x^2.mean(dim=0) -> [num_kv_heads, head_dim]
        scores = (sample_vectors.float() ** 2).mean(dim=0)

        g0_indices_list = []
        g1_indices_list = []
        for h in range(num_kv_heads):
            _, sorted_idx = scores[h].sort(descending=True)
            g0_indices_list.append(sorted_idx[:self._g0_dim])
            g1_indices_list.append(sorted_idx[self._g0_dim:])

        self._group0_indices = torch.stack(g0_indices_list).to(self.device)  # [H, g0_dim]
        self._group1_indices = torch.stack(g1_indices_list).to(self.device)  # [H, g1_dim]
        self._calibrated = True

    def calibrate_uniform(self, num_kv_heads: int = 1) -> None:
        """Split dimensions uniformly (first N = outlier, rest = regular).

        This is a data-oblivious fallback when no sample data is available.

        Args:
            num_kv_heads: Number of KV heads (all heads use the same split).
        """
        self._num_kv_heads = num_kv_heads
        g0_idx = torch.arange(self._g0_dim, device=self.device)
        g1_idx = torch.arange(self._g0_dim, self.head_dim, device=self.device)
        self._group0_indices = g0_idx.unsqueeze(0).expand(num_kv_heads, -1).contiguous()
        self._group1_indices = g1_idx.unsqueeze(0).expand(num_kv_heads, -1).contiguous()
        self._calibrated = True

    def _ensure_calibrated(self) -> None:
        if not self._calibrated:
            raise RuntimeError(
                "HybridTurboQuant: must call calibrate() or calibrate_uniform() "
                "before encode/decode."
            )

    # ================================================================
    # ENCODE
    # ================================================================

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode vectors to packed uint8 representation.

        Args:
            x: [..., head_dim] float tensor.  Supports arbitrary batch dims.
                When the second-to-last dim equals ``num_kv_heads``, per-head
                group indices are used.  Otherwise the first head's indices are
                broadcast.

        Returns:
            [..., packed_dim] uint8 tensor.
        """
        self._ensure_calibrated()
        original_shape = x.shape
        assert x.shape[-1] == self.head_dim, (
            f"Expected last dim {self.head_dim}, got {x.shape[-1]}"
        )

        # Flatten to [N, head_dim]
        x_flat = x.reshape(-1, self.head_dim).float()
        N = x_flat.shape[0]

        # Determine per-head index mapping.  If N is divisible by num_kv_heads,
        # assume the vectors cycle over heads; otherwise broadcast head 0.
        H = self._num_kv_heads
        if H > 1 and N % H == 0:
            # Reshape to [B, H, head_dim], encode per-head, then flatten back
            B = N // H
            x_bh = x_flat.reshape(B, H, self.head_dim)
            packed_list = []
            for h in range(H):
                packed_list.append(
                    self._encode_group(
                        x_bh[:, h, :],
                        self._group0_indices[h],
                        self._group1_indices[h],
                    )
                )
            # packed_list: H tensors of [B, packed_dim] -> interleave to [B, H, packed_dim]
            packed = torch.stack(packed_list, dim=1).reshape(N, self.layout.packed_dim)
        else:
            packed = self._encode_group(
                x_flat,
                self._group0_indices[0],
                self._group1_indices[0],
            )

        batch_shape = list(original_shape[:-1])
        return packed.reshape(batch_shape + [self.layout.packed_dim])

    def _encode_group(
        self,
        x: torch.Tensor,
        g0_idx: torch.Tensor,
        g1_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of vectors using the given group indices.

        Args:
            x:      [N, head_dim] float32
            g0_idx: [g0_dim] long indices for outlier group
            g1_idx: [g1_dim] long indices for regular group

        Returns:
            [N, packed_dim] uint8
        """
        N = x.shape[0]
        device = x.device

        # Gather group dimensions
        x_g0 = x[:, g0_idx]  # [N, g0_dim]
        x_g1 = x[:, g1_idx]  # [N, g1_dim]

        # Encode each group
        g0_packed = self._encode_single_group(
            x_g0,
            self._g0_had_signs,
            self._g0_qjl_signs,
            self._g0_centroids,
            self._g0_boundaries,
            self._g0_mse_bits,
            self.layout.group0,
        )
        g1_packed = self._encode_single_group(
            x_g1,
            self._g1_had_signs,
            self._g1_qjl_signs,
            self._g1_centroids,
            self._g1_boundaries,
            self._g1_mse_bits,
            self.layout.group1,
        )

        # Concatenate group0 + group1 packed bytes
        return torch.cat([g0_packed, g1_packed], dim=-1)  # [N, packed_dim]

    def _encode_single_group(
        self,
        x_group: torch.Tensor,
        had_signs: torch.Tensor,
        qjl_signs: torch.Tensor,
        centroids: torch.Tensor,
        boundaries: torch.Tensor,
        mse_bits: int,
        layout: GroupLayout,
    ) -> torch.Tensor:
        """Encode one group: normalize -> hadamard -> MSE quant -> QJL residual.

        Args:
            x_group:    [N, G] float32 input for this group.
            had_signs:  [G] Hadamard sign vector.
            qjl_signs:  [G] QJL sign vector.
            centroids:  [2^mse_bits] codebook.
            boundaries: [2^mse_bits - 1] decision boundaries.
            mse_bits:   MSE quantization bits.
            layout:     GroupLayout for byte offsets.

        Returns:
            [N, layout.total_bytes] uint8 packed tensor.
        """
        N = x_group.shape[0]
        G = x_group.shape[1]
        device = x_group.device

        # 1. Compute L2 norm
        vec_norm = x_group.norm(dim=-1)                       # [N]

        # 2. Normalize to unit sphere
        unit = x_group / (vec_norm.unsqueeze(-1) + 1e-10)     # [N, G]

        # 3. Structured Hadamard rotation
        rotated = structured_hadamard(unit, had_signs, normalized=True, inverse=False)

        # 4. MSE quantize: find nearest centroid
        indices = torch.searchsorted(boundaries, rotated)      # [N, G]
        num_levels = 1 << mse_bits
        indices = indices.clamp(0, num_levels - 1)

        # 5. Pack MSE indices
        mse_packed = pack_multibit(indices, mse_bits)          # [N, mse_packed_bytes]

        # 6. Compute residual in the rotated domain
        quantized_rotated = centroids[indices.long()]           # [N, G]
        # Inverse Hadamard to get MSE reconstruction on the unit sphere
        mse_hat_unit = structured_hadamard(
            quantized_rotated, had_signs, normalized=True, inverse=True,
        )
        residual = unit - mse_hat_unit                         # [N, G]
        res_norm = residual.norm(dim=-1)                       # [N]

        # 7. QJL projection of residual: hadamard with separate signs -> take sign bits
        qjl_projected = structured_hadamard(
            residual, qjl_signs, normalized=True, inverse=False,
        )

        # 8. Pack QJL sign bits (1 bit per coordinate)
        qjl_packed = _pack_sign_bits(qjl_projected)            # [N, qjl_packed_bytes]

        # 9. Pack norms as fp16 -> 2 uint8 bytes each
        norm_packed = _fp16_to_uint8_pair(vec_norm)            # [N, 2]
        res_norm_packed = _fp16_to_uint8_pair(res_norm)        # [N, 2]

        # 10. Concatenate: [mse_packed | qjl_packed | norm_fp16 | residual_norm_fp16]
        return torch.cat([mse_packed, qjl_packed, norm_packed, res_norm_packed], dim=-1)

    # ================================================================
    # DECODE
    # ================================================================

    def decode(self, packed: torch.Tensor) -> torch.Tensor:
        """Decode packed representation back to full vectors.

        Args:
            packed: [..., packed_dim] uint8 tensor.

        Returns:
            [..., head_dim] float32 tensor (reconstructed vectors).
        """
        self._ensure_calibrated()
        original_shape = packed.shape
        packed_dim = self.layout.packed_dim
        assert packed.shape[-1] == packed_dim, (
            f"Expected last dim {packed_dim}, got {packed.shape[-1]}"
        )

        # Flatten to [N, packed_dim]
        packed_flat = packed.reshape(-1, packed_dim)
        N = packed_flat.shape[0]

        H = self._num_kv_heads
        if H > 1 and N % H == 0:
            B = N // H
            packed_bh = packed_flat.reshape(B, H, packed_dim)
            decoded_list = []
            for h in range(H):
                decoded_list.append(
                    self._decode_group(
                        packed_bh[:, h, :],
                        self._group0_indices[h],
                        self._group1_indices[h],
                    )
                )
            decoded = torch.stack(decoded_list, dim=1).reshape(N, self.head_dim)
        else:
            decoded = self._decode_group(
                packed_flat,
                self._group0_indices[0],
                self._group1_indices[0],
            )

        batch_shape = list(original_shape[:-1])
        return decoded.reshape(batch_shape + [self.head_dim])

    def _decode_group(
        self,
        packed: torch.Tensor,
        g0_idx: torch.Tensor,
        g1_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Decode a batch using the given group indices.

        Args:
            packed: [N, packed_dim] uint8
            g0_idx: [g0_dim] long
            g1_idx: [g1_dim] long

        Returns:
            [N, head_dim] float32
        """
        N = packed.shape[0]
        g0_bytes = self.layout.group0.total_bytes
        g1_bytes = self.layout.group1.total_bytes

        g0_packed = packed[:, :g0_bytes]
        g1_packed = packed[:, g0_bytes:g0_bytes + g1_bytes]

        # Decode each group
        g0_decoded = self._decode_single_group(
            g0_packed,
            self._g0_had_signs,
            self._g0_qjl_signs,
            self._g0_centroids,
            self._g0_mse_bits,
            self.layout.group0,
            self._g0_qjl_scale,
        )
        g1_decoded = self._decode_single_group(
            g1_packed,
            self._g1_had_signs,
            self._g1_qjl_signs,
            self._g1_centroids,
            self._g1_mse_bits,
            self.layout.group1,
            self._g1_qjl_scale,
        )

        # Scatter back to full head_dim
        out = torch.zeros(N, self.head_dim, dtype=torch.float32, device=packed.device)
        out[:, g0_idx] = g0_decoded
        out[:, g1_idx] = g1_decoded
        return out

    def _decode_single_group(
        self,
        packed_group: torch.Tensor,
        had_signs: torch.Tensor,
        qjl_signs: torch.Tensor,
        centroids: torch.Tensor,
        mse_bits: int,
        layout: GroupLayout,
        qjl_scale: float,
    ) -> torch.Tensor:
        """Decode one group: unpack -> codebook -> inverse hadamard -> reconstruct.

        Args:
            packed_group: [N, layout.total_bytes] uint8.
            had_signs:    [G] Hadamard sign vector.
            qjl_signs:    [G] QJL sign vector.
            centroids:    [2^mse_bits] codebook.
            mse_bits:     MSE quantization bits.
            layout:       GroupLayout.
            qjl_scale:    sqrt(pi/2) / G scaling factor for QJL.

        Returns:
            [N, G] float32 reconstructed group vectors.
        """
        G = layout.dim
        mse_end = layout.mse_packed_bytes
        qjl_end = mse_end + layout.qjl_packed_bytes
        norm_end = qjl_end + layout.norm_bytes
        res_end = norm_end + layout.residual_norm_bytes

        # 1. Unpack MSE indices -> codebook lookup -> inverse Hadamard -> mse_hat
        mse_packed = packed_group[:, :mse_end]
        indices = unpack_multibit(mse_packed, G, mse_bits)     # [N, G] uint8
        quantized_rotated = centroids[indices.long()]           # [N, G]
        mse_hat = structured_hadamard(
            quantized_rotated, had_signs, normalized=True, inverse=True,
        )

        # 2. Unpack QJL sign bits -> +/-1 -> scale -> inverse Hadamard -> qjl_hat
        qjl_packed = packed_group[:, mse_end:qjl_end]
        qjl_signs_decoded = _unpack_sign_bits(qjl_packed, G)   # [N, G] +/-1
        qjl_scaled = qjl_signs_decoded * qjl_scale
        qjl_hat = structured_hadamard(
            qjl_scaled, qjl_signs, normalized=True, inverse=True,
        )

        # 3. Unpack norms
        norm_packed = packed_group[:, qjl_end:norm_end]
        vec_norm = _uint8_pair_to_fp16(norm_packed)            # [N]

        res_norm_packed = packed_group[:, norm_end:res_end]
        res_norm = _uint8_pair_to_fp16(res_norm_packed)        # [N]

        # 4. Reconstruct: (mse_hat + qjl_hat * residual_norm) * vector_norm
        decoded = (mse_hat + qjl_hat * res_norm.unsqueeze(-1)) * vec_norm.unsqueeze(-1)
        return decoded

    # ================================================================
    # VALIDATION & ANALYSIS
    # ================================================================

    def validate(self, num_vectors: int = 10000) -> dict:
        """Encode-decode roundtrip on random data. Report MSE and compression.

        If not yet calibrated, calls ``calibrate_uniform()`` automatically.

        Args:
            num_vectors: Number of random vectors to test.

        Returns:
            Dict with mse, cosine_similarity, compression ratios, etc.
        """
        dev = self.device
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"

        if not self._calibrated:
            self.calibrate_uniform(num_kv_heads=1)

        # Generate random vectors
        x = torch.randn(num_vectors, self.head_dim, device=dev, dtype=torch.float32)
        # Give some dimensions higher variance to test group splitting
        x[:, :self._g0_dim] *= 2.0

        # Encode -> decode
        packed = self.encode(x)
        x_hat = self.decode(packed)

        # MSE: E[||x - x_hat||^2]
        mse = (x - x_hat).pow(2).sum(dim=-1).mean().item()

        # Relative MSE: MSE / E[||x||^2]
        x_sq_norm = x.pow(2).sum(dim=-1).mean().item()
        rel_mse = mse / x_sq_norm if x_sq_norm > 0 else float("inf")

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()

        # Max absolute error
        max_err = (x - x_hat).abs().max().item()

        fp16_bytes = self.head_dim * 2
        ratio_vs_fp16 = fp16_bytes / self.layout.packed_dim
        ratio_vs_fp8 = self.head_dim / self.layout.packed_dim

        return {
            "mode": self.mode,
            "head_dim": self.head_dim,
            "num_vectors": num_vectors,
            "packed_dim_bytes": self.layout.packed_dim,
            "mse": mse,
            "relative_mse": rel_mse,
            "cosine_similarity": cos_sim,
            "max_abs_error": max_err,
            "compression_vs_fp16": ratio_vs_fp16,
            "compression_vs_fp8": ratio_vs_fp8,
            "group0": {
                "dim": self._g0_dim,
                "mse_bits": self._g0_mse_bits,
                "bytes": self.layout.group0.total_bytes,
            },
            "group1": {
                "dim": self._g1_dim,
                "mse_bits": self._g1_mse_bits,
                "bytes": self.layout.group1.total_bytes,
            },
        }

    def compression_ratio(self) -> float:
        """Compression ratio vs FP16 (2 bytes per value)."""
        return (self.head_dim * 2) / self.layout.packed_dim

    def memory_report(
        self,
        seq_len: int,
        num_layers: int = 32,
        num_kv_heads: int = 8,
    ) -> dict:
        """Compute memory usage for a KV cache configuration.

        Args:
            seq_len:      Sequence length.
            num_layers:   Number of transformer layers.
            num_kv_heads: Number of KV attention heads.

        Returns:
            Dict with memory usage in MB for FP16, FP8, and this mode.
        """
        total_vectors = seq_len * num_layers * num_kv_heads * 2  # K + V
        fp16_bytes = total_vectors * self.head_dim * 2
        fp8_bytes = total_vectors * self.head_dim
        hybrid_bytes = total_vectors * self.layout.packed_dim

        return {
            "config": (
                f"Hybrid-{self.mode.upper()} d={self.head_dim} "
                f"seq={seq_len} L={num_layers} H={num_kv_heads}"
            ),
            "fp16_mb": fp16_bytes / (1024 ** 2),
            "fp8_mb": fp8_bytes / (1024 ** 2),
            f"{self.mode}_mb": hybrid_bytes / (1024 ** 2),
            "ratio_vs_fp16": fp16_bytes / hybrid_bytes,
            "ratio_vs_fp8": fp8_bytes / hybrid_bytes,
        }

    @property
    def packed_dim(self) -> int:
        """Total bytes per encoded vector."""
        return self.layout.packed_dim

    @staticmethod
    def packed_dim_for_mode(head_dim: int, mode: str) -> int:
        """Compute packed_dim without full instantiation.

        Args:
            head_dim: Dimension per head.
            mode:     ``"tq35"`` or ``"tq25"``.

        Returns:
            Number of bytes per packed vector.
        """
        return _make_hybrid_layout(head_dim, mode).packed_dim

    def __repr__(self) -> str:
        cal = "calibrated" if self._calibrated else "uncalibrated"
        return (
            f"HybridTurboQuant(d={self.head_dim}, mode={self.mode!r}, "
            f"packed={self.layout.packed_dim}B, "
            f"compress={self.compression_ratio():.1f}x, {cal})"
        )
