"""
Bit packing/unpacking for TurboQuant quantized indices.

Packs low-bit indices into uint8 bytes for memory-efficient storage:
  4-bit: 2 values per byte  → head_dim/2 bytes per vector
  3-bit: 8 values per 3 bytes → head_dim*3/8 bytes per vector
  2-bit: 4 values per byte  → head_dim/4 bytes per vector
"""

import torch


def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack pairs of 4-bit indices into uint8 bytes.

    Input:  [..., D] with values in [0, 15]
    Output: [..., D//2] uint8
    """
    D = indices.shape[-1]
    assert D % 2 == 0, f"Dimension must be even for 4-bit packing, got {D}"
    even = indices[..., 0::2].to(torch.uint8)
    odd = indices[..., 1::2].to(torch.uint8)
    return (even << 4) | odd


def unpack_4bit(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Unpack uint8 bytes into pairs of 4-bit indices.

    Input:  [..., D//2] uint8
    Output: [..., D] uint8 with values in [0, 15]
    """
    even = (packed >> 4) & 0x0F
    odd = packed & 0x0F
    result = torch.empty(*packed.shape[:-1], dim, dtype=torch.uint8, device=packed.device)
    result[..., 0::2] = even
    result[..., 1::2] = odd
    return result


def pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack groups of 4 two-bit indices into uint8 bytes.

    Input:  [..., D] with values in [0, 3]
    Output: [..., D//4] uint8
    """
    D = indices.shape[-1]
    assert D % 4 == 0, f"Dimension must be divisible by 4 for 2-bit packing, got {D}"
    a = indices[..., 0::4].to(torch.uint8)
    b = indices[..., 1::4].to(torch.uint8)
    c = indices[..., 2::4].to(torch.uint8)
    d = indices[..., 3::4].to(torch.uint8)
    return (a << 6) | (b << 4) | (c << 2) | d


def unpack_2bit(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Unpack uint8 bytes into groups of 4 two-bit indices.

    Input:  [..., D//4] uint8
    Output: [..., D] uint8 with values in [0, 3]
    """
    a = (packed >> 6) & 0x03
    b = (packed >> 4) & 0x03
    c = (packed >> 2) & 0x03
    d_val = packed & 0x03
    result = torch.empty(*packed.shape[:-1], dim, dtype=torch.uint8, device=packed.device)
    result[..., 0::4] = a
    result[..., 1::4] = b
    result[..., 2::4] = c
    result[..., 3::4] = d_val
    return result


def pack_3bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack groups of 8 three-bit indices into 3 bytes (24 bits).

    Input:  [..., D] with values in [0, 7]
    Output: [..., D*3//8] uint8

    Bit layout per group of 8 values into 24 bits (3 bytes):
      bits[23:21] = val[0], bits[20:18] = val[1], ..., bits[2:0] = val[7]
    """
    D = indices.shape[-1]
    assert D % 8 == 0, f"Dimension must be divisible by 8 for 3-bit packing, got {D}"
    groups = D // 8
    idx = indices.to(torch.int32).reshape(*indices.shape[:-1], groups, 8)

    # Pack 8 × 3-bit values into a 24-bit integer
    bits24 = torch.zeros(*idx.shape[:-1], dtype=torch.int32, device=indices.device)
    for i in range(8):
        bits24 = bits24 | ((idx[..., i] & 0x7) << (21 - i * 3))

    # Split into 3 bytes
    byte0 = ((bits24 >> 16) & 0xFF).to(torch.uint8)
    byte1 = ((bits24 >> 8) & 0xFF).to(torch.uint8)
    byte2 = (bits24 & 0xFF).to(torch.uint8)

    packed = torch.stack([byte0, byte1, byte2], dim=-1)
    return packed.reshape(*indices.shape[:-1], groups * 3)


def unpack_3bit(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Unpack 3-bit packed bytes into indices.

    Input:  [..., D*3//8] uint8
    Output: [..., D] uint8 with values in [0, 7]
    """
    groups = dim // 8
    reshaped = packed.reshape(*packed.shape[:-1], groups, 3).to(torch.int32)

    # Reconstruct 24-bit integers
    bits24 = (reshaped[..., 0] << 16) | (reshaped[..., 1] << 8) | reshaped[..., 2]

    # Extract 8 three-bit values
    result = torch.empty(*packed.shape[:-1], groups, 8, dtype=torch.uint8, device=packed.device)
    for i in range(8):
        result[..., i] = ((bits24 >> (21 - i * 3)) & 0x7).to(torch.uint8)

    return result.reshape(*result.shape[:-2], dim)


def packed_size(dim: int, bits: int) -> int:
    """Return the number of bytes needed to store dim values at given bit-width."""
    if bits == 4:
        return dim // 2
    elif bits == 3:
        assert dim % 8 == 0
        return dim * 3 // 8
    elif bits == 2:
        return dim // 4
    else:
        raise ValueError(f"Unsupported bit-width: {bits}")
