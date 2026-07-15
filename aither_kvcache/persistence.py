"""
TQKV Binary Persistence Format for TurboQuant-compressed KV cache blocks.

Enables persistent prefix caching, cross-process sharing via mmap, and
SSD cold tier integration.

Format (v1):
    Header (32 bytes):
        magic:      4 bytes  b"TQKV"
        version:    1 byte   (1)
        bits:       1 byte   (2, 3, or 4)
        head_dim:   2 bytes  uint16 LE
        num_heads:  2 bytes  uint16 LE
        num_layers: 2 bytes  uint16 LE
        block_size: 2 bytes  uint16 LE
        num_blocks: 4 bytes  uint32 LE
        flags:      1 byte   (bit 0: has_correction, bits 1-7: reserved)
        reserved:   13 bytes (zeros)
    Block data (per block, repeated num_blocks times):
        packed_k:   [block_size, num_heads, packed_dim] uint8
        k_norms:    [block_size, num_heads] float32
        packed_v:   [block_size, num_heads, packed_dim] uint8
        v_norms:    [block_size, num_heads] float32
"""

import struct
from pathlib import Path
from typing import Optional, Tuple

import torch
import numpy as np

from turboquant.packing import packed_size

MAGIC = b"TQKV"
FORMAT_VERSION = 1
HEADER_SIZE = 32


def _pack_header(
    bits: int,
    head_dim: int,
    num_heads: int,
    num_layers: int,
    block_size: int,
    num_blocks: int,
    has_correction: bool = False,
) -> bytes:
    flags = int(has_correction) & 0x01
    header = struct.pack(
        "<4sBBHHHHIB13s",
        MAGIC,
        FORMAT_VERSION,
        bits,
        head_dim,
        num_heads,
        num_layers,
        block_size,
        num_blocks,
        flags,
        b"\x00" * 13,
    )
    assert len(header) == HEADER_SIZE
    return header


def _unpack_header(data: bytes) -> dict:
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Header too short: {len(data)} < {HEADER_SIZE}")
    magic = data[:4]
    if magic != MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}, expected {MAGIC!r}")
    (
        _magic, version, bits, head_dim, num_heads, num_layers,
        block_size, num_blocks, flags, _reserved,
    ) = struct.unpack("<4sBBHHHHIB13s", data[:HEADER_SIZE])
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported format version: {version}")
    return {
        "version": version,
        "bits": bits,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "has_correction": bool(flags & 0x01),
    }


def save_tqkv(
    path: str,
    k_packed: torch.Tensor,
    k_norms: torch.Tensor,
    v_packed: torch.Tensor,
    v_norms: torch.Tensor,
    bits: int,
    head_dim: int,
    num_layers: int = 1,
) -> None:
    """
    Save TQ-compressed KV cache blocks to a .tqkv file.

    Args:
        path: Output file path (recommend .tqkv extension).
        k_packed: [num_blocks, block_size, num_heads, packed_dim] uint8
        k_norms:  [num_blocks, block_size, num_heads] float32
        v_packed: same shape as k_packed
        v_norms:  same shape as k_norms
        bits: quantization bit-width (2, 3, or 4)
        head_dim: original head dimension
        num_layers: number of layers (metadata only)
    """
    num_blocks, block_size, num_heads = k_norms.shape
    pd = packed_size(head_dim, bits)
    assert k_packed.shape == (num_blocks, block_size, num_heads, pd)
    assert v_packed.shape == k_packed.shape
    assert v_norms.shape == k_norms.shape

    header = _pack_header(
        bits=bits,
        head_dim=head_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        block_size=block_size,
        num_blocks=num_blocks,
    )

    k_packed_np = k_packed.cpu().numpy()
    k_norms_np = k_norms.cpu().float().numpy()
    v_packed_np = v_packed.cpu().numpy()
    v_norms_np = v_norms.cpu().float().numpy()

    with open(path, "wb") as f:
        f.write(header)
        for blk in range(num_blocks):
            f.write(k_packed_np[blk].tobytes())
            f.write(k_norms_np[blk].tobytes())
            f.write(v_packed_np[blk].tobytes())
            f.write(v_norms_np[blk].tobytes())


def load_tqkv(
    path: str,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Load TQ-compressed KV cache blocks from a .tqkv file.

    Returns:
        k_packed: [num_blocks, block_size, num_heads, packed_dim] uint8
        k_norms:  [num_blocks, block_size, num_heads] float32
        v_packed: same shape as k_packed
        v_norms:  same shape as k_norms
        meta: dict with format metadata (bits, head_dim, num_heads, etc.)
    """
    data = Path(path).read_bytes()
    meta = _unpack_header(data[:HEADER_SIZE])

    bits = meta["bits"]
    head_dim = meta["head_dim"]
    num_heads = meta["num_heads"]
    block_size = meta["block_size"]
    num_blocks = meta["num_blocks"]
    pd = packed_size(head_dim, bits)

    packed_bytes = block_size * num_heads * pd
    norms_bytes = block_size * num_heads * 4  # float32
    block_data_size = 2 * (packed_bytes + norms_bytes)  # K + V

    expected = HEADER_SIZE + num_blocks * block_data_size
    if len(data) < expected:
        raise ValueError(
            f"File too short: {len(data)} < {expected} "
            f"(expected {num_blocks} blocks)"
        )

    k_packed_list = []
    k_norms_list = []
    v_packed_list = []
    v_norms_list = []

    offset = HEADER_SIZE
    for _ in range(num_blocks):
        kp = np.frombuffer(data[offset:offset + packed_bytes], dtype=np.uint8)
        kp = kp.reshape(block_size, num_heads, pd)
        offset += packed_bytes

        kn = np.frombuffer(data[offset:offset + norms_bytes], dtype=np.float32)
        kn = kn.reshape(block_size, num_heads)
        offset += norms_bytes

        vp = np.frombuffer(data[offset:offset + packed_bytes], dtype=np.uint8)
        vp = vp.reshape(block_size, num_heads, pd)
        offset += packed_bytes

        vn = np.frombuffer(data[offset:offset + norms_bytes], dtype=np.float32)
        vn = vn.reshape(block_size, num_heads)
        offset += norms_bytes

        k_packed_list.append(kp)
        k_norms_list.append(kn)
        v_packed_list.append(vp)
        v_norms_list.append(vn)

    k_packed = torch.from_numpy(np.stack(k_packed_list)).to(device)
    k_norms = torch.from_numpy(np.stack(k_norms_list)).to(device)
    v_packed = torch.from_numpy(np.stack(v_packed_list)).to(device)
    v_norms = torch.from_numpy(np.stack(v_norms_list)).to(device)

    return k_packed, k_norms, v_packed, v_norms, meta


def mmap_tqkv(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Memory-map a .tqkv file for zero-copy cross-process sharing.

    Returns numpy arrays backed by the mmap. Read-only.
    """
    data = np.memmap(path, dtype=np.uint8, mode="r")
    meta = _unpack_header(bytes(data[:HEADER_SIZE]))

    bits = meta["bits"]
    head_dim = meta["head_dim"]
    num_heads = meta["num_heads"]
    block_size = meta["block_size"]
    num_blocks = meta["num_blocks"]
    pd = packed_size(head_dim, bits)

    packed_bytes = block_size * num_heads * pd
    norms_bytes = block_size * num_heads * 4
    block_data_size = 2 * (packed_bytes + norms_bytes)

    body = data[HEADER_SIZE:HEADER_SIZE + num_blocks * block_data_size]
    body = body.reshape(num_blocks, block_data_size)

    # Split each block's data into K packed, K norms, V packed, V norms
    off = 0
    k_packed = body[:, off:off + packed_bytes].reshape(
        num_blocks, block_size, num_heads, pd)
    off += packed_bytes
    k_norms = body[:, off:off + norms_bytes].view(np.float32).reshape(
        num_blocks, block_size, num_heads)
    off += norms_bytes
    v_packed = body[:, off:off + packed_bytes].reshape(
        num_blocks, block_size, num_heads, pd)
    off += packed_bytes
    v_norms = body[:, off:off + norms_bytes].view(np.float32).reshape(
        num_blocks, block_size, num_heads)

    return k_packed, k_norms, v_packed, v_norms, meta
