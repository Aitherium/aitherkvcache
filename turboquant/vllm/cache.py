"""
TurboQuant GPU and cold tier caches for vLLM.

TQGPUCache: Per-layer TQ-compressed KV storage on VRAM with DDR5 cold tier.
ColdTierCache: Async background thread CPU cold tier (Phase 1 fallback).
"""

import logging
import os
import threading
from collections import deque
from typing import Optional, Set

import torch

logger = logging.getLogger("turboquant.vllm.cache")


# ================================================================
# TQ GPU CACHE — Phase 2 VRAM-resident compressed KV storage
# ================================================================

class TQGPUCache:
    """
    GPU-resident TQ-compressed KV cache with DDR5 cold tier.

    Stores packed uint8 indices + float32 norms per block/position/head.
    The fused Triton kernel reads directly from this storage.

    Memory per token (TQ4, 8 heads, 128 dim, 32 layers):
      2 * 32 * 8 * (64 + 4) = 34,816 bytes = 34 KB
      vs FP8: 2 * 32 * 8 * 128 = 65,536 bytes = 64 KB
      Savings: 1.88x
    """

    def __init__(self, num_layers: int, max_blocks: int, block_size: int,
                 num_kv_heads: int, head_dim: int, bits: int = 4,
                 device: str = "cuda"):
        from turboquant.quantizer import TurboQuant
        from turboquant.packing import packed_size

        self.tq = TurboQuant(head_dim=head_dim, bits=bits, device=device)
        self.num_layers = num_layers
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.packed_dim = packed_size(head_dim, bits)

        # VRAM: contiguous 5D tensors — enables bulk spill/warm
        self._k_packed = torch.zeros(
            num_layers, max_blocks, block_size, num_kv_heads, self.packed_dim,
            dtype=torch.uint8, device=device)
        self._k_norms = torch.zeros(
            num_layers, max_blocks, block_size, num_kv_heads,
            dtype=torch.float32, device=device)
        self._v_packed = torch.zeros(
            num_layers, max_blocks, block_size, num_kv_heads, self.packed_dim,
            dtype=torch.uint8, device=device)
        self._v_norms = torch.zeros(
            num_layers, max_blocks, block_size, num_kv_heads,
            dtype=torch.float32, device=device)

        # Per-layer views (zero-copy, used by fused kernel and encode_and_store)
        self.k_packed = [self._k_packed[li] for li in range(num_layers)]
        self.k_norms = [self._k_norms[li] for li in range(num_layers)]
        self.v_packed = [self._v_packed[li] for li in range(num_layers)]
        self.v_norms = [self._v_norms[li] for li in range(num_layers)]

        # DDR5 cold tier: contiguous 5D tensors in pinned memory
        # Probe pinned memory support (WSL2 Docker segfaults on pin_memory=True)
        try:
            _can_pin = torch.cuda.is_available() and torch.zeros(1).pin_memory().is_pinned()
        except Exception:
            _can_pin = False
        self._cold_k_packed = torch.zeros(
            num_layers, max_blocks, block_size, num_kv_heads, self.packed_dim,
            dtype=torch.uint8, pin_memory=_can_pin)
        self._cold_k_norms = torch.zeros(
            num_layers, max_blocks, block_size, num_kv_heads,
            dtype=torch.float32, pin_memory=_can_pin)
        self._cold_v_packed = torch.zeros(
            num_layers, max_blocks, block_size, num_kv_heads, self.packed_dim,
            dtype=torch.uint8, pin_memory=_can_pin)
        self._cold_v_norms = torch.zeros(
            num_layers, max_blocks, block_size, num_kv_heads,
            dtype=torch.float32, pin_memory=_can_pin)
        self._cold_valid = torch.zeros(max_blocks, dtype=torch.bool)
        self._spilled_set: Set[int] = set()

        compressed_mb = (
            num_layers * 2 * max_blocks * block_size * num_kv_heads
            * (self.packed_dim + 4) / (1024 ** 2)
        )
        logger.info(
            "TQGPUCache: %d layers x %d blocks @ TQ%d, %.0f MB VRAM + %.0f MB DDR5",
            num_layers, max_blocks, bits, compressed_mb, compressed_mb,
        )

    def encode_and_store(self, layer_idx: int, key: torch.Tensor,
                         value: torch.Tensor, slot_mapping: torch.Tensor):
        """TQ-encode new K,V tokens and scatter into compressed storage."""
        valid = slot_mapping >= 0
        if not valid.any():
            return

        vs = slot_mapping[valid]
        vk = key[valid]
        vv = value[valid]

        N, H, D = vk.shape
        pd = self.packed_dim

        kp, kn = self.tq.encode(vk.reshape(N * H, D))
        vp, vn = self.tq.encode(vv.reshape(N * H, D))

        bi = vs // self.block_size
        oi = vs % self.block_size

        self.k_packed[layer_idx][bi, oi] = kp.reshape(N, H, pd)
        self.k_norms[layer_idx][bi, oi] = kn.reshape(N, H)
        self.v_packed[layer_idx][bi, oi] = vp.reshape(N, H, pd)
        self.v_norms[layer_idx][bi, oi] = vn.reshape(N, H)

    def spill_blocks(self, block_indices: torch.Tensor):
        """Spill TQ-compressed blocks from VRAM to DDR5 cold tier."""
        if block_indices.numel() == 0:
            return
        bi = block_indices.long()
        self._cold_k_packed[:, bi] = self._k_packed[:, bi].cpu()
        self._cold_k_norms[:, bi] = self._k_norms[:, bi].cpu()
        self._cold_v_packed[:, bi] = self._v_packed[:, bi].cpu()
        self._cold_v_norms[:, bi] = self._v_norms[:, bi].cpu()
        self._cold_valid[bi] = True
        self._spilled_set.update(bi.tolist())

    def warm_blocks(self, block_indices: torch.Tensor):
        """Warm TQ-compressed blocks from DDR5 cold tier back to VRAM."""
        if block_indices.numel() == 0:
            return
        bi = block_indices.long()
        valid = self._cold_valid[bi]
        if not valid.any():
            return
        valid_bi = bi[valid]
        device = self._k_packed.device
        self._k_packed[:, valid_bi] = self._cold_k_packed[:, valid_bi].to(
            device, non_blocking=True)
        self._k_norms[:, valid_bi] = self._cold_k_norms[:, valid_bi].to(
            device, non_blocking=True)
        self._v_packed[:, valid_bi] = self._cold_v_packed[:, valid_bi].to(
            device, non_blocking=True)
        self._v_norms[:, valid_bi] = self._cold_v_norms[:, valid_bi].to(
            device, non_blocking=True)

    def cold_tier_stats(self) -> dict:
        """Report cold tier usage."""
        valid_blocks = self._cold_valid.sum().item()
        per_block_bytes = (
            2 * self.block_size * self.num_kv_heads
            * (self.packed_dim + 4) * self.num_layers
        )
        return {
            "cold_blocks": valid_blocks,
            "cold_tokens": valid_blocks * self.block_size,
            "cold_mb": valid_blocks * per_block_bytes / (1024 ** 2),
            "max_blocks": self.max_blocks,
        }

    def has_spilled(self, block_idx: int) -> bool:
        return block_idx in self._spilled_set

    def clear_spilled(self, block_indices: list) -> None:
        self._spilled_set.difference_update(block_indices)


# ================================================================
# COLD TIER CACHE — Phase 1 async CPU cold tier
# ================================================================

_SHARED_COLD_TIER: Optional["ColdTierCache"] = None
_LOCK = threading.Lock()


def get_shared_cold_tier(head_dim: int, num_kv_heads: int, bits: int = 4):
    """Get or create the singleton cold tier cache."""
    global _SHARED_COLD_TIER
    with _LOCK:
        if _SHARED_COLD_TIER is None:
            _SHARED_COLD_TIER = ColdTierCache(
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                bits=bits,
            )
        return _SHARED_COLD_TIER


class ColdTierCache:
    """
    Async TQ-compressed cold tier on CPU.

    compress_async() returns immediately — queues raw GPU tensor refs.
    Background thread: waits for CUDA event, copies to CPU, TQ-encodes, stores.
    """

    def __init__(self, head_dim: int, num_kv_heads: int, bits: int = 4,
                 max_blocks: int = 2048, block_size: int = 16):
        from turboquant import TurboQuant

        self.tq = TurboQuant(head_dim=head_dim, bits=bits, device="cpu")
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.bits = bits
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.packed_dim = self.tq._packed_dim

        # Per-layer storage (allocated on register_layer)
        self._layers = []
        self._layer_lock = threading.Lock()

        # Async pipeline
        self._queue = deque()
        self._queue_lock = threading.Lock()
        self._running = True
        self._copy_stream = None
        self._worker = threading.Thread(target=self._encode_loop, daemon=True)
        self._worker.start()

        # Stats
        self.tokens_encoded = 0

        logger.info(
            "ColdTierCache: %d-bit, max %d blocks, head_dim=%d, kv_heads=%d",
            bits, max_blocks, head_dim, num_kv_heads,
        )

    def register_layer(self) -> int:
        """Register a new layer. Returns layer index."""
        with self._layer_lock:
            idx = len(self._layers)
            self._layers.append({
                "packed_K": torch.zeros(
                    self.max_blocks, self.block_size, self.num_kv_heads,
                    self.packed_dim, dtype=torch.uint8),
                "norms_K": torch.zeros(
                    self.max_blocks, self.block_size, self.num_kv_heads,
                    dtype=torch.float32),
                "packed_V": torch.zeros(
                    self.max_blocks, self.block_size, self.num_kv_heads,
                    self.packed_dim, dtype=torch.uint8),
                "norms_V": torch.zeros(
                    self.max_blocks, self.block_size, self.num_kv_heads,
                    dtype=torch.float32),
                "valid": torch.zeros(self.max_blocks, dtype=torch.bool),
            })
            return idx

    def compress_async(self, layer_idx: int, key: torch.Tensor,
                       value: torch.Tensor, slot_mapping: torch.Tensor):
        """Queue K,V for background TQ encoding. Returns immediately."""
        try:
            if self._copy_stream is None and key.is_cuda:
                self._copy_stream = torch.cuda.Stream(device=key.device)

            event = None
            if self._copy_stream is not None:
                event = torch.cuda.Event()
                event.record()

            with self._queue_lock:
                self._queue.append((
                    layer_idx,
                    key.detach(),
                    value.detach(),
                    slot_mapping.detach(),
                    event,
                ))
        except Exception:
            pass

    def _encode_loop(self):
        """Background: GPU->CPU copy + TQ encode."""
        while self._running:
            item = None
            with self._queue_lock:
                if self._queue:
                    item = self._queue.popleft()

            if item is None:
                threading.Event().wait(0.005)
                continue

            layer_idx, key_gpu, value_gpu, slot_gpu, event = item

            try:
                if event is not None:
                    event.synchronize()

                if self._copy_stream is not None:
                    with torch.cuda.stream(self._copy_stream):
                        valid = slot_gpu >= 0
                        vk = key_gpu[valid].to("cpu", non_blocking=True)
                        vv = value_gpu[valid].to("cpu", non_blocking=True)
                        vs = slot_gpu[valid].to("cpu", non_blocking=True)
                    self._copy_stream.synchronize()
                else:
                    valid = slot_gpu >= 0
                    vk = key_gpu[valid].cpu()
                    vv = value_gpu[valid].cpu()
                    vs = slot_gpu[valid].cpu()

                if vk.shape[0] == 0:
                    continue

                N = vk.shape[0]
                H = self.num_kv_heads
                D = self.head_dim
                pd = self.packed_dim
                S = self.block_size

                kp, kn = self.tq.encode(vk.reshape(N * H, D).contiguous())
                vp, vn = self.tq.encode(vv.reshape(N * H, D).contiguous())

                bi = vs // S
                oi = vs % S

                layer = self._layers[layer_idx]
                layer["packed_K"][bi, oi] = kp.reshape(N, H, pd)
                layer["norms_K"][bi, oi] = kn.reshape(N, H)
                layer["packed_V"][bi, oi] = vp.reshape(N, H, pd)
                layer["norms_V"][bi, oi] = vn.reshape(N, H)
                layer["valid"].scatter_(0, bi, True)

                self.tokens_encoded += N

            except Exception as e:
                logger.debug("Cold tier encode error (L%d): %s", layer_idx, e)

    def decompress_blocks(self, layer_idx: int, block_indices: torch.Tensor):
        """Decompress blocks from cold tier. Returns (keys, values) float16 on CPU."""
        bi = block_indices.cpu()
        NB = bi.shape[0]
        S = self.block_size
        H = self.num_kv_heads
        pd = self.packed_dim

        layer = self._layers[layer_idx]
        pk = layer["packed_K"][bi].reshape(NB * S * H, pd)
        nk = layer["norms_K"][bi].reshape(NB * S * H)
        pv = layer["packed_V"][bi].reshape(NB * S * H, pd)
        nv = layer["norms_V"][bi].reshape(NB * S * H)

        dk = self.tq.decode(pk, nk).reshape(NB, S, H, self.head_dim)
        dv = self.tq.decode(pv, nv).reshape(NB, S, H, self.head_dim)
        return dk, dv

    def shutdown(self):
        self._running = False
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
