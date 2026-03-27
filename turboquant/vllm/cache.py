"""
Async TQ cold tier cache for vLLM.

Singleton shared across all attention layers. Background thread handles
GPU->CPU transfer + TQ encoding. Zero sync on the attention hot path.
"""

import logging
import os
import threading
from collections import deque
from typing import Optional

import torch

logger = logging.getLogger("turboquant.vllm.cache")

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
            # Record main-stream event so background can wait
            if self._copy_stream is None and key.is_cuda:
                self._copy_stream = torch.cuda.Stream(device=key.device)

            event = None
            if self._copy_stream is not None:
                event = torch.cuda.Event()
                event.record()  # on current (main) stream

            with self._queue_lock:
                self._queue.append((
                    layer_idx,
                    key.detach(),
                    value.detach(),
                    slot_mapping.detach(),
                    event,
                ))
        except Exception:
            pass  # never block hot path

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

                # GPU -> CPU on copy stream
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

                # TQ encode
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
