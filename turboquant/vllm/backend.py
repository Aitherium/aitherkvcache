"""
TurboQuant Custom Attention Backend for vLLM v0.15+.

Registers via vLLM's official plugin system -- no monkey-patching.

Usage:
    # In vllm serve CLI (only mechanism in vLLM v0.15.1 V1 engine):
    vllm serve model --attention-backend CUSTOM

    # Before vLLM starts (e.g., in tq_sitecustomize.py):
    from vllm.v1.attention.backends.registry import (
        register_backend, AttentionBackendEnum,
    )
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "lib.gpu.turboquant.vllm_custom_backend.TurboQuantBackend",
    )

Architecture:
  Standard path = TritonAttentionImpl.forward() (identical, zero risk).
  TQ additions layered on top:
    - Encode new K,V to TQGPUCache after each forward
    - Fused TQ decode via TQPagedAttention (single-token decode)
  Cold tier (DDR5):
    - Async spill/warm via TQGPUCache.spill_blocks()/warm_blocks()

  PRIMARY mode (AITHER_TQ_PRIMARY=1):
    - The kv_cache tensor IS the TQ cache (uint8 layout from reshape patch)
    - forward() writes TQ-encoded K,V directly into the vLLM cache tensor
    - Decode: fused TQPagedAttention reads packed+norms directly (no decompress)
    - Prefill: decompress TQ blocks into temp FP16 buffer, run standard attention
    - No separate TQGPUCache -- the vLLM cache IS the compressed cache

ALL vLLM imports are deferred via module __getattr__ to avoid circular
imports during sitecustomize (vllm.config partially initialized).
vLLM resolves "TurboQuantBackend" via getattr(module, name) which
triggers lazy creation only after all vLLM modules are loaded.
"""

import os
import logging
import time
from typing import ClassVar, Optional, Set

import torch

logger = logging.getLogger("aither.turboquant.backend")


# ================================================================
# TQ GPU CACHE -- per-layer compressed KV storage on VRAM
# ================================================================

class TQGPUCache:
    """
    Shared GPU-resident TQ-compressed KV cache across all layers.

    Supports two modes:
      - "tq4"/"tq3"/"tq2": uniform bit-width, separate packed+norms tensors
      - "tq35"/"tq25": hybrid bit-width, unified packed tensor (norms embedded)

    The fused Triton decode kernel reads directly from these tensors.
    """

    def __init__(self, num_layers: int, max_blocks: int, block_size: int,
                 num_kv_heads: int, head_dim: int, bits: int = 4,
                 device: str = "cuda", mode: str = ""):
        self.num_layers = num_layers
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Determine mode: explicit mode param, or infer from bits
        self.mode = mode if mode else f"tq{bits}"
        self.is_hybrid = self.mode in ("tq35", "tq25")

        if self.is_hybrid:
            from .hybrid_quantizer import HybridTurboQuant
            self.htq = HybridTurboQuant(
                head_dim=head_dim, mode=self.mode, device=device)
            self.htq.calibrate_uniform()
            self.tq = None  # not used in hybrid mode
            self.packed_dim = self.htq.packed_dim
        else:
            from .quantizer import TurboQuant
            from .packing import packed_size
            self.tq = TurboQuant(head_dim=head_dim, bits=bits, device=device)
            self.htq = None
            self.packed_dim = packed_size(head_dim, bits)

        # GPU-resident packed cache (both modes use k_packed/v_packed)
        self._k_packed = torch.zeros(
            num_layers, max_blocks, block_size, num_kv_heads, self.packed_dim,
            dtype=torch.uint8, device=device)
        self._v_packed = torch.zeros(
            num_layers, max_blocks, block_size, num_kv_heads, self.packed_dim,
            dtype=torch.uint8, device=device)

        # Separate norms tensors (uniform mode only -- hybrid embeds norms in packed)
        if not self.is_hybrid:
            self._k_norms = torch.zeros(
                num_layers, max_blocks, block_size, num_kv_heads,
                dtype=torch.float32, device=device)
            self._v_norms = torch.zeros(
                num_layers, max_blocks, block_size, num_kv_heads,
                dtype=torch.float32, device=device)
        else:
            self._k_norms = None
            self._v_norms = None

        self.k_packed = [self._k_packed[li] for li in range(num_layers)]
        self.v_packed = [self._v_packed[li] for li in range(num_layers)]
        if not self.is_hybrid:
            self.k_norms = [self._k_norms[li] for li in range(num_layers)]
            self.v_norms = [self._v_norms[li] for li in range(num_layers)]
        else:
            self.k_norms = None
            self.v_norms = None

        # Detect WSL2 (even from inside Docker) -- pinned memory segfaults.
        try:
            _can_pin = torch.cuda.is_available() and torch.zeros(1).pin_memory().is_pinned()
        except Exception:
            _can_pin = False

        # Cold tier (DDR5)
        self._cold_k_packed = torch.zeros(
            num_layers, max_blocks, block_size, num_kv_heads, self.packed_dim,
            dtype=torch.uint8, pin_memory=_can_pin)
        self._cold_v_packed = torch.zeros(
            num_layers, max_blocks, block_size, num_kv_heads, self.packed_dim,
            dtype=torch.uint8, pin_memory=_can_pin)
        if not self.is_hybrid:
            self._cold_k_norms = torch.zeros(
                num_layers, max_blocks, block_size, num_kv_heads,
                dtype=torch.float32, pin_memory=_can_pin)
            self._cold_v_norms = torch.zeros(
                num_layers, max_blocks, block_size, num_kv_heads,
                dtype=torch.float32, pin_memory=_can_pin)
        else:
            self._cold_k_norms = None
            self._cold_v_norms = None
        self._cold_valid = torch.zeros(max_blocks, dtype=torch.bool)
        self._spilled_set: Set[int] = set()

        if self.is_hybrid:
            per_vec = self.packed_dim  # norms embedded
        else:
            per_vec = self.packed_dim + 4  # +4 for f32 norm
        compressed_mb = (
            num_layers * 2 * max_blocks * block_size * num_kv_heads
            * per_vec / (1024 ** 2)
        )
        logger.info(
            "TQGPUCache: %d layers x %d blocks @ %s, %.0f MB VRAM + %.0f MB DDR5",
            num_layers, max_blocks, self.mode.upper(), compressed_mb, compressed_mb,
        )

    def encode_and_store(self, layer_idx: int, key: torch.Tensor,
                         value: torch.Tensor, slot_mapping: torch.Tensor):
        # Guard: slot_mapping must align with key's token dimension
        if slot_mapping.shape[0] != key.shape[0]:
            return
        valid = slot_mapping >= 0
        if not valid.any():
            return
        vs = slot_mapping[valid]
        bi = vs // self.block_size
        # Skip tokens whose blocks exceed TQ cache capacity
        in_range = bi < self.max_blocks
        if not in_range.any():
            return
        vs = vs[in_range]
        bi = bi[in_range]
        oi = vs % self.block_size
        # Force contiguous -- vLLM's masked indexing returns strided views
        vk = key[valid][in_range].contiguous()
        vv = value[valid][in_range].contiguous()
        N, H, D = vk.shape
        pd = self.packed_dim

        if self.is_hybrid:
            # Hybrid: single packed tensor with norms embedded
            kp = self.htq.encode(vk.reshape(N * H, D)).reshape(N, H, pd)
            vp = self.htq.encode(vv.reshape(N * H, D)).reshape(N, H, pd)
            self.k_packed[layer_idx][bi, oi] = kp
            self.v_packed[layer_idx][bi, oi] = vp
        else:
            kp, kn = self.tq.encode(vk.reshape(N * H, D))
            vp, vn = self.tq.encode(vv.reshape(N * H, D))
            self.k_packed[layer_idx][bi, oi] = kp.reshape(N, H, pd)
            self.k_norms[layer_idx][bi, oi] = kn.reshape(N, H)
            self.v_packed[layer_idx][bi, oi] = vp.reshape(N, H, pd)
            self.v_norms[layer_idx][bi, oi] = vn.reshape(N, H)

    def spill_blocks(self, block_indices: torch.Tensor):
        if block_indices.numel() == 0:
            return
        bi = block_indices.long()
        self._cold_k_packed[:, bi] = self._k_packed[:, bi].cpu()
        self._cold_v_packed[:, bi] = self._v_packed[:, bi].cpu()
        if not self.is_hybrid:
            self._cold_k_norms[:, bi] = self._k_norms[:, bi].cpu()
            self._cold_v_norms[:, bi] = self._v_norms[:, bi].cpu()
        self._cold_valid[bi] = True
        self._spilled_set.update(bi.tolist())

    def warm_blocks(self, block_indices: torch.Tensor):
        if block_indices.numel() == 0:
            return
        bi = block_indices.long()
        valid = self._cold_valid[bi]
        if not valid.any():
            return
        valid_bi = bi[valid]
        device = self._k_packed.device
        self._k_packed[:, valid_bi] = self._cold_k_packed[:, valid_bi].to(device, non_blocking=True)
        self._v_packed[:, valid_bi] = self._cold_v_packed[:, valid_bi].to(device, non_blocking=True)
        if not self.is_hybrid:
            self._k_norms[:, valid_bi] = self._cold_k_norms[:, valid_bi].to(device, non_blocking=True)
            self._v_norms[:, valid_bi] = self._cold_v_norms[:, valid_bi].to(device, non_blocking=True)

    def cold_tier_stats(self) -> dict:
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
# LAZY CLASS CREATION via module __getattr__
# ================================================================

def __getattr__(name):
    """Lazily create vLLM backend classes on first access.

    vLLM resolves 'TurboQuantBackend' via getattr(module, name).
    By deferring creation, we avoid importing vLLM during sitecustomize
    when vllm.config is partially initialized (circular import).
    """
    if name == "TurboQuantBackend":
        cls = _make_backend_class()
        globals()["TurboQuantBackend"] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _make_backend_class():
    from vllm.v1.attention.backend import AttentionBackend

    class TurboQuantBackend(AttentionBackend):
        supported_kv_cache_dtypes = [
            "auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2",
        ]
        # Tell vLLM Attention layer to pre-allocate output buffer for us
        accept_output_buffer = True
        # KV cache update is handled by do_kv_cache_update() on the impl,
        # called via vLLM's unified_kv_cache_update splitting op.
        # This preserves the standard graph break structure for CUDA graphs.
        forward_includes_kv_cache_update = False

        @staticmethod
        def get_name() -> str:
            return "CUSTOM"

        @staticmethod
        def get_impl_cls():
            return _get_impl_class()

        @staticmethod
        def get_builder_cls():
            from vllm.v1.attention.backends.triton_attn import (
                TritonAttentionMetadataBuilder,
            )
            from vllm.v1.attention.backend import AttentionCGSupport

            class TQMetadataBuilder(TritonAttentionMetadataBuilder):
                # TQ attention uses custom Triton kernels with Python-level
                # pre/post processing (rotation matmul, even/odd split,
                # interleave, inverse rotation) that cannot be replayed
                # from a captured CUDA graph.  Returning NEVER tells vLLM
                # to create graph breaks around attention layers so that
                # torch.compile + piecewise CUDA graphs still capture
                # MLP / norm / embedding while attention runs eagerly.
                _cudagraph_support = AttentionCGSupport.NEVER

            return TQMetadataBuilder

        @staticmethod
        def get_kv_cache_shape(num_blocks, block_size, num_kv_heads,
                               head_size, cache_dtype_str="auto"):
            return (num_blocks, 2, block_size, num_kv_heads, head_size)

        @staticmethod
        def get_supported_head_sizes():
            return [64, 96, 128, 256]

    return TurboQuantBackend


_impl_class = None


def _get_impl_class():
    global _impl_class
    if _impl_class is None:
        _impl_class = _make_impl_class()
    return _impl_class


def _make_impl_class():
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    class TurboQuantImpl(TritonAttentionImpl):
        """TritonAttentionImpl subclass that adds TQ encode after each forward.

        Two operating modes:

        SHADOW (default): parent handles all attention using FP8 cache.
            TQ encodes a compressed shadow copy for cold-tier spill and
            optional fused decode. CUDA graphs work normally.

        PRIMARY (AITHER_TQ_PRIMARY=1): the vLLM kv_cache tensor IS the TQ
            cache (uint8 layout).  forward() writes TQ-encoded K,V directly
            into the cache. Decode: fused TQPagedAttention kernel reads
            packed uint8 + f32 norms directly (no decompression buffer).
            Fallback: batch-decompress → unified_attention if fused kernel
            fails. Prefill: decompress → standard attention (amortized).
        """

        _tq_gpu_cache: ClassVar[Optional[TQGPUCache]] = None
        _tq_layer_counter: ClassVar[int] = 0
        _fused_enabled: ClassVar[bool] = (
            os.environ.get("AITHER_TQ_FUSED", "0") == "1"
        )
        _registered_blocks: ClassVar[Set[int]] = set()
        _forward_count: ClassVar[int] = 0
        _INIT_AFTER_FORWARDS: ClassVar[int] = 2
        _MIN_BLOCKS_FOR_INIT: ClassVar[int] = 16

        # --- PRIMARY mode class vars ---
        # Derive from AITHER_TQ_MODE (canonical) or AITHER_TQ_PRIMARY (legacy)
        _tq_mode_str: ClassVar[str] = os.environ.get("AITHER_TQ_MODE", "")
        _tq_primary: ClassVar[bool] = (
            _tq_mode_str.endswith("-primary") if _tq_mode_str
            else os.environ.get("AITHER_TQ_PRIMARY", "0") == "1"
        )
        _prefill_k_buf: ClassVar[Optional[torch.Tensor]] = None
        _prefill_v_buf: ClassVar[Optional[torch.Tensor]] = None
        _PREFILL_BUF_BLOCKS: ClassVar[int] = 512  # enough for ~8K token prefill
        _tq_quantizer: ClassVar[Optional[object]] = None
        # PRIMARY mode: separate float32 norm tensors (avoids .contiguous().view()
        # on the uint8 cache which copies 10+ MB per layer per forward)
        _primary_k_norms: ClassVar[Optional[torch.Tensor]] = None
        _primary_v_norms: ClassVar[Optional[torch.Tensor]] = None

        # Decode decompression buffers (reused across calls, allocated once)
        # Shape: [num_blocks, block_size, num_kv_heads, head_dim]
        _decode_k_buf: ClassVar[Optional[torch.Tensor]] = None
        _decode_v_buf: ClassVar[Optional[torch.Tensor]] = None

        # Block selector for sparse attention (DynSplit-KV inspired)
        _block_selector: ClassVar[Optional[object]] = None
        _block_select_ratio: ClassVar[float] = float(
            os.environ.get("AITHER_TQ_BLOCK_SELECT_RATIO", "1.0"))
        _block_select_min: ClassVar[int] = int(
            os.environ.get("AITHER_TQ_BLOCK_SELECT_MIN_BLOCKS", "64"))
        _block_select_sink: ClassVar[int] = int(
            os.environ.get("AITHER_TQ_BLOCK_SELECT_SINK", "2"))
        _block_select_recent: ClassVar[int] = int(
            os.environ.get("AITHER_TQ_BLOCK_SELECT_RECENT", "2"))

        # Speculative decoding: detect draft model layers by head_size mismatch.
        # First layer sets the expected target model head_size. Draft model
        # layers (different head_size) skip all TQ logic — pure passthrough.
        _target_head_size: ClassVar[Optional[int]] = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Detect draft model layers (speculative decoding)
            if TurboQuantImpl._target_head_size is None:
                TurboQuantImpl._target_head_size = self.head_size
            self._is_draft = (self.head_size != TurboQuantImpl._target_head_size)

            if self._is_draft:
                self._tq_layer_idx = -1
                self._fused_attn = None
                logger.info(
                    "TurboQuantImpl[draft]: heads=%d, kv=%d, dim=%d — TQ SKIP",
                    self.num_heads, self.num_kv_heads, self.head_size)
                return

            self._tq_layer_idx = TurboQuantImpl._tq_layer_counter
            TurboQuantImpl._tq_layer_counter += 1
            self._fused_attn = None
            if self._tq_primary:
                mode = "PRIMARY"
                if self._fused_enabled:
                    mode += "+fused"
            elif self._fused_enabled:
                mode = "fused+standard"
            else:
                mode = "standard"
            logger.info(
                "TurboQuantImpl[L%d]: heads=%d, kv=%d, dim=%d, mode=%s",
                self._tq_layer_idx, self.num_heads, self.num_kv_heads,
                self.head_size, mode,
            )

        @torch.compiler.disable
        def do_kv_cache_update(self, layer, key, value, kv_cache,
                               slot_mapping):
            """TQ-aware KV cache update, called via unified_kv_cache_update.

            This runs as a graph-break splitting op so the CUDA graph
            structure matches vLLM's expectations (same break points as
            standard attention backends).
            """
            if self._is_draft or key is None or value is None:
                return

            if not self._tq_primary:
                # SHADOW mode: parent handles standard KV write
                return

            packed_dim = self.head_size // 2
            key_cache = kv_cache[:, 0]
            value_cache = kv_cache[:, 1]
            num_blocks = key_cache.shape[0]
            block_size = key_cache.shape[1]
            num_layers = TurboQuantImpl._tq_layer_counter

            # Lazy-init norm tensors (once)
            if TurboQuantImpl._primary_k_norms is None:
                TurboQuantImpl._primary_k_norms = torch.zeros(
                    num_layers, num_blocks, block_size, self.num_kv_heads,
                    dtype=torch.float32, device=kv_cache.device)
                TurboQuantImpl._primary_v_norms = torch.zeros(
                    num_layers, num_blocks, block_size, self.num_kv_heads,
                    dtype=torch.float32, device=kv_cache.device)
                logger.info("[TQ] Primary norms: [%d layers, %d blocks, bs=%d, "
                            "%d heads] (%.1f MB)", num_layers, num_blocks,
                            block_size, self.num_kv_heads,
                            num_layers * num_blocks * block_size
                            * self.num_kv_heads * 4 * 2 / (1024 ** 2))

            try:
                self._tq_write_primary(
                    key, value, key_cache, value_cache,
                    slot_mapping, packed_dim)
            except Exception as e:
                import sys
                print(f"[TQ] do_kv_cache_update error L{self._tq_layer_idx}: "
                      f"{e}", file=sys.stderr, flush=True)

        @torch.compiler.disable
        def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                    output=None, output_scale=None, output_block_scale=None):
            # Draft model layers (speculative decoding): pure passthrough to
            # parent TritonAttentionImpl — no TQ encode, no fused decode.
            if self._is_draft:
                return super().forward(
                    layer, query, key, value, kv_cache, attn_metadata,
                    output=output, output_scale=output_scale,
                    output_block_scale=output_block_scale)

            if attn_metadata is None:
                # Profiling run -- output must exist and be zeroed
                if output is not None:
                    output.fill_(0)
                    return output
                return super().forward(
                    layer, query, key, value, kv_cache, attn_metadata,
                    output=output, output_scale=output_scale,
                    output_block_scale=output_block_scale,
                )

            assert output is not None

            # ---- PRIMARY mode: TQ IS the KV cache ----
            if self._tq_primary:
                return self._forward_primary(
                    layer, query, key, value, kv_cache, attn_metadata,
                    output, output_scale)

            # ---- SHADOW mode: unchanged from original ----
            if self._tq_layer_idx == 0:
                TurboQuantImpl._forward_count += 1

            # Initialize TQ GPU cache (deferred past profiling warmup)
            self._try_init_tq_cache(kv_cache)

            # Encode new K,V to TQ compressed GPU storage
            tq_cache = TurboQuantImpl._tq_gpu_cache
            if tq_cache is not None and key is not None:
                try:
                    if tq_cache.is_hybrid:
                        # Hybrid mode: Python encode path (no fused kernel yet)
                        tq_cache.encode_and_store(
                            self._tq_layer_idx, key, value,
                            attn_metadata.slot_mapping,
                        )
                    else:
                        # Uniform mode: fused Triton kernel
                        from .fused_kv_update import fused_encode_and_store
                        fused_encode_and_store(
                            tq_cache, self._tq_layer_idx, key, value,
                            attn_metadata.slot_mapping,
                        )
                    if self._tq_layer_idx == 0:
                        self._register_new_blocks(
                            attn_metadata.slot_mapping, tq_cache)
                except Exception as e:
                    import sys
                    print(f"[TQ] encode error L{self._tq_layer_idx}: {e}",
                          file=sys.stderr, flush=True)

            # -- Decode path: fused TQ kernel (single-token generation) --
            is_decode = (
                self._fused_enabled
                and tq_cache is not None
                and hasattr(attn_metadata, "max_query_len")
                and attn_metadata.max_query_len == 1
            )

            if is_decode:
                try:
                    self._ensure_fused_attn()
                    if self._fused_attn is not None:
                        num_tokens = query.shape[0]
                        if output is None:
                            output = torch.empty(
                                num_tokens, self.num_heads, self.head_size,
                                dtype=query.dtype, device=query.device,
                            )
                        return self._fused_decode(
                            query, attn_metadata, output)
                except Exception as e:
                    import sys
                    print(f"[TQ] fused decode fallback L{self._tq_layer_idx}: {e}",
                          file=sys.stderr, flush=True)

            # -- Prefill / fallback: delegate to TritonAttentionImpl --
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata,
                output=output, output_scale=output_scale,
                output_block_scale=output_block_scale,
            )

        # ============================================================
        # SHADOW MODE helpers (unchanged)
        # ============================================================

        def _ensure_fused_attn(self):
            """Lazily create TQPagedAttention for fused decode.

            Supports both SHADOW mode (TQGPUCache.tq) and PRIMARY mode
            (_tq_quantizer). The fused Triton kernel reads packed uint8 +
            f32 norms directly — no decompression buffer needed.
            """
            if self._fused_attn is not None:
                return
            from .fused_attention import TQPagedAttention
            tq = None
            if TurboQuantImpl._tq_gpu_cache is not None:
                tq = TurboQuantImpl._tq_gpu_cache.tq
            elif TurboQuantImpl._tq_quantizer is not None:
                tq = TurboQuantImpl._tq_quantizer
            if tq is not None:
                self._fused_attn = TQPagedAttention(
                    tq, self.num_heads,
                    block_selector=TurboQuantImpl._block_selector)

        def _fused_decode(self, query, attn_metadata, output):
            """Run fused TQ attention for single-token decode."""
            tq_cache = TurboQuantImpl._tq_gpu_cache
            li = self._tq_layer_idx

            fused_out = self._fused_attn.forward(
                query=query,
                k_packed=tq_cache.k_packed[li],
                k_norms=tq_cache.k_norms[li],
                v_packed=tq_cache.v_packed[li],
                v_norms=tq_cache.v_norms[li],
                block_tables=attn_metadata.block_table,
                context_lens=attn_metadata.seq_lens,
                block_size=tq_cache.block_size,
                num_kv_heads=self.num_kv_heads,
            )

            output[:] = fused_out
            return output

        def _try_init_tq_cache(self, kv_cache):
            if TurboQuantImpl._tq_gpu_cache is not None:
                return
            # Defer past vLLM's profiling warmup (first N forwards are profiling)
            if TurboQuantImpl._forward_count < TurboQuantImpl._INIT_AFTER_FORWARDS:
                return
            if kv_cache.dim() != 5:
                return
            num_blocks = kv_cache.shape[0]
            block_size = kv_cache.shape[2]
            if num_blocks < TurboQuantImpl._MIN_BLOCKS_FOR_INIT:
                return

            num_layers = TurboQuantImpl._tq_layer_counter
            bits = int(os.environ.get("AITHER_TQ_BITS", "4"))
            tq_mode = os.environ.get("AITHER_TQ_MODE", "")

            # Determine packed_dim for VRAM budget calculation
            if tq_mode in ("tq35", "tq25"):
                from .hybrid_quantizer import HybridTurboQuant
                pd = HybridTurboQuant.packed_dim_for_mode(self.head_size, tq_mode)
                # Hybrid packs norms inside -- no separate +4 bytes
                bytes_per_block = 2 * num_layers * block_size * self.num_kv_heads * pd
            else:
                from .packing import packed_size
                pd = packed_size(self.head_size, bits)
                bytes_per_block = 2 * num_layers * block_size * self.num_kv_heads * (pd + 4)
            try:
                free_vram, total_vram = torch.cuda.mem_get_info()
                # Hard cap: 512 MB GPU allocation max. vLLM's memory pool doesn't
                # account for external CUDA allocs, so keep TQ footprint minimal.
                MAX_TQ_VRAM_BYTES = 512 * 1024 * 1024  # 512 MiB
                cap_from_vram = min(MAX_TQ_VRAM_BYTES, int(free_vram * 0.10))
                max_tq_blocks = max(32, cap_from_vram // bytes_per_block)
                tq_blocks = min(num_blocks, max_tq_blocks)
                est_mb = tq_blocks * bytes_per_block / (1024 * 1024)
                logger.info(
                    "[TQ] VRAM: %.0f MiB free / %.0f MiB total. "
                    "Budget: %.0f MiB (min of 512 MiB cap, 10%% free)",
                    free_vram / (1024**2), total_vram / (1024**2), est_mb,
                )
            except Exception:
                tq_blocks = min(num_blocks, 128)  # very safe fallback

            mode_label = tq_mode.upper() if tq_mode else f"TQ{bits}"
            logger.info(
                "[TQ] Allocating TQGPUCache: %dL x %dB (of %d) x bs=%d @ %s",
                num_layers, tq_blocks, num_blocks, block_size, mode_label,
            )

            try:
                TurboQuantImpl._tq_gpu_cache = TQGPUCache(
                    num_layers=num_layers,
                    max_blocks=tq_blocks,
                    block_size=block_size,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_size,
                    bits=bits,
                    device=str(kv_cache.device),
                    mode=tq_mode,
                )
            except Exception as e:
                import sys
                print(f"[TQ] TQGPUCache allocation FAILED ({tq_blocks} blocks): "
                      f"{e}", file=sys.stderr, flush=True)
                logger.error("[TQ] Cache allocation failed: %s", e)
                return

            try:
                from .tier_cache_bridge import get_tier_cache_bridge
                from .block_metadata import get_block_metadata_table
                bridge = get_tier_cache_bridge()
                bridge.configure(TurboQuantImpl._tq_gpu_cache)
                logger.info("TierCacheBridge configured (%d blocks, %d layers)",
                            num_blocks, num_layers)
                _ = get_block_metadata_table()
            except Exception as e:
                logger.debug("TierCacheBridge init: %s", e)

        @staticmethod
        def _register_new_blocks(slot_mapping, tq_cache):
            try:
                valid = slot_mapping[slot_mapping >= 0]
                if valid.numel() == 0:
                    return
                block_indices = (valid // tq_cache.block_size).unique().cpu().tolist()
                new_blocks = [b for b in block_indices
                              if b not in TurboQuantImpl._registered_blocks]
                if not new_blocks:
                    return
                from .block_metadata import get_block_metadata_table
                get_block_metadata_table().register_blocks(
                    block_indices=new_blocks, source_layer="kv_cache",
                    importance=0.5, token_range=(0, 0), tenant_slug="platform",
                )
                TurboQuantImpl._registered_blocks.update(new_blocks)
            except Exception:
                pass

        # ============================================================
        # PRIMARY MODE: TQ IS the KV cache
        # ============================================================

        def _forward_primary(self, layer, query, key, value, kv_cache,
                             attn_metadata, output, output_scale):
            """Full forward pass with TQ as the primary KV cache.

            The kv_cache tensor is uint8 with shape:
                [num_blocks, 2, block_size, num_kv_heads, tq_dim]
            where tq_dim = packed_dim + 4 (packed indices + f32 norm bytes).

            kv_cache[:, 0] = K cache, kv_cache[:, 1] = V cache.

            Decode path (routes through unified_attention for CUDA graphs):
              1. TQ encode new token -> primary uint8 cache (eager)
              2. Batch-decompress active blocks -> reusable FP16 buffer (eager)
              3. Call unified_attention with FP16 buffer (graphable split point)
            VRAM savings come from TQ-sized primary cache (309K tokens).
            The attention path is standard — vLLM graphs it normally.
            """
            packed_dim = self.head_size // 2  # 64 for head_dim=128 at 4-bit

            # --- Allocate separate float32 norm tensors (once, per-layer) ---
            key_cache = kv_cache[:, 0]    # [blocks, block_size, heads, tq_dim]
            value_cache = kv_cache[:, 1]
            num_blocks = key_cache.shape[0]
            block_size = key_cache.shape[1]
            num_layers = TurboQuantImpl._tq_layer_counter
            # Norms are initialized in do_kv_cache_update (called first).

            # Lazy-init block selector for sparse attention
            if (TurboQuantImpl._block_selector is None
                    and TurboQuantImpl._block_select_ratio < 1.0
                    and TurboQuantImpl._tq_quantizer is not None):
                from .block_selector import BlockSelector
                half_d = self.head_size // 2
                TurboQuantImpl._block_selector = BlockSelector(
                    max_blocks=num_blocks,
                    num_kv_heads=self.num_kv_heads,
                    half_d=half_d,
                    device=kv_cache.device,
                    centroids=TurboQuantImpl._tq_quantizer.centroids,
                    select_ratio=TurboQuantImpl._block_select_ratio,
                    num_sink_blocks=TurboQuantImpl._block_select_sink,
                    num_recent_blocks=TurboQuantImpl._block_select_recent,
                    min_blocks_for_selection=TurboQuantImpl._block_select_min,
                )
                logger.info(
                    "[TQ] Block selector: ratio=%.2f, max_selected=%d, "
                    "sink=%d, recent=%d, min_blocks=%d",
                    TurboQuantImpl._block_select_ratio,
                    TurboQuantImpl._block_selector.max_selected,
                    TurboQuantImpl._block_select_sink,
                    TurboQuantImpl._block_select_recent,
                    TurboQuantImpl._block_select_min)

            # KV write is now handled by do_kv_cache_update() (called via
            # unified_kv_cache_update splitting op before forward).

            # --- Compute attention ---
            is_decode = (
                hasattr(attn_metadata, "max_query_len")
                and attn_metadata.max_query_len == 1
            )

            if is_decode:
                # Decode: batch-decompress -> unified_attention
                return self._forward_primary_decode(
                    layer, query, key, value, kv_cache, attn_metadata,
                    output, output_scale)
            else:
                # Prefill: decompress -> standard attention
                return self._primary_prefill(
                    layer, query, key, value, key_cache, value_cache,
                    kv_cache, attn_metadata, output, output_scale, packed_dim)

        def _tq_write_primary(self, key, value, key_cache, value_cache,
                              slot_mapping, packed_dim):
            """Encode K,V as TQ4 and scatter-write to the primary cache.

            Args:
                key: [num_tokens, num_kv_heads, head_dim] new key vectors
                value: [num_tokens, num_kv_heads, head_dim] new value vectors
                key_cache: [num_blocks, block_size, num_kv_heads, tq_dim] uint8
                value_cache: same shape as key_cache
                slot_mapping: [num_tokens] int64 -- maps tokens to cache slots
                packed_dim: number of packed index bytes (head_dim // 2 for 4-bit)
            """
            # Get or lazily create TQ quantizer
            if TurboQuantImpl._tq_quantizer is None:
                from .quantizer import TurboQuant
                TurboQuantImpl._tq_quantizer = TurboQuant(
                    head_dim=self.head_size, bits=4,
                    device=str(key.device))

            tq = TurboQuantImpl._tq_quantizer

            # Guard: slot_mapping must align with key's token dimension
            if slot_mapping.shape[0] != key.shape[0]:
                return

            valid = slot_mapping >= 0
            if not valid.any():
                return

            vs = slot_mapping[valid]
            block_size = key_cache.shape[1]  # block_size from cache shape
            bi = vs // block_size
            oi = vs % block_size

            # Bounds check
            max_blocks = key_cache.shape[0]
            in_range = bi < max_blocks
            if not in_range.any():
                return
            vs = vs[in_range]
            bi = bi[in_range]
            oi = oi[in_range]

            vk = key[valid][in_range].contiguous()   # [N, H, D]
            vv = value[valid][in_range].contiguous()  # [N, H, D]
            N, H, D = vk.shape

            # Encode: TQ returns packed indices [N*H, packed_dim] and norms [N*H]
            kp, kn = tq.encode(vk.reshape(N * H, D))
            vp, vn = tq.encode(vv.reshape(N * H, D))

            kp = kp.reshape(N, H, packed_dim)
            vp = vp.reshape(N, H, packed_dim)

            # Write packed indices to uint8 cache
            key_cache[bi, oi, :, :packed_dim] = kp
            value_cache[bi, oi, :, :packed_dim] = vp

            # Write norms to per-layer float32 tensors (no byte reinterpretation)
            li = self._tq_layer_idx
            kn_f32 = kn.reshape(N, H).to(torch.float32)
            vn_f32 = vn.reshape(N, H).to(torch.float32)
            TurboQuantImpl._primary_k_norms[li, bi, oi] = kn_f32
            TurboQuantImpl._primary_v_norms[li, bi, oi] = vn_f32

            # Update block representatives for sparse attention (layer 0 only)
            if li == 0 and TurboQuantImpl._block_selector is not None:
                TurboQuantImpl._block_selector.update_reps(
                    bi, oi, kp, kn_f32)

        # ============================================================
        # PRIMARY DECODE: graphable Triton kernel (CUDA graph capturable)
        # ============================================================
        # NOT decorated with @torch.compiler.disable — this is the split
        # point that lets piecewise CUDA graphs capture the Triton kernel.
        # All tensor shapes are static during decode (fixed by vLLM's
        # padded batch), so graph capture succeeds.

        def _primary_decode_graphable(self, query, kv_cache, attn_metadata,
                                      output):
            """Graphable PRIMARY decode — called during CUDA graph capture AND replay.

            This method contains ONLY the fused Triton kernel call with static
            tensor ops. No Python control flow, no lazy init, no try/except.
            The @torch.compiler.disable on forward() does NOT propagate here,
            so torch.compile and CUDA graphs can capture this path.
            """
            packed_dim = self.head_size // 2
            key_cache = kv_cache[:, 0]
            value_cache = kv_cache[:, 1]
            block_size = key_cache.shape[1]
            li = self._tq_layer_idx
            num_actual_tokens = attn_metadata.num_actual_tokens

            fused_out = self._fused_attn.forward(
                query=query[:num_actual_tokens].to(torch.float32),
                k_packed=key_cache[:, :, :, :packed_dim],
                k_norms=TurboQuantImpl._primary_k_norms[li],
                v_packed=value_cache[:, :, :, :packed_dim],
                v_norms=TurboQuantImpl._primary_v_norms[li],
                block_tables=attn_metadata.block_table,
                context_lens=attn_metadata.seq_lens,
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
            )
            output[:num_actual_tokens] = fused_out.to(output.dtype)
            return output

        # ============================================================
        # PRIMARY DECODE: fused TQPagedAttention (no decompress buffer)
        # ============================================================
        # The fused Triton kernel reads packed uint8 + f32 norms directly
        # from the primary cache. Falls back to batch-decompress →
        # unified_attention if fused kernel init fails.

        def _forward_primary_decode(self, layer, query, key, value,
                                     kv_cache, attn_metadata,
                                     output, output_scale,
                                     output_block_scale=None):
            """PRIMARY decode: fused TQ attention (no decompression buffer).

            The fused Triton kernel (TQPagedAttention) reads packed uint8 +
            f32 norms DIRECTLY from the primary cache, unpacks nibbles
            in-register during attention (codebook lookup + rotation).
            Zero decompression buffer. Same kernel SHADOW mode already uses.

            Falls back to _forward_primary_decode_slow (batch-decompress →
            unified_attention) only if TQPagedAttention fails to initialize.

            Expected throughput: 50-70 tok/s (up from 8-23 with decompress).
            """
            packed_dim = self.head_size // 2
            key_cache = kv_cache[:, 0]      # [B, bs, H, tq_dim] uint8
            value_cache = kv_cache[:, 1]
            num_blocks = key_cache.shape[0]
            block_size = key_cache.shape[1]
            li = self._tq_layer_idx

            # Ensure TQ quantizer + fused attention exist
            if TurboQuantImpl._tq_quantizer is None:
                from .quantizer import TurboQuant
                TurboQuantImpl._tq_quantizer = TurboQuant(
                    head_dim=self.head_size, bits=4,
                    device=str(query.device))
            self._ensure_fused_attn()

            if self._fused_attn is not None:
                try:
                    # Delegate to the graphable method — same code path that
                    # CUDA graphs capture, ensuring capture and replay match.
                    return self._primary_decode_graphable(
                        query, kv_cache, attn_metadata, output)
                except Exception as e:
                    import sys
                    print(f"[TQ] fused PRIMARY decode error L{li}: {e}",
                          file=sys.stderr, flush=True)
                    # Fall through to slow path

            # FALLBACK: old decompress path (shouldn't happen if kernel init OK)
            return self._forward_primary_decode_slow(
                layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale)

        def _forward_primary_decode_slow(self, layer, query, key, value,
                                          kv_cache, attn_metadata,
                                          output, output_scale,
                                          output_block_scale=None):
            """PRIMARY decode fallback: batch-decompress -> unified_attention.

            This is the original decode path kept as a fallback. It decompresses
            ALL active blocks into a bf16 buffer on every decode step, then calls
            unified_attention. ~18ms/token overhead due to decompression.
            """
            packed_dim = self.head_size // 2
            key_cache = kv_cache[:, 0]
            value_cache = kv_cache[:, 1]
            num_blocks = key_cache.shape[0]
            block_size = key_cache.shape[1]

            tq = TurboQuantImpl._tq_quantizer

            # Allocate decode decompression buffers (once, reused).
            buf_dtype = torch.bfloat16
            if (TurboQuantImpl._decode_k_buf is None
                    or TurboQuantImpl._decode_k_buf.shape[0] < num_blocks
                    or TurboQuantImpl._decode_k_buf.device != query.device):
                TurboQuantImpl._decode_k_buf = torch.zeros(
                    num_blocks, block_size, self.num_kv_heads, self.head_size,
                    dtype=buf_dtype, device=query.device)
                TurboQuantImpl._decode_v_buf = torch.zeros_like(
                    TurboQuantImpl._decode_k_buf)
                logger.info(
                    "[TQ] Decode buffer (slow fallback): %d blocks x bs=%d x "
                    "%d heads x %d dim (%.1f MB)",
                    num_blocks, block_size, self.num_kv_heads, self.head_size,
                    num_blocks * block_size * self.num_kv_heads * self.head_size
                    * 2 * 2 / (1024 ** 2),
                )

            k_buf = TurboQuantImpl._decode_k_buf[:num_blocks]
            v_buf = TurboQuantImpl._decode_v_buf[:num_blocks]

            # Batch-decompress active blocks (one tq.decode() call)
            self._batch_decompress_active_blocks(
                key_cache, value_cache, k_buf, v_buf, tq, packed_dim,
                attn_metadata.block_table, num_blocks)

            # Call unified_attention directly with decompressed bf16 buffers.
            from vllm.v1.attention.backends.triton_attn import unified_attention

            num_actual_tokens = attn_metadata.num_actual_tokens
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table

            num_seqs = cu_seqlens_q.shape[0] - 1
            ones_descale = torch.ones(
                num_seqs, self.num_kv_heads,
                dtype=torch.float32, device=query.device)

            q_for_attn = query[:num_actual_tokens].to(k_buf.dtype)
            out_buf = torch.empty_like(q_for_attn)

            unified_attention(
                q=q_for_attn,
                k=k_buf,
                v=v_buf,
                out=out_buf,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,
                k_descale=ones_descale,
                v_descale=ones_descale,
            )

            output[:num_actual_tokens] = out_buf.to(output.dtype)
            return output

        def _batch_decompress_active_blocks(
            self, key_cache, value_cache, k_buf, v_buf, tq, packed_dim,
            block_table, num_blocks,
        ):
            """Batch-decompress TQ blocks into FP16 buffers.

            Gathers all active blocks into flat tensors, calls tq.decode()
            once (single Triton kernel), then scatters back.

            Args:
                key_cache: [num_blocks, block_size, num_kv_heads, tq_dim] uint8
                value_cache: same shape
                k_buf: [num_blocks, block_size, num_kv_heads, head_dim] fp16 output
                v_buf: same shape
                tq: TurboQuant quantizer instance
                packed_dim: number of packed bytes per vector
                block_table: [num_seqs, max_blocks] block indices
                num_blocks: total blocks in cache
            """
            # Get unique active block indices from block_table
            active = block_table.reshape(-1).unique()
            active = active[(active >= 0) & (active < num_blocks)]

            if active.numel() == 0:
                return

            # Gather packed data: [num_active, block_size, num_kv_heads, packed_dim]
            k_packed_active = key_cache[active, :, :, :packed_dim]
            v_packed_active = value_cache[active, :, :, :packed_dim]

            # Gather norms from per-layer float32 tensors
            li = self._tq_layer_idx
            k_norms_active = TurboQuantImpl._primary_k_norms[li, active]  # [A, bs, heads]
            v_norms_active = TurboQuantImpl._primary_v_norms[li, active]

            # Flatten for batch decode: [A * bs * heads, packed_dim]
            A = active.shape[0]
            bs = key_cache.shape[1]
            H = self.num_kv_heads

            flat_kp = k_packed_active.reshape(A * bs * H, packed_dim)
            flat_kn = k_norms_active.reshape(A * bs * H)
            flat_vp = v_packed_active.reshape(A * bs * H, packed_dim)
            flat_vn = v_norms_active.reshape(A * bs * H)

            # Single tq.decode() call — one Triton kernel for all active blocks
            decoded_k = tq.decode(flat_kp, flat_kn)  # [A*bs*H, head_dim]
            decoded_v = tq.decode(flat_vp, flat_vn)

            # Scatter back into buffers
            k_buf[active] = decoded_k.reshape(A, bs, H, -1).to(k_buf.dtype)
            v_buf[active] = decoded_v.reshape(A, bs, H, -1).to(v_buf.dtype)

        def _primary_prefill(self, layer, query, key, value,
                             key_cache, value_cache, kv_cache,
                             attn_metadata, output, output_scale, packed_dim):
            """Prefill: decompress TQ cache -> temp FP16 buffer -> standard attention.

            This is the correctness-first path. The per-block per-head Python
            loop is intentionally slow -- it will be replaced with a fused Triton
            decompression kernel in a follow-up.

            Flow:
              1. Decompress all TQ blocks into a temporary float buffer
              2. Write current-chunk K,V into the decompressed buffer via
                 triton_reshape_and_cache_flash (standard vLLM scatter-write)
              3. Call unified_attention with the decompressed buffer

            Args:
                layer: vLLM layer object (has _k_scale, _v_scale)
                query: [num_actual_tokens, num_heads, head_dim]
                key: [num_actual_tokens, num_kv_heads, head_dim] or None
                value: same shape as key or None
                key_cache: [num_blocks, block_size, num_kv_heads, tq_dim] uint8
                value_cache: same shape
                kv_cache: [num_blocks, 2, block_size, num_kv_heads, tq_dim] uint8
                attn_metadata: vLLM attention metadata
                output: [num_actual_tokens, num_heads * head_dim]
                output_scale: optional output scaling
                packed_dim: number of packed index bytes
            """
            from .quantizer import TurboQuant

            # Ensure quantizer exists
            if TurboQuantImpl._tq_quantizer is None:
                TurboQuantImpl._tq_quantizer = TurboQuant(
                    head_dim=self.head_size, bits=4,
                    device=str(query.device))
            tq = TurboQuantImpl._tq_quantizer

            num_blocks = key_cache.shape[0]
            block_size = key_cache.shape[1]

            # Allocate prefill decompression buffers (reused across calls).
            # Shape: [num_blocks, block_size, num_kv_heads, head_dim]
            # Dtype: same as query (bf16 or fp16) -- NOT uint8.
            need_realloc = (
                TurboQuantImpl._prefill_k_buf is None
                or TurboQuantImpl._prefill_k_buf.shape[0] < num_blocks
                or TurboQuantImpl._prefill_k_buf.device != query.device
            )
            if need_realloc:
                buf_blocks = max(num_blocks, self._PREFILL_BUF_BLOCKS)
                # Force bfloat16 — query may be fp8 on quantized models
                TurboQuantImpl._prefill_k_buf = torch.zeros(
                    buf_blocks, block_size, self.num_kv_heads, self.head_size,
                    dtype=torch.bfloat16, device=query.device)
                TurboQuantImpl._prefill_v_buf = torch.zeros_like(
                    TurboQuantImpl._prefill_k_buf)

            k_buf = TurboQuantImpl._prefill_k_buf[:num_blocks]
            v_buf = TurboQuantImpl._prefill_v_buf[:num_blocks]

            # Decompress ONLY blocks referenced by this request's block_table.
            # block_table: [num_seqs, max_blocks_per_seq] — physical block indices
            active_blocks = attn_metadata.block_table.unique()
            active_blocks = active_blocks[active_blocks >= 0]  # filter padding
            active_blocks = active_blocks[active_blocks < num_blocks]

            # Batch-decompress all active blocks at once (not per-block Python loop)
            if active_blocks.numel() > 0:
                A = active_blocks.shape[0]
                H = self.num_kv_heads

                # Gather packed indices and norms for all active blocks
                kp_all = key_cache[active_blocks, :, :, :packed_dim]  # [A, bs, H, pd]
                vp_all = value_cache[active_blocks, :, :, :packed_dim]
                li = self._tq_layer_idx
                kn_all = TurboQuantImpl._primary_k_norms[li, active_blocks]  # [A, bs, H]
                vn_all = TurboQuantImpl._primary_v_norms[li, active_blocks]

                # Flatten and decode in one call
                flat_kp = kp_all.reshape(A * block_size * H, packed_dim)
                flat_kn = kn_all.reshape(A * block_size * H)
                flat_vp = vp_all.reshape(A * block_size * H, packed_dim)
                flat_vn = vn_all.reshape(A * block_size * H)

                decoded_k = tq.decode(flat_kp, flat_kn)  # [A*bs*H, head_dim]
                decoded_v = tq.decode(flat_vp, flat_vn)

                k_buf[active_blocks] = decoded_k.reshape(A, block_size, H, -1).to(k_buf.dtype)
                v_buf[active_blocks] = decoded_v.reshape(A, block_size, H, -1).to(v_buf.dtype)

            # Build decompressed cache in standard vLLM layout for unified_attention.
            # Shape: [num_blocks, 2, block_size, num_kv_heads, head_dim]
            fake_kv = torch.stack([k_buf, v_buf], dim=1)

            fake_key_cache = fake_kv[:, 0]    # [blocks, bs, heads, head_dim]
            fake_value_cache = fake_kv[:, 1]

            # Write current-chunk K,V into the decompressed buffer so that
            # the newly-arrived tokens are available for this prefill's attention.
            # Use direct indexing (not triton_reshape_and_cache_flash) because
            # the cache_dtype="auto" path quantizes to fp8, corrupting our bf16 buffer.
            if key is not None and value is not None:
                self._prefill_scatter_write_fallback(
                    key, value, fake_key_cache, fake_value_cache,
                    attn_metadata.slot_mapping, block_size)

            # --- Call unified_attention with the decompressed cache ---
            from vllm.v1.attention.backends.triton_attn import unified_attention

            num_actual_tokens = attn_metadata.num_actual_tokens

            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table

            # k_descale / v_descale: unified_attention expects per-sequence
            # descale factors shaped [num_seqs, num_kv_heads].
            # For our decompressed float buffer, descaling is already applied,
            # so we pass 1.0 tensors.
            num_seqs = cu_seqlens_q.shape[0] - 1
            ones_descale = torch.ones(
                num_seqs, self.num_kv_heads,
                dtype=torch.float32, device=query.device)

            # Cast query to match buffer dtype (query may be fp8 on quantized models)
            q_for_attn = query[:num_actual_tokens].to(fake_key_cache.dtype)
            out_buf = torch.empty_like(q_for_attn)

            unified_attention(
                q=q_for_attn,
                k=fake_key_cache,
                v=fake_value_cache,
                out=out_buf,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,
                k_descale=ones_descale,
                v_descale=ones_descale,
            )

            output[:num_actual_tokens] = out_buf.to(output.dtype)
            return output

        @staticmethod
        def _prefill_scatter_write_fallback(key, value, k_cache, v_cache,
                                            slot_mapping, block_size):
            """Direct scatter-write for prefill into bf16 decompression buffers.

            Uses slot_mapping to index into cache buffers. Handles padded
            num_tokens (vLLM pads to CUDA graph capture sizes).
            """
            # slot_mapping may be shorter than key if key is padded
            n = min(slot_mapping.shape[0], key.shape[0])
            sm = slot_mapping[:n]
            valid_idx = (sm >= 0).nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                return
            vs = sm[valid_idx]
            bi = vs // block_size
            oi = vs % block_size
            max_blocks = k_cache.shape[0]
            mask = bi < max_blocks
            if not mask.any():
                return
            bi = bi[mask]
            oi = oi[mask]
            src_idx = valid_idx[mask]
            k_cache[bi, oi] = key[src_idx].to(k_cache.dtype)
            v_cache[bi, oi] = value[src_idx].to(v_cache.dtype)

    return TurboQuantImpl


# ================================================================
# REGISTRATION (safe to call during sitecustomize -- no vLLM imports)
# ================================================================

def register_turboquant_backend():
    """Register TurboQuant as a CUSTOM attention backend in vLLM."""
    try:
        from vllm.v1.attention.backends.registry import (
            register_backend,
            AttentionBackendEnum,
        )
        register_backend(
            AttentionBackendEnum.CUSTOM,
            f"{__name__}.TurboQuantBackend",
        )
        logger.info("Registered TurboQuant as CUSTOM attention backend")
        return True
    except Exception as e:
        logger.warning("Failed to register TQ backend: %s", e)
        return False
