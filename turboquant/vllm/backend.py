"""
TurboQuant attention backend for vLLM v0.15+.

Phase 2: Fused TQ decode + standard Triton prefill.

Decode (single-token):
  Fused Triton kernel reads directly from TQ-compressed GPU storage.
  No decompression buffer. ~1.88x more KV capacity than FP8.

Prefill (multi-token):
  Standard Triton attention on vLLM's FP8/FP16 cache (delegated to
  TritonAttentionImpl to avoid raw kernel API drift).

Both paths encode new K,V to TQ GPU storage for decode use.
Cold tier (DDR5): spill/warm via TQGPUCache.spill_blocks()/warm_blocks().

Activation:
    vllm serve model --attention-backend CUSTOM
"""

import logging
import os
from typing import ClassVar, Optional

import torch

logger = logging.getLogger("turboquant.vllm.backend")

# Lazy imports — vLLM may not be installed
_VLLM_AVAILABLE = False
try:
    from vllm.v1.attention.backend import (
        AttentionBackend,
        AttentionImpl,
        AttentionMetadataBuilder,
        AttentionType,
    )
    _VLLM_AVAILABLE = True
except ImportError:
    pass


if _VLLM_AVAILABLE:

    class TurboQuantBackend(AttentionBackend):

        # Support all KV cache dtypes — TQ encodes from whatever dtype vLLM
        # stores, so the underlying cache dtype is irrelevant.
        supported_kv_cache_dtypes = [
            "auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2",
        ]

        @staticmethod
        def get_name() -> str:
            # Must match AttentionBackendEnum member name
            return "CUSTOM"

        @staticmethod
        def get_impl_cls() -> type["AttentionImpl"]:
            return TurboQuantImpl

        @staticmethod
        def get_builder_cls() -> type["AttentionMetadataBuilder"]:
            from vllm.v1.attention.backends.triton_attn import (
                TritonAttentionMetadataBuilder,
            )
            return TritonAttentionMetadataBuilder

        @staticmethod
        def get_kv_cache_shape(
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
            cache_dtype_str: str = "auto",
        ) -> tuple[int, ...]:
            return (num_blocks, 2, block_size, num_kv_heads, head_size)

        @staticmethod
        def get_supported_head_sizes() -> list[int]:
            return [64, 96, 128, 256]

    class TurboQuantImpl(AttentionImpl):
        """
        TQ attention with fused decode path.

        Env vars:
          AITHER_TQ_BITS: 2, 3, or 4 (default 4)
          AITHER_TQ_FUSED: "1" (default) = fused decode, "0" = standard only
        """

        _tq_gpu_cache: ClassVar[Optional["TQGPUCache"]] = None
        _layer_counter: ClassVar[int] = 0
        _fused_enabled: ClassVar[bool] = (
            os.environ.get("AITHER_TQ_FUSED", "1") == "1"
        )
        _forward_count: ClassVar[int] = 0
        _INIT_AFTER_FORWARDS: ClassVar[int] = 2

        def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int | None = None,
            alibi_slopes: list[float] | None = None,
            sliding_window: int | None = None,
            kv_cache_dtype: str = "auto",
            logits_soft_cap: float | None = None,
            attn_type: str = AttentionType.DECODER,
            kv_sharing_target_layer_name: str | None = None,
        ) -> None:
            self.num_heads = num_heads
            self.head_size = head_size
            self.scale = scale
            self.num_kv_heads = num_kv_heads or num_heads
            self.kv_cache_dtype = kv_cache_dtype
            self.sliding_window = sliding_window
            self.alibi_slopes = alibi_slopes
            self.logits_soft_cap = logits_soft_cap
            self.attn_type = attn_type

            self._layer_idx = TurboQuantImpl._layer_counter
            TurboQuantImpl._layer_counter += 1

            # Delegate standard attention to TritonAttentionImpl
            from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
            self._triton_impl = TritonAttentionImpl(
                num_heads=num_heads,
                head_size=head_size,
                scale=scale,
                num_kv_heads=num_kv_heads,
                alibi_slopes=alibi_slopes,
                sliding_window=sliding_window,
                kv_cache_dtype=kv_cache_dtype,
                logits_soft_cap=logits_soft_cap,
                attn_type=attn_type,
            )

            self._fused_attn = None

            mode = "fused+standard" if self._fused_enabled else "standard"
            logger.info(
                "TurboQuantImpl[L%d]: heads=%d, kv=%d, dim=%d, mode=%s",
                self._layer_idx, num_heads, self.num_kv_heads, head_size, mode,
            )

        def _ensure_tq_cache(self, kv_cache: torch.Tensor):
            if TurboQuantImpl._tq_gpu_cache is not None:
                return
            if TurboQuantImpl._forward_count < TurboQuantImpl._INIT_AFTER_FORWARDS:
                return

            if kv_cache.dim() == 5:
                _, num_blocks, block_size, _, _ = kv_cache.shape
            else:
                num_blocks = 2048
                block_size = 16

            num_layers = TurboQuantImpl._layer_counter
            bits = int(os.environ.get("AITHER_TQ_BITS", "4"))

            from .cache import TQGPUCache
            TurboQuantImpl._tq_gpu_cache = TQGPUCache(
                num_layers=num_layers,
                max_blocks=num_blocks,
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_size,
                bits=bits,
                device=str(kv_cache.device),
            )

        def _ensure_fused_attn(self):
            if self._fused_attn is not None:
                return
            from turboquant.fused_attention import TQPagedAttention
            cache = TurboQuantImpl._tq_gpu_cache
            if cache is not None:
                self._fused_attn = TQPagedAttention(cache.tq, self.num_heads)

        def forward(
            self,
            layer,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata,
            output: torch.Tensor | None = None,
            output_scale: torch.Tensor | None = None,
            output_block_scale: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if self._layer_idx == 0:
                TurboQuantImpl._forward_count += 1

            # Initialize TQ GPU cache (deferred past profiling)
            if self._fused_enabled:
                self._ensure_tq_cache(kv_cache)

            # Encode new K,V to TQ compressed GPU storage
            tq_cache = TurboQuantImpl._tq_gpu_cache
            if tq_cache is not None and key is not None:
                try:
                    tq_cache.encode_and_store(
                        self._layer_idx, key, value,
                        attn_metadata.slot_mapping,
                    )
                except Exception as e:
                    logger.debug("TQ encode error L%d: %s", self._layer_idx, e)

            # -- Decode path: fused TQ kernel --
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
                    logger.debug("TQ fused decode error L%d: %s",
                                 self._layer_idx, e)

            # -- Prefill / fallback: delegate to TritonAttentionImpl --
            if output is None:
                num_tokens = query.shape[0]
                output = torch.empty(
                    num_tokens, self.num_heads, self.head_size,
                    dtype=query.dtype, device=query.device,
                )
            return self._triton_impl.forward(
                layer, query, key, value, kv_cache, attn_metadata,
                output=output, output_scale=output_scale,
                output_block_scale=output_block_scale,
            )

        def _fused_decode(
            self,
            query: torch.Tensor,
            attn_metadata,
            output: torch.Tensor,
        ) -> torch.Tensor:
            tq_cache = TurboQuantImpl._tq_gpu_cache
            li = self._layer_idx

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
