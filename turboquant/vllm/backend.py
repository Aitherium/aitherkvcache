"""
TurboQuant attention backend for vLLM v0.15+.

Phase 1 (current): delegates to Triton attention for computation,
adds async TQ cold tier encoding in background.

Phase 2 (next): TQPagedAttention for decode, FlashAttn for prefill.
Phase 3 (future): TQ-native cache allocation, no FP8 buffer.
"""

import logging
import os

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

        @staticmethod
        def get_name() -> str:
            return "TURBOQUANT"

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
        def get_supported_head_sizes() -> list[int]:
            return [64, 96, 128, 256]

    class TurboQuantImpl(AttentionImpl):
        """
        Phase 1: Triton attention + async TQ cold tier encoding.

        The actual attention computation delegates to vLLM's Triton kernels
        (identical output to TRITON_ATTN backend). In parallel, every K,V
        write is TQ-compressed to a CPU cold tier via a background thread.

        Set AITHER_TQ_BITS=4 (or 2, 3) to control compression bit-width.
        Default: 4-bit.
        """

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

            # Triton attention ops (same as TRITON_ATTN backend)
            from vllm.v1.attention.ops.triton_unified_attention import (
                unified_attention,
            )
            from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
                triton_reshape_and_cache_flash,
            )
            self._reshape_and_cache = triton_reshape_and_cache_flash
            self._unified_attention = unified_attention

            # TQ cold tier (lazy init on first forward)
            self._cold_tier = None
            self._tq_bits = int(os.environ.get("AITHER_TQ_BITS", "4"))
            self._layer_idx = None  # set on first call via instance counter
            self._encode_ok = True

        def _ensure_cold_tier(self):
            """Lazy-init the shared cold tier on first use."""
            if self._cold_tier is not None:
                return

            try:
                from .cache import get_shared_cold_tier
                self._cold_tier = get_shared_cold_tier(
                    head_dim=self.head_size,
                    num_kv_heads=self.num_kv_heads,
                    bits=self._tq_bits,
                )
                # Assign a layer index
                self._layer_idx = self._cold_tier.register_layer()
                logger.info(
                    "TQ cold tier: layer %d registered (heads=%d, dim=%d, %d-bit)",
                    self._layer_idx, self.num_kv_heads, self.head_size,
                    self._tq_bits,
                )
            except Exception as e:
                logger.warning("TQ cold tier init failed: %s", e)
                self._encode_ok = False

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
            # --- Standard attention (Triton kernels) ---

            # Write new K,V to cache
            if key is not None and hasattr(attn_metadata, "slot_mapping"):
                self._reshape_and_cache(
                    key, value, kv_cache, attn_metadata.slot_mapping,
                )

            # Compute attention
            num_tokens = query.shape[0]
            if output is None:
                output = torch.empty(
                    num_tokens, self.num_heads, self.head_size,
                    dtype=query.dtype, device=query.device,
                )

            self._unified_attention(
                output=output,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                num_kv_heads=self.num_kv_heads,
                scale=self.scale,
                alibi_slopes=self.alibi_slopes,
                sliding_window=self.sliding_window,
                logits_soft_cap=self.logits_soft_cap,
            )

            # --- Async TQ cold tier encode (after attention, non-blocking) ---
            if key is not None and self._encode_ok:
                self._ensure_cold_tier()
                if self._cold_tier is not None and self._layer_idx is not None:
                    try:
                        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
                        if slot_mapping is not None:
                            self._cold_tier.compress_async(
                                self._layer_idx, key, value, slot_mapping,
                            )
                    except Exception:
                        pass  # never block attention

            return output
