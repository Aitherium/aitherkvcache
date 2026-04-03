"""
TurboQuant vLLM v0.14+ Engine Integration.

Patches vLLM v1 architecture (v0.14+) to report TQ-compressed page sizes,
causing vLLM to allocate more KV cache blocks for the same VRAM budget.

vLLM v1 architecture:
  - KVCacheSpec.page_size_bytes controls memory allocation per block
  - FullAttentionSpec.real_page_size_bytes returns actual bytes per page
  - Both AttentionSpec AND FullAttentionSpec define real_page_size_bytes
  - GPU worker calls get_kv_cache_spec() to determine cache layout
  - Raw byte tensors allocated, then viewed as [2, blocks, block_size, heads, dim]

Two operating modes:

  SHADOW mode (default, AITHER_TQ_PRIMARY=0):
    vLLM allocates FP8 KV cache normally.  TQ shadow cache encodes a
    compressed copy for cold-tier spill and (optionally) fused decode.
    No VRAM savings -- TQ shadow is capped at 512 MiB.

  PRIMARY mode (AITHER_TQ_PRIMARY=1):
    page_size patch makes vLLM allocate TQ-sized blocks (more blocks per
    VRAM budget).  Reshape intercept views raw bytes as uint8 TQ layout
    instead of FP8.  The KV cache IS the TQ cache -- no shadow copy.
    This doubles usable context from ~280K to ~548K tokens.

Apply patches BEFORE starting the vLLM server:
    from aither_kvcache.vllm.engine import apply_tq_patches
    apply_tq_patches(bits=4)
"""

import logging
import os

logger = logging.getLogger("turboquant.engine")

_TQ_BITS: int = 0
_TQ_PRIMARY: bool = False
_TQ_HYBRID: bool = False  # True for tq35/tq25 modes


def _is_primary_mode() -> bool:
    """Derive primary flag from AITHER_TQ_MODE or fallback to AITHER_TQ_PRIMARY."""
    tq_mode = os.environ.get("AITHER_TQ_MODE", "")
    if tq_mode:
        return tq_mode.endswith("-primary")
    # Legacy fallback
    return os.environ.get("AITHER_TQ_PRIMARY", "0") == "1"


def apply_tq_patches(bits: int = 4) -> bool:
    """
    Apply TurboQuant patches to vLLM v1 modules.
    Call BEFORE vllm server starts.

    When primary mode is enabled (AITHER_TQ_MODE=tq4-primary):
      - page_size patch: vLLM computes block count using TQ page size
      - max_memory patch: pre-allocation estimation uses TQ page size
      - reshape patch: raw byte buffers are viewed as TQ uint8 layout
        instead of fp8 typed tensors
    """
    global _TQ_BITS, _TQ_PRIMARY, _TQ_HYBRID
    _TQ_BITS = bits
    _TQ_PRIMARY = _is_primary_mode()
    tq_mode = os.environ.get("AITHER_TQ_MODE", "")
    _TQ_HYBRID = tq_mode.replace("-primary", "") in ("tq35", "tq25")

    if bits == 0:
        logger.info("TurboQuant disabled (bits=0)")
        return True

    logger.info("Applying TurboQuant %d-bit patches to vLLM v1...", bits)
    logger.info("  Mode: %s", "PRIMARY (TQ is the KV cache)" if _TQ_PRIMARY
                else "SHADOW (FP8 primary + TQ shadow)")

    results = []

    if _TQ_PRIMARY:
        # PRIMARY mode: TQ IS the KV cache. page_size tells vLLM that blocks
        # are TQ-compressed size so it allocates more blocks for the same VRAM.
        # reshape intercept views those blocks as uint8 TQ layout.
        results.append(("page_size", _patch_page_size()))
        results.append(("max_memory", _patch_max_memory()))
        results.append(("reshape", _patch_reshape()))
        logger.info("  page_size + max_memory + reshape: ENABLED (primary cache)")
    else:
        # SHADOW mode: vLLM uses standard FP8 cache. TQ encodes a shadow copy.
        fused = os.environ.get("AITHER_TQ_FUSED", "0") == "1"
        if fused:
            logger.info("  Fused decode: ENABLED (TQ kernel for single-token decode)")
        else:
            logger.info("  Fused decode: DISABLED (shadow encode only)")
        logger.info("  page_size/max_memory/reshape: SKIPPED (shadow mode)")

    results.append(("block_manager", _patch_block_manager()))

    for name, ok in results:
        status = "OK" if ok else "SKIPPED"
        logger.info("  Patch %-12s: %s", name, status)

    success = all(ok for _, ok in results)
    if success:
        logger.info("TurboQuant %d-bit patches active (%s mode)",
                     bits, "PRIMARY" if _TQ_PRIMARY else "SHADOW")
    else:
        logger.warning("Some patches failed -- check logs")
    return success


def _tq_page_size_bytes(block_size: int, num_kv_heads: int, head_size: int) -> int:
    """Compute TQ-compressed page size in bytes for any mode (uniform or hybrid)."""
    tq_mode = os.environ.get("AITHER_TQ_MODE", "").replace("-primary", "")
    if tq_mode in ("tq35", "tq25"):
        from ..hybrid_quantizer import HybridTurboQuant
        pd = HybridTurboQuant.packed_dim_for_mode(head_size, tq_mode)
        # Hybrid embeds norms in packed data -- no separate +4
        return 2 * block_size * num_kv_heads * pd
    else:
        from ..packing import packed_size
        pd = packed_size(head_size, _TQ_BITS)
        # Uniform: packed indices + 4 bytes for f32 norm
        return 2 * block_size * num_kv_heads * (pd + 4)


def _tq_dim_for_head(head_size: int) -> int:
    """Compute per-head TQ dimension (last axis of reshaped cache)."""
    tq_mode = os.environ.get("AITHER_TQ_MODE", "").replace("-primary", "")
    if tq_mode in ("tq35", "tq25"):
        from ..hybrid_quantizer import HybridTurboQuant
        return HybridTurboQuant.packed_dim_for_mode(head_size, tq_mode)
    else:
        from ..packing import packed_size
        return packed_size(head_size, _TQ_BITS) + 4


def _patch_page_size() -> bool:
    """
    Patch real_page_size_bytes on both AttentionSpec and FullAttentionSpec.

    FullAttentionSpec shadows AttentionSpec's property, so we must patch
    both classes for the override to take effect.
    """
    try:
        from vllm.v1.kv_cache_interface import FullAttentionSpec, AttentionSpec
    except ImportError:
        logger.warning("vLLM v1 FullAttentionSpec not found -- wrong vLLM version?")
        return False

    @property
    def _tq_real_page_size_bytes(self):
        """Return TQ-compressed page size (handles both uniform and hybrid)."""
        return _tq_page_size_bytes(self.block_size, self.num_kv_heads, self.head_size)

    # Patch BOTH classes -- FullAttentionSpec shadows parent's property
    AttentionSpec.real_page_size_bytes = _tq_real_page_size_bytes
    FullAttentionSpec.real_page_size_bytes = _tq_real_page_size_bytes

    # Log compression ratio with typical Nemotron-8B values
    try:
        standard = 2 * 16 * 8 * 128 * 2  # block=16, heads=8, dim=128, fp16
        tq = _tq_page_size_bytes(16, 8, 128)
        mode_label = os.environ.get("AITHER_TQ_MODE", "").replace("-primary", "")
        logger.info(
            "%s page: %d bytes/block (vs %d standard, %.1fx more blocks)",
            mode_label.upper() or f"TQ{_TQ_BITS}", tq, standard, standard / tq,
        )
    except Exception:
        pass

    return True


def _patch_max_memory() -> bool:
    """
    Patch FullAttentionSpec.max_memory_usage_bytes to use TQ page size.
    This affects vLLM's pre-allocation memory estimation.
    """
    try:
        from vllm.v1.kv_cache_interface import FullAttentionSpec
    except ImportError:
        return False

    def _tq_max_memory(self, vllm_config):
        from vllm.utils.math_utils import cdiv
        max_model_len = vllm_config.model_config.max_model_len
        try:
            dcp = vllm_config.parallel_config.decode_context_parallel_size
            pcp = vllm_config.parallel_config.prefill_context_parallel_size
            if dcp * pcp > 1:
                max_model_len = cdiv(max_model_len, dcp * pcp)
        except AttributeError:
            pass
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes

    FullAttentionSpec.max_memory_usage_bytes = _tq_max_memory
    return True


def _patch_reshape() -> bool:
    """
    Monkey-patch GPUModelRunner._reshape_kv_cache_tensors to produce TQ-layout
    uint8 tensors instead of fp8-typed tensors for AttentionSpec layers.

    vLLM's standard reshape flow (for AttentionSpec):
      1. num_blocks = raw_tensor.numel() // page_size_bytes
      2. shape = attn_backend.get_kv_cache_shape(...)
      3. tensor = raw_tensor.view(dtype).view(shape).permute(inv_order)

    TQ primary reshape:
      1. num_blocks computed identically (page_size already patched to TQ size)
      2. View raw bytes as uint8 (not fp8) and reshape to TQ layout:
         [num_blocks, 2, block_size, num_kv_heads, tq_dim]
         where tq_dim = packed_dim + 4 (packed indices + f32 norm as bytes)
      3. MambaSpec layers (if any) are handled by the original method

    The patched method is applied to GPUModelRunner on the class itself, so
    it takes effect in all worker processes (vLLM's multi-process architecture
    imports and instantiates GPUModelRunner in each EngineCore worker).
    """
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        logger.warning("GPUModelRunner not found -- reshape patch skipped")
        return False

    if getattr(GPUModelRunner, "_tq_reshape_patched", False):
        logger.debug("GPUModelRunner._reshape_kv_cache_tensors already patched")
        return True

    original_reshape = GPUModelRunner._reshape_kv_cache_tensors

    def _tq_reshape_kv_cache_tensors(self, kv_cache_config, kv_cache_raw_tensors,
                                      kernel_block_sizes):
        """
        TQ-primary reshape: intercept AttentionSpec layers and view raw bytes
        as uint8 TQ layout.  Non-attention layers (MambaSpec) fall through to
        the original vLLM implementation.

        Returns:
            dict[str, torch.Tensor]: layer_name -> reshaped KV cache tensor.
            For TQ layers the tensor is uint8 with shape
            [num_blocks, 2, block_size, num_kv_heads, tq_dim].
        """
        import torch
        from ..packing import packed_size

        # Check if any MambaSpec layers exist -- if so, we need the original
        # method for those.  We handle AttentionSpec layers ourselves.
        has_non_attention = False
        attention_layer_names = set()

        try:
            from vllm.v1.kv_cache_interface import AttentionSpec
        except ImportError:
            # If we can't import AttentionSpec, fall back entirely
            logger.warning("[TQ] Cannot import AttentionSpec -- falling back to "
                           "original reshape")
            return original_reshape(self, kv_cache_config,
                                    kv_cache_raw_tensors, kernel_block_sizes)

        try:
            from vllm.v1.kv_cache_interface import MambaSpec
            _has_mamba_spec = True
        except ImportError:
            _has_mamba_spec = False

        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            if group.kv_cache_group_id == len(kernel_block_sizes):
                continue
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                if isinstance(kv_cache_spec, AttentionSpec):
                    attention_layer_names.add(layer_name)
                else:
                    has_non_attention = True

        if has_non_attention:
            # Mixed model (attention + mamba): let vLLM handle mamba layers
            # via original method, then override attention layers.
            kv_caches = original_reshape(self, kv_cache_config,
                                         kv_cache_raw_tensors,
                                         kernel_block_sizes)
        else:
            kv_caches = {}

        # Now reshape attention layers with TQ layout
        tq_layer_count = 0
        last_tq_dim = 0
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            if group.kv_cache_group_id == len(kernel_block_sizes):
                continue
            if not isinstance(kv_cache_spec, AttentionSpec):
                continue

            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]

            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue

                raw_tensor = kv_cache_raw_tensors[layer_name]

                # page_size_bytes is already patched to TQ size
                page_bytes = kv_cache_spec.page_size_bytes
                assert raw_tensor.numel() % page_bytes == 0, (
                    f"[TQ] raw_tensor size {raw_tensor.numel()} not divisible "
                    f"by TQ page_size_bytes {page_bytes} for {layer_name}"
                )
                num_blocks = raw_tensor.numel() // page_bytes

                # Account for kernel block splitting (same as vLLM original)
                num_blocks_per_kv_block = (
                    kv_cache_spec.block_size // kernel_block_size
                )
                kernel_num_blocks = num_blocks * num_blocks_per_kv_block

                # TQ dimensions (handles both uniform and hybrid)
                head_size = kv_cache_spec.head_size
                num_kv_heads = kv_cache_spec.num_kv_heads
                tq_dim = _tq_dim_for_head(head_size)

                # Reshape: view as uint8, then to TQ layout
                # Shape: [kernel_num_blocks, 2(K/V), kernel_block_size,
                #         num_kv_heads, tq_dim]
                expected_bytes = (kernel_num_blocks * 2 * kernel_block_size
                                  * num_kv_heads * tq_dim)

                if raw_tensor.numel() < expected_bytes:
                    logger.error(
                        "[TQ] %s: raw tensor %d bytes < expected %d bytes "
                        "(%d blocks x 2 x bs=%d x heads=%d x tq_dim=%d). "
                        "Falling back to original reshape for this layer.",
                        layer_name, raw_tensor.numel(), expected_bytes,
                        kernel_num_blocks, kernel_block_size, num_kv_heads,
                        tq_dim,
                    )
                    # Fall back to original for this layer
                    kv_caches[layer_name] = original_reshape(
                        self, kv_cache_config, {layer_name: raw_tensor},
                        kernel_block_sizes,
                    ).get(layer_name, raw_tensor)
                    continue

                kv_caches[layer_name] = (
                    raw_tensor
                    .view(torch.uint8)
                    .view(kernel_num_blocks, 2, kernel_block_size,
                          num_kv_heads, tq_dim)
                )
                tq_layer_count += 1
                last_tq_dim = tq_dim

        if tq_layer_count > 0:
            logger.info(
                "[TQ] Primary reshape: %d layers -> uint8 "
                "[blocks, 2, bs, heads, tq_dim=%d] (TQ%d)",
                tq_layer_count, last_tq_dim, _TQ_BITS,
            )

        return kv_caches

    GPUModelRunner._reshape_kv_cache_tensors = _tq_reshape_kv_cache_tensors
    GPUModelRunner._tq_reshape_patched = True
    logger.info("Patched GPUModelRunner._reshape_kv_cache_tensors with TQ "
                "primary layout (TQ%d)", _TQ_BITS)
    return True


def _extract_block_indices_from_free(request) -> list:
    """
    Extract physical block indices from a vLLM free() request.

    vLLM's KVCacheManager.free() receives request objects whose format
    varies across versions.  We try multiple attribute paths.

    Returns:
        List of int block indices, or empty list if extraction fails.
    """
    # v1: request has .block_ids (list of ints)
    block_ids = getattr(request, "block_ids", None)
    if block_ids is not None:
        if isinstance(block_ids, (list, tuple)):
            return list(block_ids)
        # Could be a tensor
        return list(block_ids)

    # v1 alternate: request has .block_table (list/tensor of block indices)
    block_table = getattr(request, "block_table", None)
    if block_table is not None:
        try:
            return list(block_table)
        except TypeError:
            return [block_table]

    # Fallback: request might be an int directly
    if isinstance(request, int):
        return [request]

    return []


def _patch_block_manager() -> bool:
    """
    Monkey-patch vLLM's KVCacheManager.free() to spill TQ-compressed
    blocks to DDR5 cold tier before releasing them.

    This preserves evicted KV data in system RAM for potential future
    warming, following the existing patch pattern in this module.
    """
    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
    except ImportError:
        logger.debug("KVCacheManager not found -- block_manager patch skipped")
        return False

    # Guard against double-patching
    if getattr(KVCacheManager, "_tq_free_patched", False):
        logger.debug("KVCacheManager.free already patched")
        return True

    original_free = KVCacheManager.free

    def _tq_free_wrapper(self, request):
        """Spill TQ blocks to DDR5 before vLLM frees them."""
        try:
            from .backend import TurboQuantImpl
            tq_cache = TurboQuantImpl._tq_gpu_cache
            if tq_cache is not None:
                import torch
                block_indices = _extract_block_indices_from_free(request)
                if block_indices:
                    bi = torch.tensor(block_indices, dtype=torch.long)
                    tq_cache.spill_blocks(bi)

                    logger.debug(
                        "TQ spill-on-free: %d blocks spilled to DDR5",
                        len(block_indices),
                    )
        except Exception as exc:
            # Never block vLLM's free path
            logger.debug("TQ spill-on-free error: %s", exc)

        return original_free(self, request)

    KVCacheManager.free = _tq_free_wrapper
    KVCacheManager._tq_free_patched = True
    logger.info("Patched KVCacheManager.free with TQ spill-on-free hook")
    return True
