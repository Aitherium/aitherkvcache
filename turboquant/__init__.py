"""
TurboQuant — Near-optimal KV cache quantization for AitherOS.

Implements the TurboQuant algorithm from Zandieh et al. (arXiv:2504.19874):
random rotation -> optimal scalar quantization -> bit packing.

Achieves 3.5-bit KV cache compression with zero accuracy loss,
within 2.7x of the information-theoretic optimum.

Usage:
    from lib.gpu.turboquant import TurboQuant

    tq = TurboQuant(head_dim=128, bits=4, device='cuda')
    packed, norms = tq.encode(kv_vectors)
    decoded = tq.decode(packed, norms)

    # Validate correctness
    print(tq.validate())

    # Memory savings report
    print(tq.memory_report(seq_len=40000, num_layers=32, num_kv_heads=8))
"""

from .quantizer import TurboQuant, TurboQuantConfig

# NOT eager, and NOT `.gb10_fp8_rescue` directly. That module has never
# existed in git, so a direct import raised ModuleNotFoundError at PACKAGE
# import time and made the whole of lib.gpu.turboquant un-importable — every
# caller sits behind `try: ... except ImportError`, so this surfaced as the
# features being switched off rather than as an error (LLMGateway's vLLM-TQ
# tenant-token SIGNING among them). vllm_patch exposes the same two names as
# lazy wrappers that import gb10_fp8_rescue when CALLED, which defers the
# failure to the GB10/sm_121 hardware path where it belongs.
#
# vllm_patch itself is imported LAZILY here too — below, as
# get_gb10_fp8_status()/install_gb10_fp8_rescue(), matching every other
# hardware-specific entry point in this file (get_block_metadata_table,
# get_tier_cache_bridge, etc. — all import their dependency inside the
# function body, none at module level). vllm_patch is the one AitherOS-only
# file .github/workflows/sync-kvcache.yml deliberately never publishes (see
# that file's "NOT synced" list), so an EAGER import here broke the published
# aitherkvcache package's turboquant/__init__.py outright: every
# `pip install aitherkvcache; import turboquant` raised
# `ModuleNotFoundError: No module named 'turboquant.vllm_patch'` before a
# single line of the package's own code ran. Verified 2026-08-18 by staging
# the real curated payload and importing it as the published top-level
# `turboquant` package.

__all__ = [
    "TurboQuant",
    "TurboQuantConfig",
    # 3-tier cache integration (lazy imports — use these entry points)
    "get_block_metadata_table",
    "get_prefix_pin_manager",
    "get_graph_block_reserver",
    "get_cache_aware_pipeline",
    "get_tier_cache_bridge",
    "get_strata_cache_shadow",
    "get_graph_eviction_advisor",
    "install_gb10_fp8_rescue",
    "get_gb10_fp8_status",
]
__version__ = "0.9.0"


def get_block_metadata_table():
    from .block_metadata import get_block_metadata_table
    return get_block_metadata_table()


def get_prefix_pin_manager():
    from .block_metadata import get_prefix_pin_manager
    return get_prefix_pin_manager()


def get_graph_block_reserver(tq_gpu_cache=None):
    from .graph_block_reserver import get_graph_block_reserver
    return get_graph_block_reserver(tq_gpu_cache)


def get_cache_aware_pipeline():
    from .cache_aware_context import get_cache_aware_pipeline
    return get_cache_aware_pipeline()


def get_tier_cache_bridge():
    from .tier_cache_bridge import get_tier_cache_bridge
    return get_tier_cache_bridge()


def get_strata_cache_shadow():
    from .strata_shadow import get_strata_cache_shadow
    return get_strata_cache_shadow()


def get_graph_eviction_advisor():
    from .graph_eviction_advisor import get_graph_eviction_advisor
    return get_graph_eviction_advisor()


def install_gb10_fp8_rescue(force: bool = False) -> bool:
    from .vllm_patch import install_gb10_fp8_rescue
    return install_gb10_fp8_rescue(force=force)


def get_gb10_fp8_status() -> dict:
    from .vllm_patch import get_gb10_fp8_status
    return get_gb10_fp8_status()
