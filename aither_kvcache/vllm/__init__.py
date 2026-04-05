"""
aither-kvcache vLLM integration.

Usage:
    pip install aither-kvcache[vllm]

    # Automatic (plugin loaded by vLLM at startup):
    #   - TurboQuant backend registered as CUSTOM
    #   - Graph-aware eviction replaces LRU
    # Just: vllm serve model --attention-backend CUSTOM

    # Manual hook mode (deprecated, use plugin instead):
    from aither_kvcache.vllm.hooks import apply_tq_hooks
    apply_tq_hooks()

    # Graph eviction only (works with ANY attention backend):
    from aither_kvcache.vllm.eviction_plugin import install_graph_eviction
    install_graph_eviction()
"""

from .plugin import register
from .hooks import apply_tq_hooks
from .engine import apply_tq_patches

__all__ = ["register", "apply_tq_hooks", "apply_tq_patches"]
