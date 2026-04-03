"""
aither-kvcache vLLM integration.

Usage:
    pip install aither-kvcache[vllm]

    # Hook mode (recommended -- zero graph breaks, CUDA-graphable):
    from aither_kvcache.vllm.hooks import apply_tq_hooks
    apply_tq_hooks()

    # Plugin mode (legacy -- registers as CUSTOM backend):
    vllm serve model --attention-backend CUSTOM

    # Engine patches (PRIMARY mode -- TQ IS the KV cache):
    from aither_kvcache.vllm.engine import apply_tq_patches
    apply_tq_patches(bits=4)
"""

from .plugin import register
from .hooks import apply_tq_hooks
from .engine import apply_tq_patches

__all__ = ["register", "apply_tq_hooks", "apply_tq_patches"]
