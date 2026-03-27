"""
aither-kvcache vLLM integration.

Usage:
    pip install aither-kvcache[vllm]
    VLLM_ATTENTION_BACKEND=CUSTOM vllm serve ...

Or register manually:
    from turboquant.vllm import register
    register()
"""

from .plugin import register

__all__ = ["register"]
