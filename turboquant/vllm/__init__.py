"""
aither-kvcache vLLM integration.

Usage:
    pip install aither-kvcache[vllm]
    vllm serve model --attention-backend CUSTOM

Or register manually:
    from turboquant.vllm import register
    register()
"""

from .plugin import register

__all__ = ["register"]
