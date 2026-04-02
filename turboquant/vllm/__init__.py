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
from .hooks import apply_tq_hooks

__all__ = ["register", "apply_tq_hooks"]
