"""
vLLM plugin entrypoint.

Registered via pyproject.toml entry_points. vLLM loads this at startup
in every process (API server + engine workers). No import hooks needed.
"""

import logging

logger = logging.getLogger("turboquant.vllm")


def register():
    """Register TurboQuant as a CUSTOM attention backend in vLLM."""
    try:
        from vllm.v1.attention.backends.registry import (
            register_backend,
            AttentionBackendEnum,
        )
        register_backend(
            AttentionBackendEnum.CUSTOM,
            "turboquant.vllm.backend.TurboQuantBackend",
        )
        logger.info("Registered TurboQuant CUSTOM attention backend")
    except ImportError:
        logger.debug("vLLM not installed — TurboQuant backend not registered")
    except Exception as e:
        logger.warning("Failed to register TurboQuant backend: %s", e)
