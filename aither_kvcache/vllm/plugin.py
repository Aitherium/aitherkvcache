"""
vLLM plugin entrypoint.

Registered via pyproject.toml entry_points under vllm.general_plugins.
vLLM loads this at startup in every process (API server + engine workers).

Registers:
  1. TurboQuant CUSTOM attention backend (--attention-backend CUSTOM)
  2. Graph-aware eviction (replaces LRU with semantic scoring)
"""

import logging
import os

logger = logging.getLogger("aither_kvcache.vllm")


def register():
    """Register aither-kvcache components in vLLM."""

    # 1. Register TurboQuant attention backend
    try:
        from vllm.v1.attention.backends.registry import (
            register_backend,
            AttentionBackendEnum,
        )
        register_backend(
            AttentionBackendEnum.CUSTOM,
            "aither_kvcache.vllm.backend.TurboQuantBackend",
        )
        logger.info("Registered TurboQuant CUSTOM attention backend")
    except ImportError:
        logger.debug("vLLM v1 not available — backend not registered")
    except Exception as e:
        logger.warning("Failed to register TurboQuant backend: %s", e)

    # 2. Install graph-aware eviction (unless disabled)
    if os.environ.get("AITHER_TQ_NO_GRAPH_EVICTION") != "1":
        try:
            from .eviction_plugin import install_graph_eviction
            install_graph_eviction()
        except ImportError:
            logger.debug("Graph eviction not available")
        except Exception as e:
            logger.warning("Failed to install graph eviction: %s", e)
