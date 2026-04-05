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
    import sys

    print("[aither-kvcache] Plugin register() called", file=sys.stderr, flush=True)

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
        print("[aither-kvcache] Registered TurboQuant CUSTOM backend",
              file=sys.stderr, flush=True)
    except ImportError:
        pass  # vLLM v1 not available
    except Exception as e:
        print(f"[aither-kvcache] Backend registration failed: {e}",
              file=sys.stderr, flush=True)

    # 2. Install graph-aware eviction (unless disabled)
    if os.environ.get("AITHER_TQ_NO_GRAPH_EVICTION") != "1":
        try:
            from .eviction_plugin import install_graph_eviction
            install_graph_eviction()
            print("[aither-kvcache] Graph-aware eviction installed",
                  file=sys.stderr, flush=True)
        except ImportError:
            pass
        except Exception as e:
            print(f"[aither-kvcache] Eviction install failed: {e}",
                  file=sys.stderr, flush=True)
