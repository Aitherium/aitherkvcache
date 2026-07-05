"""
vLLM plugin entrypoint for the aither-kvcache wheel.

Registered via pyproject.toml entry_points under vllm.general_plugins.
vLLM loads this at startup in every process (API server + engine workers).

Registers:
  1. TurboQuant CUSTOM attention backend (--attention-backend CUSTOM)
  2. Graph-aware eviction (replaces LRU with semantic scoring)
  3. Per-request tenant isolation (default OFF; AITHER_REQUEST_TENANT_ISOLATION=1)

CANONICAL SOURCE. Synced by .github/workflows/sync-kvcache.yml into the public
Aitherium/aitherkvcache repo as aither_kvcache/vllm/plugin.py, alongside
tenant_isolation.py. Edit here, not in the public repo.
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
            AttentionBackendEnum,
            register_backend,
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

    # 3. Per-request tenant isolation (default OFF; AITHER_REQUEST_TENANT_ISOLATION=1).
    # Rides this plugin so it loads in every process (API server + EngineCore
    # workers). Non-fatal: a failure leaves the process on the instance tenant.
    if os.environ.get("AITHER_REQUEST_TENANT_ISOLATION") == "1":
        try:
            from .tenant_isolation import install_tenant_isolation
            install_tenant_isolation()
        except ImportError:
            pass
        except Exception as e:
            print(f"[aither-kvcache] Tenant isolation install failed: {e}",
                  file=sys.stderr, flush=True)
