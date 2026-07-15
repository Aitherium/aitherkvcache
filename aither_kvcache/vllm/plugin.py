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

    # 1b. Apply TQ engine patches when armed (D-395 fix). Registering the
    # CUSTOM backend alone is NOT activation: PRIMARY mode also needs the
    # engine patches (page_size + uint8 reshape + spill-on-free) or the KV
    # cache stays a float tensor and the backend fail-closes at first
    # forward. The plugin loads in every vLLM process (API server + engine
    # workers) BEFORE model init, so class-level patches land in time.
    # Armed by the same envs the backend reads: AITHER_TQ_MODE / AITHER_TQ_BITS.
    # No envs set -> no patches -> stock behavior (backend stays dormant).
    tq_mode = os.environ.get("AITHER_TQ_MODE", "")
    tq_bits_str = os.environ.get("AITHER_TQ_BITS", "")
    if tq_mode.endswith("-primary"):
        # D-411: PRIMARY is EAGER-ONLY today. Under torch.compile +
        # piecewise CUDA graphs (fork 0.19.1) the padded splitting-op
        # inputs make the layer-0 hook write trip a device-side index
        # assert during decode replay and the engine dies mid-generation.
        print("[aither-kvcache] WARNING: TQ PRIMARY mode is EAGER-ONLY — "
              "pass --enforce-eager (VLLM_TQ_ENFORCE_EAGER=1). Compiled/"
              "cudagraph mode crashes with a device-side assert (D-411).",
              file=sys.stderr, flush=True)
    if tq_mode or tq_bits_str in ("2", "3", "4"):
        try:
            try:
                from .engine import apply_tq_patches  # wheel layout
            except ImportError:
                from lib.gpu.turboquant.vllm_engine import (  # monorepo layout
                    apply_tq_patches,
                )
            bits = int(tq_bits_str) if tq_bits_str in ("2", "3", "4") else 4
            ok = apply_tq_patches(bits=bits)
            print(f"[aither-kvcache] Engine patches "
                  f"{'OK' if ok else 'PARTIAL'} (mode={tq_mode or 'shadow'}, "
                  f"bits={bits})", file=sys.stderr, flush=True)
        except ImportError:
            print("[aither-kvcache] Engine patch module unavailable — TQ "
                  "PRIMARY will fail-closed if selected", file=sys.stderr,
                  flush=True)
        except Exception as e:
            print(f"[aither-kvcache] Engine patches failed: {e}",
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
    else:
        # Warn if multi-tenant environment but isolation disabled
        if os.environ.get("AITHER_MULTI_TENANT", "0") == "1":
            print("[aither-kvcache] WARNING: AITHER_MULTI_TENANT=1 but "
                  "AITHER_REQUEST_TENANT_ISOLATION is OFF; blocks will not be "
                  "isolated per-request", file=sys.stderr, flush=True)

    # 3a. Isolation status endpoint (GET /tq/isolation).
    # Installs only in the API-server process (not engine workers).
    # Non-fatal: if serving layer unavailable, skips silently.
    try:
        from .isolation_status_endpoint import install_isolation_status_endpoint
        install_isolation_status_endpoint()
    except ImportError:
        pass
    except Exception as e:
        print(f"[aither-kvcache] Isolation endpoint install failed: {e}",
              file=sys.stderr, flush=True)

    # 4. Strata shadow persistence (default OFF; AITHER_STRATA_SHADOW_LIFECYCLE=1).
    # Shadows KV cache state to Strata for crash recovery. Runs in background
    # thread, never on the hot path. Non-fatal if Strata unavailable.
    if os.environ.get("AITHER_STRATA_SHADOW_LIFECYCLE") == "1":
        try:
            from .strata_shadow_plugin import install_strata_shadow
            install_strata_shadow()
            print("[aither-kvcache] Strata shadow installed",
                  file=sys.stderr, flush=True)
        except ImportError:
            pass
        except Exception as e:
            print(f"[aither-kvcache] Strata shadow install failed: {e}",
                  file=sys.stderr, flush=True)

    # 5. KV warmth reporting (default OFF; AITHER_NEXUS_KV_ENABLED=1).
    # Reports cache allocations/frees to Nexus for telemetry. Background thread,
    # fire-and-forget, never blocks token generation.
    if os.environ.get("AITHER_NEXUS_KV_ENABLED") == "1":
        try:
            from .kv_warmth_client import install_kv_warmth
            install_kv_warmth()
            print("[aither-kvcache] KV warmth reporter installed",
                  file=sys.stderr, flush=True)
        except ImportError:
            pass
        except Exception as e:
            print(f"[aither-kvcache] KV warmth install failed: {e}",
                  file=sys.stderr, flush=True)
