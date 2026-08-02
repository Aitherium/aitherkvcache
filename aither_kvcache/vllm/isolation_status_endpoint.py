"""
GET /tq/isolation endpoint — expose tenant isolation status.

When the serving layer imports (API-server process only), non-fatally adds
a FastAPI route returning isolation status + counts. Requires API-Key auth.

Returns JSON:
  {
    "enabled": bool,
    "serving_patched": bool,
    "engine_patched": bool,
    "block_manager_patched": bool,
    "key_present": bool,
    "verified": bool,
    "n_tagged_blocks": int,
    "tenants": [list of tenant slugs]
  }
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("aither.kvcache.isolation_endpoint")


def _attach_route(vllm_app) -> bool:
    """Attach GET /tq/isolation to an already-built FastAPI app. Idempotent."""
    if vllm_app is None:
        return False
    # Don't double-register if build_app runs more than once.
    for r in getattr(vllm_app, "routes", []):
        if getattr(r, "path", None) == "/tq/isolation":
            return True

    try:
        from .tenant_isolation import (
            block_tenant_count,
            get_isolation_status,
        )
    except (ImportError, AttributeError):
        logger.debug("Tenant isolation helpers not found; endpoint skipped")
        return False

    from fastapi import Request
    from fastapi.responses import JSONResponse

    @vllm_app.get("/tq/isolation", tags=["diagnostics"])
    async def get_isolation_status_endpoint(request: Request) -> Any:
        """GET /tq/isolation — Return isolation status + block counts.

        Requires X-API-Key header matching AITHER_TENANT_SIGNING_KEY. Does not
        enumerate tenant slugs to prevent tenant-discovery attacks.
        """
        expected_key = os.environ.get("AITHER_TENANT_SIGNING_KEY", "").strip()
        provided_key = request.headers.get("X-API-Key", "").strip()
        if not expected_key or provided_key != expected_key:
            logger.warning("Unauthorized access to /tq/isolation endpoint")
            return JSONResponse({"error": "Unauthorized"}, status_code=403)

        status = get_isolation_status()
        return JSONResponse({**status, "n_tagged_blocks": block_tenant_count()})

    logger.info("[aither-kvcache] Installed GET /tq/isolation endpoint")
    return True


def install_isolation_status_endpoint() -> bool:
    """
    Install GET /tq/isolation into the vLLM OpenAI API server.

    vLLM v1 builds the served FastAPI app lazily via ``build_app(args)`` at
    startup — the module-level ``app`` is ``None`` at plugin-register time.
    So we wrap ``build_app`` to attach our route to the REAL app once it is
    constructed. Runs only in the API-server process (where build_app exists).
    Non-fatal: any failure logs and returns False without crashing vLLM.
    """
    try:
        from vllm.entrypoints.openai import api_server as _api
    except (ImportError, AttributeError):
        logger.debug("vLLM api_server not importable; isolation endpoint skipped")
        return False

    # Fast path: if an app already exists (older vLLM), attach directly.
    existing = getattr(_api, "app", None)
    if existing is not None:
        return _attach_route(existing)

    wrapped_any = False

    # Primary hook: wrap build_app (fires if register runs before app construction).
    build_app = getattr(_api, "build_app", None)
    if build_app is not None and not getattr(build_app, "_tq_wrapped", False):
        def _wrapped_build_app(*args, **kwargs):
            app = build_app(*args, **kwargs)
            try:
                _attach_route(app)
            except Exception as exc:  # noqa: BLE001 — never break app construction
                logger.warning("Failed to attach /tq/isolation via build_app: %s", exc)
            return app

        _wrapped_build_app._tq_wrapped = True
        _api.build_app = _wrapped_build_app
        wrapped_any = True

    # Belt-and-suspenders: wrap serve_http, the LAST startup step. vLLM calls
    # `serve_http(app, ...)` after build_app + init_app_state, so this fires even
    # when plugin register() runs AFTER build_app (the real-startup ordering that
    # made the build_app-only wrap lose the race). serve_http resolves via the
    # module global at call time, so reassigning it here is honoured.
    serve_http = getattr(_api, "serve_http", None)
    if serve_http is not None and not getattr(serve_http, "_tq_wrapped", False):
        def _wrapped_serve_http(app, *args, **kwargs):
            try:
                _attach_route(app)
            except Exception as exc:  # noqa: BLE001 — never break serving
                logger.warning("Failed to attach /tq/isolation via serve_http: %s", exc)
            return serve_http(app, *args, **kwargs)

        _wrapped_serve_http._tq_wrapped = True
        _api.serve_http = _wrapped_serve_http
        wrapped_any = True

    if wrapped_any:
        logger.info("[aither-kvcache] Hooked app startup for /tq/isolation endpoint")
    return wrapped_any
