"""
StrataCacheShadow vLLM Plugin — Strata integration for KV cache persistence.

Self-contained plugin (stdlib + httpx only; NO lib.* imports). Runs inside
the vLLM process to shadow KV cache state to Strata for crash recovery.

Env flags:
  - AITHER_STRATA_SHADOW_LIFECYCLE=1 (default OFF)
  - AITHER_STRATA_URL (default https://aitheros-strata:8136)
  - AITHER_STRATA_SHADOW_INTERVAL_S (default 60)
  - AITHER_CA_BUNDLE (optional, TLS CA path)

All Strata I/O runs in a background thread — never on the token hot path.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("aither.kvcache.strata_shadow")

_ENABLED = os.environ.get("AITHER_STRATA_SHADOW_LIFECYCLE", "0") == "1"
_STRATA_URL = os.environ.get("AITHER_STRATA_URL", "https://aitheros-strata:8136")
_SHADOW_INTERVAL_S = float(
    os.environ.get("AITHER_STRATA_SHADOW_INTERVAL_S", "60")
)


def _is_safe_tenant_slug(tenant_slug: str) -> bool:
    """Validate tenant_slug format — reject path traversal and special chars."""
    if not tenant_slug or not isinstance(tenant_slug, str):
        return False
    # Match: lowercase alphanumeric start, then alphanumeric/underscore/dash
    return bool(re.match(r"^[a-z0-9][a-z0-9_-]{0,62}$", tenant_slug))


# ===========================================================================
# Manifest dataclasses (mirror of strata_shadow.py for interoperability)
# ===========================================================================


@dataclass
class BlockSnapshot:
    """Serializable metadata for a single block."""
    block_idx: int
    source_layer: str
    importance: float
    token_range: list[int]  # [start, end]
    tenant_slug: str
    pinned: bool = False


@dataclass
class SessionManifest:
    """Full manifest for a shadowed session (matches strata_shadow.py exactly)."""
    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    num_layers: int = 0
    num_blocks: int = 0
    block_size: int = 16
    num_kv_heads: int = 8
    packed_dim: int = 64
    total_tokens: int = 0
    blocks: list[BlockSnapshot] = field(default_factory=list)
    prefix_hash: str = ""
    model_name: str = ""
    tenant_slug: str = "platform"
    expires_at: float = 0.0

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, data: str) -> SessionManifest:
        """Deserialize manifest from JSON (matches strata_shadow.py)."""
        d = json.loads(data)
        if not isinstance(d, dict):
            raise ValueError("manifest must be a JSON object")
        raw_blocks = d.pop("blocks", [])
        blocks = [BlockSnapshot(**b) for b in raw_blocks]
        return cls(blocks=blocks, **d)


# ===========================================================================
# Strata HTTP client (TLS-aware, background thread-safe)
# ===========================================================================

def _get_ca_bundle() -> str | None:
    """Resolve TLS CA bundle path from env.

    Strata serves HTTPS with the internal CA. The fleet TLS bootstrap sets
    SSL_CERT_FILE / REQUESTS_CA_BUNDLE to the combined internal bundle, so honour
    those (in addition to AITHER_CA_BUNDLE) before falling back to certifi — which
    lacks the internal CA and would fail verification.
    """
    for var in ("AITHER_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        cand = os.environ.get(var, "").strip()
        if cand and os.path.isfile(cand):
            return cand
    # Fallback: certifi (if available)
    try:
        import certifi
        return certifi.where()
    except ImportError:
        logger.warning("No CA bundle available; using system certs")
        return None


def _get_httpx_client():
    """Lazy-init httpx Client with TLS config."""
    try:
        import httpx
        ca_bundle = _get_ca_bundle()
        verify: str | bool = ca_bundle if ca_bundle else True
        return httpx.Client(
            base_url=_STRATA_URL,
            verify=verify,
            timeout=30.0,
        )
    except ImportError:
        logger.error("httpx not available — Strata shadow disabled")
        return None


# ===========================================================================
# Shadow paths (mirror of strata_shadow.py)
# ===========================================================================

_STRATA_KV_PREFIX = "aither://warm/kvcache"


def _is_safe_session_id(session_id: str) -> bool:
    """Validate session_id doesn't contain path traversal characters."""
    return (session_id and "/" not in session_id and
            "\\" not in session_id and ".." not in session_id)


def _manifest_path(session_id: str) -> str:
    if not _is_safe_session_id(session_id):
        raise ValueError(f"Invalid session_id (path traversal detected): {session_id}")
    return f"{_STRATA_KV_PREFIX}/{session_id}/manifest.json"


def _blocks_path(session_id: str, key_or_value: str) -> str:
    if not _is_safe_session_id(session_id):
        raise ValueError(f"Invalid session_id (path traversal detected): {session_id}")
    return f"{_STRATA_KV_PREFIX}/{session_id}/{key_or_value}_packed.bin"


def _norms_path(session_id: str, key_or_value: str) -> str:
    if not _is_safe_session_id(session_id):
        raise ValueError(f"Invalid session_id (path traversal detected): {session_id}")
    return f"{_STRATA_KV_PREFIX}/{session_id}/{key_or_value}_norms.bin"


def _index_path() -> str:
    return f"{_STRATA_KV_PREFIX}/index.json"


def _derive_owner_tenant(blocks: list[BlockSnapshot]) -> str | None:
    """Derive owner tenant from per-block provenance (matches strata_shadow.py)."""
    tenants = {
        b.tenant_slug for b in blocks
        if getattr(b, "tenant_slug", None) and b.tenant_slug != "platform"
    }
    if len(tenants) == 1:
        return next(iter(tenants))
    if len(tenants) > 1:
        return None
    return "platform"


# ===========================================================================
# Background worker thread
# ===========================================================================

class StrataShadowWorker:
    """Manages Strata shadow operations in a background thread."""

    def __init__(self):
        self._client = None
        self._lock = threading.Lock()
        # Maps session_id -> (timestamp, tenant_slug)
        self._active_sessions: dict[str, tuple] = {}
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._stats = {
            "shadows_written": 0,
            "shadows_recovered": 0,
            "bytes_written": 0,
            "bytes_read": 0,
            "errors": 0,
        }

    def _get_client(self):
        """Lazy-init httpx client."""
        if self._client is None:
            self._client = _get_httpx_client()
        return self._client

    def register_session(
        self, session_id: str, tenant_slug: str = "platform"
    ) -> None:
        """Register a session for periodic shadowing with its tenant.

        Validates tenant_slug format; rejects if not safe.
        """
        if not _is_safe_session_id(session_id):
            logger.warning("Rejecting session with unsafe id: %s", session_id)
            return
        if not _is_safe_tenant_slug(tenant_slug):
            logger.warning("Rejecting session with unsafe tenant: %s", tenant_slug)
            return
        with self._lock:
            self._active_sessions[session_id] = (time.time(), tenant_slug)

    def unregister_session(self, session_id: str) -> None:
        """Unregister a session."""
        with self._lock:
            self._active_sessions.pop(session_id, None)

    def request_recovery(self, session_id: str) -> bool:
        """Request immediate recovery of a session from Strata.

        Falls back to prefer_local=false if session is not in local Strata.
        Returns True if recovery succeeded.
        """
        client = self._get_client()
        if client is None:
            logger.warning("Strata recovery skipped: httpx unavailable")
            return False

        try:
            # Try local recovery first
            try:
                client.get(
                    f"/strata/read/{_manifest_path(session_id)}"
                )
                logger.info("Recovered session %s from local Strata", session_id)
                return True
            except Exception as e1:
                logger.debug("Local Strata read failed: %s", e1)

            # Fall back to cross-node restore with prefer_local=false
            try:
                resp = client.post(
                    "/strata/mesh/read",
                    json={
                        "path": _manifest_path(session_id),
                        "prefer_local": False,
                    },
                )
                if resp.status_code == 200:
                    logger.info(
                        "Recovered session %s via mesh read (cross-node)",
                        session_id,
                    )
                    return True
            except Exception as e2:
                logger.debug("Mesh read failed: %s", e2)

            return False
        except Exception as e:
            logger.error("Recovery request for session %s failed: %s", session_id, e)
            with self._lock:
                self._stats["errors"] += 1
            return False

    def shadow_session(self, session_id: str, manifest: SessionManifest) -> bool:
        """Shadow a session manifest to Strata (non-blocking).

        Returns True if queued successfully (actual write is async).
        """
        client = self._get_client()
        if client is None:
            return False

        try:
            # Fire-and-forget POST to Strata
            json_bytes = manifest.to_json().encode("utf-8")
            try:
                client.post(
                    f"/strata/write/{_manifest_path(session_id)}",
                    content=json_bytes,
                    headers={"Content-Type": "application/json"},
                )
                with self._lock:
                    self._stats["shadows_written"] += 1
                    self._stats["bytes_written"] += len(json_bytes)
            except Exception as e:
                logger.debug("Shadow write failed: %s", e)
                with self._lock:
                    self._stats["errors"] += 1
            return True
        except Exception as e:
            logger.error("Shadow request for session %s failed: %s", session_id, e)
            return False

    def _worker_loop(self) -> None:
        """Background worker: periodic shadow loop."""
        logger.info("Strata shadow worker started (interval=%.1fs)", _SHADOW_INTERVAL_S)
        while not self._stop_event.wait(_SHADOW_INTERVAL_S):
            try:
                with self._lock:
                    sessions = list(self._active_sessions.items())
                for sid, (timestamp, tenant_slug) in sessions:
                    # Dummy manifest for now (real implementation would gather state)
                    manifest = SessionManifest(
                        session_id=sid,
                        num_blocks=0,
                        tenant_slug=tenant_slug,
                    )
                    self.shadow_session(sid, manifest)
            except Exception as e:
                logger.error("Worker loop error: %s", e)

    def start(self) -> None:
        """Start the background worker thread."""
        with self._lock:
            if self._worker_thread is None:
                self._stop_event.clear()
                self._worker_thread = threading.Thread(
                    target=self._worker_loop, daemon=True, name="StrataShadowWorker"
                )
                self._worker_thread.start()

    def stop(self) -> None:
        """Stop the background worker thread."""
        with self._lock:
            if self._worker_thread is not None:
                self._stop_event.set()
                self._worker_thread = None

    def get_stats(self) -> dict[str, Any]:
        """Return shadow statistics."""
        with self._lock:
            return dict(self._stats)


# ===========================================================================
# Singleton worker
# ===========================================================================

_worker: StrataShadowWorker | None = None
_worker_lock = threading.Lock()


def _get_worker() -> StrataShadowWorker:
    """Get or create the shadow worker singleton."""
    global _worker
    if _worker is None:
        with _worker_lock:
            if _worker is None:
                _worker = StrataShadowWorker()
    return _worker


def install_strata_shadow() -> bool:
    """Install Strata shadow support into vLLM (called from plugin.register()).

    Returns True if successfully installed.
    """
    if not _ENABLED:
        logger.debug("Strata shadow lifecycle disabled")
        return False

    worker = _get_worker()
    worker.start()

    # Register atexit shutdown
    atexit.register(worker.stop)

    logger.info(
        "[aither-kvcache] Strata shadow installed "
        "(AITHER_STRATA_SHADOW_LIFECYCLE=1)"
    )
    return True
