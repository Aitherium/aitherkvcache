"""
KV Cache Warmth Reporter — Fire-and-forget telemetry for KV cache allocations.

Enqueues warmth registrations (allocate/free) and batches them to Nexus via
POST /kv-cache/register-warmth. All I/O is in a background thread.

Env flags:
  - AITHER_NEXUS_KV_ENABLED=1 (default OFF)
  - AITHER_NEXUS_URL (default https://aitheros-nexus:8122)
  - AITHER_NODE_ID (default hostname)
  - AITHER_CA_BUNDLE (optional, TLS CA path)
"""

from __future__ import annotations

import collections
import logging
import os
import socket
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger("aither.kvcache.warmth")

_ENABLED = os.environ.get("AITHER_NEXUS_KV_ENABLED", "0") == "1"
_NEXUS_URL = os.environ.get("AITHER_NEXUS_URL", "https://aitheros-nexus:8122")
_NODE_ID = os.environ.get("AITHER_NODE_ID", "").strip() or socket.gethostname()
# Platform credential — Nexus /kv-cache/* requires X-Internal-Key (require_nexus_auth).
# Read lazily at send time so a key rotated into the env after import is picked up.
_BATCH_INTERVAL_S = 1.0
_MAX_QUEUE_SIZE = 10000


def _is_safe_tenant_slug(tenant_slug: str) -> bool:
    """Validate tenant_slug format — reject path traversal and special chars."""
    import re
    if not tenant_slug or not isinstance(tenant_slug, str):
        return False
    # Match: lowercase alphanumeric start, then alphanumeric/underscore/dash
    return bool(re.match(r"^[a-z0-9][a-z0-9_-]{0,62}$", tenant_slug))


# ===========================================================================
# Warmth event dataclass
# ===========================================================================


@dataclass
class WarmthEvent:
    """A single KV cache warmth event (allocate or free)."""
    tenant_slug: str
    session_id: str
    node_id: str
    tier: str  # 'vram', 'ddr5', 'disk'
    block_count: int
    operation: str  # 'allocate' or 'free'
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ===========================================================================
# Background reporter thread
# ===========================================================================


class KVWarmthReporter:
    """Queues and batches KV cache warmth events to Nexus."""

    def __init__(self):
        self._queue: collections.deque[WarmthEvent] = collections.deque(
            maxlen=_MAX_QUEUE_SIZE
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._client = None
        self._stats = {
            "enqueued": 0,
            "dropped": 0,
            "sent": 0,
            "batches": 0,
            "errors": 0,
        }

    def _get_client(self):
        """Lazy-init httpx client."""
        if self._client is None:
            try:
                import httpx
                # Honour the fleet's internal-CA trust: the TLS bootstrap
                # (_05_tls.py) sets SSL_CERT_FILE / REQUESTS_CA_BUNDLE to the
                # combined internal CA bundle. Prefer AITHER_CA_BUNDLE if given,
                # then those standard vars, then certifi. Nexus serves HTTPS with
                # the internal CA, so certifi alone fails (CERTIFICATE_VERIFY_FAILED).
                verify: str | bool = True
                ca_path = ""
                for var in (
                    "AITHER_CA_BUNDLE",
                    "REQUESTS_CA_BUNDLE",
                    "SSL_CERT_FILE",
                ):
                    cand = os.environ.get(var, "").strip()
                    if cand and os.path.isfile(cand):
                        ca_path = cand
                        break
                if ca_path:
                    verify = ca_path
                else:
                    try:
                        import certifi
                        verify = certifi.where()
                    except ImportError:
                        logger.warning(
                            "No CA bundle available; using system certs"
                        )
                self._client = httpx.Client(
                    base_url=_NEXUS_URL,
                    verify=verify,
                    timeout=30.0,
                )
            except ImportError:
                logger.error("httpx not available — KV warmth disabled")
                return None
        return self._client

    def enqueue(
        self,
        tenant_slug: str,
        session_id: str,
        block_count: int,
        operation: str,
    ) -> None:
        """Enqueue a warmth event (allocate or free).

        Non-blocking: event is added to queue; actual POST happens in background.
        """
        if not _ENABLED:
            return
        # Validate inputs: tenant_slug format, session_id non-empty, block_count
        if not (isinstance(tenant_slug, str) and _is_safe_tenant_slug(tenant_slug)
                and isinstance(session_id, str) and session_id.strip() and
                isinstance(block_count, int)):
            return

        event = WarmthEvent(
            tenant_slug=tenant_slug,
            session_id=session_id,
            node_id=_NODE_ID,
            tier="vram",
            block_count=max(0, block_count),
            operation=operation,
            timestamp=time.time(),
        )

        with self._lock:
            # Deque silently drops oldest when full; track overflow
            was_full = len(self._queue) >= self._queue.maxlen
            self._queue.append(event)
            if was_full:
                self._stats["dropped"] += 1
            else:
                self._stats["enqueued"] += 1

    def _drain_and_send(self) -> None:
        """Drain queue and send events to Nexus.

        Nexus `/kv-cache/register-warmth` takes ONE `RegisterWarmthRequest` per
        call (tenant_slug/session_id/node_id/tier/block_count/released), NOT a
        batch, and requires the platform `X-Internal-Key`. We collapse the drained
        queue to the latest event per (tenant, session) — the index only cares
        about current warmth — and POST each with the internal key.
        """
        with self._lock:
            if not self._queue:
                return
            events = list(self._queue)
            self._queue.clear()

        # Collapse to latest state per (tenant, session): a later 'free' or a
        # newer 'allocate' supersedes an earlier one, so we don't spam the index.
        latest: dict[tuple[str, str], WarmthEvent] = {}
        for e in events:
            latest[(e.tenant_slug, e.session_id)] = e

        client = self._get_client()
        if client is None:
            logger.warning(
                "Nexus client unavailable; dropping %d warmth events", len(latest)
            )
            return

        internal_key = os.environ.get("AITHER_INTERNAL_SECRET", "").strip()
        headers = {"X-Internal-Key": internal_key} if internal_key else {}
        sent = 0
        for e in latest.values():
            body = {
                "tenant_slug": e.tenant_slug,
                "session_id": e.session_id,
                "node_id": e.node_id,
                "tier": e.tier,
                "block_count": e.block_count,
                "released": e.operation == "free",
            }
            try:
                response = client.post(
                    "/kv-cache/register-warmth", json=body, headers=headers
                )
                if response.status_code == 200:
                    sent += 1
                else:
                    with self._lock:
                        self._stats["errors"] += 1
                    logger.warning(
                        "Nexus returned %d for warmth (%s/%s)",
                        response.status_code, e.tenant_slug, e.session_id,
                    )
            except Exception as exc:  # noqa: BLE001 — telemetry is non-fatal
                with self._lock:
                    self._stats["errors"] += 1
                logger.error("Failed to send warmth event: %s", exc)
        if sent:
            with self._lock:
                self._stats["sent"] += sent
                self._stats["batches"] += 1
            logger.debug("Sent %d warmth events to Nexus", sent)

    def _worker_loop(self) -> None:
        """Background worker: periodic drain loop."""
        logger.info(
            "KV warmth reporter started (batch interval=%.1fs)",
            _BATCH_INTERVAL_S,
        )
        while not self._stop_event.wait(_BATCH_INTERVAL_S):
            try:
                self._drain_and_send()
            except Exception as e:
                logger.error("Warmth worker error: %s", e)

    def start(self) -> None:
        """Start the background worker thread."""
        with self._lock:
            if self._worker_thread is None:
                self._stop_event.clear()
                self._worker_thread = threading.Thread(
                    target=self._worker_loop, daemon=True, name="KVWarmthReporter"
                )
                self._worker_thread.start()

    def stop(self) -> None:
        """Stop the background worker and flush remaining events."""
        with self._lock:
            if self._worker_thread is not None:
                self._stop_event.set()
                self._worker_thread = None

        # Final flush
        self._drain_and_send()

    def get_stats(self) -> dict[str, Any]:
        """Return reporter statistics."""
        with self._lock:
            return dict(self._stats)


# ===========================================================================
# Singleton reporter
# ===========================================================================

_reporter: KVWarmthReporter | None = None
_reporter_lock = threading.Lock()


def _get_reporter() -> KVWarmthReporter:
    """Get or create the warmth reporter singleton."""
    global _reporter
    if _reporter is None:
        with _reporter_lock:
            if _reporter is None:
                _reporter = KVWarmthReporter()
    return _reporter


def install_kv_warmth() -> bool:
    """Install KV warmth reporter (called from plugin.register()).

    Returns True if successfully installed.
    """
    if not _ENABLED:
        logger.debug("KV warmth reporter disabled")
        return False

    reporter = _get_reporter()
    reporter.start()

    logger.info(
        "[aither-kvcache] KV warmth reporter installed "
        "(AITHER_NEXUS_KV_ENABLED=1)"
    )
    return True


def report_allocate(
    tenant_slug: str, session_id: str, block_count: int
) -> None:
    """Report KV cache allocation (called from tenant_isolation.tag_blocks seam)."""
    reporter = _get_reporter()
    reporter.enqueue(tenant_slug, session_id, block_count, "allocate")


def report_free(tenant_slug: str, session_id: str, block_count: int) -> None:
    """Report KV cache free (called from tenant_isolation.untag_blocks seam)."""
    reporter = _get_reporter()
    reporter.enqueue(tenant_slug, session_id, block_count, "free")
