"""
GraphEvictionAdvisor — Background thread that pre-computes eviction/prefetch
rankings from KVCacheGraph so the hot decode path never blocks on graph queries.

The advisor periodically calls graph.suggest_eviction() and caches the result.
Readers (your decode loop) atomically read the pre-computed list — no lock
contention on the decode path.

Standalone — no AitherOS dependencies. Requires turboquant.kvcache_graph.

Usage:
    from turboquant.kvcache_graph import KVCacheGraph
    from turboquant.eviction_advisor import GraphEvictionAdvisor

    graph = KVCacheGraph(protected_sources={"system"})
    advisor = GraphEvictionAdvisor(graph)
    advisor.start()

    # Hot decode path — zero lock contention:
    candidates = advisor.get_eviction_candidates(n=16)
    prefetch = advisor.get_prefetch_candidates(active_blocks, n=8)

    # Shutdown:
    advisor.stop()
"""

import atexit
import logging
import threading
import time
from typing import Dict, List, Optional

from .kvcache_graph import KVCacheGraph

logger = logging.getLogger("turboquant.eviction_advisor")


class GraphEvictionAdvisor:
    """Background thread that pre-computes eviction/prefetch rankings.

    Hot path reads a pre-computed list (atomic reference swap, no lock).
    Background thread recomputes every ``interval`` seconds.
    Falls back to None (caller uses FIFO) if ranking is stale.

    Args:
        graph: The KVCacheGraph instance to query. If None, the advisor
            attempts to use the module-level singleton via
            ``get_kvcache_graph()``.
        interval: Seconds between recomputation cycles. Default 0.5.
        max_stale: Maximum age (seconds) of a ranking before it is
            considered stale and ``get_eviction_candidates`` returns
            None. Default 2.0.
        eviction_batch: Number of eviction candidates to pre-compute
            per cycle. Default 256.
    """

    def __init__(
        self,
        graph: Optional[KVCacheGraph] = None,
        interval: float = 0.5,
        max_stale: float = 2.0,
        eviction_batch: int = 256,
    ):
        self._graph = graph
        self._interval = interval
        self._max_stale = max_stale
        self._eviction_batch = eviction_batch

        # Pre-computed rankings — atomically swapped by background thread.
        # Readers never acquire a lock; they just read the reference.
        self._eviction_ranking: Optional[List[int]] = None
        self._ranking_ts: float = 0.0

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = False

        # Stats
        self._recompute_count = 0
        self._stale_fallbacks = 0

    # ------------------------------------------------------------------
    # PUBLIC API (called from hot path — no locks)
    # ------------------------------------------------------------------

    def get_eviction_candidates(self, n: int) -> Optional[List[int]]:
        """Return top-n eviction candidates from pre-computed ranking.

        Returns None if ranking is stale (older than max_stale seconds),
        signalling the caller to fall back to FIFO/LRU.

        This method is lock-free and safe to call from the decode path.
        """
        ranking = self._eviction_ranking
        if ranking is None:
            return None
        age = time.monotonic() - self._ranking_ts
        if age > self._max_stale:
            self._stale_fallbacks += 1
            return None
        return ranking[:n]

    def get_prefetch_candidates(
        self, active_block_idxs: List[int], n: int = 8
    ) -> Optional[List[int]]:
        """Suggest cold-tier blocks to prefetch based on graph neighbors.

        Queries the graph directly (lightweight — only neighbor lookup).
        Returns None if graph unavailable.
        """
        graph = self._get_graph()
        if graph is None:
            return None
        try:
            return graph.suggest_prefetch(active_block_idxs, max_suggestions=n)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background recompute thread."""
        if self._started:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker,
            name="kvcache-graph-advisor",
            daemon=True,
        )
        self._thread.start()
        self._started = True
        atexit.register(self.stop)
        logger.info(
            "GraphEvictionAdvisor started (interval=%.1fs, max_stale=%.1fs)",
            self._interval,
            self._max_stale,
        )

    def stop(self) -> None:
        """Stop the background thread."""
        if not self._started:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._started = False
        self._thread = None
        logger.info(
            "GraphEvictionAdvisor stopped (recomputes=%d, stale_fallbacks=%d)",
            self._recompute_count,
            self._stale_fallbacks,
        )

    @property
    def is_running(self) -> bool:
        return self._started and self._thread is not None and self._thread.is_alive()

    def get_stats(self) -> Dict:
        return {
            "running": self.is_running,
            "recompute_count": self._recompute_count,
            "stale_fallbacks": self._stale_fallbacks,
            "ranking_size": len(self._eviction_ranking) if self._eviction_ranking else 0,
            "ranking_age_s": round(time.monotonic() - self._ranking_ts, 2)
            if self._ranking_ts
            else None,
            "interval": self._interval,
            "max_stale": self._max_stale,
        }

    # ------------------------------------------------------------------
    # BACKGROUND WORKER
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        """Background loop: recompute eviction ranking every interval."""
        while not self._stop.is_set():
            try:
                self._recompute()
            except Exception as exc:
                logger.debug("GraphEvictionAdvisor recompute error: %s", exc)
            self._stop.wait(self._interval)

    def _recompute(self) -> None:
        """Recompute eviction ranking from KVCacheGraph."""
        graph = self._get_graph()
        if graph is None:
            return
        ranking = graph.suggest_eviction(self._eviction_batch)
        # Atomic swap — readers see either old or new, never partial
        self._eviction_ranking = ranking
        self._ranking_ts = time.monotonic()
        self._recompute_count += 1

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    def _get_graph(self) -> Optional[KVCacheGraph]:
        """Get the graph instance."""
        if self._graph is not None:
            return self._graph
        try:
            from .kvcache_graph import get_kvcache_graph
            return get_kvcache_graph()
        except Exception:
            return None


def reorder_by_ranking(
    block_indices: List[int], ranked: List[int]
) -> List[int]:
    """Reorder block_indices so graph-recommended eviction candidates come first.

    Blocks in ``ranked`` appear first (in ranking order), followed by
    remaining blocks in their original order.

    Useful for integrating with existing eviction code that expects an
    ordered list of candidates.
    """
    ranked_set = set(ranked)
    front = [b for b in ranked if b in set(block_indices)]
    front_set = set(front)
    tail = [b for b in block_indices if b not in front_set]
    return front + tail
