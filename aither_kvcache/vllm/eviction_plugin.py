"""
Graph-aware eviction plugin for vLLM's KV cache block manager.

Replaces vLLM's default LRU eviction with graph-informed scoring:
  - System prompt blocks are never evicted (protected sources)
  - Blocks with high co-attendance are kept together
  - Semantically similar blocks are preserved as a group
  - Prefix cache hits increase retention priority

Integration: monkey-patches FreeKVCacheBlockQueue.popleft() to consult
the GraphEvictionAdvisor before returning the "least important" block
instead of the "least recently used" block.

Usage:
    from aither_kvcache.vllm.eviction_plugin import install_graph_eviction
    install_graph_eviction()  # call after vLLM is initialized

Or via plugin entry point (automatic if aither-kvcache is installed):
    # pyproject.toml registers this in vllm.general_plugins
"""

import logging
import threading

logger = logging.getLogger("aither.kvcache.eviction")

_installed = False
_graph = None
_advisor = None
_lock = threading.Lock()


def _get_graph():
    """Lazy-init the KVCacheGraph singleton."""
    global _graph
    if _graph is None:
        from ..kvcache_graph import KVCacheGraph
        _graph = KVCacheGraph(protected_sources={"system", "tools"})
    return _graph


def _get_advisor():
    """Lazy-init the GraphEvictionAdvisor singleton."""
    global _advisor
    if _advisor is None:
        from ..eviction_advisor import GraphEvictionAdvisor
        _advisor = GraphEvictionAdvisor(
            graph=_get_graph(),
            interval=0.5,   # recompute rankings every 500ms
            max_stale=2.0,  # fall back to LRU if ranking > 2s old
        )
        _advisor.start()
        logger.info("GraphEvictionAdvisor started (interval=0.5s, max_stale=2.0s)")
    return _advisor


def register_block(block_id: int, source_label: str = "user",
                   importance: float = 0.5, token_range: tuple = (0, 0)):
    """Register a block in the eviction graph when it's allocated."""
    graph = _get_graph()
    graph.add_block(block_id, source_label, importance, token_range)


def unregister_block(block_id: int):
    """Remove a block from the eviction graph when it's freed."""
    graph = _get_graph()
    try:
        graph.remove_block(block_id)
    except KeyError:
        pass  # block wasn't registered


def on_attention_step(active_block_ids: list):
    """Feed attention patterns to the graph for co-attendance tracking."""
    graph = _get_graph()
    graph.on_attention_step(active_block_ids)


def on_prefix_hit(request_id: str, block_ids: list):
    """Track prefix cache reuse events."""
    graph = _get_graph()
    graph.on_prefix_hit(request_id, block_ids)


def install_graph_eviction():
    """Monkey-patch vLLM's FreeKVCacheBlockQueue to use graph-aware eviction.

    Instead of always popping the LRU block, consults the GraphEvictionAdvisor
    for the least important block. Falls back to LRU if the advisor has no
    ranking (cold start or stale data).
    """
    global _installed
    if _installed:
        return

    try:
        from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue
    except ImportError:
        logger.debug("vLLM v1 not available — graph eviction not installed")
        return

    _original_popleft = FreeKVCacheBlockQueue.popleft

    def _graph_aware_popleft(self):
        """Pop the least important block according to graph scoring.

        Falls back to standard LRU popleft if:
        - Advisor has no ranking yet (cold start)
        - Ranking is stale (> max_stale seconds old)
        - Only 1 block in queue (nothing to reorder)
        """
        if self.num_free_blocks <= 1:
            return _original_popleft(self)

        advisor = _get_advisor()
        candidates = advisor.get_eviction_candidates(n=1)

        if candidates is None:
            # No ranking available — fall back to LRU
            return _original_popleft(self)

        target_id = candidates[0]

        # Walk the free list to find the target block
        curr = self.fake_free_list_head.next_free_block
        while curr is not None and curr is not self.fake_free_list_tail:
            if curr.block_id == target_id:
                # Found it — remove from current position
                self.remove(curr)
                return curr
            curr = curr.next_free_block

        # Target not in free list (already allocated) — fall back to LRU
        return _original_popleft(self)

    FreeKVCacheBlockQueue.popleft = _graph_aware_popleft
    _installed = True
    logger.info("Graph-aware eviction installed (replaces LRU popleft)")


def get_stats() -> dict:
    """Get eviction graph and advisor statistics."""
    stats = {}
    if _graph is not None:
        stats["graph"] = _graph.get_stats()
    if _advisor is not None:
        stats["advisor"] = _advisor.get_stats()
    return stats
