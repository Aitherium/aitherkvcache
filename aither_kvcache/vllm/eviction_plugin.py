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
from typing import Optional

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
    """Install graph-aware eviction into vLLM.

    Patches two things:
    1. FreeKVCacheBlockQueue.popleft — graph-scored eviction instead of LRU
    2. TritonAttentionImpl.forward — auto-registers blocks and tracks
       co-attendance from slot_mapping and block_table in attn_metadata
    """
    global _installed
    if _installed:
        return

    try:
        from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue
    except ImportError:
        logger.debug("vLLM v1 not available — graph eviction not installed")
        return

    # ── Patch 1: Graph-aware eviction on the free queue ──────────────

    _original_popleft = FreeKVCacheBlockQueue.popleft

    def _graph_aware_popleft(self):
        """Pop the least important block instead of LRU.
        Falls back to LRU if advisor has no ranking."""
        if self.num_free_blocks <= 1:
            return _original_popleft(self)

        advisor = _get_advisor()
        candidates = advisor.get_eviction_candidates(n=1)

        if not candidates:
            return _original_popleft(self)

        target_id = candidates[0]

        curr = self.fake_free_list_head.next_free_block
        while curr is not None and curr is not self.fake_free_list_tail:
            if curr.block_id == target_id:
                self.remove(curr)
                return curr
            curr = curr.next_free_block

        return _original_popleft(self)

    FreeKVCacheBlockQueue.popleft = _graph_aware_popleft

    # ── Patch 2: Auto-register blocks + track co-attendance ──────────

    _patch_attention_forward()

    _installed = True
    logger.info("Graph-aware eviction installed (eviction + attention tracking)")


# Track which blocks we've already registered to avoid re-registration
_registered_blocks = set()
# Only track co-attendance every N forward calls (reduce overhead)
_fwd_counter = 0
_TRACK_INTERVAL = 10  # track every 10th forward call per layer 0


def _patch_attention_forward():
    """Patch TritonAttentionImpl.forward to feed block data to the graph."""
    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
    except ImportError:
        logger.debug("TritonAttentionImpl not found — attention tracking skipped")
        return

    _original_forward = TritonAttentionImpl.forward

    def _tracking_forward(self, layer, query, key, value, kv_cache,
                          attn_metadata, output=None, output_scale=None,
                          output_block_scale=None):
        """Wrapped forward that feeds block info to the eviction graph."""
        global _fwd_counter

        # Run the real attention first
        result = _original_forward(
            self, layer, query, key, value, kv_cache, attn_metadata,
            output=output, output_scale=output_scale,
            output_block_scale=output_block_scale)

        # Only track on layer 0 to avoid 28x overhead
        if (attn_metadata is not None
                and hasattr(self, 'kv_cache_dtype')
                and hasattr(attn_metadata, 'slot_mapping')):

            layer_idx = getattr(self, '_tq_layer_idx', None)
            if layer_idx is not None and layer_idx != 0:
                return result
            # For non-TQ backends, use a simple counter
            _fwd_counter += 1

            try:
                _auto_register_blocks(attn_metadata, kv_cache)
                if _fwd_counter % _TRACK_INTERVAL == 0:
                    _auto_track_coattendance(attn_metadata)
            except Exception:
                pass  # never break attention for graph tracking

        return result

    TritonAttentionImpl.forward = _tracking_forward
    logger.info("Attention forward patched for block tracking")


def _auto_register_blocks(attn_metadata, kv_cache):
    """Register new blocks in the graph from slot_mapping."""
    global _registered_blocks

    if not hasattr(attn_metadata, 'slot_mapping'):
        return

    sm = attn_metadata.slot_mapping
    if sm is None or sm.numel() == 0:
        return

    block_size = kv_cache.shape[2] if kv_cache.ndim >= 3 else 16

    # Get unique block IDs from slot mapping
    valid = sm[sm >= 0]
    if valid.numel() == 0:
        return

    block_ids = (valid // block_size).unique().tolist()

    # Determine source label from context
    # Heuristic: first chunk of a sequence = likely system/user prompt
    # Later chunks = assistant generation
    is_prefill = (hasattr(attn_metadata, 'max_query_len')
                  and attn_metadata.max_query_len > 1)

    graph = _get_graph()
    for bid in block_ids:
        if bid not in _registered_blocks:
            # First blocks in a sequence are higher importance (prompt)
            # Later blocks are lower importance (generation)
            importance = 0.7 if is_prefill else 0.3
            source = "user" if is_prefill else "assistant"
            token_start = bid * block_size
            graph.add_block(bid, source, importance,
                            (token_start, token_start + block_size))
            _registered_blocks.add(bid)


def _auto_track_coattendance(attn_metadata):
    """Feed active block IDs to the graph for co-attendance tracking."""
    if not hasattr(attn_metadata, 'block_table'):
        return

    bt = attn_metadata.block_table
    if bt is None or bt.numel() == 0:
        return

    seq_lens = getattr(attn_metadata, 'seq_lens', None)
    if seq_lens is None:
        return

    graph = _get_graph()
    # For each sequence, get its active blocks and record co-attendance
    for i in range(min(bt.shape[0], seq_lens.shape[0])):
        ctx_len = seq_lens[i].item()
        if ctx_len <= 0:
            continue
        block_size = 16  # standard vLLM block size
        n_blocks = (ctx_len + block_size - 1) // block_size
        active = bt[i, :n_blocks]
        active = active[active >= 0].tolist()
        if len(active) > 1:
            graph.on_attention_step(active)


def get_stats() -> dict:
    """Get eviction graph and advisor statistics."""
    stats = {}
    if _graph is not None:
        stats["graph"] = _graph.get_stats()
    if _advisor is not None:
        stats["advisor"] = _advisor.get_stats()
    return stats
