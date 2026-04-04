"""Tests for standalone KVCacheGraph and GraphEvictionAdvisor."""
import time

import pytest

from aither_kvcache.kvcache_graph import KVCacheGraph, EdgeType
from aither_kvcache.eviction_advisor import GraphEvictionAdvisor, reorder_by_ranking


@pytest.fixture
def graph():
    return KVCacheGraph()


@pytest.fixture
def populated():
    """Graph with blocks and edges for eviction/prefetch testing."""
    g = KVCacheGraph(protected_sources={"system"})
    g.add_block(0, "system", 0.95, (0, 16))
    g.add_block(1, "system", 0.9, (16, 32))
    g.add_block(2, "user", 0.6, (32, 48))
    g.add_block(3, "assistant", 0.3, (48, 64))
    g.add_block(4, "assistant", 0.2, (64, 80))
    g.add_block(5, "generation", 0.1, (80, 96))
    g.add_edge(0, 1, EdgeType.TEMPORAL)
    g.add_edge(1, 2, EdgeType.TEMPORAL)
    g.add_edge(2, 3, EdgeType.CO_ATTEND, weight=3.0)
    g.on_spill([4, 5])
    return g


# =====================================================================
# GRAPH BASICS
# =====================================================================

class TestGraphBasics:
    def test_add_and_get_block(self, graph):
        node = graph.add_block(0, "system", 0.9, (0, 16))
        assert node.block_idx == 0
        assert node.source_label == "system"
        assert graph.get_block(0) is node
        assert graph.get_block(999) is None

    def test_upsert_block(self, graph):
        graph.add_block(0, "system", 0.5, (0, 16))
        graph.add_block(0, "user", 0.8, (0, 16))  # upsert
        assert graph.get_block(0).source_label == "user"
        assert graph.get_block(0).importance == 0.8

    def test_remove_block_cleans_edges(self, graph):
        graph.add_block(0, "a", 0.5, (0, 16))
        graph.add_block(1, "b", 0.5, (16, 32))
        graph.add_edge(0, 1, EdgeType.TEMPORAL)
        assert graph.get_stats()["total_edges"] == 1
        graph.remove_block(0)
        assert graph.get_block(0) is None
        assert graph.get_stats()["total_edges"] == 0

    def test_add_edge_self_loop_rejected(self, graph):
        graph.add_block(0, "a", 0.5, (0, 16))
        assert graph.add_edge(0, 0, EdgeType.TEMPORAL) is None

    def test_add_edge_missing_node_rejected(self, graph):
        graph.add_block(0, "a", 0.5, (0, 16))
        assert graph.add_edge(0, 99, EdgeType.TEMPORAL) is None

    def test_stats(self, populated):
        stats = populated.get_stats()
        assert stats["total_nodes"] == 6
        assert stats["total_edges"] == 3
        assert stats["spilled_nodes"] == 2
        assert stats["hot_nodes"] == 4
        assert "system" in stats["protected_sources"]


# =====================================================================
# EVICTION
# =====================================================================

class TestEviction:
    def test_protected_sources_excluded(self, populated):
        victims = populated.suggest_eviction(10)
        # system blocks (0, 1) must never appear in eviction list
        assert 0 not in victims
        assert 1 not in victims

    def test_spilled_blocks_excluded(self, populated):
        victims = populated.suggest_eviction(10)
        # blocks 4, 5 are spilled — not occupying VRAM, skip
        assert 4 not in victims
        assert 5 not in victims

    def test_low_importance_evicted_first(self):
        g = KVCacheGraph(protected_sources=set())
        g.add_block(0, "a", 0.9, (0, 16))
        g.add_block(1, "b", 0.1, (16, 32))
        victims = g.suggest_eviction(1)
        assert victims == [1]

    def test_connected_blocks_kept(self):
        g = KVCacheGraph(protected_sources=set())
        g.add_block(0, "a", 0.5, (0, 16))
        g.add_block(1, "a", 0.5, (16, 32))
        g.add_block(2, "a", 0.5, (32, 48))
        # 0 and 1 are connected; 2 is isolated
        g.add_edge(0, 1, EdgeType.CO_ATTEND, weight=5.0)
        victims = g.suggest_eviction(1)
        assert victims == [2]

    def test_custom_protected_sources(self):
        g = KVCacheGraph(protected_sources={"system", "tools"})
        g.add_block(0, "system", 0.9, (0, 16))
        g.add_block(1, "tools", 0.8, (16, 32))
        g.add_block(2, "user", 0.5, (32, 48))
        victims = g.suggest_eviction(10)
        assert 0 not in victims
        assert 1 not in victims
        assert 2 in victims


# =====================================================================
# PREFETCH
# =====================================================================

class TestPrefetch:
    def test_suggests_spilled_neighbors(self, populated):
        # Add edges from active block to spilled blocks
        populated.add_edge(3, 4, EdgeType.TEMPORAL)
        populated.add_edge(3, 5, EdgeType.TEMPORAL)
        result = populated.suggest_prefetch([3])
        assert set(result) == {4, 5}

    def test_ignores_hot_neighbors(self, populated):
        # Block 2 is a neighbor of 3 but not spilled
        result = populated.suggest_prefetch([2])
        # Only spilled neighbors should appear
        for bidx in result:
            assert populated.get_block(bidx).is_spilled

    def test_empty_when_no_spilled(self):
        g = KVCacheGraph()
        g.add_block(0, "a", 0.5, (0, 16))
        g.add_block(1, "b", 0.5, (16, 32))
        g.add_edge(0, 1, EdgeType.TEMPORAL)
        assert g.suggest_prefetch([0]) == []


# =====================================================================
# CO-ATTENTION TRACKING
# =====================================================================

class TestCoAttention:
    def test_edge_after_threshold(self, graph):
        graph.add_block(0, "a", 0.5, (0, 16))
        graph.add_block(1, "b", 0.5, (16, 32))
        # 3 co-occurrences needed (default threshold)
        graph.on_attention_step([0, 1])
        graph.on_attention_step([0, 1])
        assert graph.get_stats()["edges_by_type"].get("co_attend", 0) == 0
        graph.on_attention_step([0, 1])  # Threshold hit
        assert graph.get_stats()["edges_by_type"].get("co_attend", 0) > 0

    def test_edge_weight_increases(self, graph):
        graph.add_block(0, "a", 0.5, (0, 16))
        graph.add_block(1, "b", 0.5, (16, 32))
        for _ in range(5):
            graph.on_attention_step([0, 1])
        # Weight should be > 1.0 after extra co-occurrences
        edge = graph._edges.get((0, 1, EdgeType.CO_ATTEND))
        assert edge is not None
        assert edge.weight > 1.0


# =====================================================================
# SEMANTIC EDGES
# =====================================================================

class TestSemanticEdges:
    def test_created_on_high_similarity(self, graph):
        emb1 = [1.0, 0.0, 0.0, 0.0]
        emb2 = [0.99, 0.01, 0.0, 0.0]
        graph.add_block(0, "a", 0.5, (0, 16), embedding=emb1)
        graph.add_block(1, "b", 0.5, (16, 32), embedding=emb2)
        assert graph.get_stats()["edges_by_type"].get("semantic", 0) > 0

    def test_not_created_on_low_similarity(self, graph):
        emb1 = [1.0, 0.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0, 0.0]  # orthogonal
        graph.add_block(0, "a", 0.5, (0, 16), embedding=emb1)
        graph.add_block(1, "b", 0.5, (16, 32), embedding=emb2)
        assert graph.get_stats()["edges_by_type"].get("semantic", 0) == 0

    def test_cosine_identical(self, graph):
        assert abs(graph._cosine_similarity([1, 0], [1, 0]) - 1.0) < 1e-6

    def test_cosine_zero_norm(self, graph):
        assert graph._cosine_similarity([0, 0], [1, 0]) == 0.0


# =====================================================================
# TEMPORAL SEQUENCES
# =====================================================================

class TestTemporalSequence:
    def test_creates_chain(self, graph):
        for i in range(4):
            graph.add_block(i, "gen", 0.5, (i * 16, (i + 1) * 16))
        graph.on_temporal_sequence([0, 1, 2, 3])
        assert graph.get_stats()["total_edges"] == 3  # 0-1, 1-2, 2-3


# =====================================================================
# PREFIX CACHE HITS
# =====================================================================

class TestPrefixHits:
    def test_increments_hit_count(self, graph):
        graph.add_block(0, "a", 0.5, (0, 16))
        graph.on_prefix_hit("req1", [0])
        graph.on_prefix_hit("req2", [0])
        assert graph.get_block(0).hit_count == 2


# =====================================================================
# SPILL / WARM
# =====================================================================

class TestSpillWarm:
    def test_spill_marks_blocks(self, graph):
        graph.add_block(0, "a", 0.5, (0, 16))
        graph.on_spill([0])
        assert graph.get_block(0).is_spilled is True

    def test_warm_clears_flag(self, graph):
        graph.add_block(0, "a", 0.5, (0, 16))
        graph.on_spill([0])
        graph.on_warm([0])
        assert graph.get_block(0).is_spilled is False


# =====================================================================
# NEIGHBORS / SUBGRAPH
# =====================================================================

class TestNeighbors:
    def test_direct_neighbors(self, populated):
        nbrs = populated.neighbors(1)
        assert 0 in nbrs  # temporal edge
        assert 2 in nbrs  # temporal edge

    def test_filtered_by_type(self, populated):
        nbrs = populated.neighbors(2, edge_type=EdgeType.CO_ATTEND)
        assert 3 in nbrs
        assert 1 not in nbrs  # that's a TEMPORAL edge

    def test_multi_hop(self, populated):
        nbrs = populated.neighbors(0, max_depth=2)
        assert 2 in nbrs  # 0 -> 1 -> 2

    def test_subgraph(self, populated):
        sg = populated.subgraph([0, 1, 2])
        assert len(sg["nodes"]) == 3
        assert len(sg["edges"]) == 2  # 0-1 temporal, 1-2 temporal


# =====================================================================
# ADVISOR LIFECYCLE
# =====================================================================

class TestAdvisorLifecycle:
    def test_start_stop(self):
        g = KVCacheGraph()
        adv = GraphEvictionAdvisor(g, interval=0.05)
        assert not adv.is_running
        adv.start()
        assert adv.is_running
        adv.stop()
        assert not adv.is_running

    def test_double_start_noop(self):
        g = KVCacheGraph()
        adv = GraphEvictionAdvisor(g, interval=0.05)
        adv.start()
        adv.start()
        assert adv.is_running
        adv.stop()

    def test_double_stop_noop(self):
        g = KVCacheGraph()
        adv = GraphEvictionAdvisor(g, interval=0.05)
        adv.start()
        adv.stop()
        adv.stop()  # no-op
        assert not adv.is_running

    def test_stats_initial(self):
        g = KVCacheGraph()
        adv = GraphEvictionAdvisor(g)
        stats = adv.get_stats()
        assert stats["running"] is False
        assert stats["recompute_count"] == 0


# =====================================================================
# ADVISOR EVICTION CANDIDATES
# =====================================================================

class TestAdvisorEviction:
    def test_none_when_no_ranking(self):
        g = KVCacheGraph()
        adv = GraphEvictionAdvisor(g)
        assert adv.get_eviction_candidates(5) is None

    def test_returns_top_n(self):
        g = KVCacheGraph()
        adv = GraphEvictionAdvisor(g, max_stale=10.0)
        adv._eviction_ranking = [5, 4, 3, 2, 1]
        adv._ranking_ts = time.monotonic()
        assert adv.get_eviction_candidates(3) == [5, 4, 3]

    def test_none_when_stale(self):
        g = KVCacheGraph()
        adv = GraphEvictionAdvisor(g, max_stale=0.01)
        adv._eviction_ranking = [5, 4, 3]
        adv._ranking_ts = time.monotonic() - 1.0
        assert adv.get_eviction_candidates(3) is None
        assert adv._stale_fallbacks == 1


# =====================================================================
# ADVISOR PREFETCH
# =====================================================================

class TestAdvisorPrefetch:
    def test_prefetch_returns_spilled_neighbors(self, populated):
        populated.add_edge(3, 4, EdgeType.TEMPORAL)
        populated.add_edge(3, 5, EdgeType.TEMPORAL)
        adv = GraphEvictionAdvisor(populated)
        result = adv.get_prefetch_candidates([3], n=4)
        assert result is not None
        assert set(result) == {4, 5}

    def test_prefetch_none_when_no_graph(self):
        adv = GraphEvictionAdvisor.__new__(GraphEvictionAdvisor)
        adv._graph = None
        # Override _get_graph to prevent singleton fallback
        adv._get_graph = lambda: None
        result = adv.get_prefetch_candidates([0, 1], n=4)
        assert result is None


# =====================================================================
# ADVISOR BACKGROUND RECOMPUTE
# =====================================================================

class TestAdvisorRecompute:
    def test_recompute_populates_ranking(self, populated):
        adv = GraphEvictionAdvisor(populated, interval=0.05, eviction_batch=10)
        adv._recompute()
        assert adv._eviction_ranking is not None
        assert len(adv._eviction_ranking) > 0
        assert adv._recompute_count == 1

    def test_worker_runs_periodically(self, populated):
        adv = GraphEvictionAdvisor(populated, interval=0.05, max_stale=5.0)
        adv.start()
        time.sleep(0.25)
        adv.stop()
        assert adv._recompute_count >= 2
        assert adv._eviction_ranking is not None


# =====================================================================
# REORDER BY RANKING
# =====================================================================

class TestReorderByRanking:
    def test_basic(self):
        assert reorder_by_ranking([1, 2, 3, 4, 5], [3, 5]) == [3, 5, 1, 2, 4]

    def test_ranking_not_in_indices(self):
        assert reorder_by_ranking([1, 2, 3], [99, 2]) == [2, 1, 3]

    def test_empty_ranking(self):
        assert reorder_by_ranking([1, 2, 3], []) == [1, 2, 3]

    def test_empty_indices(self):
        assert reorder_by_ranking([], [1, 2]) == []
