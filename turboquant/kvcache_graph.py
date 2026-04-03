"""
KVCacheGraph — Relationship graph over KV cache blocks.

Models KV cache blocks as graph nodes with edges encoding:
- prefix_share:    blocks reused across requests (prefix caching)
- co_attend:       blocks frequently attended together
- semantic:        blocks with similar key vector embeddings
- temporal:        blocks from the same generation sequence
- spill_link:      hot block -> cold DDR5 copy

Enables graph-based prefetching (warm blocks that are graph-neighbors of
the current working set), semantic eviction (evict the least-connected
subgraph), and cross-session KV reuse.

Standalone — no AitherOS dependencies.

Usage:
    from turboquant.kvcache_graph import KVCacheGraph, EdgeType

    graph = KVCacheGraph()
    graph.add_block(0, "system", 0.95, (0, 16))
    graph.add_block(1, "user", 0.6, (16, 32))
    graph.add_edge(0, 1, EdgeType.TEMPORAL)

    # Who should we evict?
    victims = graph.suggest_eviction(n_blocks=4)

    # What should we prefetch from cold tier?
    prefetch = graph.suggest_prefetch(active_block_idxs=[0, 1])
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger("turboquant.kvcache_graph")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class EdgeType(str, Enum):
    PREFIX_SHARE = "prefix_share"   # Shared prefix between requests
    CO_ATTEND = "co_attend"         # Blocks attended together frequently
    SEMANTIC = "semantic"           # Similar key embeddings
    TEMPORAL = "temporal"           # Sequential in same generation
    SPILL_LINK = "spill_link"       # Hot -> cold tier link


@dataclass
class KVBlockNode:
    """A KV cache block as a graph node."""
    block_idx: int
    source_label: str           # system, user, assistant, tool, generation, etc.
    importance: float           # 0-1 importance score
    token_range: Tuple[int, int]  # (start, end) token positions
    created_at: float = field(default_factory=time.time)
    last_attended: float = field(default_factory=time.time)
    last_hit_at: float = 0.0    # Last prefix cache hit
    hit_count: int = 0          # Prefix cache hit count
    is_spilled: bool = False    # Currently in cold tier (DDR5/CPU)
    embedding: Optional[List[float]] = None  # Block representative vector


@dataclass
class KVEdge:
    """Weighted typed edge between KV cache blocks."""
    source: int     # block_idx
    target: int     # block_idx
    edge_type: EdgeType
    weight: float = 1.0
    created_at: float = field(default_factory=time.time)


# ============================================================================
# KVCACHE GRAPH
# ============================================================================

class KVCacheGraph:
    """
    Graph over KV cache blocks for intelligent prefetching and eviction.

    Nodes = physical KV cache blocks.
    Edges = relationships (prefix sharing, co-attention, semantic similarity).

    Use cases:
      - Prefetch: warm cold-tier blocks that are graph-neighbors of current working set
      - Evict: drop the least-connected subgraph first
      - Reuse: find blocks from prior sessions via graph links
      - Analyze: visualize attention patterns as a graph

    Thread-safe. All mutations under _lock.

    Args:
        protected_sources: Set of source labels that should never be evicted.
            Defaults to ``{"system"}``. Pass your own set to protect
            whichever roles matter for your use case.
        coattend_threshold: Number of co-occurrences before a CO_ATTEND
            edge is created between two blocks. Default 3.
        semantic_threshold: Cosine similarity threshold for creating
            SEMANTIC edges between blocks with embeddings. Default 0.8.
    """

    def __init__(
        self,
        protected_sources: Optional[Set[str]] = None,
        coattend_threshold: int = 3,
        semantic_threshold: float = 0.8,
    ):
        self._lock = threading.Lock()
        self._nodes: Dict[int, KVBlockNode] = {}          # block_idx -> node
        self._edges: Dict[Tuple[int, int, EdgeType], KVEdge] = {}
        self._adj: Dict[int, Set[int]] = defaultdict(set)  # adjacency list
        self._adj_by_type: Dict[EdgeType, Dict[int, Set[int]]] = {
            t: defaultdict(set) for t in EdgeType
        }
        # Co-attention tracking: (block_a, block_b) -> count
        self._coattend_counts: Dict[FrozenSet[int], int] = defaultdict(int)
        self._coattend_threshold: int = coattend_threshold
        self._semantic_threshold: float = semantic_threshold
        self._protected_sources: Set[str] = (
            protected_sources if protected_sources is not None else {"system"}
        )
        # Prefix sharing: request_id -> set of block_idxs
        self._request_blocks: Dict[str, Set[int]] = {}
        self._stats = {
            "nodes_added": 0, "nodes_removed": 0,
            "edges_added": 0, "prefetch_suggestions": 0,
            "eviction_suggestions": 0,
        }

    # ------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------

    @property
    def protected_sources(self) -> Set[str]:
        """Source labels that are excluded from eviction."""
        return set(self._protected_sources)

    @protected_sources.setter
    def protected_sources(self, sources: Set[str]) -> None:
        self._protected_sources = set(sources)

    # ------------------------------------------------------------------
    # NODE OPS
    # ------------------------------------------------------------------

    def add_block(
        self,
        block_idx: int,
        source_label: str,
        importance: float,
        token_range: Tuple[int, int],
        embedding: Optional[List[float]] = None,
    ) -> KVBlockNode:
        """Register a KV cache block as a graph node.

        If the block already exists, updates its metadata (upsert).
        When ``embedding`` is provided, computes cosine similarity with
        existing neighbors and creates SEMANTIC edges for similarity
        above the threshold.

        Args:
            block_idx: Integer block index in the KV cache.
            source_label: Label identifying the content source (e.g.
                "system", "user", "assistant", "tool", "generation").
            importance: 0-1 importance score. Higher = harder to evict.
            token_range: (start, end) token positions this block covers.
            embedding: Optional representative vector for semantic edges.

        Returns:
            The created or updated KVBlockNode.
        """
        with self._lock:
            if block_idx in self._nodes:
                node = self._nodes[block_idx]
                node.source_label = source_label
                node.importance = importance
                node.token_range = token_range
                if embedding:
                    old_had_embedding = node.embedding is not None
                    node.embedding = embedding
                    if not old_had_embedding:
                        self._create_semantic_edges_unlocked(block_idx, embedding)
                return node

            node = KVBlockNode(
                block_idx=block_idx,
                source_label=source_label,
                importance=importance,
                token_range=token_range,
                embedding=embedding,
            )
            self._nodes[block_idx] = node
            self._stats["nodes_added"] += 1

            if embedding is not None:
                self._create_semantic_edges_unlocked(block_idx, embedding)

            return node

    def remove_block(self, block_idx: int) -> None:
        """Remove a block and all its edges."""
        with self._lock:
            if block_idx not in self._nodes:
                return
            neighbors = list(self._adj.get(block_idx, set()))
            for neighbor in neighbors:
                for etype in EdgeType:
                    key = (block_idx, neighbor, etype)
                    rev_key = (neighbor, block_idx, etype)
                    self._edges.pop(key, None)
                    self._edges.pop(rev_key, None)
                    self._adj_by_type[etype][block_idx].discard(neighbor)
                    self._adj_by_type[etype][neighbor].discard(block_idx)
                self._adj[neighbor].discard(block_idx)
            self._adj.pop(block_idx, None)
            del self._nodes[block_idx]
            self._stats["nodes_removed"] += 1

    def get_block(self, block_idx: int) -> Optional[KVBlockNode]:
        """Get a block node by index, or None if not found."""
        return self._nodes.get(block_idx)

    # ------------------------------------------------------------------
    # EDGE OPS
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source: int,
        target: int,
        edge_type: EdgeType,
        weight: float = 1.0,
    ) -> Optional[KVEdge]:
        """Add or update a typed edge between blocks.

        Returns None if source == target or either block does not exist.
        """
        if source == target:
            return None
        with self._lock:
            if source not in self._nodes or target not in self._nodes:
                return None
            key = (source, target, edge_type)
            if key in self._edges:
                self._edges[key].weight = weight
                return self._edges[key]
            edge = KVEdge(
                source=source, target=target,
                edge_type=edge_type, weight=weight,
            )
            self._edges[key] = edge
            self._adj[source].add(target)
            self._adj[target].add(source)
            self._adj_by_type[edge_type][source].add(target)
            self._adj_by_type[edge_type][target].add(source)
            self._stats["edges_added"] += 1
            return edge

    # ------------------------------------------------------------------
    # EVENT HOOKS (call from your inference loop)
    # ------------------------------------------------------------------

    def on_attention_step(self, active_block_idxs: List[int]) -> None:
        """Called after each attention step with the active block indices.

        Tracks co-attendance: blocks that appear together in the same
        attention step get CO_ATTEND edges after enough co-occurrences.
        Also updates last_attended timestamps.
        """
        now = time.time()
        active_set = set(active_block_idxs)

        with self._lock:
            for bidx in active_set:
                node = self._nodes.get(bidx)
                if node:
                    node.last_attended = now

            blocks = sorted(active_set)
            max_pairs = 100
            count = 0
            for i, a in enumerate(blocks):
                if count >= max_pairs:
                    break
                for b in blocks[i + 1:]:
                    if count >= max_pairs:
                        break
                    pair = frozenset((a, b))
                    self._coattend_counts[pair] += 1
                    if self._coattend_counts[pair] == self._coattend_threshold:
                        self._add_edge_unlocked(
                            a, b, EdgeType.CO_ATTEND, weight=1.0)
                    elif self._coattend_counts[pair] > self._coattend_threshold:
                        key = (a, b, EdgeType.CO_ATTEND)
                        if key in self._edges:
                            self._edges[key].weight = min(
                                10.0, self._edges[key].weight + 0.1)
                    count += 1

    def on_prefix_hit(self, request_id: str, block_idxs: List[int]) -> None:
        """Called when prefix caching reuses blocks for a new request.

        Creates PREFIX_SHARE edges between blocks that co-appear across
        different requests (shared prefixes).
        """
        with self._lock:
            for bidx in block_idxs:
                node = self._nodes.get(bidx)
                if node:
                    node.hit_count += 1
                    node.last_hit_at = time.time()

            current = set(block_idxs)
            for req_id, prev_blocks in self._request_blocks.items():
                if req_id == request_id:
                    continue
                shared = current & prev_blocks
                for bidx in shared:
                    for other in prev_blocks - shared:
                        if other in self._nodes:
                            self._add_edge_unlocked(
                                bidx, other, EdgeType.PREFIX_SHARE, weight=1.0)
            self._request_blocks[request_id] = current

            if len(self._request_blocks) > 100:
                oldest = sorted(self._request_blocks.keys())[:50]
                for k in oldest:
                    del self._request_blocks[k]

    def on_spill(self, block_idxs: List[int]) -> None:
        """Called when blocks are spilled to cold tier (DDR5/CPU)."""
        with self._lock:
            for bidx in block_idxs:
                node = self._nodes.get(bidx)
                if node:
                    node.is_spilled = True

    def on_warm(self, block_idxs: List[int]) -> None:
        """Called when blocks are warmed from cold tier back to VRAM."""
        with self._lock:
            for bidx in block_idxs:
                node = self._nodes.get(bidx)
                if node:
                    node.is_spilled = False

    def on_temporal_sequence(self, block_idxs: List[int]) -> None:
        """Called with blocks from the same generation in order.

        Creates TEMPORAL edges between consecutive blocks.
        """
        with self._lock:
            for i in range(len(block_idxs) - 1):
                a, b = block_idxs[i], block_idxs[i + 1]
                if a in self._nodes and b in self._nodes:
                    self._add_edge_unlocked(
                        a, b, EdgeType.TEMPORAL, weight=1.0)

    # ------------------------------------------------------------------
    # GRAPH QUERIES
    # ------------------------------------------------------------------

    def suggest_prefetch(
        self,
        active_block_idxs: List[int],
        max_suggestions: int = 16,
    ) -> List[int]:
        """Suggest cold-tier blocks to prefetch based on graph proximity.

        Returns block indices that are:
          1. Currently spilled to cold tier
          2. Graph-neighbors of the active working set
          3. Ranked by edge weight + neighbor importance
        """
        with self._lock:
            active = set(active_block_idxs)
            candidates: Dict[int, float] = {}

            for bidx in active:
                for neighbor in self._adj.get(bidx, set()):
                    if neighbor in active:
                        continue
                    node = self._nodes.get(neighbor)
                    if not node or not node.is_spilled:
                        continue
                    score = candidates.get(neighbor, 0.0)
                    for etype in EdgeType:
                        key = (bidx, neighbor, etype)
                        rev = (neighbor, bidx, etype)
                        edge = self._edges.get(key) or self._edges.get(rev)
                        if edge:
                            score += edge.weight
                    score += node.importance * 2.0
                    candidates[neighbor] = score

            ranked = sorted(candidates.items(), key=lambda x: -x[1])
            result = [bidx for bidx, _ in ranked[:max_suggestions]]
            self._stats["prefetch_suggestions"] += len(result)
            return result

    def suggest_eviction(
        self,
        n_blocks: int,
        protect_sources: Optional[Set[str]] = None,
    ) -> List[int]:
        """Suggest blocks to evict based on graph connectivity.

        Prefers evicting blocks that are:
          1. Least connected (fewest/weakest edges)
          2. Oldest (largest last_attended gap)
          3. Lowest importance
          4. Not in protected sources

        Args:
            n_blocks: Maximum number of eviction candidates to return.
            protect_sources: Override the instance-level protected sources.
                If None, uses ``self._protected_sources``.
        """
        protect = protect_sources if protect_sources is not None else self._protected_sources

        with self._lock:
            now = time.time()
            scored: List[Tuple[float, int]] = []

            for bidx, node in self._nodes.items():
                if node.source_label in protect:
                    continue
                if node.is_spilled:
                    continue  # Already cold, not occupying VRAM

                degree = len(self._adj.get(bidx, set()))
                edge_weight_sum = sum(
                    e.weight for k, e in self._edges.items()
                    if k[0] == bidx or k[1] == bidx
                )

                age = now - node.last_attended

                # Eviction score: higher = evict first
                score = (
                    age * 0.01                    # Older = more evictable
                    - degree * 5.0                # More connected = keep
                    - edge_weight_sum * 2.0       # Stronger edges = keep
                    - node.importance * 20.0      # More important = keep
                    - node.hit_count * 3.0        # More prefix hits = keep
                )
                scored.append((score, bidx))

            scored.sort(key=lambda x: -x[0])  # Highest score = evict first
            result = [bidx for _, bidx in scored[:n_blocks]]
            self._stats["eviction_suggestions"] += len(result)
            return result

    def neighbors(
        self,
        block_idx: int,
        edge_type: Optional[EdgeType] = None,
        max_depth: int = 1,
    ) -> Set[int]:
        """Get graph neighbors of a block, optionally filtered by edge type."""
        with self._lock:
            if edge_type:
                direct = self._adj_by_type[edge_type].get(block_idx, set())
            else:
                direct = self._adj.get(block_idx, set())

            if max_depth <= 1:
                return set(direct)

            visited = {block_idx}
            frontier = set(direct)
            for _ in range(max_depth - 1):
                next_frontier = set()
                for n in frontier:
                    if edge_type:
                        nbrs = self._adj_by_type[edge_type].get(n, set())
                    else:
                        nbrs = self._adj.get(n, set())
                    next_frontier |= nbrs - visited
                visited |= frontier
                frontier = next_frontier
            visited |= frontier
            visited.discard(block_idx)
            return visited

    def subgraph(self, block_idxs: List[int]) -> Dict[str, Any]:
        """Extract a subgraph as a serializable dict (for visualization)."""
        with self._lock:
            nodes = []
            edges = []
            block_set = set(block_idxs)
            for bidx in block_idxs:
                node = self._nodes.get(bidx)
                if node:
                    nodes.append({
                        "id": bidx,
                        "source_label": node.source_label,
                        "importance": node.importance,
                        "token_range": list(node.token_range),
                        "is_spilled": node.is_spilled,
                        "hit_count": node.hit_count,
                        "degree": len(self._adj.get(bidx, set())),
                    })
            for key, edge in self._edges.items():
                if edge.source in block_set and edge.target in block_set:
                    edges.append({
                        "source": edge.source,
                        "target": edge.target,
                        "type": edge.edge_type.value,
                        "weight": round(edge.weight, 2),
                    })
            return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return graph statistics."""
        with self._lock:
            spilled = sum(1 for n in self._nodes.values() if n.is_spilled)
            edge_counts = defaultdict(int)
            for key in self._edges:
                edge_counts[key[2].value] += 1
            return {
                "total_nodes": len(self._nodes),
                "total_edges": len(self._edges),
                "spilled_nodes": spilled,
                "hot_nodes": len(self._nodes) - spilled,
                "edges_by_type": dict(edge_counts),
                "tracked_requests": len(self._request_blocks),
                "coattend_pairs": len(self._coattend_counts),
                "protected_sources": sorted(self._protected_sources),
                **self._stats,
            }

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    def _create_semantic_edges_unlocked(
        self, block_idx: int, embedding: List[float]
    ) -> None:
        """Create SEMANTIC edges to similar neighbors. Caller holds _lock."""
        max_comparisons = 200
        count = 0
        for other_idx, other_node in self._nodes.items():
            if other_idx == block_idx or other_node.embedding is None:
                continue
            if count >= max_comparisons:
                break
            sim = self._cosine_similarity(embedding, other_node.embedding)
            if sim >= self._semantic_threshold:
                self._add_edge_unlocked(
                    block_idx, other_idx, EdgeType.SEMANTIC, weight=sim
                )
            count += 1

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors (pure Python)."""
        if len(a) != len(b) or len(a) == 0:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _add_edge_unlocked(
        self, source: int, target: int, edge_type: EdgeType, weight: float
    ) -> None:
        """Add edge without acquiring lock (caller holds it)."""
        if source == target:
            return
        key = (source, target, edge_type)
        if key not in self._edges:
            self._edges[key] = KVEdge(
                source=source, target=target,
                edge_type=edge_type, weight=weight,
            )
            self._adj[source].add(target)
            self._adj[target].add(source)
            self._adj_by_type[edge_type][source].add(target)
            self._adj_by_type[edge_type][target].add(source)
            self._stats["edges_added"] += 1


# ============================================================================
# SINGLETON
# ============================================================================

_instance: Optional[KVCacheGraph] = None
_instance_lock = threading.Lock()


def get_kvcache_graph(**kwargs) -> KVCacheGraph:
    """Get or create the singleton KVCacheGraph instance.

    Keyword arguments are forwarded to KVCacheGraph() on first call only.
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = KVCacheGraph(**kwargs)
    return _instance
