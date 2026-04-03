"""
TurboQuant — Near-optimal KV cache quantization for LLM inference.

Implements the TurboQuant algorithm from Zandieh et al. (arXiv:2504.19874):
random rotation -> optimal scalar quantization -> bit packing.

Two quantizer variants:
  TurboQuant       — uniform bit-width (2/3/4-bit), data-oblivious
  HybridTurboQuant — split-group (tq35/tq25), variance-based + QJL residual

Graph-aware KV cache management:
  KVCacheGraph          — relationship graph over KV cache blocks
  GraphEvictionAdvisor  — background thread for zero-latency eviction decisions

Usage:
    from aither_kvcache import TurboQuant, HybridTurboQuant

    # Uniform 4-bit
    tq = TurboQuant(head_dim=128, bits=4, device='cuda')
    packed, norms = tq.encode(kv_vectors)
    decoded = tq.decode(packed, norms)

    # Graph-aware eviction
    from aither_kvcache import KVCacheGraph, GraphEvictionAdvisor

    graph = KVCacheGraph(protected_sources={"system"})
    advisor = GraphEvictionAdvisor(graph)
    advisor.start()
    candidates = advisor.get_eviction_candidates(n=16)
"""

from .quantizer import TurboQuant, TurboQuantConfig
from .hybrid_quantizer import HybridTurboQuant, HybridLayout, GroupLayout
from .kvcache_graph import KVCacheGraph, KVBlockNode, KVEdge, EdgeType, get_kvcache_graph
from .eviction_advisor import GraphEvictionAdvisor, reorder_by_ranking

__all__ = [
    # Quantizers
    "TurboQuant", "TurboQuantConfig",
    "HybridTurboQuant", "HybridLayout", "GroupLayout",
    # KV cache graph
    "KVCacheGraph", "KVBlockNode", "KVEdge", "EdgeType", "get_kvcache_graph",
    # Eviction advisor
    "GraphEvictionAdvisor", "reorder_by_ranking",
]
__version__ = "1.1.0"
