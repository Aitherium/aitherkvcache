"""
aither-kvcache — Near-optimal KV cache compression for LLM inference.

Two compression engines:
  TurboQuant    — Vector quantization (Zandieh et al., arXiv:2504.19874)
                  Random rotation → Lloyd-Max scalar quantization → bit packing.
                  2/3/4-bit, 3.8-7.1× compression vs FP16.

  TriAttention  — Spectral KV compression via trigonometric series (NEW in v2.0)
                  Top-F RoPE frequency pairs → 4/8/16-bit coefficient quantization.
                  ~10× compression with bounded approximation error.

Hybrid variant:
  HybridTurboQuant — split-group (tq35/tq25), variance-based + QJL residual

Graph-aware KV cache management:
  KVCacheGraph          — relationship graph over KV cache blocks
  GraphEvictionAdvisor  — background thread for zero-latency eviction decisions

Usage:
    from aither_kvcache import TurboQuant, HybridTurboQuant

    # TurboQuant: 4-bit vector quantization (3.8× compression)
    tq = TurboQuant(head_dim=128, bits=4, device='cuda')
    packed, norms = tq.encode(kv_vectors)
    decoded = tq.decode(packed, norms)

    # TriAttention: spectral compression (~10× compression)
    from aither_kvcache.triattention import TriAttention, TriAttentionConfig
    config = TriAttentionConfig(head_dim=128, num_freqs=12, coeff_bits=4)
    tri = TriAttention(config)
    k_enc, v_enc = tri.encode_kv(keys, values)
    output = tri.decode_step(query, k_enc, v_enc, query_pos, key_positions)

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
    # TurboQuant — vector quantization
    "TurboQuant", "TurboQuantConfig",
    "HybridTurboQuant", "HybridLayout", "GroupLayout",
    # KV cache graph
    "KVCacheGraph", "KVBlockNode", "KVEdge", "EdgeType", "get_kvcache_graph",
    # Eviction advisor
    "GraphEvictionAdvisor", "reorder_by_ranking",
    # TriAttention — lazy-loaded via subpackage
    # from aither_kvcache.triattention import TriAttention, TriAttentionConfig
]
__version__ = "2.0.1"
