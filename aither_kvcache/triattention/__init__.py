"""
TriAttention — Trigonometric Series Attention for extreme KV cache compression.

Core insight: transformer attention with RoPE is naturally a trigonometric
series in position difference. By storing only the dominant frequency
components of each key/value vector (in the RoPE pairing basis), we achieve
~10× KV cache compression with minimal quality loss.

Mathematical foundation:
    score(q, k, m, n) = (1/√d) Σ_i [c_i cos(Δθ_i) + s_i sin(Δθ_i)]

    where  c_i = q_{2i}k_{2i} + q_{2i+1}k_{2i+1}   (pair dot product)
           s_i = q_{2i+1}k_{2i} - q_{2i}k_{2i+1}   (pair cross product)
           Δ = n − m                                  (position difference)
           θ_i = base^{-2i/d}                         (RoPE frequency)

Only top-F frequency pairs (by energy) are retained per key/value token.
With F=12 and 4-bit coefficient quantization: 26 bytes/token vs 256 → ~10×.

Calibrated for Qwen3.5 family with per-layer spectral profiles.
"""

__version__ = "0.1.0"

__all__ = [
    "TriAttention",
    "TriAttentionConfig",
    "SpectralKVEncoder",
    "SpectralKVCache",
    "TrigSeriesScorer",
    "SpectralEncoding",
]

from .config import TriAttentionConfig


def __getattr__(name):
    """Lazy imports to avoid heavy torch dependency at package level."""
    if name == "TriAttention":
        from .attention import TriAttention
        return TriAttention
    if name == "SpectralKVEncoder":
        from .encoder import SpectralKVEncoder
        return SpectralKVEncoder
    if name == "SpectralEncoding":
        from .encoder import SpectralEncoding
        return SpectralEncoding
    if name == "SpectralKVCache":
        from .cache import SpectralKVCache
        return SpectralKVCache
    if name == "TrigSeriesScorer":
        from .scorer import TrigSeriesScorer
        return TrigSeriesScorer
    raise AttributeError(f"module 'triattention' has no attribute {name!r}")
