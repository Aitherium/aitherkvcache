"""
TurboQuant — Near-optimal KV cache quantization for LLM inference.

Implements the TurboQuant algorithm from Zandieh et al. (arXiv:2504.19874):
random rotation -> optimal scalar quantization -> bit packing.

Two quantizer variants:
  TurboQuant       — uniform bit-width (2/3/4-bit), data-oblivious
  HybridTurboQuant — split-group (tq35/tq25), variance-based + QJL residual

Usage:
    from turboquant import TurboQuant, HybridTurboQuant

    # Uniform 4-bit
    tq = TurboQuant(head_dim=128, bits=4, device='cuda')
    packed, norms = tq.encode(kv_vectors)
    decoded = tq.decode(packed, norms)

    # Hybrid 3.5-bit (better quality, same compression as TQ4)
    htq = HybridTurboQuant(head_dim=128, mode='tq35', device='cuda')
    htq.calibrate_uniform()  # or htq.calibrate(sample_data)
    packed = htq.encode(kv_vectors)
    decoded = htq.decode(packed)
"""

from .quantizer import TurboQuant, TurboQuantConfig
from .hybrid_quantizer import HybridTurboQuant, HybridLayout, GroupLayout

__all__ = [
    "TurboQuant", "TurboQuantConfig",
    "HybridTurboQuant", "HybridLayout", "GroupLayout",
]
__version__ = "1.0.0"
