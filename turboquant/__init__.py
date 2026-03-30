"""
TurboQuant — Near-optimal KV cache quantization for AitherOS.

Implements the TurboQuant algorithm from Zandieh et al. (arXiv:2504.19874):
random rotation -> optimal scalar quantization -> bit packing.

Achieves 3.5-bit KV cache compression with zero accuracy loss,
within 2.7x of the information-theoretic optimum.

Usage:
    from lib.gpu.turboquant import TurboQuant

    tq = TurboQuant(head_dim=128, bits=4, device='cuda')
    packed, norms = tq.encode(kv_vectors)
    decoded = tq.decode(packed, norms)

    # Validate correctness
    print(tq.validate())

    # Memory savings report
    print(tq.memory_report(seq_len=40000, num_layers=32, num_kv_heads=8))
"""

from .quantizer import TurboQuant, TurboQuantConfig

__all__ = ["TurboQuant", "TurboQuantConfig"]
__version__ = "0.5.0"
