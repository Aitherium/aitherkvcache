"""
TurboQuant: Near-optimal vector quantizer for KV cache compression.

Algorithm (MSE-optimal):
  1. Compute ||x|| and normalize to unit sphere
  2. Apply random orthogonal rotation Π (data-oblivious)
  3. Scalar-quantize each coordinate using precomputed Lloyd-Max codebook
  4. Pack indices into low-bit representation
  5. Store packed indices + scalar norm

Dequantization reverses steps 4→3→2→1.

Achieves MSE within 2.7× of the information-theoretic lower bound
across all bit-widths, with zero calibration data and zero retraining.

Reference: Zandieh et al., "TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate", arXiv:2504.19874, April 2025.
"""

import time
import torch
from dataclasses import dataclass
from typing import Optional, Tuple

from .codebook import get_codebook, get_theory_bounds
from .rotation import random_orthogonal, randomized_hadamard_matrix
from .packing import (
    pack_4bit, unpack_4bit,
    pack_3bit, unpack_3bit,
    pack_2bit, unpack_2bit,
    packed_size,
)


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant quantizer."""
    head_dim: int = 128
    bits: int = 4           # 2, 3, or 4
    seed: int = 42
    use_hadamard: bool = False   # True = RHT, False = full random orthogonal
    hadamard_rounds: int = 3
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    use_triton: bool = True      # Try Triton kernels when available


class TurboQuant:
    """
    TurboQuant KV cache quantizer.

    Compresses KV cache vectors from FP16 (16 bits/value) to b bits/value
    with near-optimal distortion (within 2.7x of information-theoretic bound).

    Compression ratios vs FP16:
        4-bit: 3.8x  (68 bytes vs 256 bytes per 128-dim vector)
        3-bit: 4.9x  (52 bytes vs 256 bytes)
        2-bit: 7.1x  (36 bytes vs 256 bytes)

    Usage:
        tq = TurboQuant(head_dim=128, bits=4)
        packed, norms = tq.encode(kv_vectors)
        decoded = tq.decode(packed, norms)
    """

    def __init__(self, config: Optional[TurboQuantConfig] = None, **kwargs):
        if config is None:
            config = TurboQuantConfig(**kwargs)
        self.config = config
        self.head_dim = config.head_dim
        self.bits = config.bits
        self.num_levels = 1 << config.bits
        self.device = config.device
        self.dtype = config.dtype

        # Validate
        if config.bits not in (2, 3, 4):
            raise ValueError(f"Supported bit-widths: 2, 3, 4. Got {config.bits}")
        if config.head_dim <= 0 or (config.head_dim & (config.head_dim - 1)) != 0:
            raise ValueError(f"head_dim must be power of 2, got {config.head_dim}")

        # Precompute rotation matrix
        if config.use_hadamard:
            self.rotation = randomized_hadamard_matrix(
                config.head_dim, seed=config.seed, device=config.device,
                num_rounds=config.hadamard_rounds,
            )
        else:
            self.rotation = random_orthogonal(
                config.head_dim, seed=config.seed, device=config.device,
            )

        # Precompute codebook
        centroids_np, boundaries_np, mse_total = get_codebook(config.head_dim, config.bits)
        self.centroids = torch.tensor(centroids_np, dtype=torch.float32, device=config.device)
        self.boundaries_inner = torch.tensor(
            boundaries_np, dtype=torch.float32, device=config.device,
        )
        self.mse_total = mse_total

        # Theory bounds
        self.theory_lower, self.theory_upper = get_theory_bounds(config.bits)

        # Pack/unpack dispatch
        self._pack_fn = {2: pack_2bit, 3: pack_3bit, 4: pack_4bit}[config.bits]
        self._unpack_fn = {2: unpack_2bit, 3: unpack_3bit, 4: unpack_4bit}[config.bits]
        self._packed_dim = packed_size(config.head_dim, config.bits)

        # Try Triton kernels (auto-enabled on Blackwell SM_100+)
        self._use_triton = False
        if config.use_triton:
            try:
                from . import triton_ops
                if triton_ops.HAS_TRITON and torch.cuda.is_available():
                    self._triton = triton_ops
                    # Triton kernels available for 2-bit and 4-bit
                    self._use_triton = config.bits in (2, 4)
            except ImportError:
                pass

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize vectors to packed low-bit representation.

        Args:
            x: [..., head_dim] float tensor

        Returns:
            packed: [..., packed_dim] uint8 packed indices
            norms: [...] float32 L2 norms
        """
        original_shape = x.shape
        assert x.shape[-1] == self.head_dim, \
            f"Expected last dim {self.head_dim}, got {x.shape[-1]}"

        # Flatten to [N, D] — force contiguous for safe matmul/Triton ops.
        # vLLM's paged KV cache passes strided views; non-contiguous tensors
        # cause CUDA segfaults on Blackwell (sm_100) in matmul and Triton JIT.
        x_flat = x.reshape(-1, self.head_dim).float().contiguous()
        x_flat.shape[0]

        # 1) Compute norms
        norms = x_flat.norm(dim=-1)

        # 2) Normalize to unit sphere
        x_norm = x_flat / (norms.unsqueeze(-1) + 1e-10)

        # 3) Random rotation: y = x_norm @ Π^T
        y = torch.matmul(x_norm, self.rotation.T).contiguous()

        # 4-5) Quantize + pack
        # Blackwell (SM_100+) auto-detected at init — Triton enabled automatically.
        use_triton = self._use_triton and x.is_cuda

        if use_triton:
            packed = self._triton_encode(y)
        else:
            packed = self._pytorch_encode(y)

        # Reshape to match input batch dims
        batch_shape = list(original_shape[:-1])
        return (
            packed.reshape(batch_shape + [self._packed_dim]),
            norms.reshape(batch_shape),
        )

    def decode(self, packed: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """
        Dequantize packed representation back to vectors.

        Args:
            packed: [..., packed_dim] uint8 packed indices
            norms: [...] float32 L2 norms

        Returns:
            x_hat: [..., head_dim] reconstructed vectors
        """
        batch_shape = list(norms.shape)

        # Flatten
        packed_flat = packed.reshape(-1, packed.shape[-1])
        norms_flat = norms.reshape(-1).float()

        # Unpack + codebook lookup
        if self._use_triton and packed.is_cuda:
            y_hat = self._triton_decode(packed_flat)
        else:
            y_hat = self._pytorch_decode(packed_flat)

        # Inverse rotation: x_hat = y_hat @ Π
        x_hat = torch.matmul(y_hat, self.rotation)

        # Rescale
        x_hat = x_hat * norms_flat.unsqueeze(-1)

        return x_hat.reshape(batch_shape + [self.head_dim]).to(self.dtype)

    # ================================================================
    # INTERNAL: PyTorch path (CPU + GPU fallback)
    # ================================================================

    def _pytorch_encode(self, y: torch.Tensor) -> torch.Tensor:
        """Quantize + pack using pure PyTorch ops."""
        indices = torch.searchsorted(self.boundaries_inner, y)
        indices = indices.clamp(0, self.num_levels - 1)
        return self._pack_fn(indices)

    def _pytorch_decode(self, packed: torch.Tensor) -> torch.Tensor:
        """Unpack + codebook lookup using pure PyTorch ops."""
        indices = self._unpack_fn(packed, self.head_dim)
        return self.centroids[indices.long()]

    # ================================================================
    # INTERNAL: Triton path (GPU only)
    # ================================================================

    def _triton_encode(self, y: torch.Tensor) -> torch.Tensor:
        if self.bits == 4:
            return self._triton.triton_quantize_4bit(y, self.boundaries_inner)
        elif self.bits == 2:
            return self._triton.triton_quantize_2bit(y, self.boundaries_inner)
        return self._pytorch_encode(y)

    def _triton_decode(self, packed: torch.Tensor) -> torch.Tensor:
        if self.bits == 4:
            return self._triton.triton_dequantize_4bit(packed, self.centroids, self.head_dim)
        elif self.bits == 2:
            return self._triton.triton_dequantize_2bit(packed, self.centroids, self.head_dim)
        return self._pytorch_decode(packed)

    # ================================================================
    # ANALYSIS
    # ================================================================

    def compression_ratio(self) -> float:
        """Compression ratio vs FP16."""
        fp16_bytes = self.head_dim * 2
        quant_bytes = self._packed_dim + 4  # packed indices + float32 norm
        return fp16_bytes / quant_bytes

    def compression_ratio_vs_fp8(self) -> float:
        """Compression ratio vs FP8."""
        fp8_bytes = self.head_dim * 1
        quant_bytes = self._packed_dim + 4
        return fp8_bytes / quant_bytes

    def memory_report(self, seq_len: int, num_layers: int = 32,
                      num_kv_heads: int = 8) -> dict:
        """Compute memory usage for a KV cache configuration."""
        # Total vectors = seq × layers × kv_heads × 2 (keys + values)
        total_vectors = seq_len * num_layers * num_kv_heads * 2

        fp16_bytes = total_vectors * self.head_dim * 2
        fp8_bytes = total_vectors * self.head_dim
        tq_bytes = total_vectors * (self._packed_dim + 4)

        return {
            "config": f"TQ{self.bits} d={self.head_dim} seq={seq_len} "
                      f"L={num_layers} H={num_kv_heads}",
            "fp16_mb": fp16_bytes / (1024 ** 2),
            "fp8_mb": fp8_bytes / (1024 ** 2),
            f"tq{self.bits}_mb": tq_bytes / (1024 ** 2),
            "ratio_vs_fp16": fp16_bytes / tq_bytes,
            "ratio_vs_fp8": fp8_bytes / tq_bytes,
        }

    def validate(self, num_vectors: int = 10000,
                 device: Optional[str] = None) -> dict:
        """
        Run validation: encode→decode random unit vectors, measure distortion.
        Returns dict with MSE, inner product error, and theory bounds.
        """
        dev = device or self.device
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"

        # Generate random unit vectors
        x = torch.randn(num_vectors, self.head_dim, device=dev, dtype=torch.float32)
        x = x / x.norm(dim=-1, keepdim=True)

        # Encode → decode
        packed, norms = self.encode(x)
        x_hat = self.decode(packed, norms).float()

        # MSE: E[||x - x̂||²]
        mse = (x - x_hat).pow(2).sum(dim=-1).mean().item()

        # Inner product error (one-sided quantization)
        n = min(num_vectors, 5000)
        idx_a = torch.randint(0, num_vectors, (n,), device=dev)
        idx_b = torch.randint(0, num_vectors, (n,), device=dev)
        true_ip = (x[idx_a] * x[idx_b]).sum(dim=-1)
        est_ip = (x_hat[idx_a] * x[idx_b]).sum(dim=-1)
        ip_mse = (true_ip - est_ip).pow(2).mean().item()
        ip_bias = (est_ip - true_ip).mean().item()

        # Max absolute error
        max_err = (x - x_hat).abs().max().item()

        return {
            "bits": self.bits,
            "head_dim": self.head_dim,
            "num_vectors": num_vectors,
            "mse": mse,
            "mse_theory_lower": self.theory_lower,
            "mse_theory_upper": self.theory_upper,
            "mse_ratio_to_lower": mse / self.theory_lower if self.theory_lower > 0 else float("inf"),
            "ip_mse": ip_mse,
            "ip_bias": ip_bias,
            "max_abs_error": max_err,
            "compression_vs_fp16": self.compression_ratio(),
            "compression_vs_fp8": self.compression_ratio_vs_fp8(),
            "triton_active": self._use_triton,
        }

    def benchmark(self, num_vectors: int = 32768, warmup: int = 10,
                  iters: int = 100, device: Optional[str] = None) -> dict:
        """
        Benchmark encode/decode throughput.
        Returns dict with timings in microseconds.
        """
        dev = device or self.device
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"

        x = torch.randn(num_vectors, self.head_dim, device=dev, dtype=torch.float16)

        # Warmup
        for _ in range(warmup):
            p, n = self.encode(x)
            _ = self.decode(p, n)

        if dev == "cuda":
            torch.cuda.synchronize()

        # Benchmark encode
        t0 = time.perf_counter()
        for _ in range(iters):
            p, n = self.encode(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        encode_us = (time.perf_counter() - t0) / iters * 1e6

        # Benchmark decode
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = self.decode(p, n)
        if dev == "cuda":
            torch.cuda.synchronize()
        decode_us = (time.perf_counter() - t0) / iters * 1e6

        return {
            "bits": self.bits,
            "num_vectors": num_vectors,
            "encode_us": encode_us,
            "decode_us": decode_us,
            "encode_throughput_mvec_s": num_vectors / encode_us,
            "decode_throughput_mvec_s": num_vectors / decode_us,
            "triton_active": self._use_triton,
            "device": dev,
        }

    def __repr__(self):
        mode = "Triton" if self._use_triton else "PyTorch"
        return (
            f"TurboQuant(d={self.head_dim}, bits={self.bits}, "
            f"compress={self.compression_ratio():.1f}x, "
            f"mse={self.mse_total:.4f}, engine={mode})"
        )
