"""
Spectral KV Encoder — compress key/value vectors to spectral representation.

The encoder selects the top-F RoPE frequency pairs by energy and stores
only those pair values (optionally quantized to 4-bit or 8-bit), plus
a scale factor. This achieves ~10× compression for 4-bit with F=12.

Storage layout per token (int4 mode, F=12):
    indices:  12 × uint8   = 12 bytes   (frequency pair indices, 0..63)
    packed:   12 × uint8   = 12 bytes   (two 4-bit values per byte)
    scale:     1 × float16 =  2 bytes   (max absolute coefficient value)
    ─────────────────────────────────────
    Total:                   26 bytes   (vs 256 bytes FP16 → 9.85× compression)

Encoding pipeline:
    1. Reshape to pairs: x[..., d] → pairs[..., d/2, 2]
    2. Compute pair energies: E_i = pairs[i, 0]² + pairs[i, 1]²
    3. Select top-F pairs by energy (sorted ascending for cache coherence)
    4. Extract pair values at selected indices
    5. Quantize pair values (int4/int8) with symmetric uniform quantizer
    6. Pack two 4-bit values per byte (for int4 mode)

Decoding:
    1. Unpack quantized values
    2. Dequantize: val = (code / max_code) * 2 - 1) * scale
    3. Scatter pair values back to full head_dim vector
"""

import math
import torch
from typing import Tuple, Optional, NamedTuple

from .config import TriAttentionConfig


class SpectralEncoding(NamedTuple):
    """Container for spectral-encoded K or V token(s).

    All tensors share the same batch prefix shape [...].
    """
    indices: torch.Tensor    # [..., F] uint8 — frequency pair indices
    packed: torch.Tensor     # [..., F] uint8 (int4) or [..., F, 2] int8/fp16
    scales: torch.Tensor     # [...] float16 — per-token scale factors


class SpectralKVEncoder:
    """Encode/decode key and value vectors to spectral representation.

    Usage:
        encoder = SpectralKVEncoder(config)
        enc_k = encoder.encode(keys)           # compress
        dec_k = encoder.decode(enc_k)          # reconstruct
        mse = ((keys - dec_k) ** 2).mean()     # measure quality
    """

    def __init__(self, config: TriAttentionConfig):
        self.config = config
        self.head_dim = config.head_dim
        self.num_pairs = config.num_pairs
        self.num_freqs = config.num_freqs
        self.coeff_bits = config.coeff_bits

        # Quantization levels for symmetric uniform quantizer
        if self.coeff_bits == 4:
            self._max_code = 15
            self._half_code = 7.5
        elif self.coeff_bits == 8:
            self._max_code = 255
            self._half_code = 127.5

    # ── ENCODE ────────────────────────────────────────────────────

    def encode(
        self,
        x: torch.Tensor,
        num_freqs: Optional[int] = None,
    ) -> SpectralEncoding:
        """Encode vectors to spectral representation.

        Args:
            x: Input tensor [..., head_dim].
            num_freqs: Override num_freqs (for adaptive mode).

        Returns:
            SpectralEncoding with indices, packed coefficients, and scales.
        """
        F = num_freqs or self.num_freqs

        # Step 1: Reshape to pairs
        pairs = x.view(*x.shape[:-1], self.num_pairs, 2).float()

        # Step 2: Pair energies
        energies = (pairs ** 2).sum(dim=-1)  # [..., num_pairs]

        # Step 3: Top-F selection (sorted ascending for cache coherence)
        _, top_idx = energies.topk(F, dim=-1)  # [..., F]
        top_idx = top_idx.sort(dim=-1).values

        # Step 4: Gather pair values
        idx_expanded = top_idx.unsqueeze(-1).expand(*top_idx.shape, 2)
        top_pairs = torch.gather(pairs, -2, idx_expanded)  # [..., F, 2]

        # Step 5-6: Quantize and pack
        indices = top_idx.to(torch.uint8)

        if self.coeff_bits == 16:
            # No quantization — store raw float16
            return SpectralEncoding(
                indices=indices,
                packed=top_pairs.to(torch.float16),
                scales=torch.ones(
                    *x.shape[:-1], dtype=torch.float16, device=x.device
                ),
            )

        # Compute per-token scale: max absolute value across all retained coeffs
        scales = top_pairs.abs().reshape(*top_pairs.shape[:-2], -1).amax(dim=-1)
        scales = scales.clamp(min=1e-8)  # avoid division by zero

        # Normalize to [-1, 1]
        normalized = top_pairs / scales.unsqueeze(-1).unsqueeze(-1)

        if self.coeff_bits == 4:
            # Symmetric uniform quantization to 4-bit [0, 15]
            quantized = ((normalized + 1) * self._half_code).round().clamp(
                0, self._max_code
            ).to(torch.uint8)
            # Pack: high nibble = dim_even, low nibble = dim_odd
            packed = (quantized[..., 0] << 4) | quantized[..., 1]
            return SpectralEncoding(
                indices=indices,
                packed=packed,
                scales=scales.to(torch.float16),
            )

        elif self.coeff_bits == 8:
            quantized = ((normalized + 1) * self._half_code).round().clamp(
                0, self._max_code
            ).to(torch.uint8)
            # Store both components as separate uint8 values
            # Reshape: [..., F, 2] → [..., F*2] for flat storage
            packed = quantized.reshape(*quantized.shape[:-2], F * 2)
            return SpectralEncoding(
                indices=indices,
                packed=packed,
                scales=scales.to(torch.float16),
            )

        raise ValueError(f"Unsupported coeff_bits: {self.coeff_bits}")

    # ── DECODE ────────────────────────────────────────────────────

    def decode(self, enc: SpectralEncoding) -> torch.Tensor:
        """Reconstruct full vectors from spectral encoding.

        Args:
            enc: SpectralEncoding from encode().

        Returns:
            Reconstructed tensor [..., head_dim].
        """
        indices = enc.indices.long()
        device = enc.indices.device

        if self.coeff_bits == 16:
            # Unpack raw float16 pairs
            pair_vals = enc.packed.float()  # [..., F, 2]
        elif self.coeff_bits == 4:
            # Unpack 4-bit values
            v0 = (enc.packed >> 4).float()          # high nibble
            v1 = (enc.packed & 0x0F).float()        # low nibble
            # Dequantize: [0, 15] → [-1, 1] → scaled
            scales = enc.scales.float()
            v0 = (v0 / self._half_code - 1.0) * scales.unsqueeze(-1)
            v1 = (v1 / self._half_code - 1.0) * scales.unsqueeze(-1)
            pair_vals = torch.stack([v0, v1], dim=-1)  # [..., F, 2]
        elif self.coeff_bits == 8:
            F = indices.shape[-1]
            raw = enc.packed.float()
            raw = raw.reshape(*raw.shape[:-1], F, 2)
            scales = enc.scales.float()
            pair_vals = (raw / self._half_code - 1.0) * scales.unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported coeff_bits: {self.coeff_bits}")

        # Scatter back to full vector
        batch_shape = indices.shape[:-1]
        result = torch.zeros(
            *batch_shape, self.head_dim,
            dtype=pair_vals.dtype, device=device,
        )

        # Scatter even components (dim 2*i)
        even_indices = indices * 2
        result.scatter_(-1, even_indices, pair_vals[..., 0])

        # Scatter odd components (dim 2*i + 1)
        odd_indices = indices * 2 + 1
        result.scatter_(-1, odd_indices, pair_vals[..., 1])

        return result

    # ── UTILITIES ─────────────────────────────────────────────────

    def encode_adaptive(
        self,
        x: torch.Tensor,
        min_energy_ratio: Optional[float] = None,
        max_freqs: Optional[int] = None,
    ) -> Tuple[SpectralEncoding, torch.Tensor]:
        """Encode with adaptive frequency selection.

        Increases num_freqs per token until the specified energy ratio
        is achieved, up to max_freqs.

        Returns:
            (encoding, actual_num_freqs_per_token)
        """
        from .spectral import pair_energies

        if min_energy_ratio is None:
            min_energy_ratio = self.config.min_energy_ratio
        if max_freqs is None:
            max_freqs = self.num_pairs

        energies = pair_energies(x)
        total = energies.sum(dim=-1, keepdim=True)
        sorted_energies = energies.sort(dim=-1, descending=True).values
        cumsum = sorted_energies.cumsum(dim=-1)
        ratio = cumsum / (total + 1e-12)

        # Find minimum F per token where ratio >= threshold
        above = ratio >= min_energy_ratio
        # First index where above is True, +1 for count
        first_above = above.float().argmax(dim=-1) + 1
        # Clamp to [num_freqs, max_freqs]
        actual_freqs = first_above.clamp(self.num_freqs, max_freqs)

        # For simplicity, use the max across the batch
        max_f = int(actual_freqs.max().item())
        return self.encode(x, num_freqs=max_f), actual_freqs

    def compression_stats(self, x: torch.Tensor) -> dict:
        """Compute compression statistics for input vectors.

        Returns dict with compression ratio, energy retention, MSE, etc.
        """
        from .spectral import spectral_concentration

        enc = self.encode(x)
        dec = self.decode(enc)

        mse = ((x.float() - dec.float()) ** 2).mean().item()
        energy_ratio = spectral_concentration(x, self.num_freqs).mean().item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            x.float().reshape(-1, self.head_dim),
            dec.float().reshape(-1, self.head_dim),
            dim=-1,
        ).mean().item()

        return {
            "num_freqs": self.num_freqs,
            "coeff_bits": self.coeff_bits,
            "bytes_per_token": self.config.bytes_per_kv_token,
            "compression_ratio": self.config.compression_ratio,
            "mse": mse,
            "energy_ratio": energy_ratio,
            "cosine_similarity": cosine_sim,
        }
