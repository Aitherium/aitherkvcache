"""
TriAttention configuration — spectral KV cache compression parameters.

Controls frequency budget, quantization, RoPE settings, and per-layer
adaptive schedules. Designed for Qwen3.5 family but generic enough
for any RoPE-based transformer.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TriAttentionConfig:
    """Configuration for TriAttention spectral KV cache compression.

    Core parameters:
        head_dim:       Dimension per attention head (128 for Qwen3.5).
        num_freqs:      Number of RoPE frequency pairs to retain per token.
                        Higher = better quality, lower compression. Default 12
                        gives ~10× compression with 4-bit coefficients.
        coeff_bits:     Bit-width for coefficient quantization (4, 8, or 16).
        pre_rope:       If True (default), store keys BEFORE RoPE and handle
                        position via analytic trig phases. Better spectral
                        concentration → better compression.

    Model parameters:
        num_kv_heads:   Number of KV heads (for GQA).
        num_query_heads: Number of query heads.
        rope_base:      RoPE base frequency (1e6 for Qwen3.5).
        rope_dim:       RoPE rotation dimension (defaults to head_dim).

    Quality control:
        min_energy_ratio: Minimum fraction of pair energy to retain.
                          If adaptive_freqs=True, num_freqs increases
                          until this threshold is met.
        adaptive_freqs:   Dynamically choose num_freqs per token.

    Scheduling:
        layer_freq_schedule: Per-layer num_freqs overrides.
                             Early layers often need more frequencies.
    """

    # ── Core ──────────────────────────────────────────────────────
    head_dim: int = 128
    num_freqs: int = 12
    coeff_bits: int = 4
    pre_rope: bool = True
    block_size: int = 16

    # ── Model ─────────────────────────────────────────────────────
    num_kv_heads: int = 8
    num_query_heads: int = 32
    rope_base: float = 1_000_000.0
    rope_dim: Optional[int] = None
    model_family: str = "generic"

    # ── Quality control ───────────────────────────────────────────
    min_energy_ratio: float = 0.85
    adaptive_freqs: bool = False

    # ── Scheduling ────────────────────────────────────────────────
    layer_freq_schedule: Optional[List[int]] = None

    def __post_init__(self):
        if self.rope_dim is None:
            self.rope_dim = self.head_dim
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {self.head_dim}")
        if self.num_freqs > self.num_pairs:
            raise ValueError(
                f"num_freqs ({self.num_freqs}) cannot exceed "
                f"num_pairs ({self.num_pairs})"
            )
        if self.coeff_bits not in (4, 8, 16):
            raise ValueError(f"coeff_bits must be 4, 8, or 16, got {self.coeff_bits}")
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads ({self.num_query_heads}) must be divisible "
                f"by num_kv_heads ({self.num_kv_heads})"
            )

    # ── Derived properties ────────────────────────────────────────

    @property
    def num_pairs(self) -> int:
        """Number of RoPE frequency pairs = head_dim // 2."""
        return self.head_dim // 2

    @property
    def gqa_ratio(self) -> int:
        """Number of query heads per KV head."""
        return self.num_query_heads // self.num_kv_heads

    @property
    def bytes_per_kv_token(self) -> int:
        """Storage bytes per single K or V token in spectral form.

        Layout:
            indices:  F × uint8     (frequency pair indices)
            coeffs:   depends on coeff_bits
            norm:     1 × float16   (scale factor)

        int4:  F + F + 2     = 2F + 2 bytes
        int8:  F + 2F + 2    = 3F + 2 bytes
        fp16:  F + 4F + 2    = 5F + 2 bytes
        """
        F = self.num_freqs
        if self.coeff_bits == 4:
            return 2 * F + 2          # packed: 2 nibbles per byte
        elif self.coeff_bits == 8:
            return 3 * F + 2          # 2 int8 per pair
        else:
            return 5 * F + 2          # 2 float16 per pair

    @property
    def fp16_bytes_per_kv_token(self) -> int:
        """Baseline FP16 storage per K or V token."""
        return self.head_dim * 2

    @property
    def compression_ratio(self) -> float:
        """Single-tensor (K or V) compression ratio vs FP16."""
        return self.fp16_bytes_per_kv_token / self.bytes_per_kv_token

    @property
    def kv_compression_ratio(self) -> float:
        """Combined K+V compression ratio (same ratio for both)."""
        return self.compression_ratio

    def freq_budget(self, layer_idx: int) -> int:
        """Get num_freqs for a specific layer."""
        if self.layer_freq_schedule and layer_idx < len(self.layer_freq_schedule):
            return self.layer_freq_schedule[layer_idx]
        return self.num_freqs

    def summary(self) -> str:
        """Human-readable configuration summary."""
        lines = [
            f"TriAttention Config ({self.model_family})",
            f"  head_dim={self.head_dim}, num_freqs={self.num_freqs}, "
            f"coeff_bits={self.coeff_bits}",
            f"  heads: {self.num_query_heads}Q / {self.num_kv_heads}KV "
            f"(GQA {self.gqa_ratio}:1)",
            f"  rope: base={self.rope_base:.0f}, pre_rope={self.pre_rope}",
            f"  storage: {self.bytes_per_kv_token} bytes/token "
            f"(vs {self.fp16_bytes_per_kv_token} FP16)",
            f"  compression: {self.compression_ratio:.1f}× per K/V tensor",
        ]
        return "\n".join(lines)
