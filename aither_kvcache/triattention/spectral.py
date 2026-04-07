"""
Spectral primitives for TriAttention.

Provides RoPE-pair energy analysis, top-k frequency selection, and
optional DCT transforms. The primary spectral basis is the RoPE pairing
structure: dimensions (2i, 2i+1) form frequency pair i with RoPE
frequency θ_i = base^{-2i/d}.

Key insight: transformer attention with RoPE is naturally a trigonometric
series in position difference Δ = n − m:

    score(q, k, m, n) = (1/√d) Σ_i [c_i cos(Δθ_i) + s_i sin(Δθ_i)]

where c_i = q_{2i}k_{2i} + q_{2i+1}k_{2i+1}  (pair dot product)
      s_i = q_{2i+1}k_{2i} - q_{2i}k_{2i+1}  (pair cross product)

The pair energy E_i = k_{2i}² + k_{2i+1}² determines how much each
frequency contributes. Retaining only high-energy pairs gives a
truncated trig series with bounded approximation error.

Also includes orthonormal DCT-II / iDCT-II for alternative spectral
analysis (value compression, spectral profiling).
"""

import math
import torch
from typing import Tuple, Optional


# ============================================================================
# ROPE FREQUENCY ANALYSIS
# ============================================================================

def rope_frequencies(
    head_dim: int,
    base: float = 1_000_000.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute RoPE frequencies θ_i = base^{-2i/d} for i = 0..d/2-1.

    Args:
        head_dim: Attention head dimension (must be even).
        base: RoPE base frequency (1e6 for Qwen3.5).

    Returns:
        Tensor of shape [head_dim // 2] with frequencies θ_i.
    """
    num_pairs = head_dim // 2
    i = torch.arange(num_pairs, device=device, dtype=dtype)
    return base ** (-2.0 * i / head_dim)


def pair_energies(x: torch.Tensor) -> torch.Tensor:
    """Compute energy per RoPE frequency pair.

    Energy E_i = x_{2i}² + x_{2i+1}² for each pair i.

    Args:
        x: Tensor of shape [..., head_dim] (head_dim must be even).

    Returns:
        Tensor of shape [..., head_dim // 2] with pair energies.
    """
    pairs = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
    return (pairs ** 2).sum(dim=-1)


def topk_pairs(
    x: torch.Tensor,
    k: int,
    return_values: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Select top-k RoPE frequency pairs by energy.

    Args:
        x: Tensor of shape [..., head_dim].
        k: Number of frequency pairs to retain.
        return_values: If True, also return the pair values.

    Returns:
        indices: [..., k] uint8 tensor of pair indices (sorted ascending).
        values: [..., k, 2] tensor of pair values (if return_values=True),
                or None.
    """
    energies = pair_energies(x)  # [..., num_pairs]
    _, top_idx = energies.topk(k, dim=-1)  # [..., k]
    top_idx = top_idx.sort(dim=-1).values  # sorted for cache coherence

    if not return_values:
        return top_idx.to(torch.uint8), None

    pairs = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)  # [..., num_pairs, 2]
    # Gather top pair values
    idx_expanded = top_idx.unsqueeze(-1).expand(*top_idx.shape, 2)
    top_values = torch.gather(pairs, -2, idx_expanded)  # [..., k, 2]

    return top_idx.to(torch.uint8), top_values


def spectral_concentration(
    x: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Fraction of total pair energy captured by top-k pairs.

    Returns a scalar (or per-batch) value in [0, 1].

    Args:
        x: Tensor of shape [..., head_dim].
        k: Number of top frequency pairs.

    Returns:
        Energy ratio: sum(top-k energies) / sum(all energies).
    """
    energies = pair_energies(x)  # [..., num_pairs]
    total_energy = energies.sum(dim=-1, keepdim=True)  # [..., 1]
    top_energies, _ = energies.topk(k, dim=-1)  # [..., k]
    top_energy = top_energies.sum(dim=-1, keepdim=True)
    return (top_energy / (total_energy + 1e-12)).squeeze(-1)


def pair_energy_profile(x: torch.Tensor) -> torch.Tensor:
    """Sorted (descending) energy profile for spectral analysis.

    Useful for determining optimal num_freqs for a given tensor.

    Args:
        x: Tensor of shape [..., head_dim].

    Returns:
        Sorted energies [..., num_pairs], descending.
    """
    energies = pair_energies(x)
    return energies.sort(dim=-1, descending=True).values


# ============================================================================
# TRIGONOMETRIC SERIES COMPONENTS
# ============================================================================

def trig_series_coefficients(
    q: torch.Tensor,
    k: torch.Tensor,
    freq_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute cosine and sine coefficients of the attention trig series.

    For frequency pair i:
        c_i = q_{2i} k_{2i} + q_{2i+1} k_{2i+1}   (pair dot product)
        s_i = q_{2i+1} k_{2i} - q_{2i} k_{2i+1}   (pair cross product)

    Args:
        q: Query vectors [..., head_dim].
        k: Key vectors [..., head_dim].
        freq_indices: Optional [..., F] indices to compute for.
                      If None, compute for all pairs.

    Returns:
        c: Cosine coefficients [..., num_pairs] or [..., F].
        s: Sine coefficients [..., num_pairs] or [..., F].
    """
    q_even = q[..., 0::2]  # [..., D/2]
    q_odd = q[..., 1::2]   # [..., D/2]
    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]

    if freq_indices is not None:
        idx = freq_indices.long()
        q_even = torch.gather(q_even, -1, idx)
        q_odd = torch.gather(q_odd, -1, idx)
        k_even = torch.gather(k_even, -1, idx)
        k_odd = torch.gather(k_odd, -1, idx)

    c = q_even * k_even + q_odd * k_odd     # pair dot product
    s = q_odd * k_even - q_even * k_odd      # pair cross product
    return c, s


def rope_phase_matrix(
    positions_q: torch.Tensor,
    positions_k: torch.Tensor,
    theta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE phase cos/sin matrices for trig series scoring.

    Args:
        positions_q: [B] or [B, 1] query positions.
        positions_k: [B, S] key positions.
        theta: [num_pairs] or [F] RoPE frequencies.

    Returns:
        cos_phases: [B, S, num_freqs] cos(Δ θ_i).
        sin_phases: [B, S, num_freqs] sin(Δ θ_i).
    """
    if positions_q.dim() == 1:
        positions_q = positions_q.unsqueeze(-1)  # [B, 1]

    delta = positions_k - positions_q  # [B, S]
    # phases: [B, S, num_freqs]
    phases = delta.unsqueeze(-1) * theta.unsqueeze(0).unsqueeze(0)
    return torch.cos(phases), torch.sin(phases)


# ============================================================================
# DCT-II / iDCT-II (Discrete Cosine Transform, orthonormal)
# ============================================================================

def dct_matrix(
    N: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Construct orthonormal DCT-II matrix of size N × N.

    D[k, n] = w_k * sqrt(2/N) * cos(π(2n+1)k / (2N))

    where w_0 = 1/sqrt(2), w_k = 1 for k > 0.

    Satisfies D @ D^T = I (orthogonal).

    Args:
        N: Dimension (positive integer).

    Returns:
        [N, N] orthonormal DCT-II matrix.
    """
    n = torch.arange(N, device=device, dtype=dtype)
    k = torch.arange(N, device=device, dtype=dtype)
    # D[k, n] = cos(π(2n+1)k / (2N))
    D = torch.cos(math.pi * k.unsqueeze(1) * (2 * n.unsqueeze(0) + 1) / (2 * N))
    # Orthonormal scaling
    D *= math.sqrt(2.0 / N)
    D[0] *= 1.0 / math.sqrt(2.0)
    return D


def dct(x: torch.Tensor, D: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply orthonormal DCT-II to last dimension of x.

    Args:
        x: Input tensor [..., N].
        D: Pre-computed DCT matrix [N, N]. If None, computed on the fly.

    Returns:
        DCT coefficients [..., N].
    """
    N = x.shape[-1]
    if D is None:
        D = dct_matrix(N, device=x.device, dtype=x.dtype)
    return x @ D.T


def idct(x: torch.Tensor, D: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply inverse DCT-II (= DCT-III, orthonormal) to last dimension.

    Since D is orthonormal, D^{-1} = D^T.

    Args:
        x: DCT coefficients [..., N].
        D: Pre-computed DCT matrix [N, N]. If None, computed on the fly.

    Returns:
        Reconstructed signal [..., N].
    """
    N = x.shape[-1]
    if D is None:
        D = dct_matrix(N, device=x.device, dtype=x.dtype)
    return x @ D  # D^T @ x in the other convention; here x @ D = x @ (D^T)^T
