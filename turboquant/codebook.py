"""
TurboQuant Codebook -- Optimal Lloyd-Max scalar quantizers for sphere coordinates.

Precomputed codebooks for the Beta distribution of coordinates on S^{d-1}.
For d >= 64, the Gaussian approximation N(0, 1/d) is excellent.

The standard N(0,1) Lloyd-Max centroids are hardcoded from quantization theory.
For non-standard dimensions, scipy can compute them on the fly.

Reference: Zandieh et al., "TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate", arXiv:2504.19874, April 2025.
"""

import math
import numpy as np
from typing import Dict, Tuple

# ============================================================================
# STANDARD N(0,1) LLOYD-MAX CODEBOOKS
# ============================================================================
# These are the optimal MSE scalar quantizers for the standard normal
# distribution, from Max (1960) and Lloyd (1982). To use with N(0, 1/d),
# multiply centroids and boundaries by 1/sqrt(d).
#
# Total vector MSE = MSE_standard (independent of d, for Gaussian approx).
# ============================================================================

_STANDARD_CODEBOOKS: Dict[int, Dict] = {
    1: {
        "centroids": np.array([-0.798284, 0.798284]),
        "mse": 0.363480,
    },
    2: {
        "centroids": np.array([-1.510469, -0.452781, 0.452781, 1.510469]),
        "mse": 0.117517,
    },
    3: {
        "centroids": np.array([
            -2.152090, -1.344134, -0.756031, -0.245104,
            0.245104, 0.756031, 1.344134, 2.152090,
        ]),
        "mse": 0.034506,
    },
    4: {
        "centroids": np.array([
            -2.733266, -2.069016, -1.618002, -1.256233,
            -0.942391, -0.656804, -0.388089, -0.128350,
            0.128350, 0.388089, 0.656804, 0.942391,
            1.256233, 1.618002, 2.069016, 2.733266,
        ]),
        "mse": 0.009497,
    },
}


def _compute_boundaries(centroids: np.ndarray) -> np.ndarray:
    """Compute decision boundaries as midpoints between consecutive centroids."""
    inner = (centroids[:-1] + centroids[1:]) / 2
    return inner


def get_codebook(d: int, bits: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Get optimal codebook for dimension d and bit-width.

    Returns:
        centroids: array of 2^bits optimal reconstruction values
        boundaries_inner: array of (2^bits - 1) decision boundaries (for searchsorted)
        mse_total: total MSE E[||x - x_hat||^2] for unit vectors on S^{d-1}
    """
    if bits not in _STANDARD_CODEBOOKS:
        raise ValueError(f"Supported bit-widths: {list(_STANDARD_CODEBOOKS.keys())}. Got {bits}")

    entry = _STANDARD_CODEBOOKS[bits]
    sigma = 1.0 / math.sqrt(d)

    # Scale standard N(0,1) centroids to N(0, 1/d)
    centroids = entry["centroids"] * sigma
    boundaries_inner = _compute_boundaries(centroids)
    mse_total = entry["mse"]

    return centroids, boundaries_inner, mse_total


def get_theory_bounds(bits: int) -> Tuple[float, float]:
    """Return (lower_bound, upper_bound) on MSE for b-bit quantization."""
    lower = 1.0 / (4 ** bits)
    upper = (3 * math.pi / 2) / (4 ** bits)
    return lower, upper


def compute_codebook_scipy(d: int, bits: int, max_iter: int = 2000,
                           tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute optimal codebook from scratch using Lloyd-Max algorithm.
    Requires scipy. Use this for non-standard configurations or validation.
    """
    from scipy.integrate import quad
    from scipy.special import gammaln

    num_levels = 1 << bits

    if d >= 64:
        sigma_sq = 1.0 / d
        sigma = math.sqrt(sigma_sq)
        support = (-5 * sigma, 5 * sigma)

        def pdf(x):
            return (1 / math.sqrt(2 * math.pi * sigma_sq)) * math.exp(
                -x ** 2 / (2 * sigma_sq)
            )
    else:
        support = (-1.0, 1.0)

        log_coeff = gammaln(d / 2) - 0.5 * math.log(math.pi) - gammaln((d - 1) / 2)

        def pdf(x):
            if abs(x) >= 1.0:
                return 0.0
            return math.exp(log_coeff + ((d - 3) / 2) * math.log(1 - x * x))

    lo, hi = support
    centroids = np.linspace(lo + (hi - lo) / (2 * num_levels),
                            hi - (hi - lo) / (2 * num_levels),
                            num_levels)

    for _ in range(max_iter):
        # Boundaries = midpoints
        boundaries = np.empty(num_levels + 1)
        boundaries[0] = lo
        boundaries[-1] = hi
        for i in range(num_levels - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2

        # Centroids = conditional means
        new_centroids = np.empty(num_levels)
        for i in range(num_levels):
            b_lo, b_hi = boundaries[i], boundaries[i + 1]
            if b_hi - b_lo < 1e-15:
                new_centroids[i] = (b_lo + b_hi) / 2
                continue
            num, _ = quad(lambda x: x * pdf(x), b_lo, b_hi)
            den, _ = quad(lambda x: pdf(x), b_lo, b_hi)
            new_centroids[i] = num / den if abs(den) > 1e-15 else (b_lo + b_hi) / 2

        if np.max(np.abs(centroids - new_centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    # Compute MSE
    boundaries = np.empty(num_levels + 1)
    boundaries[0] = lo
    boundaries[-1] = hi
    for i in range(num_levels - 1):
        boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2

    mse_total = 0.0
    for i in range(num_levels):
        val, _ = quad(lambda x, c=centroids[i]: (x - c) ** 2 * pdf(x),
                      boundaries[i], boundaries[i + 1])
        mse_total += val
    mse_total *= d  # per-coord MSE x d = total MSE

    boundaries_inner = _compute_boundaries(centroids)
    return centroids, boundaries_inner, mse_total
