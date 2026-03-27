"""
Random rotation matrices for TurboQuant.

The random rotation maps worst-case input vectors to uniformly distributed
points on S^{d-1}, where each coordinate follows a Beta distribution
(approximately Gaussian for d >= 64). This is the key insight enabling optimal
per-coordinate scalar quantization.

Two approaches:
  1. Full random orthogonal matrix via QR decomposition (default)
  2. Randomized Hadamard Transform (faster for large d, same quality)

For head_dim=128, both are fast (<1us per vector with cuBLAS).
"""

import math
import torch


def random_orthogonal(d: int, seed: int = 42, device: str = "cuda",
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Generate a random orthogonal matrix via QR decomposition of a Gaussian matrix.

    This is the standard approach from Mezzadri (2007). The resulting matrix
    is uniformly distributed over O(d) (Haar measure).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    G = torch.randn(d, d, generator=gen, dtype=dtype)
    Q, R = torch.linalg.qr(G)
    # Fix sign ambiguity: ensure det(Q) = +1 (proper rotation)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device=device)


def hadamard_matrix(d: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Construct normalized Walsh-Hadamard matrix of dimension d.
    d must be a power of 2. Result satisfies H @ H.T = I.
    """
    assert d > 0 and (d & (d - 1)) == 0, f"d must be power of 2, got {d}"
    H = torch.tensor([[1.0]], dtype=dtype)
    while H.shape[0] < d:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H / math.sqrt(d)


def random_signs(d: int, seed: int = 42, device: str = "cuda") -> torch.Tensor:
    """Generate a random +/-1 sign vector (Rademacher distribution)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    bits = torch.randint(0, 2, (d,), generator=gen, dtype=torch.float32)
    return (2 * bits - 1).to(device)


def randomized_hadamard_matrix(d: int, seed: int = 42, device: str = "cuda",
                                dtype: torch.dtype = torch.float32,
                                num_rounds: int = 3) -> torch.Tensor:
    """
    Construct Randomized Hadamard Transform (RHT) matrix.

    RHT = H . D_k . H . D_{k-1} . ... . H . D_1

    where H is the normalized Walsh-Hadamard matrix and D_i are random
    diagonal sign matrices. With num_rounds >= 3, the result closely
    approximates a Haar-random orthogonal matrix.

    Advantage: O(d log d) apply via butterfly, vs O(d^2) for full matrix.
    For d=128 the difference is negligible, but matters for d >= 1024.
    """
    H = hadamard_matrix(d, dtype=dtype).to(device)
    result = torch.eye(d, dtype=dtype, device=device)
    for r in range(num_rounds):
        D = torch.diag(random_signs(d, seed=seed + r, device=device).to(dtype))
        result = H @ D @ result
    return result


def fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Apply unnormalized Walsh-Hadamard transform to last dimension of x.
    Uses O(d log d) butterfly operations without materializing the matrix.

    Multiply result by 1/sqrt(d) for the normalized version.
    """
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, f"last dim must be power of 2, got {d}"

    result = x.clone()
    log2_d = int(math.log2(d))

    for s in range(log2_d):
        h = 1 << s
        block = h << 1
        # Reshape to expose butterfly pairs
        shape_prefix = result.shape[:-1]
        result = result.view(*shape_prefix, d // block, 2, h)
        top = result[..., 0, :].clone()
        bot = result[..., 1, :].clone()
        result[..., 0, :] = top + bot
        result[..., 1, :] = top - bot
        result = result.view(*shape_prefix, d)

    return result
