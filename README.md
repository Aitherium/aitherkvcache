# TurboQuant

Near-optimal KV cache quantization for LLM inference. Implements the algorithm
from [Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal
Distortion Rate" (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874).

Compresses KV cache vectors to 2-4 bits per value with MSE within 2.7x of
the information-theoretic lower bound. No calibration data. No retraining.
Works online (one vector at a time).


## Installation

```bash
pip install turboquant
```

Optional extras:

```bash
pip install turboquant[triton]   # GPU-fused quantize/dequantize kernels
pip install turboquant[scipy]    # Custom codebook computation via Lloyd-Max
pip install turboquant[dev]      # pytest for running tests
```


## Quick Start

```python
from turboquant import TurboQuant

tq = TurboQuant(head_dim=128, bits=4, device="cuda")

# Encode: FP16 vectors -> packed uint8 + norms
packed, norms = tq.encode(kv_vectors)   # kv_vectors: [..., 128] float16

# Decode: packed representation -> reconstructed vectors
decoded = tq.decode(packed, norms)      # decoded: [..., 128] float16

# Validate MSE against theory
print(tq.validate())
```


## Algorithm

TurboQuant applies three steps to each KV cache vector:

1. **Normalize** -- extract L2 norm, project onto the unit sphere S^{d-1}.
2. **Random rotation** -- multiply by a fixed orthogonal matrix Pi. This makes
   each coordinate approximately Gaussian N(0, 1/d), regardless of the input
   distribution. The rotation is data-oblivious (generated once from a seed).
3. **Optimal scalar quantization** -- quantize each coordinate independently
   using a precomputed Lloyd-Max codebook for N(0, 1/d). Pack indices into
   uint8 bytes.

Storage per vector: `ceil(d * bits / 8)` bytes for indices + 4 bytes for the
float32 norm.

Decoding reverses the process: unpack indices, look up codebook centroids,
apply inverse rotation Pi^T, rescale by the stored norm.


## Compression Ratios

Ratios for head_dim=128 (256 bytes at FP16, 128 bytes at FP8):

| Bits | Packed Size | Ratio vs FP16 | Ratio vs FP8 |
|------|-------------|---------------|--------------|
| 4    | 68 bytes    | 3.8x          | 1.9x         |
| 3    | 52 bytes    | 4.9x          | 2.5x         |
| 2    | 36 bytes    | 7.1x          | 3.6x         |


## Validated MSE

MSE for unit vectors on S^{127} (d=128). Theory bounds from the paper:

| Bits | MSE (measured) | Theory Lower | Theory Upper | Ratio to LB |
|------|---------------|--------------|--------------|-------------|
| 4    | 0.0095        | 0.0039       | 0.0184       | 2.4x        |
| 3    | 0.0345        | 0.0156       | 0.0736       | 2.2x        |
| 2    | 0.1175        | 0.0625       | 0.2945       | 1.9x        |

All measured values are within the paper's upper bound and well below the
worst-case ratio of 3*pi/2 = 4.71x.


## Fused Attention (TQPagedAttention)

The key optimization for inference: compute attention scores and accumulate
values **in the rotated domain** without ever materializing a decompression
buffer.

```python
from turboquant.fused_attention import TQPagedAttention

attn = TQPagedAttention(tq, num_query_heads=32)
output = attn.forward(
    query,          # [num_seqs, num_query_heads, head_dim]
    k_packed,       # [num_blocks, block_size, num_kv_heads, packed_dim]
    k_norms,        # [num_blocks, block_size, num_kv_heads]
    v_packed,       # same layout as k_packed
    v_norms,        # same layout as k_norms
    block_tables,   # [num_seqs, max_blocks_per_seq]
    context_lens,   # [num_seqs]
)
```

The math:

```
q_rot = Pi @ q                                          # rotate query once
score_i = ||k_i|| * dot(q_rot, y_hat_k_i) / sqrt(d)    # score in rotated domain
acc += softmax_weight_i * ||v_i|| * y_hat_v_i           # accumulate rotated V
output = Pi^T @ normalize(acc)                           # rotate back once
```

This reads packed uint8 indices directly, avoids the O(seq_len * head_dim)
decompression buffer, and uses the even/odd nibble split to compute dot
products without interleaving after 4-bit unpacking.


## API Reference

### TurboQuant

```python
class TurboQuant:
    def __init__(self, config=None, *, head_dim=128, bits=4, seed=42,
                 use_hadamard=False, device="cuda", dtype=torch.float16,
                 use_triton=True): ...

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]: ...
    def decode(self, packed: Tensor, norms: Tensor) -> Tensor: ...
    def validate(self, num_vectors=10000, device=None) -> dict: ...
    def benchmark(self, num_vectors=32768, warmup=10, iters=100) -> dict: ...
    def compression_ratio(self) -> float: ...
    def memory_report(self, seq_len, num_layers=32, num_kv_heads=8) -> dict: ...
```

### TurboQuantConfig

```python
@dataclass
class TurboQuantConfig:
    head_dim: int = 128          # Must be power of 2
    bits: int = 4                # 2, 3, or 4
    seed: int = 42               # RNG seed for rotation matrix
    use_hadamard: bool = False   # True = Randomized Hadamard Transform
    hadamard_rounds: int = 3     # RHT rounds (>= 3 for near-Haar)
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    use_triton: bool = True      # Try fused Triton kernels on GPU
```

### TQPagedAttention

```python
class TQPagedAttention:
    def __init__(self, tq: TurboQuant, num_query_heads: int): ...

    def forward(self, query, k_packed, k_norms, v_packed, v_norms,
                block_tables, context_lens, block_size=16,
                num_kv_heads=None) -> Tensor: ...
```


## Benchmark

Run the built-in benchmark to validate correctness and measure throughput:

```bash
python -m turboquant.bench
```

This reports:
- MSE vs theoretical bounds for each bit-width
- Encode/decode throughput (vectors/second)
- KV cache memory usage for common model configurations
- Maximum context length estimates for given GPU memory


## Codebook Computation

The package includes hardcoded Lloyd-Max codebooks for 1-4 bit quantization
of N(0,1). For custom configurations, compute codebooks from scratch:

```python
from turboquant.codebook import compute_codebook_scipy

centroids, boundaries, mse = compute_codebook_scipy(d=128, bits=3)
```

Requires `scipy` (install with `pip install turboquant[scipy]`).


## Reference

```
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Han, Insu and Daliri, Majid and Karbasi, Amin},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}
```


## License

CC BY 4.0 -- see LICENSE file.
