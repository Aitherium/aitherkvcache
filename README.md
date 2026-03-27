# aither-kvcache

Near-optimal KV cache quantization for LLM inference. Implements the TurboQuant
algorithm from [Zandieh et al. (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874).

Compresses KV cache vectors to 2-4 bits per value with MSE within 2.7x of
the information-theoretic lower bound. No calibration data. No retraining.
Works on streaming tokens.

## Installation

```bash
pip install aither-kvcache
```

## What This Does

Every transformer layer stores key-value vectors in a KV cache during inference.
At FP16, each 128-dim vector costs 256 bytes. At FP8, 128 bytes. This library
compresses them to 36-68 bytes (2-4 bit) with near-zero quality loss.

| Bits | Bytes per vector | vs FP16 | vs FP8 |
|------|-----------------|---------|--------|
| 4    | 68              | 3.8x    | 1.9x   |
| 3    | 52              | 4.9x    | 2.5x   |
| 2    | 36              | 7.1x    | 3.6x   |

## Where This Fits

### In a custom inference loop

If you manage your own KV cache, drop `encode()` where you write to cache
and `decode()` where you read from it:

```python
from turboquant import TurboQuant
import torch

tq = TurboQuant(head_dim=128, bits=4, device="cuda")

# Your model produces key/value projections each step
# key_proj: [batch, num_kv_heads, head_dim]

# BEFORE: store raw FP16
# kv_cache[layer][block, pos] = key_proj          # 256 bytes per vector

# AFTER: store compressed
packed, norms = tq.encode(key_proj)                # 68 bytes per vector
# Store packed (uint8) and norms (float32) instead of the raw tensor

# When attention needs the keys:
key_restored = tq.decode(packed, norms)             # back to FP16
# Use key_restored in your attention computation
```

### In a paged KV cache (like vLLM's block manager)

Paged caches store KV data in fixed-size blocks indexed by a block table.
The library handles arbitrary batch dimensions, so it works with paged layouts:

```python
tq = TurboQuant(head_dim=128, bits=4, device="cuda")

# Block-structured cache: [num_blocks, block_size, num_kv_heads, head_dim]
# Compress the entire cache or individual blocks:

# Compress a block of 16 tokens across 8 heads
block = cache[block_idx]                           # [16, 8, 128] FP16
packed, norms = tq.encode(block)                   # [16, 8, 64] uint8 + [16, 8] f32

# Decompress when needed for attention
restored = tq.decode(packed, norms)                # [16, 8, 128] FP16
```

### Zero-buffer attention (the real goal)

The fused attention kernel computes attention **directly from compressed data**
without decompressing first. This is where the memory savings actually happen —
no decompression buffer means the compressed cache IS the only cache:

```python
from turboquant import TurboQuant
from turboquant.fused_attention import TQPagedAttention

tq = TurboQuant(head_dim=128, bits=4, device="cuda")
attn = TQPagedAttention(tq, num_query_heads=32)

# query:        [num_seqs, num_query_heads, head_dim]  — from model
# k_packed:     [num_blocks, block_size, num_kv_heads, 64]  — compressed keys
# k_norms:      [num_blocks, block_size, num_kv_heads]  — key norms
# block_tables: [num_seqs, max_blocks_per_seq]  — maps sequences to blocks
# context_lens: [num_seqs]  — tokens per sequence

output = attn.forward(
    query, k_packed, k_norms, v_packed, v_norms,
    block_tables, context_lens,
)
```

The math: rotate the query forward once (`Pi @ q`), dot-product against
codebook-decoded values in the rotated domain, accumulate weighted values
in the rotated domain, rotate back once (`Pi^T @ acc`). Two matrix multiplies
total regardless of context length. Everything else is codebook lookups and
dot products.

This is a **PyTorch reference implementation** (correct, tested, works on CPU
and GPU). A production Triton kernel is the next step.

### As a research tool

Validate the paper's theoretical bounds on your own data:

```python
tq = TurboQuant(head_dim=128, bits=4)
result = tq.validate(num_vectors=50000)
print(f"MSE: {result['mse']:.6f}")
print(f"Theory range: [{result['mse_theory_lower']:.6f}, {result['mse_theory_upper']:.6f}]")
print(f"Ratio to lower bound: {result['mse_ratio_to_lower']:.2f}x (paper claims <= 2.7x)")
```

Compare compression vs quality at different bit widths:

```bash
python -m turboquant.bench
```

Compute custom codebooks for non-standard dimensions:

```python
from turboquant.codebook import compute_codebook_scipy
centroids, boundaries, mse = compute_codebook_scipy(d=256, bits=3)
```

## What This Does NOT Do (Yet)

- **Does not integrate with vLLM/llama.cpp/TensorRT-LLM out of the box.**
  The encode/decode primitives are framework-agnostic. Framework integration
  requires hooking into the specific cache allocation and attention paths.
  We have a working vLLM v0.15 integration in our production stack
  (280K tokens on RTX 5090) but it's not portable yet.

- **The fused attention is a PyTorch reference, not a Triton kernel.**
  It's correct (16/16 tests pass, validated against decompress+FlashAttn)
  but not optimized for production throughput. The Triton kernel is next.

- **Does not handle KV cache management** (block allocation, eviction,
  prefix caching). It compresses and decompresses vectors. Your inference
  framework handles the rest.

## Algorithm

1. **Normalize**: extract L2 norm, project onto unit sphere
2. **Rotate**: multiply by a fixed random orthogonal matrix Pi (data-oblivious,
   generated once from a seed). Makes each coordinate ~N(0, 1/d).
3. **Quantize**: each coordinate independently via precomputed Lloyd-Max codebook
4. **Pack**: indices into uint8 bytes (2 values/byte at 4-bit, 4 values/byte at 2-bit)
5. **Store**: packed bytes + float32 norm

Decoding reverses steps 4-1.

## Validated MSE

| Bits | MSE | Theory Lower | Theory Upper | Ratio to LB |
|------|-----|-------------|-------------|-------------|
| 4    | 0.0095 | 0.0039 | 0.0184 | 2.4x |
| 3    | 0.0345 | 0.0156 | 0.0736 | 2.2x |
| 2    | 0.1175 | 0.0625 | 0.2945 | 1.9x |

## API Reference

```python
class TurboQuant:
    def __init__(self, head_dim=128, bits=4, seed=42, device="cuda", ...)
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]    # -> (packed, norms)
    def decode(self, packed: Tensor, norms: Tensor) -> Tensor
    def validate(self, num_vectors=10000) -> dict
    def benchmark(self, num_vectors=32768) -> dict
    def compression_ratio(self) -> float
    def memory_report(self, seq_len, num_layers=32, num_kv_heads=8) -> dict

class TQPagedAttention:
    def __init__(self, tq: TurboQuant, num_query_heads: int)
    def forward(self, query, k_packed, k_norms, v_packed, v_norms,
                block_tables, context_lens, block_size=16) -> Tensor
```

## Reference

```
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}
```

## License

CC BY 4.0
