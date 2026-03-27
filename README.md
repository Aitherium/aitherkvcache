# aither-kvcache

Near-optimal KV cache quantization for LLM inference. Implements the TurboQuant
algorithm from [Zandieh et al. (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874).

Compresses KV cache vectors to 2-4 bits per value with MSE within 2.7x of
the information-theoretic lower bound. No calibration data. No retraining.
Works on streaming tokens.

## Installation

```bash
pip install aither-kvcache            # core library
pip install aither-kvcache[vllm]      # + vLLM plugin (v0.15+)
pip install aither-kvcache[triton]    # + fused GPU kernels
pip install aither-kvcache[all]       # everything
```

## Quick Start

```python
from turboquant import TurboQuant

tq = TurboQuant(head_dim=128, bits=4, device="cuda")

packed, norms = tq.encode(kv_vectors)   # [..., 128] float16 -> [..., 64] uint8 + [...] f32
decoded = tq.decode(packed, norms)       # [..., 64] uint8 + [...] f32 -> [..., 128] float16
```

## vLLM Integration

Works with vLLM v0.15+ via the official plugin system. No monkey-patching.

```bash
pip install aither-kvcache[vllm]
VLLM_ATTENTION_BACKEND=CUSTOM vllm serve your-model
```

The plugin auto-registers at startup in all vLLM processes (API server + engine workers)
via Python entry points. It provides:

- **TurboQuantBackend**: registered as the `CUSTOM` attention backend
- **TurboQuantImpl**: handles attention using vLLM's Triton kernels + async TQ compression
- **ColdTierCache**: background GPU-to-CPU transfer + TQ encode on a separate thread, zero sync on the attention hot path

Every token is TQ-compressed to a CPU cold tier in the background. The cold tier
provides `decompress_blocks()` for future block warming (prefix cache from compressed data).

```python
# Or register manually in your own code:
from turboquant.vllm import register
register()
```

## Where This Fits

### Custom inference loop

If you manage your own KV cache, drop `encode()` where you write and `decode()` where you read:

```python
from turboquant import TurboQuant

tq = TurboQuant(head_dim=128, bits=4, device="cuda")

# Write to cache: compress
packed, norms = tq.encode(key_proj)       # [batch, heads, 128] -> [batch, heads, 64] uint8

# Read from cache: decompress
key_restored = tq.decode(packed, norms)   # -> [batch, heads, 128] float16
```

### Paged KV cache

Works with block-structured caches (like vLLM's). Handles arbitrary batch dimensions:

```python
# Compress a block of 16 tokens across 8 heads
block = cache[block_idx]                   # [16, 8, 128]
packed, norms = tq.encode(block)           # [16, 8, 64] uint8 + [16, 8] f32
restored = tq.decode(packed, norms)        # [16, 8, 128]
```

### Zero-buffer fused attention

Compute attention directly from compressed data without ever decompressing:

```python
from turboquant.fused_attention import TQPagedAttention

attn = TQPagedAttention(tq, num_query_heads=32)
output = attn.forward(
    query, k_packed, k_norms, v_packed, v_norms,
    block_tables, context_lens,
)
```

The math: rotate the query forward once, dot-product in the rotated domain against
codebook-decoded values, accumulate weighted values in the rotated domain, rotate back
once. Two matrix multiplies total regardless of context length.

This is a PyTorch reference implementation. A production Triton kernel is next.

### Research / benchmarking

```python
tq = TurboQuant(head_dim=128, bits=4)
print(tq.validate(num_vectors=50000))
```

```bash
python -m turboquant.bench
```

## Compression Ratios

For head_dim=128:

| Bits | Bytes/vector | vs FP16 | vs FP8 |
|------|-------------|---------|--------|
| 4    | 68          | 3.8x    | 1.9x   |
| 3    | 52          | 4.9x    | 2.5x   |
| 2    | 36          | 7.1x    | 3.6x   |

## Validated MSE

| Bits | MSE | Theory Lower | Theory Upper | Ratio to LB |
|------|-----|-------------|-------------|-------------|
| 4    | 0.0095 | 0.0039 | 0.0184 | 2.4x |
| 3    | 0.0345 | 0.0156 | 0.0736 | 2.2x |
| 2    | 0.1175 | 0.0625 | 0.2945 | 1.9x |

## Algorithm

1. **Normalize**: extract L2 norm, project onto unit sphere
2. **Rotate**: multiply by a fixed random orthogonal matrix (data-oblivious). Makes each coordinate ~N(0, 1/d).
3. **Quantize**: each coordinate via precomputed Lloyd-Max codebook
4. **Pack**: indices into uint8 bytes
5. **Store**: packed bytes + float32 norm

Decoding reverses steps 4-1.

## API Reference

```python
class TurboQuant:
    def __init__(self, head_dim=128, bits=4, seed=42, device="cuda", ...)
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]
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

```bibtex
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}
```

## License

CC BY 4.0
