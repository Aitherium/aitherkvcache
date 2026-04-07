# aither-kvcache

Near-optimal KV cache compression for LLM inference. Two compression engines:

- **TurboQuant** — Vector quantization ([Zandieh et al., arXiv:2504.19874](https://arxiv.org/abs/2504.19874)).
  2-4 bit, 3.8-7.1× compression vs FP16. No calibration data. Works on streaming tokens.

- **TriAttention** *(NEW in v2.0)* — Spectral KV compression via trigonometric series.
  Retains top RoPE frequency pairs, scores via trig series without materializing full K/V.
  ~10× compression with bounded approximation error. Calibrated for Qwen3.5 family.

## Installation

```bash
pip install aither-kvcache            # core library
pip install aither-kvcache[vllm]      # + vLLM plugin (v0.15+)
pip install aither-kvcache[triton]    # + fused GPU kernels
pip install aither-kvcache[all]       # everything
```

## Quick Start — TurboQuant

```python
from aither_kvcache import TurboQuant

tq = TurboQuant(head_dim=128, bits=4, device="cuda")

packed, norms = tq.encode(kv_vectors)   # [..., 128] float16 -> [..., 64] uint8 + [...] f32
decoded = tq.decode(packed, norms)       # [..., 64] uint8 + [...] f32 -> [..., 128] float16
```

## Quick Start — TriAttention (v2.0+)

```python
from aither_kvcache.triattention import TriAttention, TriAttentionConfig

# Configure: 12 frequency pairs, 4-bit coefficients → ~10× compression
config = TriAttentionConfig(
    head_dim=128, num_freqs=12, coeff_bits=4,
    num_kv_heads=8, num_query_heads=32,
    rope_base=1_000_000.0,       # Qwen3.5 RoPE base
)
tri = TriAttention(config, device="cuda")

# Encode K/V to spectral representation (pre-RoPE keys)
k_enc, v_enc = tri.encode_kv(keys, values)
# keys: [B, S, num_kv_heads, head_dim] → spectral: 26 bytes/token vs 256 FP16

# Decode: score via trig series, accumulate values
output = tri.decode_step(query, k_enc, v_enc, query_pos, key_positions)
```

### Qwen3.5 calibration profiles

```python
from aither_kvcache.triattention.calibration import get_config_for_model

config = get_config_for_model("Qwen3.5-8B", coeff_bits=4)
# Per-layer frequency schedule: early layers get more frequencies,
# middle layers are most spectrally concentrated.
print(config.summary())
# TriAttention Config (qwen3.5)
#   head_dim=128, num_freqs=12, coeff_bits=4
#   heads: 32Q / 8KV (GQA 4:1)
#   storage: 26 bytes/token (vs 256 FP16)
#   compression: 9.8× per K/V tensor
```

### How it works

Transformer attention with RoPE is naturally a trigonometric series in position difference Δ = n − m:

```
score(q, k, m, n) = (1/√d) Σᵢ [cᵢ cos(Δθᵢ) + sᵢ sin(Δθᵢ)]
```

where `cᵢ = q₂ᵢk₂ᵢ + q₂ᵢ₊₁k₂ᵢ₊₁` (pair dot product) and `θᵢ = base^{-2i/d}` (RoPE frequency).

Most key energy concentrates in a few frequency pairs. By retaining only the top-F pairs (by
energy E_i = k₂ᵢ² + k₂ᵢ₊₁²) and quantizing coefficients to 4-bit, we store 26 bytes per token
instead of 256 — a ~10× reduction with bounded approximation error.

### TriAttention compression ratios

| Mode | Coeff Bits | Bytes/Token | Compression vs FP16 |
|------|-----------|-------------|---------------------|
| F=12, int4 | 4 | 26 | **9.85×** |
| F=12, int8 | 8 | 38 | 6.74× |
| F=16, int4 | 4 | 34 | 7.53× |
| F=8, int4 | 4 | 18 | **14.2×** |

## vLLM Integration

### Native kv-cache-dtype (recommended)

With the [TQ-patched vLLM fork](https://github.com/Aitherium/aitherkvcache), TurboQuant is a **native KV cache dtype**.
One flag, no hooks, no env vars:

```bash
vllm serve your-model --kv-cache-dtype tq-t4nc
```

Supported dtypes:

| Dtype | Bits | Compression | Notes |
|-------|------|------------|-------|
| `tq-t4nc` | 4 | 3.8x | **Recommended** — best quality/compression tradeoff |
| `tq-t3nc` | 3 | 4.9x | More aggressive |
| `tq-t35nc` | 3.5 | 4.4x | Hybrid split-group quantization |
| `tq-t2nc` | 2 | 7.1x | Maximum compression |

Production Docker example:

```yaml
services:
  vllm:
    image: aither-vllm-tq:latest
    command: >
      python -m vllm.entrypoints.openai.api_server
        --model your-model
        --kv-cache-dtype tq-t4nc
        --gpu-memory-utilization 0.85
        --max-model-len 40960
        --compilation-config '{"cudagraph_mode":"piecewise","max_cudagraph_capture_size":16}'
        --max-num-seqs 16
        --enable-prefix-caching
```

**Boundary layer protection**: First/last 2 layers auto-use FlashAttn (`auto`)
to preserve embedding quality and output mixing.

**Production results** (RTX 5090, Nemotron-8B-AWQ, `tq-t4nc`):

| Metric | Value |
|--------|-------|
| KV cache | **250,928 tokens** (19.1 GiB) |
| Max concurrency (40K ctx) | 6.1x simultaneous |
| CUDA graphs | 10 piecewise, 6s capture |
| Model load | 6.4 GiB, 15s |

### Plugin mode (stock vLLM, v0.15+)

For stock vLLM without the TQ fork:

```bash
pip install aither-kvcache[vllm]
vllm serve your-model --attention-backend CUSTOM
```

### Hook-based integration (legacy, v0.9.1+)

For older vLLM or when you need monkey-patching:

```python
import os
os.environ["AITHER_TQ_MODE"] = "tq4-primary"
os.environ["AITHER_TQ_BITS"] = "4"

from aither_kvcache.vllm import apply_tq_patches, apply_tq_hooks
apply_tq_patches(bits=4)   # BEFORE vLLM starts

from vllm import LLM
llm = LLM(model="your-model", gpu_memory_utilization=0.90)
apply_tq_hooks()            # AFTER model loads
```

### Hybrid modes (tq35/tq25)

Split-group quantization with QJL residual encoding:

```python
from aither_kvcache import HybridTurboQuant
htq = HybridTurboQuant(head_dim=128, mode="tq35", device="cuda")
htq.calibrate_uniform()
packed = htq.encode(kv_vectors)
decoded = htq.decode(packed)
```

| Mode | Avg Bits | Strategy |
|------|----------|----------|
| tq35 | 3.5 | 50% dims @ 4-bit + 50% @ 3-bit (MSE + QJL) |
| tq25 | 2.5 | 25% dims @ 3-bit + 75% @ 2-bit (MSE + QJL) |

### Zero graph breaks (v1.0+)

All modes register `torch.library.custom_op` for zero-graph-break decode.
Requires PyTorch 2.4+. Falls back to `@torch.compiler.disable` on older versions.

## Where This Fits

### Custom inference loop

If you manage your own KV cache, drop `encode()` where you write and `decode()` where you read:

```python
from aither_kvcache import TurboQuant

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
from aither_kvcache.fused_attention import TQPagedAttention

attn = TQPagedAttention(tq, num_query_heads=32)
output = attn.forward(
    query, k_packed, k_norms, v_packed, v_norms,
    block_tables, context_lens,
)
```

The math: rotate the query forward once, dot-product in the rotated domain against
codebook-decoded values, accumulate weighted values in the rotated domain, rotate back
once. Two matrix multiplies total regardless of context length.

Uses fused Triton kernels on GPU (Ampere through Blackwell). Falls back to PyTorch reference on CPU.
Set `AITHER_TQ_FORCE_TRITON=1` on Blackwell (SM_120) GPUs -- validated on RTX 5090 at 26 tok/s.

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

class HybridTurboQuant:
    def __init__(self, head_dim=128, mode="tq35", seed=42, device="cuda")
    def calibrate_uniform(self, num_kv_heads=1)
    def calibrate(self, sample_vectors: Tensor)
    def encode(self, x: Tensor) -> Tensor           # packed only (norms embedded)
    def decode(self, packed: Tensor) -> Tensor
    def validate(self, num_vectors=10000) -> dict
    def compression_ratio(self) -> float

    @staticmethod
    def packed_dim_for_mode(head_dim: int, mode: str) -> int

class TQPagedAttention:
    def __init__(self, tq: TurboQuant, num_query_heads: int)
    def forward(self, query, k_packed, k_norms, v_packed, v_norms,
                block_tables, context_lens, block_size=16) -> Tensor
```

## Graph-Aware KV Cache Eviction (v1.1+)

Standard KV cache eviction (LRU/FIFO) doesn't know the difference between your system
prompt and throwaway generation tokens. `KVCacheGraph` builds a relationship graph over
physical KV cache blocks so eviction decisions understand what the blocks actually mean.

### Quick Start

```python
from aither_kvcache import KVCacheGraph, GraphEvictionAdvisor, EdgeType

# 1. Create the graph — protect system prompt blocks from eviction
graph = KVCacheGraph(protected_sources={"system", "tools"})

# 2. Register blocks as they enter the KV cache
graph.add_block(0, "system", importance=0.95, token_range=(0, 16))
graph.add_block(1, "system", importance=0.90, token_range=(16, 32))
graph.add_block(2, "user",   importance=0.60, token_range=(32, 48))
graph.add_block(3, "assistant", importance=0.30, token_range=(48, 64))

# 3. Feed attention patterns — edges form automatically
graph.on_attention_step([0, 1, 2, 3])   # track co-attendance
graph.on_temporal_sequence([2, 3])       # sequential generation
graph.on_prefix_hit("req_42", [0, 1])   # prefix cache reuse

# 4. Ask who to evict (system blocks are structurally protected)
victims = graph.suggest_eviction(n_blocks=2)
# -> returns least-connected, lowest-importance, non-protected blocks

# 5. Ask what to prefetch from cold tier
graph.on_spill([3])  # block 3 moved to DDR5
prefetch = graph.suggest_prefetch(active_block_idxs=[0, 1, 2])
# -> returns spilled blocks that are graph-neighbors of active set
```

### Background Advisor (zero decode-path overhead)

For hot inference loops where you can't afford graph queries on the decode path:

```python
from aither_kvcache import GraphEvictionAdvisor

advisor = GraphEvictionAdvisor(graph, interval=0.5, max_stale=2.0)
advisor.start()  # background thread recomputes rankings every 0.5s

# Hot decode path — lock-free, zero overhead:
candidates = advisor.get_eviction_candidates(n=16)   # pre-computed list or None
prefetch = advisor.get_prefetch_candidates([0, 1], n=8)  # graph neighbor lookup

advisor.stop()
```

The advisor pre-computes eviction rankings on a background thread. The decode path reads
an atomically-swapped reference — no lock, no mutex, no blocking. If the ranking goes
stale (>2s), returns None and the caller falls back to FIFO.

### How Eviction Scoring Works

The `suggest_eviction()` method scores every non-protected, non-spilled block:

```
score = age × 0.01          # older = more evictable
      − degree × 5.0        # more graph connections = keep
      − edge_weight × 2.0   # stronger edges = keep
      − importance × 20.0   # higher importance = keep
      − hit_count × 3.0     # more prefix cache hits = keep
```

Protected source labels are excluded entirely — they cannot be eviction candidates.

### Six Edge Types

| Edge Type | Created By | Meaning |
|-----------|-----------|---------|
| `PREFIX_SHARE` | `on_prefix_hit()` | Blocks reused across requests |
| `CO_ATTEND` | `on_attention_step()` | Blocks frequently attended together |
| `SEMANTIC` | `add_block(embedding=...)` | Similar key vector embeddings (cosine > 0.8) |
| `TEMPORAL` | `on_temporal_sequence()` | Consecutive in same generation |
| `SPILL_LINK` | `on_spill()` / `on_warm()` | Hot ↔ cold tier tracking |

### Integration with Any Inference Engine

The graph has no vLLM dependency. It works with any paged KV cache:

1. Call `add_block()` when blocks are allocated
2. Call `remove_block()` when blocks are freed
3. Call `on_attention_step()` with active block indices each decode step
4. Call `suggest_eviction()` when you need to free VRAM
5. Call `suggest_prefetch()` to warm cold-tier blocks preemptively

### API

```python
class KVCacheGraph:
    def __init__(self, protected_sources={"system"}, coattend_threshold=3,
                 semantic_threshold=0.8)
    def add_block(self, block_idx, source_label, importance, token_range,
                  embedding=None) -> KVBlockNode
    def remove_block(self, block_idx) -> None
    def add_edge(self, source, target, edge_type, weight=1.0) -> Optional[KVEdge]
    def on_attention_step(self, active_block_idxs: List[int]) -> None
    def on_prefix_hit(self, request_id: str, block_idxs: List[int]) -> None
    def on_spill(self, block_idxs: List[int]) -> None
    def on_warm(self, block_idxs: List[int]) -> None
    def on_temporal_sequence(self, block_idxs: List[int]) -> None
    def suggest_eviction(self, n_blocks, protect_sources=None) -> List[int]
    def suggest_prefetch(self, active_block_idxs, max_suggestions=16) -> List[int]
    def neighbors(self, block_idx, edge_type=None, max_depth=1) -> Set[int]
    def subgraph(self, block_idxs) -> Dict
    def get_stats(self) -> Dict

class GraphEvictionAdvisor:
    def __init__(self, graph=None, interval=0.5, max_stale=2.0, eviction_batch=256)
    def start(self) -> None
    def stop(self) -> None
    def get_eviction_candidates(self, n: int) -> Optional[List[int]]
    def get_prefetch_candidates(self, active_block_idxs, n=8) -> Optional[List[int]]
    def get_stats(self) -> Dict

def reorder_by_ranking(block_indices: List[int], ranked: List[int]) -> List[int]
```

## Benchmarks

### KV Cache Memory by Model (at 32K context)

| Model | Layers | KV Heads | FP16 | FP8 | TQ4 (4-bit) | TQ3 (3-bit) | TQ2 (2-bit) |
|-------|--------|----------|------|-----|-------------|-------------|-------------|
| Llama 3.1 8B | 32 | 8 | 4.0 GB | 2.0 GB | **1.1 GB** | 0.8 GB | 0.6 GB |
| Mistral 7B v0.3 | 32 | 8 | 4.0 GB | 2.0 GB | **1.1 GB** | 0.8 GB | 0.6 GB |
| Qwen2.5 14B | 40 | 8 | 5.0 GB | 2.5 GB | **1.3 GB** | 1.0 GB | 0.7 GB |
| Llama 3.1 70B | 80 | 8 | 10.0 GB | 5.0 GB | **2.7 GB** | 2.0 GB | 1.4 GB |
| Qwen2.5 72B | 80 | 8 | 10.0 GB | 5.0 GB | **2.7 GB** | 2.0 GB | 1.4 GB |

### KV Cache Memory by Context Length (Llama 3.1 8B)

| Context | FP16 | FP8 | TQ4 | TQ3 | TQ2 |
|---------|------|-----|-----|-----|-----|
| 8K | 1.0 GB | 512 MB | **272 MB** | 208 MB | 144 MB |
| 32K | 4.0 GB | 2.0 GB | **1.1 GB** | 0.8 GB | 0.6 GB |
| 128K | 16.0 GB | 8.0 GB | **4.3 GB** | 3.3 GB | 2.3 GB |

### Decode Throughput (RTX 5090, Llama 3.1 8B)

| Integration | Single Request | 5x Concurrent | CUDA Graphs |
|-------------|---------------|---------------|-------------|
| Hook mode (recommended) | **40 tok/s** | 120 tok/s | 7/7 captured |
| Plugin mode (CUSTOM backend) | 23.6 tok/s | 120 tok/s | 7/7 captured |
| Baseline (FP8, no TQ) | 45 tok/s | 130 tok/s | 7/7 captured |

Hook mode reaches ~89% of baseline FP8 throughput while storing 3.8x more KV cache blocks.

### Max Context Window (RTX 5090 32GB, single model)

Shows maximum tokens that fit in KV cache VRAM after model weights.

| Model | Weights | FP8 | TQ4 | TQ3 | TQ2 |
|-------|---------|-----|-----|-----|-----|
| Llama 3.1 8B (util=0.90) | ~5 GB | 353K | **665K** | 869K | 1.26M |
| Qwen2.5 14B (util=0.90) | ~9 GB | 247K | **466K** | 609K | 880K |
| Llama 3.1 70B (util=0.90) | ~37 GB | N/A | N/A | N/A | N/A |

70B requires multi-GPU or offloading — KV savings still apply per-GPU.

### Quantization Quality (head_dim=128, 50K vectors)

| Mode | Avg Bits | MSE | Compression vs FP16 | Compression vs FP8 |
|------|----------|-----|---------------------|---------------------|
| TQ4 | 4.0 | 0.0095 | 3.8x | 1.9x |
| tq35 | 3.5 | 0.0130 | 4.4x | 2.2x |
| TQ3 | 3.0 | 0.0345 | 4.9x | 2.5x |
| tq25 | 2.5 | 0.0520 | 5.8x | 2.9x |
| TQ2 | 2.0 | 0.1175 | 7.1x | 3.6x |

All MSE values within 2.7x of the information-theoretic lower bound (matches paper).

Run `python -m aither_kvcache.bench` to reproduce on your hardware.

## Quickstart Notebook

See [`notebooks/vllm_quickstart.ipynb`](notebooks/vllm_quickstart.ipynb) for a step-by-step
walkthrough covering installation, validation, vLLM integration, and graph-aware eviction.

## Reference

```bibtex
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}
```

## Community

- [GitHub Discussions](https://github.com/Aitherium/aitherkvcache/discussions) — Questions, ideas, show & tell
- [GitHub Issues](https://github.com/Aitherium/aitherkvcache/issues) — Bug reports and feature requests

## License

CC BY 4.0

## vLLM Integration (upstream)

Native integration via [TQ-patched fork](https://github.com/Aitherium/aitherkvcache).
Upstream PR: [vllm-project/vllm#39008](https://github.com/vllm-project/vllm/pull/39008).

```bash
# TQ fork (works now):
vllm serve your-model --kv-cache-dtype tq-t4nc

# Stock vLLM (plugin mode):
pip install aither-kvcache[vllm]
vllm serve your-model --attention-backend CUSTOM
```
