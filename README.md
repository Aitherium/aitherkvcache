# aither-kvcache

Near-optimal KV cache compression for LLM inference. Two compression engines:

- **TurboQuant** — Vector quantization ([Zandieh et al., arXiv:2504.19874](https://arxiv.org/abs/2504.19874)).
  2-4 bit, 3.8-7.1× compression vs FP16. No calibration data. Works on streaming tokens.

- **TriAttention** *(v2.0)* — Spectral KV compression via trigonometric series.
  Retains top RoPE frequency pairs, scores via trig series without materializing full K/V.
  14–26× compression at F=8–16. **Calibration-dependent — read this before using it:**
  which frequency pairs carry the energy is a property of the model, and off-profile the
  ranking degrades while reconstruction still looks fine. Measured on an *uncalibrated*
  model at the F=12 default: cosine 0.91, but mean top-32 attention overlap **0.41** over
  64 query directions. Profiles ship for Qwen3.5, Nemotron, DeepSeek-R1 and Llama 3.1;
  anything else now emits a `RuntimeWarning` instead of falling back silently.

- **KVTransfer** *(NEW in v2.4)* — Cross-model KV transfer. Convert one model's KV cache
  into another model's so the receiving model **skips prefill entirely**. Per-(layer, head)
  linear maps fitted by ridge; source layers chosen per target layer by held-out R².
  Same model family only — the two models must share a tokenizer.

## Installation

```bash
pip install aither-kvcache            # core library
pip install aither-kvcache[vllm]      # + vLLM plugin (v0.15+)
pip install aither-kvcache[triton]    # + fused GPU kernels
pip install aither-kvcache[transfer]  # + cross-model KV transfer
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

## Quick Start — KVTransfer (v2.4+)

Prefill is the tax you pay before a model says anything, and a KV cache only works on the
model that produced it. Route a conversation to a different model and the accumulated
cache becomes dead weight. KVTransfer fits a map that converts one model's cache into
another's, so the receiving model can skip prefill.

```bash
# 1. capture aligned KV from both models over one corpus
python -m aither_kvcache.kvtransfer.capture \
    --source Qwen/Qwen3-0.6B --target Qwen/Qwen3-4B \
    --out ./cap --corpus ./my-corpus --seq-len 256 --train-seqs 64 --val-seqs 24

# 2. fit the mapper pack
python -m aither_kvcache.kvtransfer.fit --capture ./cap --out ./pack --top-k 8

# 3. measure what the RECEIVING model does with a translated cache, and record it
python -m aither_kvcache.kvtransfer.transfer \
    --pack ./pack --source Qwen/Qwen3-0.6B --target Qwen/Qwen3-4B \
    --corpus ./held-out --sequences 24 --cuts 32 64 96 128 160 192 224 --record
```

```python
from aither_kvcache.kvtransfer import load_pack
from aither_kvcache.kvtransfer.transfer import translate_cache

pack = load_pack("./pack", live_source=src_geom, live_target=tgt_geom)
layers = translate_cache(src_k, src_v, pack, positions)   # -> the target's KV cache
```

### Step 3 is not optional

`load_pack` **refuses** any pack that carries no downstream acceptance evidence. That is
the core design decision here, and it is not defensive programming — it is the only thing
standing between an interesting R² and a broken deployment.

A converted cache does not fail loudly. An approximate one makes the receiving model
produce fluent, on-topic, **confidently wrong** text: no exception, no error status, no
unhealthy process, nothing in a log. Every cheap signal is green.

So a pack must carry a measurement, and the measurement has **four arms**:

| arm | what it is |
|---|---|
| `reference` | the target prefills the context itself — the ceiling |
| `translated` | the target is handed the converted cache — the candidate |
| `control` | the map is run on a **different document** — catches a mapper ignoring its input |
| `nocontext` | no cache at all — the floor |

The control arm is the one that matters. A mapper that has quietly learned the target's
*average* key/value statistics scores respectably against `reference` alone. Without a
floor, "68% agreement" is unreadable — it could be excellent, or exactly what a dead
mapper gets.

A minimum sample size is enforced too. Top-1 agreement is a proportion, so a mean over a
few dozen positions has a standard error near ten percentage points — and once recorded it
is indistinguishable from a mean over ten thousand.

### Preconditions that are definitional, not quality knobs

- **The two models must tokenize identically.** The map sends position *i* to position
  *i*; different tokenizers mean row *i* is not the same token and the fit is regressing
  misaligned data. Nothing downstream can see this — it reads as "this pair transfers
  poorly" rather than "these rows do not correspond". Enforced by vocabulary digest, and
  by comparing token ids per document at capture time.
- **The RoPE schedule must be exactly reproducible.** Keys carry a position rotation, so
  the map is fitted in position-free space: stripped with the *source* schedule, re-applied
  with the *target's* (not a no-op — the two models often disagree on `rope_theta`).
  `default`, `linear` and `yarn` are implemented; anything else is refused rather than
  approximated.

`check_pair()` separates definitional refusals from merely-unproven regimes — mismatched
KV-head counts, differing head dimension, cross-family pairs. The latter are recorded as
flags and decided by measurement, because a gate keyed on "heads must match" would refuse
the very experiment that tests whether heads need to match.

### Latent attention (MLA)

Models such as DeepSeek V2/V3/V4 cache a compressed latent plus a decoupled RoPE key
rather than per-head K and V. That is a different cache **layout**, not an obstacle — the
latent is dense, and often smaller than a conventional cache for the same context.
`KVGeometry.roles()` returns `("c", "kr")` for those models, and only `kr` carries a
rotation because the latent is position-free by construction.

The latent is captured at the compression projection rather than from `past_key_values`:
HuggingFace materialises per-head K/V before caching, so reading the cache would yield
tensors a real serving engine never stores, and a mapper fitted to them could not drive a
production deployment while looking entirely correct end to end.

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

<!-- aitherium-ecosystem:start -->
## Aitherium open-source ecosystem

This repo is one piece of a connected set. All public, MIT/BSL-licensed:

| repo | what it is | pages |
|---|---|---|
| [awrecover](https://github.com/Aitherium/awrecover) | Labelled snapshots with an all-or-nothing restore | [docs](https://aitherium.github.io/awrecover/) |
| [awshare](https://github.com/Aitherium/awshare) | Publish an artifact and fetch it back verified | [docs](https://aitherium.github.io/awshare/) |
| [awseal](https://github.com/Aitherium/awseal) | Sign an artifact so a stranger can verify it | [docs](https://aitherium.github.io/awseal/) |
| [awnode](https://github.com/Aitherium/awnode) | Lightweight local gateway — your apps to backends you chose | [docs](https://aitherium.github.io/awnode/) |
| [awnix](https://github.com/Aitherium/awnix) | A bootable, immutable Linux base for agent-run machines | [docs](https://aitherium.github.io/awnix/) |
| [awdk](https://github.com/Aitherium/awdk) | Build AI agent fleets — 3 lines, any backend | [docs](https://aitherium.github.io/awdk/) |
| [awskills](https://github.com/Aitherium/awskills) | Free agent skills, scripts & automations | [docs](https://aitherium.github.io/awskills/) |
| [AitherZero](https://github.com/Aitherium/AitherZero) | PowerShell 7+ automation framework | [docs](https://aitherium.github.io/AitherZero/) |
| [awgit](https://github.com/Aitherium/awgit) | Semantic version control on top of git | [docs](https://aitherium.github.io/awgit/) |
| [awgraph](https://github.com/Aitherium/awgraph) | Code knowledge graph for AI agents | [docs](https://aitherium.github.io/awgraph/) |
| [aitherkvcache](https://github.com/Aitherium/aitherkvcache) | Near-optimal KV cache quantization | [docs](https://aitherium.github.io/aitherkvcache/) |
| [awrelay](https://github.com/Aitherium/awrelay) | Agent-to-agent messaging over any chat server | [docs](https://aitherium.github.io/awrelay/) |
| [awm](https://github.com/Aitherium/awm) | A small world model (LeWM JEPA + MLP) to bootstrap your own | [docs](https://aitherium.github.io/awm/) |
| [AitherConnect](https://github.com/Aitherium/AitherConnect) | Browser extension: federated AI search & desktop bridge | — |
| [homebrew-tap](https://github.com/Aitherium/homebrew-tap) | `brew tap aitherium/tap` | — |

Built by [Aitherium](https://aitherium.com).
<!-- aitherium-ecosystem:end -->

<!-- aither-ecosystem:start GENERATED from the ecosystem registry. Edits here are overwritten; change the registry instead. -->

## The aw family

Standalone tools that share one idea: **replace something you would otherwise have to _trust_ with something you can _check_.**

Each installs on its own, works offline, and needs no account.

| | instead of trusting | you check |
|---|---|---|
| [awdk](https://github.com/Aitherium/awdk) | a framework's idea of how your agents should run | one loop you can read, pointed at a backend you already pay for |
| [awskills](https://github.com/Aitherium/awskills) | that an agent knows your procedure | the procedure written down, versioned, and loadable by any agent |
| [awm](https://github.com/Aitherium/awm) | that memory stayed in its lane | tenant:user:project scopes, so a write cannot cross a boundary |
| [awnode](https://github.com/Aitherium/awnode) | a vendor's cloud with every prompt | a local gateway routing to backends you chose |
| [awgraph](https://github.com/Aitherium/awgraph) | that grep found everything | an AST + tree-sitter call graph an agent can traverse |
| [awgit](https://github.com/Aitherium/awgit) | that no one else is editing this file | a lease, refused at commit time if you do not hold it |
| [awseal](https://github.com/Aitherium/awseal) | that the artifact came from who you think | an Ed25519 seal — the key that verifies is not the key that forges |
| [awshare](https://github.com/Aitherium/awshare) | that the download is intact | content-addressed bundles, verified on fetch |
| [awnest](https://github.com/Aitherium/awnest) | that there is a person on the other end | a verdict with evidence, where "we could not tell" is not "yes" |
| [awnboard](https://github.com/Aitherium/awnboard) | a share link anyone who sees it can use | an invitation addressed to one person, for one gate, revocable |
| [awnix](https://github.com/Aitherium/awnix) | that the box is what you left it as | an immutable image you built, with atomic rollback |
| [awrecover](https://github.com/Aitherium/awrecover) | that the restore worked | a restore that fully lands or does not land at all |
| [awrelay](https://github.com/Aitherium/awrelay) | a SaaS in the middle of your agents | findings, alerts and coordination over your own transport |
| [awmail](https://github.com/Aitherium/awmail) | a mailbox somebody else can read | mail your agents send and receive over your own server |
| [awfind](https://github.com/Aitherium/awfind) | one vendor's idea of the web | results from whichever providers you configured |
| [awbrowse](https://github.com/Aitherium/awbrowse) | that the page said what you were told | the render, the DOM and the requests it made |
| **aitherkvcache** _(you are here)_ | a vendor's quantisation defaults | sub-byte KV cache kernels you can benchmark yourself |
| [AitherZero](https://github.com/Aitherium/AitherZero) | a pile of scripts nobody has numbered | numbered, discoverable automation with declarative playbooks |
| [AitherConnect](https://github.com/Aitherium/AitherConnect) | what a page tells your browser to do | a federated search and desktop bridge you host |
| [awreason](https://github.com/Aitherium/awreason) | a confident paragraph | the phases it went through, and every tool call it made to get there |
| [awrecurse](https://github.com/Aitherium/awrecurse) | that everything you pasted in was actually read | which slices it opened, and what it concluded from each |
| [awprism](https://github.com/Aitherium/awprism) | the first explanation that fits | the ranked alternatives, and the observation that separates them |
| [awrepl](https://github.com/Aitherium/awrepl) | what the agent believes the value is | the value, printed from the live session |
| [awresearch](https://github.com/Aitherium/awresearch) | a summary of pages nobody opened | every claim against the source it came from |
| [awkno](https://github.com/Aitherium/awkno) | that the docs site is up, or that you remember the family | the whole ecosystem in your terminal, with no network at all |

[**awnix**](https://github.com/Aitherium/awnix) is the ground floor — A Linux you can hand to an agent — immutable base, capabilities included.

## The Aitherium ecosystem

Every repository here is public. Each publishes an `aither-manifest.json` beside its page, so any surface can read every sibling's — the network is browsable from any node in it.

| repo | what it is | pages |
|---|---|---|
| [awdk](https://github.com/Aitherium/awdk) | Build AI agent fleets — 3 lines, any backend, local or cloud | [docs](https://aitherium.github.io/awdk/) |
| [awskills](https://github.com/Aitherium/awskills) | Portable agent skills — self-contained procedures an agent loads on demand | [docs](https://aitherium.github.io/awskills/) |
| [awm](https://github.com/Aitherium/awm) | A portable, scoped agent memory | [docs](https://aitherium.github.io/awm/) |
| [awnode](https://github.com/Aitherium/awnode) | A lightweight local gateway — bridges your apps to the AI backends you chose | [docs](https://aitherium.github.io/awnode/) |
| [awrun](https://github.com/Aitherium/awrun) | A priority-aware queue and dispatcher for agentic runs and ad-hoc CI builds | [docs](https://aitherium.github.io/awrun/) |
| [awgraph](https://github.com/Aitherium/awgraph) | A semantic code graph for agents — AST + tree-sitter, call graphs | [docs](https://aitherium.github.io/awgraph/) |
| [awgit](https://github.com/Aitherium/awgit) | Semantic version control on top of git — edit-ops and leases | [docs](https://aitherium.github.io/awgit/) |
| [awseal](https://github.com/Aitherium/awseal) | Sign an artifact so a stranger can verify it | [docs](https://aitherium.github.io/awseal/) |
| [awshare](https://github.com/Aitherium/awshare) | Publish an artifact and fetch it back verified | [docs](https://aitherium.github.io/awshare/) |
| [awnest](https://github.com/Aitherium/awnest) | Prove there is a human before you let them into the nest | [docs](https://aitherium.github.io/awnest/) |
| [awnboard](https://github.com/Aitherium/awnboard) | A front gate you can put in front of anything, and hand someone the key to | [docs](https://aitherium.github.io/awnboard/) |
| [awnix](https://github.com/Aitherium/awnix) | A Linux you can hand to an agent — immutable base, capabilities included | [docs](https://aitherium.github.io/awnix/) |
| [awrecover](https://github.com/Aitherium/awrecover) | Labelled snapshots with an all-or-nothing restore | [docs](https://aitherium.github.io/awrecover/) |
| [awrelay](https://github.com/Aitherium/awrelay) | Portable agent messaging — findings, alerts, coordination | [docs](https://aitherium.github.io/awrelay/) |
| [awmail](https://github.com/Aitherium/awmail) | Give an agent an email address — send, and actually receive | [docs](https://aitherium.github.io/awmail/) |
| [awfind](https://github.com/Aitherium/awfind) | A portable search client — query, results, ranking | [docs](https://aitherium.github.io/awfind/) |
| [awbrowse](https://github.com/Aitherium/awbrowse) | A portable browser client — navigate, console, network, DOM, screenshot | [docs](https://aitherium.github.io/awbrowse/) |
| **aitherkvcache** _(you are here)_ | Near-optimal KV cache quantization for LLM inference — sub-byte compression | [docs](https://aitherium.github.io/aitherkvcache/) |
| [AitherZero](https://github.com/Aitherium/AitherZero) | PowerShell 7+ automation framework — numbered, self-describing scripts | [docs](https://aitherium.github.io/AitherZero/) |
| [AitherConnect](https://github.com/Aitherium/AitherConnect) | Browser extension — federated AI search, page context, and the Living OS overlay | [docs](https://aitherium.github.io/AitherConnect/) |
| [awreason](https://github.com/Aitherium/awreason) | A portable reasoning client — sessions, phases, thoughts, and the chain that produced the answer | [docs](https://aitherium.github.io/awreason/) |
| [awrecurse](https://github.com/Aitherium/awrecurse) | Answer a question over a context far larger than the window — recursively, with the trace kept | [docs](https://aitherium.github.io/awrecurse/) |
| [awprism](https://github.com/Aitherium/awprism) | Turn a failure into ranked hypotheses — and say what would confirm each one | [docs](https://aitherium.github.io/awprism/) |
| [awrepl](https://github.com/Aitherium/awrepl) | A REPL an agent can actually use — state that survives between turns | [docs](https://aitherium.github.io/awrepl/) |
| [awresearch](https://github.com/Aitherium/awresearch) | Ask a research question, get a cited report you can check | [docs](https://aitherium.github.io/awresearch/) |
| [awkno](https://github.com/Aitherium/awkno) | The man page for the Aither World — every brick, stack and law, offline | [docs](https://aitherium.github.io/awkno/) |

<!-- aither-ecosystem:end -->
