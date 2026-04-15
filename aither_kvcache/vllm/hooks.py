"""
TurboQuant Hook-Based Integration for vLLM v0.15.1+.

Monkey-patches TritonAttentionImpl.forward() to add TQ encode/decode
WITHOUT registering as a custom attention backend. This preserves
torch.compile + piecewise CUDA graphs (which break with custom backends
due to an Inductor tensor corruption bug at splitting op boundaries).

Architecture:
  - vLLM uses standard Triton attention backend (compiled, fast)
  - TQ hooks intercept forward() to encode/decode KV cache transparently
  - Decode: merged encode+fused attention in ONE graph break per layer
    (halves Python overhead vs separate encode + decode calls)
  - Prefill: decompress TQ blocks -> temp fp8 buffer -> standard attention
  - @torch.compiler.disable on encode/decompress creates clean graph breaks
  - The attention kernel itself remains compiled + CUDA-graphable

Apply hooks AFTER vLLM engine init (model loaded, layers instantiated):
    from aither_kvcache.vllm.hooks import apply_tq_hooks
    apply_tq_hooks()

Requires: turboquant.vllm.engine.apply_tq_patches() already applied (page_size, reshape).

Debug logging: set AITHER_TQ_DEBUG=1 to enable detailed per-step diagnostics.
Set AITHER_TQ_DEBUG_STEPS=N to log the first N forward calls per layer-0
(default 10). Logs go to stderr via the tq_dbg logger.
"""

import logging
import os
import sys
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger("turboquant.hooks")

# -- Mode detection -------------------------------------------------------
# Derive TQ mode at import time. Hooks support both uniform (tq2/3/4)
# and hybrid (tq35/tq25) quantizers. The main difference:
#   uniform: encode -> (packed, norms), separate norm tensors, fused Triton decode
#   hybrid:  encode -> packed (norms embedded), decompress+SDPA for decode

_TQ_MODE = os.environ.get("AITHER_TQ_MODE", "").replace("-primary", "")
_TQ_BITS = int(os.environ.get("AITHER_TQ_BITS", "4"))
_TQ_IS_HYBRID = _TQ_MODE in ("tq35", "tq25")

# -- Debug logger ----------------------------------------------------------
# Controlled by AITHER_TQ_DEBUG=1. Prints to stderr so it shows in
# `docker logs` without buffering issues.

_TQ_DEBUG = os.environ.get("AITHER_TQ_DEBUG", "0") == "1"
_TQ_DEBUG_STEPS = int(os.environ.get("AITHER_TQ_DEBUG_STEPS", "10"))


def _dbg(msg: str) -> None:
    """Print debug message to stderr if TQ debug is enabled."""
    if _TQ_DEBUG:
        print(f"[TQ] {msg}", file=sys.stderr, flush=True)


class _TQStats:
    """Accumulates encode/decode stats for debug logging."""
    __slots__ = (
        "encode_calls", "encode_tokens", "encode_blocks",
        "fwd_count", "decode_count",
    )

    def __init__(self):
        self.encode_calls = 0
        self.encode_tokens = 0
        self.encode_blocks: set = set()
        self.fwd_count = 0
        self.decode_count = 0


_stats = _TQStats()


# -- Module-level state ----------------------------------------------------

_hooks_applied = False
_original_forward = None  # Saved reference to TritonAttentionImpl.forward

# TQ state shared across all layers (class-level in the patched impl)
# Per-head-dim quantizers for heterogeneous models (e.g. Gemma 4: 256 + 512)
_tq_quantizers: Dict[int, object] = {}    # head_dim -> TurboQuant instance
_tq_packed_dims: Dict[int, int] = {}      # head_dim -> packed bytes per head
_tq_fused_attn: dict = {}   # layer_idx -> TQPagedAttention instance
_primary_k_norms: Optional[torch.Tensor] = None  # None for hybrid (norms embedded)
_primary_v_norms: Optional[torch.Tensor] = None
_layer_counter = 0
_num_layers = 0

# Prefill decompression buffer (shared across layers, allocated once)
_prefill_k_buf: Optional[torch.Tensor] = None
_prefill_v_buf: Optional[torch.Tensor] = None
_PREFILL_BUF_BLOCKS = 512

def _get_tq(head_size: int) -> Tuple[object, int]:
    """Get quantizer and packed_dim for a given head_size."""
    return _tq_quantizers[head_size], _tq_packed_dims[head_size]


def _any_tq() -> Optional[object]:
    """Get any initialized quantizer (for operations that don't care about head_dim)."""
    if _tq_quantizers:
        return next(iter(_tq_quantizers.values()))
    return None


# Custom op flag -- set by _register_custom_ops()
_USE_CUSTOM_OP = False


# -- Custom op: zero graph breaks, CUDA-graphable -------------------------

def _register_custom_ops():
    """Register TQ decode as a torch custom op (PyTorch 2.4+).
    Eliminates ALL graph breaks from decode -- enables CUDA graph capture."""
    global _USE_CUSTOM_OP

    if not hasattr(torch.library, "custom_op"):
        logger.info("TQ custom op: torch.library.custom_op not available "
                     "(need PyTorch 2.4+), using graph-break fallback")
        return

    try:
        @torch.library.custom_op("tq::decode_step",
                                  mutates_args=("kv_cache", "output",
                                                "k_norms", "v_norms"))
        def _decode_step_op(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            output: torch.Tensor,
            k_norms: torch.Tensor,
            v_norms: torch.Tensor,
            slot_mapping: torch.Tensor,
            block_table: torch.Tensor,
            seq_lens: torch.Tensor,
            layer_idx: int,
            num_heads: int,
            num_kv_heads: int,
            head_size: int,
            scale: float,
            num_actual: int,
        ) -> torch.Tensor:
            """Encode new token + fused TQ attention. No graph break.
            Accesses module globals (_tq_quantizers, _tq_fused_attn) in body --
            these are initialized during vLLM warmup before CUDA graph capture."""
            tq, packed_dim = _get_tq(head_size)
            key_cache = kv_cache[:, 0]
            value_cache = kv_cache[:, 1]
            block_size = key_cache.shape[1]

            # Encode -- branchless, no CPU-GPU sync
            bi = slot_mapping // block_size
            oi = slot_mapping % block_size
            N, H, D = key.shape

            kp, kn = tq.encode(key.reshape(N * H, D))
            vp, vn = tq.encode(value.reshape(N * H, D))

            kp = kp.reshape(N, H, packed_dim)
            vp = vp.reshape(N, H, packed_dim)

            key_cache[bi, oi, :, :packed_dim] = kp
            value_cache[bi, oi, :, :packed_dim] = vp

            k_norms[layer_idx, bi, oi] = kn.reshape(N, H).to(torch.float32)
            v_norms[layer_idx, bi, oi] = vn.reshape(N, H).to(torch.float32)

            _stats.encode_calls += 1
            _stats.encode_tokens += N

            # Fused attention (lazy-init persists across CUDA graph replay)
            fused = _tq_fused_attn.get(layer_idx)
            if fused is None:
                from ..fused_attention import TQPagedAttention
                fused = TQPagedAttention(tq, num_heads)
                _tq_fused_attn[layer_idx] = fused

            fused_out = fused.forward(
                query=query[:num_actual],
                k_packed=key_cache[:, :, :, :packed_dim],
                k_norms=k_norms[layer_idx],
                v_packed=value_cache[:, :, :, :packed_dim],
                v_norms=v_norms[layer_idx],
                block_tables=block_table,
                context_lens=seq_lens,
            )
            output[:num_actual] = fused_out.to(output.dtype)
            return output.clone()

        @_decode_step_op.register_fake
        def _decode_step_fake(
            query, key, value, kv_cache, output, k_norms, v_norms,
            slot_mapping, block_table, seq_lens,
            layer_idx, num_heads, num_kv_heads, head_size, scale,
            num_actual,
        ):
            """Shape inference for torch.compile tracing."""
            return output.clone()

        _USE_CUSTOM_OP = True
        logger.info("TQ custom op registered: tq::decode_step "
                     "(zero graph breaks, CUDA-graphable)")
    except Exception as e:
        logger.warning("TQ custom op registration failed: %s -- "
                       "using graph-break fallback", e)
        _USE_CUSTOM_OP = False


# -- Hybrid custom op: zero graph breaks for tq35/tq25 --------------------
# Strategy: clamp-gather decompress -> masked batched SDPA.
# All fixed-size tensor ops. Fully CUDA-graphable.

_USE_HYBRID_CUSTOM_OP = False

# Pre-allocated decompress buffers (avoid per-step VRAM allocation)
_hybrid_dk_buf: Optional[torch.Tensor] = None
_hybrid_dv_buf: Optional[torch.Tensor] = None


def _register_hybrid_custom_ops():
    """Register hybrid TQ decode as a torch custom op (PyTorch 2.4+).
    Zero graph breaks -- CUDA-graphable hybrid decode."""
    global _USE_HYBRID_CUSTOM_OP

    if not hasattr(torch.library, "custom_op"):
        logger.info("TQ hybrid custom op: torch.library.custom_op not available "
                     "(need PyTorch 2.4+), using graph-break fallback")
        return

    try:
        import torch.nn.functional as F

        @torch.library.custom_op("tq::hybrid_decode_step",
                                  mutates_args=("kv_cache", "output"))
        def _hybrid_decode_step_op(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            output: torch.Tensor,
            slot_mapping: torch.Tensor,
            block_table: torch.Tensor,
            seq_lens: torch.Tensor,
            layer_idx: int,
            num_heads: int,
            num_kv_heads: int,
            head_size: int,
            scale: float,
            num_actual: int,
        ) -> torch.Tensor:
            """Hybrid TQ: encode + clamp-gather decompress + masked SDPA.
            Zero graph breaks. All fixed-size tensor ops.

            Architecture:
              1. Encode new K/V token into TQ cache (branchless index math)
              2. Clamp block_table indices to [0, max) -- padding reads block 0
              3. Batch decompress ALL referenced blocks in one tq.decode() call
              4. Reshape to [num_seqs, max_ctx, kv_heads, head_dim]
              5. Build attention mask from seq_lens (arange < seq_lens)
              6. GQA expand + batched SDPA
              7. Write output

            No Python loops. No .item() / .any() CPU-GPU syncs.
            CUDA graph captures the exact kernel launch sequence.
            """
            global _hybrid_dk_buf, _hybrid_dv_buf

            tq, packed_dim = _get_tq(head_size)
            key_cache = kv_cache[:, 0]
            value_cache = kv_cache[:, 1]
            block_size = key_cache.shape[1]

            # -- 1. Encode new K/V into TQ cache (branchless) ------
            bi = slot_mapping // block_size
            oi = slot_mapping % block_size
            N, H, D = key.shape

            kp = tq.encode(key.reshape(N * H, D)).reshape(N, H, packed_dim)
            vp = tq.encode(value.reshape(N * H, D)).reshape(N, H, packed_dim)

            key_cache[bi, oi, :, :packed_dim] = kp
            value_cache[bi, oi, :, :packed_dim] = vp

            _stats.encode_calls += 1
            _stats.encode_tokens += N

            # -- 2. Clamp block indices (padding -> block 0, masked out) --
            num_seqs = block_table.shape[0]
            max_blocks_per_seq = block_table.shape[1]
            max_ctx = max_blocks_per_seq * block_size
            num_cache_blocks = key_cache.shape[0]

            bt_clamped = block_table.clamp(min=0, max=num_cache_blocks - 1)
            flat_bt = bt_clamped.reshape(-1)  # [num_seqs * max_blocks]

            # -- 3. Batch decompress (one call, no loop) -----------
            kp_gather = key_cache[flat_bt, :, :, :packed_dim]
            vp_gather = value_cache[flat_bt, :, :, :packed_dim]
            # Shape: [S*max_blocks, block_size, kv_heads, packed_dim]

            total_slots = flat_bt.shape[0] * block_size * num_kv_heads
            dk = tq.decode(kp_gather.reshape(total_slots, packed_dim))
            dv = tq.decode(vp_gather.reshape(total_slots, packed_dim))
            # Shape: [total_slots, head_dim]

            dk = dk.reshape(num_seqs, max_ctx, num_kv_heads, head_size)
            dv = dv.reshape(num_seqs, max_ctx, num_kv_heads, head_size)

            # -- 4. Attention mask from seq_lens -------------------
            positions = torch.arange(max_ctx, device=query.device, dtype=seq_lens.dtype)
            # [max_ctx] < [num_seqs, 1] -> [num_seqs, max_ctx]
            valid_mask = positions.unsqueeze(0) < seq_lens[:num_seqs].unsqueeze(1)
            # Expand to [num_seqs, 1, 1, max_ctx] for SDPA broadcast
            attn_mask = torch.where(
                valid_mask.unsqueeze(1).unsqueeze(1),
                torch.zeros(1, dtype=torch.bfloat16, device=query.device),
                torch.tensor(float('-inf'), dtype=torch.bfloat16, device=query.device),
            )

            # -- 5. SDPA (batched, no loop) ------------------------
            q = query[:num_actual].to(torch.bfloat16)
            # [num_seqs, 1, num_heads, head_dim] -> [num_seqs, num_heads, 1, head_dim]
            q = q.unsqueeze(1).transpose(1, 2)

            k = dk.to(torch.bfloat16).transpose(1, 2)  # [S, kv_H, max_ctx, D]
            v = dv.to(torch.bfloat16).transpose(1, 2)

            # GQA expansion
            gqa_ratio = num_heads // num_kv_heads
            if gqa_ratio > 1:
                k = k.repeat_interleave(gqa_ratio, dim=1)
                v = v.repeat_interleave(gqa_ratio, dim=1)

            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, scale=scale)
            # [S, H, 1, D] -> [S, H, D]
            output[:num_actual] = attn_out.squeeze(2).to(output.dtype)
            return output.clone()

        @_hybrid_decode_step_op.register_fake
        def _hybrid_decode_step_fake(
            query, key, value, kv_cache, output,
            slot_mapping, block_table, seq_lens,
            layer_idx, num_heads, num_kv_heads, head_size, scale,
            num_actual,
        ):
            """Shape inference for torch.compile tracing."""
            return output.clone()

        _USE_HYBRID_CUSTOM_OP = True
        logger.info("TQ hybrid custom op registered: tq::hybrid_decode_step "
                     "(zero graph breaks, CUDA-graphable decompress+SDPA)")
    except Exception as e:
        logger.warning("TQ hybrid custom op registration failed: %s -- "
                       "using graph-break fallback", e)
        _USE_HYBRID_CUSTOM_OP = False


# -- Helpers (run eagerly, create graph breaks) ----------------------------

@torch.compiler.disable
def _tq_init_layer(self_impl):
    """Assign TQ layer index to a TritonAttentionImpl instance. Called once."""
    global _layer_counter
    if hasattr(self_impl, '_tq_layer_idx'):
        return  # Already initialized
    self_impl._tq_layer_idx = _layer_counter
    _layer_counter += 1
    logger.info("TQ hook L%d: heads=%d, kv=%d, dim=%d",
                self_impl._tq_layer_idx, self_impl.num_heads,
                self_impl.num_kv_heads, self_impl.head_size)


@torch.compiler.disable
def _ensure_quantizer(device, head_size, num_heads, num_kv_heads):
    """Lazy-init TQ quantizer for a given head_size. Creates/caches per unique head_dim.
    Supports heterogeneous head dimensions (e.g. Gemma 4: 256 local + 512 global)."""
    global _num_layers
    if head_size in _tq_quantizers:
        return  # Already initialized for this head_dim

    if not _tq_quantizers:  # First quantizer -- snapshot layer count
        _num_layers = _layer_counter

    if _TQ_IS_HYBRID:
        from ..hybrid_quantizer import HybridTurboQuant
        tq = HybridTurboQuant(
            head_dim=head_size, mode=_TQ_MODE, device=str(device),
            num_kv_heads=num_kv_heads)
        tq.calibrate_uniform()
        pd = tq.layout.packed_dim
        _tq_quantizers[head_size] = tq
        _tq_packed_dims[head_size] = pd
        logger.info("TQ hybrid quantizer initialized: head_dim=%d, mode=%s, "
                    "packed_dim=%d, %d layers (total quantizers: %d)",
                    head_size, _TQ_MODE, pd, _num_layers, len(_tq_quantizers))
    else:
        from ..quantizer import TurboQuant
        from ..packing import packed_size
        tq = TurboQuant(
            head_dim=head_size, bits=_TQ_BITS, device=str(device))
        pd = packed_size(head_size, _TQ_BITS)
        _tq_quantizers[head_size] = tq
        _tq_packed_dims[head_size] = pd
        logger.info("TQ quantizer initialized: head_dim=%d, bits=%d, "
                    "packed_dim=%d, %d layers (total quantizers: %d)",
                    head_size, _TQ_BITS, pd, _num_layers, len(_tq_quantizers))


@torch.compiler.disable
def _ensure_norms(num_layers, num_blocks, block_size, num_kv_heads, device):
    """Lazy-init per-layer float32 norm tensors (uniform mode only).
    Hybrid mode embeds norms in packed data -- no separate tensors needed."""
    global _primary_k_norms, _primary_v_norms
    if _TQ_IS_HYBRID:
        return  # Norms embedded in packed data
    if _primary_k_norms is not None:
        return
    _primary_k_norms = torch.zeros(
        num_layers, num_blocks, block_size, num_kv_heads,
        dtype=torch.float32, device=device)
    _primary_v_norms = torch.zeros_like(_primary_k_norms)
    logger.info("TQ norms: [%d layers, %d blocks, bs=%d, %d heads] (%.1f MB)",
                num_layers, num_blocks, block_size, num_kv_heads,
                num_layers * num_blocks * block_size * num_kv_heads * 4 * 2
                / (1024 ** 2))


@torch.compiler.disable
def _tq_encode_phase(layer_idx, key, value, kv_cache, slot_mapping):
    """Encode new K,V tokens into TQ primary cache. Runs eagerly.
    Supports both uniform (separate norms) and hybrid (norms embedded) modes."""
    global _primary_k_norms, _primary_v_norms
    head_size = key.shape[-1]
    if head_size not in _tq_quantizers:
        return
    tq, packed_dim = _get_tq(head_size)

    key_cache = kv_cache[:, 0]    # [blocks, block_size, kv_heads, tq_dim]
    value_cache = kv_cache[:, 1]
    block_size = key_cache.shape[1]

    # Guard: slot_mapping must align with key's token dimension
    if slot_mapping.shape[0] != key.shape[0]:
        _dbg(f"ENCODE GUARD REJECT L{layer_idx}: "
             f"slot_mapping={slot_mapping.shape[0]} != key={key.shape[0]}")
        return

    valid = slot_mapping >= 0
    if not valid.any():
        return

    vs = slot_mapping[valid]
    bi = vs // block_size
    oi = vs % block_size

    max_blocks = key_cache.shape[0]
    in_range = bi < max_blocks
    if not in_range.any():
        return
    bi = bi[in_range]
    oi = oi[in_range]

    vk = key[valid][in_range].contiguous()
    vv = value[valid][in_range].contiguous()
    N, H, D = vk.shape

    if _TQ_IS_HYBRID:
        # Hybrid: encode returns packed-only (norms embedded in packed data)
        kp = tq.encode(vk.reshape(N * H, D))
        vp = tq.encode(vv.reshape(N * H, D))

        kp = kp.reshape(N, H, packed_dim)
        vp = vp.reshape(N, H, packed_dim)

        key_cache[bi, oi, :, :packed_dim] = kp
        value_cache[bi, oi, :, :packed_dim] = vp
    else:
        # Uniform: encode returns (packed, norms) separately
        kp, kn = tq.encode(vk.reshape(N * H, D))
        vp, vn = tq.encode(vv.reshape(N * H, D))

        kp = kp.reshape(N, H, packed_dim)
        vp = vp.reshape(N, H, packed_dim)

        key_cache[bi, oi, :, :packed_dim] = kp
        value_cache[bi, oi, :, :packed_dim] = vp

        _primary_k_norms[layer_idx, bi, oi] = kn.reshape(N, H).to(torch.float32)
        _primary_v_norms[layer_idx, bi, oi] = vn.reshape(N, H).to(torch.float32)

    # VALIDATION: verify encode/decode roundtrip in cache
    if _TQ_DEBUG and _stats.encode_calls < 2 and not _TQ_IS_HYBRID:
        import torch.nn.functional as _F
        kp_back = key_cache[bi[:1], oi[:1], :, :packed_dim]
        kn_back = _primary_k_norms[layer_idx, bi[:1], oi[:1]]
        k_dec = tq.decode(kp_back.reshape(-1, packed_dim), kn_back.reshape(-1))
        k_orig = vk[:1].reshape(-1, D).float()
        cos = _F.cosine_similarity(k_orig, k_dec.float(), dim=-1).mean().item()
        _dbg(f"VALIDATE L{layer_idx}: encode/decode cosine={cos:.4f} "
             f"orig[0]={k_orig[0,:3].tolist()} dec[0]={k_dec[0,:3].float().tolist()}")

    # Stats
    _stats.encode_calls += 1
    _stats.encode_tokens += N
    _stats.encode_blocks.update(bi.tolist())

    if _TQ_DEBUG and _stats.encode_calls <= _TQ_DEBUG_STEPS:
        _dbg(f"ENCODE#{_stats.encode_calls} L{layer_idx}: N={N}, "
             f"mode={'hybrid' if _TQ_IS_HYBRID else 'uniform'}, "
             f"bi={bi[:min(4,len(bi))].tolist()}")


@torch.compiler.disable
def _tq_decompress_active(layer_idx, kv_cache, block_table, num_kv_heads,
                          head_size, device):
    """Decompress active TQ blocks into a temporary fp8/bf16 buffer.
    Returns a standard-shaped kv_cache tensor for TritonAttention.
    Runs eagerly."""
    global _primary_k_norms, _primary_v_norms
    global _prefill_k_buf, _prefill_v_buf

    if head_size not in _tq_quantizers:
        return kv_cache  # Fallback: return original (will fail but shouldn't happen)
    tq, packed_dim = _get_tq(head_size)

    key_cache = kv_cache[:, 0]
    value_cache = kv_cache[:, 1]
    num_blocks = key_cache.shape[0]
    block_size = key_cache.shape[1]

    # Allocate or reuse decompress buffers
    need_realloc = (
        _prefill_k_buf is None
        or _prefill_k_buf.shape[0] < num_blocks
        or _prefill_k_buf.device != device
    )
    if need_realloc:
        buf_blocks = max(num_blocks, _PREFILL_BUF_BLOCKS)
        _prefill_k_buf = torch.zeros(
            buf_blocks, block_size, num_kv_heads, head_size,
            dtype=torch.bfloat16, device=device)
        _prefill_v_buf = torch.zeros_like(_prefill_k_buf)

    k_buf = _prefill_k_buf[:num_blocks]
    v_buf = _prefill_v_buf[:num_blocks]

    # Find active blocks from block_table
    active_blocks = block_table.unique()
    active_blocks = active_blocks[active_blocks >= 0]
    active_blocks = active_blocks[active_blocks < num_blocks]

    if active_blocks.numel() > 0:
        A = active_blocks.shape[0]
        H = num_kv_heads

        kp_all = key_cache[active_blocks, :, :, :packed_dim]
        vp_all = value_cache[active_blocks, :, :, :packed_dim]

        flat_kp = kp_all.reshape(A * block_size * H, packed_dim)
        flat_vp = vp_all.reshape(A * block_size * H, packed_dim)

        if _TQ_IS_HYBRID:
            decoded_k = tq.decode(flat_kp)
            decoded_v = tq.decode(flat_vp)
        else:
            kn_all = _primary_k_norms[layer_idx, active_blocks]
            vn_all = _primary_v_norms[layer_idx, active_blocks]
            flat_kn = kn_all.reshape(A * block_size * H)
            flat_vn = vn_all.reshape(A * block_size * H)
            decoded_k = tq.decode(flat_kp, flat_kn)
            decoded_v = tq.decode(flat_vp, flat_vn)

        k_buf[active_blocks] = decoded_k.reshape(A, block_size, H, -1).to(k_buf.dtype)
        v_buf[active_blocks] = decoded_v.reshape(A, block_size, H, -1).to(v_buf.dtype)

    # Build standard vLLM cache layout: [num_blocks, 2, block_size, kv_heads, head_dim]
    fake_kv = torch.stack([k_buf, v_buf], dim=1)
    return fake_kv


@torch.compiler.disable
def _tq_fused_decode(layer_idx, query, kv_cache, attn_metadata, output,
                     num_heads, num_kv_heads, head_size, scale):
    """Fused TQ decode using TQPagedAttention kernel. No decompress needed.
    Runs eagerly (graph break)."""
    global _tq_fused_attn, _primary_k_norms, _primary_v_norms

    tq, packed_dim = _get_tq(head_size)

    # Lazy-init fused attention for this layer
    if layer_idx not in _tq_fused_attn:
        from ..fused_attention import TQPagedAttention
        _tq_fused_attn[layer_idx] = TQPagedAttention(tq, num_heads)

    fused = _tq_fused_attn[layer_idx]

    key_cache = kv_cache[:, 0]
    value_cache = kv_cache[:, 1]

    num_actual = attn_metadata.num_actual_tokens
    fused_out = fused.forward(
        query=query[:num_actual],
        k_packed=key_cache[:, :, :, :packed_dim],
        k_norms=_primary_k_norms[layer_idx],
        v_packed=value_cache[:, :, :, :packed_dim],
        v_norms=_primary_v_norms[layer_idx],
        block_tables=attn_metadata.block_table,
        context_lens=attn_metadata.seq_lens,
    )
    output[:num_actual] = fused_out.to(output.dtype)
    return output


# -- Inline helpers (no decorator -- called from within @torch.compiler.disable) --

def _encode_inline(layer_idx, key, value, kv_cache, slot_mapping):
    """Encode K/V into TQ cache. No decorator -- called from _tq_decode_step.
    Branchless for decode: no .any() CPU-GPU syncs. Works for any N tokens.
    Supports both uniform and hybrid modes."""
    tq, packed_dim = _get_tq(key.shape[-1])
    key_cache = kv_cache[:, 0]
    value_cache = kv_cache[:, 1]
    block_size = key_cache.shape[1]

    # Decode: slot_mapping is [N] with N usually 1, all valid.
    # Avoid .any() which triggers CPU-GPU sync. Direct index math.
    bi = slot_mapping // block_size
    oi = slot_mapping % block_size

    N, H, D = key.shape

    if _TQ_IS_HYBRID:
        kp = tq.encode(key.reshape(N * H, D))
        vp = tq.encode(value.reshape(N * H, D))

        kp = kp.reshape(N, H, packed_dim)
        vp = vp.reshape(N, H, packed_dim)

        key_cache[bi, oi, :, :packed_dim] = kp
        value_cache[bi, oi, :, :packed_dim] = vp
    else:
        kp, kn = tq.encode(key.reshape(N * H, D))
        vp, vn = tq.encode(value.reshape(N * H, D))

        kp = kp.reshape(N, H, packed_dim)
        vp = vp.reshape(N, H, packed_dim)

        key_cache[bi, oi, :, :packed_dim] = kp
        value_cache[bi, oi, :, :packed_dim] = vp

        _primary_k_norms[layer_idx, bi, oi] = kn.reshape(N, H).to(torch.float32)
        _primary_v_norms[layer_idx, bi, oi] = vn.reshape(N, H).to(torch.float32)

    _stats.encode_calls += 1
    _stats.encode_tokens += N


@torch.compiler.disable
def _tq_decode_step(layer_idx, query, key, value, kv_cache, attn_metadata,
                    output, num_heads, num_kv_heads, head_size, scale):
    """Single graph break per layer: encode + fused attention.
    Eliminates 3 extra graph breaks from init guards + removes CPU-GPU syncs."""
    # Encode new K/V into TQ cache (branchless for decode)
    if key is not None and value is not None:
        _encode_inline(layer_idx, key, value, kv_cache,
                       attn_metadata.slot_mapping)

    tq, packed_dim = _get_tq(head_size)

    # Fused attention -- dict lookup for cached TQPagedAttention
    fused = _tq_fused_attn.get(layer_idx)
    if fused is None:
        from ..fused_attention import TQPagedAttention
        fused = TQPagedAttention(tq, num_heads)
        _tq_fused_attn[layer_idx] = fused
    key_cache = kv_cache[:, 0]
    value_cache = kv_cache[:, 1]

    num_actual = attn_metadata.num_actual_tokens

    fused_out = fused.forward(
        query=query[:num_actual],
        k_packed=key_cache[:, :, :, :packed_dim],
        k_norms=_primary_k_norms[layer_idx],
        v_packed=value_cache[:, :, :, :packed_dim],
        v_norms=_primary_v_norms[layer_idx],
        block_tables=attn_metadata.block_table,
        context_lens=attn_metadata.seq_lens,
    )
    output[:num_actual] = fused_out.to(output.dtype)
    return output


# -- Prefill scatter-write (for writing raw K/V into decompress buffer) --

@torch.compiler.disable
def _prefill_scatter_write(key, value, k_cache, v_cache, slot_mapping, block_size):
    """Write raw K/V into decompressed buffer at correct slot positions."""
    n = min(slot_mapping.shape[0], key.shape[0])
    sm = slot_mapping[:n]
    valid_idx = (sm >= 0).nonzero(as_tuple=True)[0]
    if valid_idx.numel() == 0:
        return
    vs = sm[valid_idx]
    bi = vs // block_size
    oi = vs % block_size
    max_blocks = k_cache.shape[0]
    mask = bi < max_blocks
    if not mask.any():
        return
    bi = bi[mask]
    oi = oi[mask]
    src_idx = valid_idx[mask]
    k_cache[bi, oi] = key[src_idx].to(k_cache.dtype)
    v_cache[bi, oi] = value[src_idx].to(v_cache.dtype)


@torch.compiler.disable
def _tq_prefill_sdpa(layer_idx, query, key, value, kv_cache, attn_metadata,
                     output, num_heads, num_kv_heads, head_size, scale):
    """Prefill/decode using torch SDPA with raw K/V + decompressed prior context.
    Manual GQA expansion. is_causal=True for prefill, is_causal=False for decode
    (single query attends to all prior positions -- no mask needed)."""
    import torch.nn.functional as F
    global _primary_k_norms, _primary_v_norms

    if head_size not in _tq_quantizers:
        output.fill_(0)
        return output
    tq_inst, packed_dim = _get_tq(head_size)

    num_actual = attn_metadata.num_actual_tokens
    cu_seqlens = attn_metadata.query_start_loc
    seq_lens = attn_metadata.seq_lens
    num_seqs = cu_seqlens.shape[0] - 1

    q = query[:num_actual].to(torch.bfloat16)
    k_raw = key[:num_actual].to(torch.bfloat16) if key is not None else None
    v_raw = value[:num_actual].to(torch.bfloat16) if value is not None else None

    key_cache = kv_cache[:, 0]
    value_cache = kv_cache[:, 1]
    block_size = key_cache.shape[1]
    block_table = attn_metadata.block_table

    for i in range(num_seqs):
        q_start = cu_seqlens[i].item()
        q_end = cu_seqlens[i + 1].item()
        q_len = q_end - q_start
        ctx_len = seq_lens[i].item()
        if q_len == 0:
            continue

        qi = q[q_start:q_end]

        if k_raw is not None and ctx_len == q_len:
            # First prefill: all context is in current K/V
            ki = k_raw[q_start:q_end]
            vi = v_raw[q_start:q_end]
        elif k_raw is not None and tq_inst is not None:
            # Continuation: decompress prior blocks + append current
            prior_len = ctx_len - q_len
            prior_blocks = (prior_len + block_size - 1) // block_size
            bt = block_table[i, :prior_blocks]
            bt = bt[bt >= 0]

            if bt.numel() > 0:
                tq = tq_inst
                H = num_kv_heads

                kp = key_cache[bt, :, :, :packed_dim]
                vp = value_cache[bt, :, :, :packed_dim]

                flat_kp = kp.reshape(-1, packed_dim)
                flat_vp = vp.reshape(-1, packed_dim)

                if _TQ_IS_HYBRID:
                    dk = tq.decode(flat_kp)
                    dv = tq.decode(flat_vp)
                else:
                    kn = _primary_k_norms[layer_idx, bt]
                    vn = _primary_v_norms[layer_idx, bt]
                    dk = tq.decode(flat_kp, kn.reshape(-1))
                    dv = tq.decode(flat_vp, vn.reshape(-1))

                dk = dk.reshape(-1, H, head_size).to(torch.bfloat16)[:prior_len]
                dv = dv.reshape(-1, H, head_size).to(torch.bfloat16)[:prior_len]

                ki = torch.cat([dk, k_raw[q_start:q_end]], dim=0)
                vi = torch.cat([dv, v_raw[q_start:q_end]], dim=0)
            else:
                ki = k_raw[q_start:q_end]
                vi = v_raw[q_start:q_end]
        else:
            continue

        kv_len = ki.shape[0]

        # [1, heads, seq_len, D] for SDPA
        qi_4d = qi.transpose(0, 1).unsqueeze(0)  # [1, num_heads, q_len, D]
        ki_4d = ki.transpose(0, 1).unsqueeze(0)   # [1, num_kv_heads, kv_len, D]
        vi_4d = vi.transpose(0, 1).unsqueeze(0)

        # Manual GQA expansion (safe across all SDPA backends)
        gqa_ratio = num_heads // num_kv_heads
        if gqa_ratio > 1:
            ki_4d = ki_4d.repeat_interleave(gqa_ratio, dim=1)
            vi_4d = vi_4d.repeat_interleave(gqa_ratio, dim=1)

        # Prefill (q_len == kv_len): is_causal=True for standard causal mask.
        # Decode (q_len < kv_len): is_causal=False -- single query row attends
        # to all KV positions (no future tokens exist to mask out).
        # NOTE: is_causal=True with q_len < kv_len is documented to work in
        # PyTorch 2.2+ but produces degenerate (constant) output on some
        # model/backend combinations. is_causal=False is correct and safe.
        # Causal masking: for continuation (q_len < kv_len), we need a
        # proper causal mask that accounts for the prior context offset.
        # is_causal=True only works when q_len == kv_len.
        # For q_len < kv_len, build an explicit mask.
        if q_len == kv_len:
            out_i = F.scaled_dot_product_attention(
                qi_4d, ki_4d, vi_4d, is_causal=True, scale=scale)
        else:
            # Build causal mask: query position j can attend to KV position k
            # where k <= (kv_len - q_len) + j  (prior context + current pos)
            prior_len = kv_len - q_len
            q_pos = torch.arange(q_len, device=qi_4d.device).unsqueeze(1)
            k_pos = torch.arange(kv_len, device=qi_4d.device).unsqueeze(0)
            causal_mask = k_pos <= (prior_len + q_pos)
            # SDPA expects [1, 1, q_len, kv_len] bool mask or float -inf mask
            attn_mask = torch.where(
                causal_mask.unsqueeze(0).unsqueeze(0),
                torch.tensor(0.0, device=qi_4d.device),
                torch.tensor(float('-inf'), device=qi_4d.device),
            )
            out_i = F.scaled_dot_product_attention(
                qi_4d, ki_4d, vi_4d, attn_mask=attn_mask, scale=scale)

        out_i = out_i.squeeze(0).transpose(0, 1)
        output[q_start:q_end] = out_i.to(output.dtype)

    return output


# -- Main hook: patched forward --------------------------------------------

def _make_tq_forward(original_fwd):
    """Create the TQ-hooked forward that wraps standard TritonAttention."""

    def tq_forward(self, layer, query, key, value, kv_cache, attn_metadata,
                   output=None, output_scale=None, output_block_scale=None):
        # === FAST PATH: decode via custom op (Dynamo-traceable, CUDA-graphable) ===
        # No try/except, no init, no graph breaks. Dynamo can trace straight through.
        # Runs during BOTH normal execution AND CUDA graph capture.
        # Falls through to slow path for: init, prefill, missing key/value.
        if (hasattr(self, '_tq_layer_idx')
                and self.head_size in _tq_quantizers
                and attn_metadata is not None
                and output is not None
                and key is not None and value is not None
                and hasattr(attn_metadata, 'max_query_len')
                and attn_metadata.max_query_len == 1):
            num_actual = attn_metadata.num_actual_tokens

            # Hybrid custom op: clamp-gather decompress + masked SDPA
            if _USE_HYBRID_CUSTOM_OP and _TQ_IS_HYBRID:
                return torch.ops.tq.hybrid_decode_step(
                    query, key[:num_actual], value[:num_actual],
                    kv_cache, output,
                    attn_metadata.slot_mapping,
                    attn_metadata.block_table,
                    attn_metadata.seq_lens,
                    self._tq_layer_idx,
                    self.num_heads, self.num_kv_heads,
                    self.head_size, self.scale, num_actual)

            # Uniform custom op: fused rotated-domain attention
            if _USE_CUSTOM_OP and not _TQ_IS_HYBRID and _primary_k_norms is not None:
                return torch.ops.tq.decode_step(
                    query, key[:num_actual], value[:num_actual],
                    kv_cache, output,
                    _primary_k_norms, _primary_v_norms,
                    attn_metadata.slot_mapping,
                    attn_metadata.block_table,
                    attn_metadata.seq_lens,
                    self._tq_layer_idx,
                    self.num_heads, self.num_kv_heads,
                    self.head_size, self.scale, num_actual)

        # === SLOW PATH: init, prefill, fallback ===
        # --- Layer init (once per instance, guarded to avoid graph break) ---
        if not hasattr(self, '_tq_layer_idx'):
            _tq_init_layer(self)

        # --- CUDA graph capture bypass (slow path only -- fast path handles capture) ---
        if attn_metadata is None or torch.cuda.is_current_stream_capturing():
            if output is not None:
                output.fill_(0)
                return output
            return original_fwd(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output=output, output_scale=output_scale,
                output_block_scale=output_block_scale)

        # --- Init quantizer + norms (guarded: no graph break after first call) ---
        if self.head_size not in _tq_quantizers:
            _ensure_quantizer(query.device, self.head_size,
                             self.num_heads, self.num_kv_heads)
        if not _TQ_IS_HYBRID and _primary_k_norms is None:
            num_blocks = kv_cache.shape[0]
            block_size = kv_cache.shape[2]
            _ensure_norms(_layer_counter, num_blocks, block_size,
                         self.num_kv_heads, kv_cache.device)

        assert output is not None

        # --- Optional bypass for A/B testing ---
        if not hasattr(tq_forward, '_bypass'):
            tq_forward._bypass = os.environ.get("TQ_BYPASS", "") == "1"
            if tq_forward._bypass:
                logger.warning("TQ BYPASS MODE: using original forward")
        if tq_forward._bypass:
            return original_fwd(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output=output, output_scale=output_scale,
                output_block_scale=output_block_scale)

        # --- Debug logging (L0 only) ---
        if _TQ_DEBUG and self._tq_layer_idx == 0:
            _stats.fwd_count += 1
            if _stats.fwd_count <= _TQ_DEBUG_STEPS:
                _sm = attn_metadata.slot_mapping
                _dbg(f"FWD#{_stats.fwd_count} L0: "
                     f"q={list(query.shape)}, "
                     f"k={'None' if key is None else list(key.shape)}, "
                     f"max_q={getattr(attn_metadata, 'max_query_len', '?')}, "
                     f"actual={attn_metadata.num_actual_tokens}, "
                     f"sm={_sm[:min(4,_sm.shape[0])].tolist()}")

        # --- Determine phase early (decode uses merged encode+decode) ---
        is_decode = (
            hasattr(attn_metadata, "max_query_len")
            and attn_metadata.max_query_len == 1
        )

        # Debug: decode metadata
        if _TQ_DEBUG and is_decode and self._tq_layer_idx == 0:
            _stats.decode_count += 1
            if _stats.decode_count <= _TQ_DEBUG_STEPS:
                sl = attn_metadata.seq_lens
                _dbg(f"DECODE#{_stats.decode_count} L0: "
                     f"ctx={sl[0].item()}, "
                     f"enc_calls={_stats.encode_calls}, "
                     f"enc_tok={_stats.encode_tokens}, "
                     f"blocks={sorted(_stats.encode_blocks)}")

        # --- Phase 2a: Decode (fused TQ attention for uniform, custom op for hybrid) ---
        # Hybrid mode uses tq::hybrid_decode_step custom op (zero graph breaks),
        # or falls through to encode+SDPA path if custom op not available.
        if is_decode and self.head_size in _tq_quantizers and _TQ_IS_HYBRID:
            if _USE_HYBRID_CUSTOM_OP and key is not None and value is not None:
                num_actual = attn_metadata.num_actual_tokens
                try:
                    result = torch.ops.tq.hybrid_decode_step(
                        query, key[:num_actual], value[:num_actual],
                        kv_cache, output,
                        attn_metadata.slot_mapping,
                        attn_metadata.block_table,
                        attn_metadata.seq_lens,
                        self._tq_layer_idx,
                        self.num_heads, self.num_kv_heads,
                        self.head_size, self.scale, num_actual)
                    return result
                except Exception as e:
                    logger.error("TQ hybrid decode error L%d: %s",
                                 self._tq_layer_idx, e)
                    # Fall through to graph-break SDPA

        if is_decode and self.head_size in _tq_quantizers and not _TQ_IS_HYBRID:
            num_actual = attn_metadata.num_actual_tokens
            try:
                if _TQ_DEBUG and self._tq_layer_idx == 0:
                    if not hasattr(tq_forward, '_dec_times'):
                        tq_forward._dec_times = []
                    import time as _time
                    _t0 = _time.perf_counter()

                if (_USE_CUSTOM_OP and key is not None and value is not None
                        and _primary_k_norms is not None):
                    # Zero graph breaks -- CUDA-graphable
                    result = torch.ops.tq.decode_step(
                        query,
                        key[:num_actual], value[:num_actual],
                        kv_cache, output,
                        _primary_k_norms, _primary_v_norms,
                        attn_metadata.slot_mapping,
                        attn_metadata.block_table,
                        attn_metadata.seq_lens,
                        self._tq_layer_idx,
                        self.num_heads, self.num_kv_heads,
                        self.head_size, self.scale,
                        num_actual)
                else:
                    # Fallback: 1 graph break per layer
                    result = _tq_decode_step(
                        self._tq_layer_idx, query,
                        key[:num_actual] if key is not None else None,
                        value[:num_actual] if value is not None else None,
                        kv_cache, attn_metadata, output,
                        self.num_heads, self.num_kv_heads,
                        self.head_size, self.scale)

                if _TQ_DEBUG and self._tq_layer_idx == 0:
                    tq_forward._dec_times.append((_time.perf_counter() - _t0) * 1000)
                    if len(tq_forward._dec_times) == 50:
                        avg = sum(tq_forward._dec_times) / len(tq_forward._dec_times)
                        mode = "CUSTOM_OP" if _USE_CUSTOM_OP else "GRAPH_BREAK"
                        _dbg(f"{mode}_DECODE avg={avg:.2f}ms over 50 calls")
                return result
            except Exception as e:
                logger.error("TQ decode error L%d: %s",
                             self._tq_layer_idx, e)
                # Fall through to SDPA path

        # --- Prefill: encode to TQ cache + use ORIGINAL attention ---
        # The original TritonAttentionImpl.forward handles:
        #   - Paged cache write (do_kv_cache_update)
        #   - Fused triton attention kernel (reads from paged cache)
        #   - Chunked prefill with correct causal masking
        # We just need to ALSO encode to our TQ cache for future decode.
        # The original forward writes to the SAME kv_cache tensor (now uint8),
        # which works because TritonAttention writes raw bytes via
        # reshape_and_cache_flash (FP8_KV_CACHE=False path stores as-is).
        if key is not None and value is not None:
            try:
                num_actual = attn_metadata.num_actual_tokens
                _tq_encode_phase(
                    self._tq_layer_idx,
                    key[:num_actual], value[:num_actual],
                    kv_cache, attn_metadata.slot_mapping)
            except Exception as e:
                logger.error("TQ encode error L%d: %s", self._tq_layer_idx, e)

        # Use original vLLM forward for attention (handles chunked prefill correctly)
        return original_fwd(
            self, layer, query, key, value, kv_cache, attn_metadata,
            output=output, output_scale=output_scale,
            output_block_scale=output_block_scale)

    return tq_forward


# -- Public API ------------------------------------------------------------

def apply_tq_hooks() -> bool:
    """
    Monkey-patch TritonAttentionImpl.forward with TQ encode/decode hooks.
    Call AFTER vLLM model is loaded (layers instantiated).

    Does NOT register a custom attention backend -- uses standard Triton.
    torch.compile + piecewise CUDA graphs work normally.
    """
    global _hooks_applied, _original_forward

    if _hooks_applied:
        logger.info("TQ hooks already applied, skipping")
        return True

    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
    except ImportError:
        logger.warning("TritonAttentionImpl not found -- vLLM v0.15.1+ required")
        return False

    _original_forward = TritonAttentionImpl.forward
    TritonAttentionImpl.forward = _make_tq_forward(_original_forward)
    _hooks_applied = True

    # Register custom ops for zero-graph-break decode
    if _TQ_IS_HYBRID:
        _register_hybrid_custom_ops()
    else:
        _register_custom_ops()

    debug_status = "ENABLED" if _TQ_DEBUG else "disabled"
    if _TQ_IS_HYBRID and _USE_HYBRID_CUSTOM_OP:
        op_status = f"hybrid {_TQ_MODE} custom_op (CUDA-graphable decompress+SDPA)"
    elif _TQ_IS_HYBRID:
        op_status = f"hybrid {_TQ_MODE} (graph-break fallback)"
    elif _USE_CUSTOM_OP:
        op_status = "custom_op (CUDA-graphable fused rotated-domain)"
    else:
        op_status = "graph-break fallback"
    logger.info("TQ hooks applied to TritonAttentionImpl.forward "
                "(%s, debug %s)", op_status, debug_status)
    return True

# TEMP DEBUG: monkey-patch _tq_prefill_sdpa to dump decompress info
_orig_prefill_sdpa = _tq_prefill_sdpa

def _debug_prefill_sdpa(layer_idx, query, key, value, kv_cache, attn_metadata,
                         output, num_heads, num_kv_heads, head_size, scale):
    import sys
    if layer_idx == 0:
        cu = attn_metadata.query_start_loc
        sl = attn_metadata.seq_lens
        ns = cu.shape[0] - 1
        for i in range(ns):
            qs = cu[i].item()
            qe = cu[i+1].item()
            ql = qe - qs
            cl = sl[i].item()
            print(f"[TQ-DBG] L0 seq{i}: q_len={ql}, ctx_len={cl}, prior={cl-ql}",
                  file=sys.stderr, flush=True)
            if cl > ql and key is not None:
                # This is a continuation — prior blocks need decompressing
                bt = attn_metadata.block_table
                block_size = kv_cache[:, 0].shape[1]
                prior_len = cl - ql
                prior_blocks = (prior_len + block_size - 1) // block_size
                block_ids = bt[i, :prior_blocks]
                print(f"[TQ-DBG] L0 seq{i}: reading blocks={block_ids.tolist()}, "
                      f"block_size={block_size}, prior_len={prior_len}",
                      file=sys.stderr, flush=True)

                # Check norms
                if _primary_k_norms is not None:
                    kn = _primary_k_norms[0, block_ids[0], :4, :2]
                    print(f"[TQ-DBG] L0 norms block0 first 4 slots x 2 heads: {kn.tolist()}",
                          file=sys.stderr, flush=True)

                # Decode first few vectors and check
                tq, packed_dim = _get_tq(head_size)
                kc = kv_cache[:, 0]
                kp0 = kc[block_ids[0], 0, 0, :packed_dim]  # first slot, first head
                kn0 = _primary_k_norms[0, block_ids[0], 0, 0]  # first slot, first head
                dk0 = tq.decode(kp0.unsqueeze(0), kn0.unsqueeze(0))
                print(f"[TQ-DBG] L0 decoded block0_slot0_head0 first5: {dk0[0,:5].float().tolist()} norm={dk0.float().norm().item():.2f}",
                      file=sys.stderr, flush=True)

                # Compare with raw key if available
                if key is not None:
                    raw_k0 = key[qs, 0, :5].float()  # first token of current chunk, head 0
                    print(f"[TQ-DBG] L0 raw_key current_chunk[0] head0 first5: {raw_k0.tolist()}",
                          file=sys.stderr, flush=True)

    return _orig_prefill_sdpa(layer_idx, query, key, value, kv_cache, attn_metadata,
                               output, num_heads, num_kv_heads, head_size, scale)

_tq_prefill_sdpa = _debug_prefill_sdpa
