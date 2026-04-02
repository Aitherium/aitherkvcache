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
  - Prefill: decompress TQ blocks → temp fp8 buffer → standard attention
  - @torch.compiler.disable on encode/decompress creates clean graph breaks
  - The attention kernel itself remains compiled + CUDA-graphable

Apply hooks AFTER vLLM engine init (model loaded, layers instantiated):
    from lib.gpu.turboquant.vllm_hooks import apply_tq_hooks
    apply_tq_hooks()

Requires: vllm_engine.apply_tq_patches() already applied (page_size, reshape).

Debug logging: set AITHER_TQ_DEBUG=1 to enable detailed per-step diagnostics.
Set AITHER_TQ_DEBUG_STEPS=N to log the first N forward calls per layer-0
(default 10). Logs go to stderr via the tq_dbg logger.
"""

import logging
import math
import os
import sys
from typing import ClassVar, Optional

import torch

logger = logging.getLogger("aither.turboquant.hooks")

# ── Debug logger ──────────────────────────────────────────────────
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


# ── Module-level state ──────────────────────────────────────────────

_hooks_applied = False
_original_forward = None  # Saved reference to TritonAttentionImpl.forward

# TQ state shared across all layers (class-level in the patched impl)
_tq_quantizer: Optional[object] = None
_tq_fused_attn: dict = {}  # layer_idx → TQPagedAttention instance
_primary_k_norms: Optional[torch.Tensor] = None
_primary_v_norms: Optional[torch.Tensor] = None
_layer_counter = 0
_num_layers = 0

# Prefill decompression buffer (shared across layers, allocated once)
_prefill_k_buf: Optional[torch.Tensor] = None
_prefill_v_buf: Optional[torch.Tensor] = None
_PREFILL_BUF_BLOCKS = 512


# ── Helpers (run eagerly, create graph breaks) ──────────────────────

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
    """Lazy-init TQ quantizer and fused attention on first real forward."""
    global _tq_quantizer, _num_layers
    if _tq_quantizer is not None:
        return

    from .quantizer import TurboQuant
    _tq_quantizer = TurboQuant(head_dim=head_size, bits=4, device=str(device))
    _num_layers = _layer_counter
    logger.info("TQ quantizer initialized: head_dim=%d, bits=4, %d layers",
                head_size, _num_layers)


@torch.compiler.disable
def _ensure_norms(num_layers, num_blocks, block_size, num_kv_heads, device):
    """Lazy-init per-layer float32 norm tensors."""
    global _primary_k_norms, _primary_v_norms
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
    """Encode new K,V tokens into TQ primary cache. Runs eagerly."""
    global _tq_quantizer, _primary_k_norms, _primary_v_norms
    tq = _tq_quantizer
    if tq is None:
        return

    packed_dim = key.shape[-1] // 2  # head_dim // 2 for 4-bit

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

    # Encode: TQ returns packed indices [N*H, packed_dim] and norms [N*H]
    kp, kn = tq.encode(vk.reshape(N * H, D))
    vp, vn = tq.encode(vv.reshape(N * H, D))

    kp = kp.reshape(N, H, packed_dim)
    vp = vp.reshape(N, H, packed_dim)

    # Write packed indices to uint8 cache
    key_cache[bi, oi, :, :packed_dim] = kp
    value_cache[bi, oi, :, :packed_dim] = vp

    # Write norms to per-layer float32 tensors
    _primary_k_norms[layer_idx, bi, oi] = kn.reshape(N, H).to(torch.float32)
    _primary_v_norms[layer_idx, bi, oi] = vn.reshape(N, H).to(torch.float32)

    # Stats
    _stats.encode_calls += 1
    _stats.encode_tokens += N
    _stats.encode_blocks.update(bi.tolist())

    if _TQ_DEBUG and _stats.encode_calls <= _TQ_DEBUG_STEPS:
        _dbg(f"ENCODE#{_stats.encode_calls} L{layer_idx}: N={N}, "
             f"bi={bi[:min(4,len(bi))].tolist()}, "
             f"kn=[{kn.min().item():.1f},{kn.max().item():.1f}], "
             f"vn=[{vn.min().item():.3f},{vn.max().item():.3f}]")


@torch.compiler.disable
def _tq_decompress_active(layer_idx, kv_cache, block_table, num_kv_heads,
                          head_size, device):
    """Decompress active TQ blocks into a temporary fp8/bf16 buffer.
    Returns a standard-shaped kv_cache tensor for TritonAttention.
    Runs eagerly."""
    global _tq_quantizer, _primary_k_norms, _primary_v_norms
    global _prefill_k_buf, _prefill_v_buf

    tq = _tq_quantizer
    if tq is None:
        return kv_cache  # Fallback: return original (will fail but shouldn't happen)

    key_cache = kv_cache[:, 0]
    value_cache = kv_cache[:, 1]
    num_blocks = key_cache.shape[0]
    block_size = key_cache.shape[1]
    packed_dim = head_size // 2

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
        kn_all = _primary_k_norms[layer_idx, active_blocks]
        vn_all = _primary_v_norms[layer_idx, active_blocks]

        flat_kp = kp_all.reshape(A * block_size * H, packed_dim)
        flat_kn = kn_all.reshape(A * block_size * H)
        flat_vp = vp_all.reshape(A * block_size * H, packed_dim)
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
    global _tq_fused_attn, _tq_quantizer, _primary_k_norms, _primary_v_norms

    # Lazy-init fused attention for this layer
    if layer_idx not in _tq_fused_attn:
        from .fused_attention import TQPagedAttention
        _tq_fused_attn[layer_idx] = TQPagedAttention(_tq_quantizer, num_heads)

    fused = _tq_fused_attn[layer_idx]
    packed_dim = head_size // 2

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


# ── Inline helpers (no decorator — called from within @torch.compiler.disable) ──

def _encode_inline(layer_idx, key, value, kv_cache, slot_mapping):
    """Encode K/V into TQ cache. No decorator — called from _tq_decode_step.
    Branchless for decode: no .any() CPU-GPU syncs. Works for any N tokens."""
    tq = _tq_quantizer
    packed_dim = key.shape[-1] // 2
    key_cache = kv_cache[:, 0]
    value_cache = kv_cache[:, 1]
    block_size = key_cache.shape[1]

    # Decode: slot_mapping is [N] with N usually 1, all valid.
    # Avoid .any() which triggers CPU-GPU sync. Direct index math.
    bi = slot_mapping // block_size
    oi = slot_mapping % block_size

    N, H, D = key.shape

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

    # Fused attention — dict lookup for cached TQPagedAttention
    fused = _tq_fused_attn.get(layer_idx)
    if fused is None:
        from .fused_attention import TQPagedAttention
        fused = TQPagedAttention(_tq_quantizer, num_heads)
        _tq_fused_attn[layer_idx] = fused

    packed_dim = head_size // 2
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


# ── Prefill scatter-write (for writing raw K/V into decompress buffer) ──

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
    (single query attends to all prior positions — no mask needed)."""
    import torch.nn.functional as F
    global _tq_quantizer, _primary_k_norms, _primary_v_norms

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
    packed_dim = head_size // 2
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
        elif k_raw is not None and _tq_quantizer is not None:
            # Continuation: decompress prior blocks + append current
            prior_len = ctx_len - q_len
            prior_blocks = (prior_len + block_size - 1) // block_size
            bt = block_table[i, :prior_blocks]
            bt = bt[bt >= 0]

            if bt.numel() > 0:
                tq = _tq_quantizer
                H = num_kv_heads

                kp = key_cache[bt, :, :, :packed_dim]
                vp = value_cache[bt, :, :, :packed_dim]
                kn = _primary_k_norms[layer_idx, bt]
                vn = _primary_v_norms[layer_idx, bt]

                dk = tq.decode(kp.reshape(-1, packed_dim), kn.reshape(-1))
                dv = tq.decode(vp.reshape(-1, packed_dim), vn.reshape(-1))

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
        # Decode (q_len < kv_len): is_causal=False — single query row attends
        # to all KV positions (no future tokens exist to mask out).
        # NOTE: is_causal=True with q_len < kv_len is documented to work in
        # PyTorch 2.2+ but produces degenerate (constant) output on some
        # model/backend combinations. is_causal=False is correct and safe.
        use_causal = (q_len == kv_len)
        out_i = F.scaled_dot_product_attention(
            qi_4d, ki_4d, vi_4d, is_causal=use_causal, scale=scale)

        out_i = out_i.squeeze(0).transpose(0, 1)
        output[q_start:q_end] = out_i.to(output.dtype)

    return output


# ── Main hook: patched forward ──────────────────────────────────────

def _make_tq_forward(original_fwd):
    """Create the TQ-hooked forward that wraps standard TritonAttention."""

    def tq_forward(self, layer, query, key, value, kv_cache, attn_metadata,
                   output=None, output_scale=None, output_block_scale=None):
        # --- Layer init (once per instance, guarded to avoid graph break) ---
        if not hasattr(self, '_tq_layer_idx'):
            _tq_init_layer(self)

        # --- CUDA graph capture / profiling: bypass TQ entirely ---
        if attn_metadata is None or torch.cuda.is_current_stream_capturing():
            if output is not None:
                output.fill_(0)
                return output
            return original_fwd(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output=output, output_scale=output_scale,
                output_block_scale=output_block_scale)

        # --- Init quantizer + norms (guarded: no graph break after first call) ---
        if _tq_quantizer is None:
            _ensure_quantizer(query.device, self.head_size,
                             self.num_heads, self.num_kv_heads)
        if _primary_k_norms is None:
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

        # --- Phase 2a: Merged decode (single graph break: encode + fused) ---
        # Halves graph breaks from 72->36 per token by combining encode +
        # fused attention into one @torch.compiler.disable call.
        if is_decode and _tq_quantizer is not None:
            num_actual = attn_metadata.num_actual_tokens
            try:
                if _TQ_DEBUG and self._tq_layer_idx == 0:
                    if not hasattr(tq_forward, '_dec_times'):
                        tq_forward._dec_times = []
                    import time as _time
                    _t0 = _time.perf_counter()
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
                        _dbg(f"MERGED_DECODE avg={avg:.2f}ms over 50 calls")
                return result
            except Exception as e:
                logger.error("TQ merged decode error L%d: %s",
                             self._tq_layer_idx, e)
                # Fall through to SDPA path

        # --- Phase 1: TQ encode new K,V (prefill only) ---
        # Decode path encodes inside _tq_decode_step above.
        # vLLM v1 pads key/value to chunk size but slot_mapping has only
        # num_actual_tokens entries. Slice to match.
        if key is not None and value is not None:
            try:
                num_actual = attn_metadata.num_actual_tokens
                if _TQ_DEBUG and self._tq_layer_idx == 0:
                    if not hasattr(tq_forward, '_enc_times'):
                        tq_forward._enc_times = []
                    import time as _time
                    _t0 = _time.perf_counter()
                _tq_encode_phase(
                    self._tq_layer_idx,
                    key[:num_actual], value[:num_actual],
                    kv_cache, attn_metadata.slot_mapping)
                if _TQ_DEBUG and self._tq_layer_idx == 0:
                    tq_forward._enc_times.append((_time.perf_counter() - _t0) * 1000)
                    if len(tq_forward._enc_times) == 50:
                        avg = sum(tq_forward._enc_times) / len(tq_forward._enc_times)
                        _dbg(f"ENCODE avg={avg:.2f}ms over 50 calls")
            except Exception as e:
                logger.error("TQ encode error L%d: %s", self._tq_layer_idx, e)

        # Prefill / decode: SDPA with raw K/V + decompressed prior context
        try:
            return _tq_prefill_sdpa(
                self._tq_layer_idx, query, key, value, kv_cache,
                attn_metadata, output, self.num_heads, self.num_kv_heads,
                self.head_size, self.scale)
        except Exception as e:
            logger.error("TQ SDPA error L%d: %s", self._tq_layer_idx, e)
            output.fill_(0)
            return output

    return tq_forward


# ── Public API ──────────────────────────────────────────────────────

def apply_tq_hooks() -> bool:
    """
    Monkey-patch TritonAttentionImpl.forward with TQ encode/decode hooks.
    Call AFTER vLLM model is loaded (layers instantiated).

    Does NOT register a custom attention backend — uses standard Triton.
    torch.compile + piecewise CUDA graphs work normally.
    """
    global _hooks_applied, _original_forward

    if _hooks_applied:
        logger.info("TQ hooks already applied, skipping")
        return True

    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
    except ImportError:
        logger.warning("TritonAttentionImpl not found — vLLM v0.15.1+ required")
        return False

    _original_forward = TritonAttentionImpl.forward
    TritonAttentionImpl.forward = _make_tq_forward(_original_forward)
    _hooks_applied = True

    debug_status = "ENABLED" if _TQ_DEBUG else "disabled"
    logger.info("TQ hooks applied to TritonAttentionImpl.forward "
                "(standard backend, torch.compile compatible, "
                "debug %s)", debug_status)
    return True
