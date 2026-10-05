"""Runtime translation of a source KV cache into a target KV cache, and the
downstream measurement that decides whether a pack may be used at all.

## The runtime path

``translate_cache`` is the whole trick in one function: slice the selected source
layers, apply the affine map, reshape to the target's head layout, and re-apply
the TARGET model's RoPE at the correct ABSOLUTE positions. Values skip the
rotation entirely.

The positions argument is not optional and does not default to ``range(n)``. A
translated cache is very often a *suffix* — the reusable head of a conversation
that already has tokens before it — and starting the rotation at zero produces a
cache that is well-formed, correctly shaped, and rotated to the wrong angles.
The model then attends confidently to positions that do not exist.

## Why the evaluation has four arms

Reconstruction R^2 answers "did the map learn the mapping". It cannot answer
"can the receiving model still work", and the gap between those is where a
cross-model KV feature would quietly ship broken. So the acceptance run measures
what the target model actually predicts, against three reference points:

* **reference** — the target prefills the context itself. The ceiling.
* **translated** — the target is handed our converted cache. The candidate.
* **control** — the target is handed a cache translated from a DIFFERENT
  document. The map runs, at full cost, on the wrong content. This is the arm
  that catches a mapper which has learned the target's average key/value
  statistics and ignores its input: such a mapper scores respectably against
  reference and IDENTICALLY against control.
* **nocontext** — no cache at all. The floor.

A headline of "68% top-1 agreement" is unreadable alone. Next tokens are often
forced by the last few tokens regardless of context, so a dead mapper scores far
above zero. ``translated`` must beat ``control`` by a wide margin, or the
feature is a no-op that reports success — the exact class
the silent-no-op class: a feature that returns success-shaped output while
doing nothing, and therefore passes every test that only asserts it did not
crash.

## Span-aligned packs (different tokenizers)

For a pack whose manifest declares span alignment, the source never sees text
the target has not read: for a cut at target position ``cut`` the cache carries
target positions ``0..cut-2``, so the source is run on ``text[:char_end(cut-2)]``
ONLY — re-tokenized, exactly as a server would see that prefix — and pooled onto
target rows through the causal CSR. Any CSR entry that would pool a source token
ending after its target token raises instead of scoring (``causal_violations``).
``--leak-canary`` deliberately breaks both (full-window text, non-causal
pooling) to prove the evaluation can see a leak; it may never be recorded.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .align import (
    build_span_csr,
    causal_violations,
    encode_with_offsets,
    require_fast_tokenizer,
)
from .capture import _extract_kv, _stack_layers, gather_corpus
from .geometry import KVGeometry
from .pack import (
    AcceptanceMetrics,
    LoadedPack,
    load_pack,
    record_acceptance,
)
from .rope import rotate_to_target

logger = logging.getLogger("kvtransfer.transfer")


def translate_cache(
    source_k_flat: np.ndarray,
    source_v_flat: np.ndarray,
    pack: LoadedPack,
    positions: Sequence[int] | np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Convert a source KV cache into the target model's KV cache.

    Args:
        source_k_flat: (n_tokens, n_src_layers * src_width), RoPE ALREADY
            STRIPPED — the same representation the fit was performed in.
        source_v_flat: (n_tokens, n_src_layers * src_width), as captured.
        pack: a verified pack (``load_pack`` has already enforced every rule).
        positions: absolute position of each token. See the module docstring.

    Returns:
        One (K, V) pair per target layer, each shaped
        (1, n_kv_heads, n_tokens, head_dim) — the layout every HF attention
        implementation expects.
    """
    tgt = pack.target
    src_block = pack.source.per_layer_width
    n_tokens = source_k_flat.shape[0]
    pos = np.asarray(positions, dtype=np.int64).reshape(-1)
    if pos.shape[0] != n_tokens:
        raise ValueError(f"got {pos.shape[0]} positions for {n_tokens} tokens")
    expected = pack.source.flat_width
    for name, arr in (("K", source_k_flat), ("V", source_v_flat)):
        if arr.shape[1] != expected:
            raise ValueError(
                f"source {name} has width {arr.shape[1]}, pack expects {expected}"
            )

    rope = tgt.rope_spec()
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for layer in range(tgt.n_layers):
        pair: List[np.ndarray] = []
        for role, flat in (("k", source_k_flat), ("v", source_v_flat)):
            w, b, src_layers = pack.layer_map(role, layer)
            cols = np.concatenate([
                np.arange(s * src_block, (s + 1) * src_block, dtype=np.int64)
                for s in src_layers
            ])
            x = flat[:, cols].astype(np.float32, copy=False)
            y = x @ w.astype(np.float32) + b.astype(np.float32)
            y = y.reshape(n_tokens, tgt.n_kv_heads, tgt.head_dim)
            y = np.ascontiguousarray(y.transpose(1, 0, 2))       # (h, tokens, d)
            if role == "k":
                y = rotate_to_target(y, rope, pos)
            pair.append(y[None, ...])                            # (1, h, tokens, d)
        out.append((pair[0], pair[1]))
    return out


def translate_span(
    source_k_flat: np.ndarray,
    source_v_flat: np.ndarray,
    pack: LoadedPack,
    *,
    src_offsets: Sequence[Tuple[int, int]],
    tgt_offsets: Sequence[Tuple[int, int]],
    positions: Sequence[int] | np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Translate a source cache with its OWN token rows through a span pack.

    ``src_offsets``/``tgt_offsets`` are both tokenizers' character offsets over
    the SAME text; the source rows are pooled onto target rows by the causal CSR
    and then mapped exactly as ``translate_cache`` maps positional rows.
    """
    spec = pack.alignment
    if spec is None or spec.method != "span":
        raise ValueError("translate_span needs a span-aligned pack (manifest alignment)")
    if not spec.causal:
        raise ValueError("refusing a non-causal alignment spec at serve time")
    if source_k_flat.shape[0] != len(src_offsets):
        raise ValueError(f"{source_k_flat.shape[0]} source rows for "
                         f"{len(src_offsets)} source offsets")
    csr = build_span_csr(src_offsets, tgt_offsets, causal=True)
    leaks = causal_violations(csr, src_offsets, tgt_offsets)
    if leaks:
        raise RuntimeError(f"span CSR pools future text into the cache: {leaks[:3]}")
    k = csr.apply(source_k_flat.astype(np.float32, copy=False))
    v = csr.apply(source_v_flat.astype(np.float32, copy=False))
    return translate_cache(k, v, pack, positions)


def _to_cache(layers: List[Tuple[np.ndarray, np.ndarray]], torch_mod: Any, device: str,
              dtype: Any) -> Any:
    """Wrap numpy KV in whatever Cache class this transformers ships."""
    from transformers import DynamicCache

    legacy = tuple(
        (
            torch_mod.tensor(k, dtype=dtype, device=device),
            torch_mod.tensor(v, dtype=dtype, device=device),
        )
        for k, v in layers
    )
    from_legacy = getattr(DynamicCache, "from_legacy_cache", None)
    if callable(from_legacy):
        try:
            return from_legacy(legacy)
        except (TypeError, AttributeError) as exc:
            logger.debug("from_legacy_cache unusable (%s); falling back to update()", exc)
    cache = DynamicCache()
    for idx, (k, v) in enumerate(legacy):
        cache.update(k, v, idx)
    return cache


def _slice_layers(
    layers: List[Tuple[np.ndarray, np.ndarray]], upto: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    return [(k[:, :, :upto, :], v[:, :, :upto, :]) for k, v in layers]


def evaluate_acceptance(
    pack_dir: Path,
    *,
    source_id: str,
    target_id: str,
    corpus_roots: Sequence[Path],
    cuts: Sequence[int] = (64, 128, 192),
    seq_len: int = 256,
    n_sequences: int = 8,
    device: str = "cpu",
    dtype: str = "float32",
    skip_docs: int = 0,
    leak_canary: bool = False,
) -> AcceptanceMetrics:
    """Measure what the target model does with a translated cache.

    Held-out by construction: ``skip_docs`` moves past the documents used for
    fitting, so this never scores the mapper on text it was trained on. A
    span-aligned pack is measured in span mode (module docstring);
    ``leak_canary`` is span-only and its result must never be recorded.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(target_id)
    cfg_src = AutoConfig.from_pretrained(source_id)
    cfg_tgt = AutoConfig.from_pretrained(target_id)
    tok_src = AutoTokenizer.from_pretrained(source_id)
    geom_src = KVGeometry.from_hf(source_id, cfg_src, tok_src, dtype)
    geom_tgt = KVGeometry.from_hf(target_id, cfg_tgt, tok, dtype)

    pack = load_pack(pack_dir, live_source=geom_src, live_target=geom_tgt,
                     require_acceptance=False)
    span = pack.alignment is not None and pack.alignment.method == "span"
    if leak_canary and not span:
        raise ValueError("--leak-canary applies to span-aligned packs only")
    if span:
        require_fast_tokenizer(tok_src, "source")
        require_fast_tokenizer(tok, "target")

    docs = gather_corpus(corpus_roots, (".md", ".py", ".txt"), skip_docs + n_sequences * 4)
    docs = docs[skip_docs:]
    seqs: List[List[int]] = []
    seq_docs: List[str] = []
    seq_offsets: List[List[Tuple[int, int]]] = []
    for doc in docs:
        if len(seqs) >= n_sequences:
            break
        if span:
            ids, offs = encode_with_offsets(tok, doc)
        else:
            ids, offs = tok(doc, add_special_tokens=False)["input_ids"], []
        if len(ids) >= seq_len:
            seqs.append(ids[:seq_len])
            seq_docs.append(doc)
            seq_offsets.append(offs[:seq_len])
    if len(seqs) < 2:
        raise RuntimeError(
            f"need at least 2 eval sequences of {seq_len} tokens, got {len(seqs)}"
        )

    positions = np.arange(seq_len, dtype=np.int64)
    src_model = AutoModelForCausalLM.from_pretrained(
        source_id, dtype=torch_dtype, attn_implementation="eager").to(device).eval()
    # Translated caches are built lazily and only the two in use (this sequence and its
    # control) are kept: ~150 MB each for an 8B target, so pre-building all of them held
    # ~20 GB at 136 sequences and OOM-killed a CPU run (2026-10-01). The source model
    # therefore stays loaded beside the target (~1.2 GB for 0.6B): a small fixed cost.
    translated: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}

    def full_translation(i: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        if i not in translated:
            ids = torch.tensor([seqs[i]], dtype=torch.long, device=device)
            with torch.no_grad():
                out = src_model(ids, use_cache=True)
            k_flat, v_flat = _stack_layers(_extract_kv(out.past_key_values), geom_src,
                                           positions)
            translated[i] = translate_cache(k_flat.astype(np.float32),
                                            v_flat.astype(np.float32), pack, positions)
        return translated[i]

    span_stats = {"carried": 0, "empty": 0, "canary_leaks": 0}

    def context_cache(i: int, cut: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Translated cache for target positions 0..cut-2 of sequence i."""
        if not span:
            return _slice_layers(full_translation(i), cut - 1)
        n_ctx = cut - 1
        t_off = seq_offsets[i]
        end_char = t_off[(seq_len if leak_canary else n_ctx) - 1][1]
        src_ids, src_offs = encode_with_offsets(tok_src, seq_docs[i][:end_char])
        if not src_ids:
            raise RuntimeError(f"source tokenized text[:{end_char}] to nothing")
        with torch.no_grad():
            out = src_model(torch.tensor([src_ids], dtype=torch.long, device=device),
                            use_cache=True)
        k_flat, v_flat = _stack_layers(_extract_kv(out.past_key_values), geom_src,
                                       np.arange(len(src_ids), dtype=np.int64))
        csr = build_span_csr(src_offs, t_off[:n_ctx], causal=not leak_canary)
        leaks = causal_violations(csr, src_offs, t_off[:n_ctx])
        if leaks and not leak_canary:
            raise RuntimeError(f"span CSR leaks future text at cut {cut}: {leaks[:3]}")
        span_stats["canary_leaks"] += len(leaks)
        span_stats["carried"] += csr.n_carried
        span_stats["empty"] += csr.n_empty
        return translate_cache(csr.apply(k_flat.astype(np.float32)),
                               csr.apply(v_flat.astype(np.float32)), pack,
                               np.arange(n_ctx, dtype=np.int64))

    tgt_model = AutoModelForCausalLM.from_pretrained(
        target_id, dtype=torch_dtype, attn_implementation="eager").to(device).eval()

    tallies: Dict[str, Dict[str, float]] = {
        arm: {"agree": 0.0, "nll": 0.0, "n": 0.0}
        for arm in ("translated", "control", "nocontext")
    }
    ref_nll_total = 0.0
    n_positions = 0

    for idx, seq in enumerate(seqs):
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        with torch.no_grad():
            ref_logits = tgt_model(ids, use_cache=False).logits[0]
        control_idx = (idx + 1) % len(seqs)
        for stale in [k for k in translated if k not in (idx, control_idx)]:
            del translated[stale]

        for cut in cuts:
            if cut < 2 or cut >= seq_len:
                continue
            # Predict token[cut] from context tokens[0:cut]. The cache carries
            # positions 0..cut-2; the model computes position cut-1 itself.
            gold = seq[cut]
            ref_row = ref_logits[cut - 1].float()
            ref_lp = torch.log_softmax(ref_row, dim=-1)
            ref_pred = int(torch.argmax(ref_row))
            ref_nll_total += float(-ref_lp[gold])
            n_positions += 1

            last = torch.tensor([[seq[cut - 1]]], dtype=torch.long, device=device)
            arms = {
                "translated": context_cache(idx, cut),
                "control": context_cache(control_idx, cut),
                "nocontext": None,
            }
            for arm, layers in arms.items():
                with torch.no_grad():
                    if layers is None:
                        out = tgt_model(last, use_cache=False)
                    else:
                        cache = _to_cache(layers, torch, device, torch_dtype)
                        out = tgt_model(
                            last,
                            past_key_values=cache,
                            use_cache=True,
                            cache_position=torch.tensor([cut - 1], device=device),
                            attention_mask=torch.ones(
                                (1, cut), dtype=torch.long, device=device),
                        )
                row = out.logits[0, -1].float()
                lp = torch.log_softmax(row, dim=-1)
                tallies[arm]["agree"] += float(int(torch.argmax(row)) == ref_pred)
                tallies[arm]["nll"] += float(-lp[gold])
                tallies[arm]["n"] += 1.0
        logger.info("[eval] sequence %d/%d", idx + 1, len(seqs))

    del tgt_model, src_model
    translated.clear()
    if n_positions == 0:
        raise RuntimeError("no evaluation positions were scored — check --cuts vs --seq-len")

    def mean(arm: str, key: str) -> float:
        return tallies[arm][key] / max(tallies[arm]["n"], 1.0)

    ref_nll = ref_nll_total / n_positions
    nll_tr = mean("translated", "nll")
    return AcceptanceMetrics(
        top1_agreement=mean("translated", "agree"),
        nll_translated=nll_tr,
        nll_reference=ref_nll,
        nll_delta=nll_tr - ref_nll,
        n_positions=n_positions,
        eval_sequences=len(seqs),
        top1_control=mean("control", "agree"),
        nll_control=mean("control", "nll"),
        top1_nocontext=mean("nocontext", "agree"),
        nll_nocontext=mean("nocontext", "nll"),
        notes=(f"cuts={list(cuts)} seq_len={seq_len} skip_docs={skip_docs}"
               + (f" align=span digest={pack.alignment.digest()[:16]} "  # type: ignore[union-attr]
                  f"carried={span_stats['carried']} empty={span_stats['empty']}"
                  if span else "")
               + (f" LEAK-CANARY leaks={span_stats['canary_leaks']}" if leak_canary else "")),
    )


# ── self-test ───────────────────────────────────────────────────────────────


def _self_test() -> int:
    import shutil
    import tempfile

    from .mapper import LayerMap
    from .pack import write_pack

    failures: List[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(f"{name}: {detail}")

    def geom(mid: str, layers: int, kv: int, dh: int, theta: float) -> KVGeometry:
        return KVGeometry(
            model_id=mid, family="llama", n_layers=layers, n_kv_heads=kv, head_dim=dh,
            hidden_size=kv * dh, rope_theta=theta, rope_type="default",
            rope_scaling_factor=1.0, tokenizer_sha256="z" * 64, torch_dtype="float32")

    src = geom("s", 4, 2, 4, 10000.0)
    tgt = geom("t", 2, 3, 4, 20000.0)
    tmp = Path(tempfile.mkdtemp(prefix="kvtxfer"))
    try:
        d_sel = 2 * src.per_layer_width          # top_k = 2
        maps = {
            role: [
                LayerMap(target_layer=t, role=role, source_layers=[0, 2], lam_rel=1e-3,
                         weight=np.eye(d_sel, tgt.per_layer_width, dtype=np.float32),
                         bias=np.zeros(tgt.per_layer_width, dtype=np.float32),
                         val_r2_mean=0.9, val_r2_per_head=[0.9] * tgt.n_kv_heads)
                for t in range(tgt.n_layers)
            ]
            for role in ("k", "v")
        }
        write_pack(tmp, maps, source=src, target=tgt, top_k=2, regime_flags=[],
                   train_tokens=100, val_tokens=50, corpus_sha256="q" * 64,
                   created_at=0.0, weight_dtype="float32")
        pack = load_pack(tmp, live_source=src, live_target=tgt, require_acceptance=False)

        rng = np.random.default_rng(11)
        n_tok = 7
        k_flat = rng.standard_normal((n_tok, src.flat_width)).astype(np.float32)
        v_flat = rng.standard_normal((n_tok, src.flat_width)).astype(np.float32)
        layers = translate_cache(k_flat, v_flat, pack, np.arange(n_tok))

        check("layer_count", len(layers) == tgt.n_layers, f"got {len(layers)}")
        check("shape", layers[0][0].shape == (1, 3, n_tok, 4),
              f"got {layers[0][0].shape}")

        # V must be the raw affine output (no rotation applied).
        cols = np.concatenate([np.arange(0, 8), np.arange(16, 24)])
        want_v = (v_flat[:, cols] @ np.eye(d_sel, tgt.per_layer_width, dtype=np.float32))
        want_v = want_v.reshape(n_tok, 3, 4).transpose(1, 0, 2)[None, ...]
        check("v_unrotated", np.allclose(layers[0][1], want_v, atol=1e-5),
              "V path is not the plain affine map")

        # K must NOT equal the raw affine output — the target rotation is applied.
        want_k_raw = (k_flat[:, cols] @ np.eye(d_sel, tgt.per_layer_width,
                                               dtype=np.float32))
        want_k_raw = want_k_raw.reshape(n_tok, 3, 4).transpose(1, 0, 2)[None, ...]
        check("k_rotated", not np.allclose(layers[0][0], want_k_raw, atol=1e-4),
              "K path did not apply the target rotation — rotate_to_target is inert")

        # ...but row 0 IS unrotated, because position 0 has angle 0. This pins
        # that the rotation is position-indexed rather than a constant twist.
        check("k_pos0_identity",
              np.allclose(layers[0][0][:, :, 0, :], want_k_raw[:, :, 0, :], atol=1e-5),
              "position 0 was rotated — positions are not being threaded through")

        # Offset positions must change the result: a suffix cache is the common
        # case and defaulting to range(n) would silently mis-rotate it.
        shifted = translate_cache(k_flat, v_flat, pack, np.arange(100, 100 + n_tok))
        check("positions_matter",
              not np.allclose(shifted[0][0], layers[0][0], atol=1e-4),
              "absolute positions had no effect on the translated keys")

        for name, fn in (
            ("reject_position_count",
             lambda: translate_cache(k_flat, v_flat, pack, np.arange(3))),
            ("reject_width",
             lambda: translate_cache(k_flat[:, :4], v_flat, pack, np.arange(n_tok))),
        ):
            try:
                fn()
                check(name, False, "accepted malformed input")
            except ValueError as exc:
                check(f"{name}_message", str(exc).strip() != "",
                      "refused with an empty message")

        check("slice_layers", _slice_layers(layers, 3)[0][0].shape == (1, 3, 3, 4),
              f"got {_slice_layers(layers, 3)[0][0].shape}")

        # Span packs: identical offsets reproduce translate_cache exactly, and a
        # positional pack is refused by translate_span.
        from .align import AlignmentSpec

        span_dir = tmp / "span"
        src_s = KVGeometry(**{**src.to_dict(), "tokenizer_sha256": "y" * 64})
        write_pack(span_dir, maps, source=src_s, target=tgt, top_k=2, regime_flags=[],
                   train_tokens=100, val_tokens=50, corpus_sha256="q" * 64,
                   created_at=0.0, weight_dtype="float32",
                   alignment=AlignmentSpec.span("y" * 64, "z" * 64))
        span_pack = load_pack(span_dir, live_source=src_s, live_target=tgt,
                              require_acceptance=False)
        offs = [(i, i + 1) for i in range(n_tok)]
        via_span = translate_span(k_flat, v_flat, span_pack, src_offsets=offs,
                                  tgt_offsets=offs, positions=np.arange(n_tok))
        check("span_identity_matches_positional",
              all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                  for a, b in zip(via_span, layers)),
              "identity span translation differs from positional translation")
        # Two source tokens per target token: target rows are the source means.
        src_offs2 = [(i, i + 1) for i in range(2 * n_tok)]
        tgt_offs2 = [(2 * i, 2 * i + 2) for i in range(n_tok)]
        k2 = np.repeat(k_flat, 2, axis=0)
        v2 = np.repeat(v_flat, 2, axis=0)
        via_pool = translate_span(k2, v2, span_pack, src_offsets=src_offs2,
                                  tgt_offsets=tgt_offs2, positions=np.arange(n_tok))
        check("span_pooling_means",
              np.allclose(via_pool[0][1], layers[0][1], atol=1e-5),
              "mean of two identical source rows did not reproduce the row")
        try:
            translate_span(k_flat, v_flat, pack, src_offsets=offs, tgt_offsets=offs,
                           positions=np.arange(n_tok))
            check("span_refuses_positional_pack", False, "accepted a positional pack")
        except ValueError as exc:
            check("span_refuses_positional_reason", "span-aligned" in str(exc), str(exc))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for f in failures:
        print(f"FAIL {f}")
    print(f"transfer self-test: {'PASS' if not failures else 'FAIL'} ({len(failures)} failures)")
    return 1 if failures else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Measure and record pack acceptance")
    ap.add_argument("--pack", type=Path)
    ap.add_argument("--source")
    ap.add_argument("--target")
    ap.add_argument("--corpus", type=Path, nargs="*", default=[Path(".")])
    ap.add_argument("--cuts", type=int, nargs="*", default=[64, 128, 192])
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--sequences", type=int, default=8)
    ap.add_argument("--skip-docs", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--record", action="store_true",
                    help="write the measurement into the pack manifest")
    ap.add_argument("--leak-canary", action="store_true",
                    help="span packs: deliberately leak future text (never recordable)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.self_test:
        return _self_test()
    if not (args.pack and args.source and args.target):
        ap.error("--pack, --source and --target are required")
    if args.leak_canary and args.record:
        ap.error("--leak-canary measures a deliberately leaking cache; never --record it")

    t0 = time.time()
    metrics = evaluate_acceptance(
        args.pack, source_id=args.source, target_id=args.target,
        corpus_roots=args.corpus, cuts=args.cuts, seq_len=args.seq_len,
        n_sequences=args.sequences, device=args.device, dtype=args.dtype,
        skip_docs=args.skip_docs, leak_canary=args.leak_canary,
    )
    if args.record:
        record_acceptance(args.pack, metrics)
    payload = asdict(metrics)
    payload["seconds"] = round(time.time() - t0, 1)
    payload["headroom_over_control"] = round(
        metrics.top1_agreement - (metrics.top1_control or 0.0), 4)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
