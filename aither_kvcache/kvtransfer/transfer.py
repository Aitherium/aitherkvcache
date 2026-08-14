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
) -> AcceptanceMetrics:
    """Measure what the target model does with a translated cache.

    Held-out by construction: ``skip_docs`` moves past the documents used for
    fitting, so this never scores the mapper on text it was trained on.
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

    docs = gather_corpus(corpus_roots, (".md", ".py", ".txt"), skip_docs + n_sequences * 4)
    docs = docs[skip_docs:]
    seqs: List[List[int]] = []
    for doc in docs:
        if len(seqs) >= n_sequences:
            break
        ids = tok(doc, add_special_tokens=False)["input_ids"]
        if len(ids) >= seq_len:
            seqs.append(ids[:seq_len])
    if len(seqs) < 2:
        raise RuntimeError(
            f"need at least 2 eval sequences of {seq_len} tokens, got {len(seqs)}"
        )

    positions = np.arange(seq_len, dtype=np.int64)
    src_model = AutoModelForCausalLM.from_pretrained(
        source_id, dtype=torch_dtype, attn_implementation="eager").to(device).eval()
    translated: List[List[Tuple[np.ndarray, np.ndarray]]] = []
    for seq in seqs:
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        with torch.no_grad():
            out = src_model(ids, use_cache=True)
        k_flat, v_flat = _stack_layers(_extract_kv(out.past_key_values), geom_src, positions)
        translated.append(translate_cache(k_flat.astype(np.float32),
                                          v_flat.astype(np.float32), pack, positions))
    del src_model

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
                "translated": _slice_layers(translated[idx], cut - 1),
                "control": _slice_layers(translated[control_idx], cut - 1),
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

    del tgt_model
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
        notes=f"cuts={list(cuts)} seq_len={seq_len} skip_docs={skip_docs}",
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
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.self_test:
        return _self_test()
    if not (args.pack and args.source and args.target):
        ap.error("--pack, --source and --target are required")

    t0 = time.time()
    metrics = evaluate_acceptance(
        args.pack, source_id=args.source, target_id=args.target,
        corpus_roots=args.corpus, cuts=args.cuts, seq_len=args.seq_len,
        n_sequences=args.sequences, device=args.device, dtype=args.dtype,
        skip_docs=args.skip_docs,
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
