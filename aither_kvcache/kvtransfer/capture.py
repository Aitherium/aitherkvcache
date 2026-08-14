"""Capture aligned KV activations from a (source, target) model pair.

Runs both models over the SAME token ids, pulls the KV cache out of each, strips
each model's own RoPE from the keys, and writes flat ``(tokens, layers*width)``
arrays to disk.

## Why the capture is a separate, cached artifact

Fitting is cheap (Gram algebra); running two transformers over tens of thousands
of tokens is not. Persisting the capture means a re-fit with different top-k,
lambdas, or layer subsets costs seconds instead of re-running the models, which
is the difference between "we measured one configuration" and "we swept it".

## Alignment is asserted per sequence, not assumed from the digest

``geometry.check_pair`` refuses a tokenizer-digest mismatch, but a matching
digest is a claim about the vocabulary, not about what these two tokenizers did
to *this* string — chat templates, added tokens and normalisation can still
diverge. So every sequence is tokenized by BOTH tokenizers and the ids compared
exactly. A mismatch raises; it never truncates to the shorter one, because a
silently truncated pair produces a perfectly well-formed capture of misaligned
rows, which is the single failure mode that no downstream number can reveal.

## Splits are by DOCUMENT

Not by token (adjacent tokens are near-duplicates) and not even by sequence
(two 256-token windows of one document share topic, vocabulary and often literal
text). Held-out R^2 is only worth quoting if the validation documents were never
seen, so the split happens before chunking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import KVGeometry, check_pair
from .rope import strip_rope

logger = logging.getLogger("kvtransfer.capture")


@dataclass
class CaptureManifest:
    """What a capture directory contains, and what produced it."""

    source: Dict[str, Any]
    target: Dict[str, Any]
    seq_len: int
    n_train_tokens: int
    n_val_tokens: int
    n_train_docs: int
    n_val_docs: int
    corpus_sha256: str
    regime_flags: List[str]
    created_at: float
    dtype: str = "float16"

    def write(self, out_dir: Path) -> None:
        (out_dir / "capture.manifest.json").write_text(
            json.dumps(self.__dict__, indent=2), encoding="utf-8"
        )

    @classmethod
    def read(cls, out_dir: Path) -> "CaptureManifest":
        data = json.loads((out_dir / "capture.manifest.json").read_text(encoding="utf-8"))
        return cls(**data)


def _extract_kv(past: object) -> List[Tuple[Any, Any]]:
    """Pull (K, V) per layer out of whatever cache object this transformers ships.

    The cache API has changed repeatedly (tuples -> DynamicCache.key_cache ->
    Cache.layers). Each accessor is tried in turn and the FIRST that yields a
    non-empty list wins; running out raises rather than returning an empty list,
    because an empty capture would sail through every shape check downstream and
    fit a mapper on nothing.
    """
    to_legacy = getattr(past, "to_legacy_cache", None)
    if callable(to_legacy):
        legacy = list(to_legacy())
        if legacy:
            return [(k, v) for k, v in legacy]
    layers = getattr(past, "layers", None)
    if layers:
        got = [(lyr.keys, lyr.values) for lyr in layers
               if getattr(lyr, "keys", None) is not None]
        if got:
            return got
    keys = getattr(past, "key_cache", None)
    values = getattr(past, "value_cache", None)
    if keys and values:
        return list(zip(keys, values))
    if isinstance(past, (list, tuple)) and past:
        return [(k, v) for k, v in past]
    raise RuntimeError(
        f"cannot extract KV from cache object of type {type(past).__name__} — "
        "transformers changed its cache API; add an accessor rather than "
        "letting the capture come back empty"
    )


# Module leaf-names that produce an MLA cache. DeepSeek V2/V3/V4 all use these.
# Discovery is by module name rather than by config, because the config says a
# model IS latent while only the module tree says WHERE the latent is.
_MLA_LATENT_MODULES = ("kv_a_layernorm",)
_MLA_PROJ_MODULES = ("kv_a_proj_with_mqa",)


def _layer_index(qualified_name: str) -> Optional[int]:
    """Pull the layer number out of e.g. 'model.layers.7.self_attn.kv_a_layernorm'."""
    parts = qualified_name.split(".")
    for i, part in enumerate(parts):
        if part in ("layers", "h", "blocks") and i + 1 < len(parts):
            if parts[i + 1].isdigit():
                return int(parts[i + 1])
    return None


class LatentCapturer:
    """Capture an MLA model's cache by hooking the compression projection.

    ``past_key_values`` is the wrong source for a latent model. HuggingFace's
    DeepSeek modelling code MATERIALISES per-head keys and values before
    caching them, so reading the cache yields the expanded tensors an efficient
    engine never stores — 64 heads x 192 rather than one 512-dim latent. Fitting
    to that would produce a mapper that cannot drive a real vLLM or llama.cpp
    deployment, while looking entirely correct end to end.

    So the latent is taken where it is actually formed:

    * ``c``  — the output of ``kv_a_layernorm``, i.e. the compressed KV AFTER
      normalisation, which is precisely what gets cached.
    * ``kr`` — the tail of ``kv_a_proj_with_mqa``'s output, the decoupled RoPE
      key, sliced off after ``kv_lora_rank`` columns.

    **Both are position-free at this point**, because the hook fires BEFORE
    ``apply_rotary_pos_emb``. That is the opposite of the conventional path,
    where captured keys are post-rotation and must be stripped. Getting this
    backwards would strip a rotation that was never applied.

    Discovery is fail-closed: if the expected modules are not found on every
    layer, this raises rather than capturing a partial or mis-specified tensor.
    A model with only SOME latent layers (hybrid attention) is refused
    explicitly — silently capturing the ones that match would produce a cache
    with holes and no error.
    """

    def __init__(self, model: Any, geom: KVGeometry) -> None:
        self.geom = geom
        self._handles: List[Any] = []
        self._latent: Dict[int, Any] = {}
        self._rope: Dict[int, Any] = {}
        self._lat_rank = geom.role_width("c")

        latent_layers, proj_layers = set(), set()
        for name, module in model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            idx = _layer_index(name)
            if idx is None:
                continue
            if leaf in _MLA_LATENT_MODULES:
                latent_layers.add(idx)
                self._handles.append(
                    module.register_forward_hook(self._latent_hook(idx)))
            elif leaf in _MLA_PROJ_MODULES:
                proj_layers.add(idx)
                self._handles.append(
                    module.register_forward_hook(self._rope_hook(idx)))

        missing_latent = set(range(geom.n_layers)) - latent_layers
        missing_proj = set(range(geom.n_layers)) - proj_layers
        if missing_latent or missing_proj:
            self.close()
            raise RuntimeError(
                f"{geom.model_id}: latent capture found "
                f"{len(latent_layers)}/{geom.n_layers} {_MLA_LATENT_MODULES} and "
                f"{len(proj_layers)}/{geom.n_layers} {_MLA_PROJ_MODULES}. A partial "
                "match means either a hybrid attention stack or a different MLA "
                "implementation; capturing only the matching layers would yield a "
                "cache with holes and no error."
            )

    def _latent_hook(self, idx: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            self._latent[idx] = output.detach()
        return hook

    def _rope_hook(self, idx: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            # (batch, seq, kv_lora_rank + qk_rope_head_dim) -> keep the tail.
            self._rope[idx] = output.detach()[..., self._lat_rank:]
        return hook

    def collect(self) -> Dict[str, np.ndarray]:
        """Return {'c': (b*s, layers*rank), 'kr': (b*s, layers*rope_dim)} float16."""
        out: Dict[str, np.ndarray] = {}
        for role, store in (("c", self._latent), ("kr", self._rope)):
            width = self.geom.role_width(role)
            per_layer = []
            for idx in range(self.geom.n_layers):
                if idx not in store:
                    raise RuntimeError(
                        f"{self.geom.model_id}: no {role!r} captured for layer {idx} — "
                        "the forward pass did not reach every layer"
                    )
                arr = store[idx].to("cpu").float().numpy()
                if arr.shape[-1] != width:
                    raise RuntimeError(
                        f"{self.geom.model_id} layer {idx} role {role!r}: captured "
                        f"width {arr.shape[-1]}, geometry says {width}"
                    )
                per_layer.append(arr.reshape(-1, width))
            out[role] = np.concatenate(per_layer, axis=1).astype(np.float16)
        return out

    def reset(self) -> None:
        self._latent.clear()
        self._rope.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _stack_layers(
    kv_pairs: Sequence[Tuple[Any, Any]],
    geom: KVGeometry,
    positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """(K, V) -> two (batch*seq, n_layers*width) float16 arrays, K de-rotated.

    Input tensors are (batch, n_kv_heads, seq, head_dim), which is the layout
    every HF attention implementation writes.
    """
    if len(kv_pairs) != geom.n_layers:
        raise RuntimeError(
            f"{geom.model_id}: captured {len(kv_pairs)} layers, config says {geom.n_layers}"
        )
    spec = geom.rope_spec()
    k_out: List[np.ndarray] = []
    v_out: List[np.ndarray] = []
    for layer_idx, (k_t, v_t) in enumerate(kv_pairs):
        k = k_t.detach().to("cpu", dtype=None).float().numpy()
        v = v_t.detach().to("cpu", dtype=None).float().numpy()
        if k.shape[1] != geom.n_kv_heads or k.shape[-1] != geom.head_dim:
            raise RuntimeError(
                f"{geom.model_id} layer {layer_idx}: captured K shape {k.shape} "
                f"contradicts geometry (n_kv_heads={geom.n_kv_heads}, "
                f"head_dim={geom.head_dim})"
            )
        # (b, h, s, d) -> de-rotate along the position axis -> (b, s, h*d)
        k_unrot = strip_rope(k, spec, positions)
        k_out.append(np.ascontiguousarray(k_unrot.transpose(0, 2, 1, 3)).reshape(
            k.shape[0], k.shape[2], -1))
        v_out.append(np.ascontiguousarray(v.transpose(0, 2, 1, 3)).reshape(
            v.shape[0], v.shape[2], -1))
    # (layers, b, s, w) -> (b*s, layers*w) with layer-major columns, which is the
    # layout mapper._block_columns assumes.
    k_arr = np.stack(k_out, axis=2)   # (b, s, layers, w)
    v_arr = np.stack(v_out, axis=2)
    b, s = k_arr.shape[0], k_arr.shape[1]
    return (
        k_arr.reshape(b * s, -1).astype(np.float16),
        v_arr.reshape(b * s, -1).astype(np.float16),
    )


def gather_corpus(roots: Sequence[Path], suffixes: Sequence[str], limit: int) -> List[str]:
    """Read up to ``limit`` documents from the tree, deterministically ordered."""
    docs: List[str] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if len(docs) >= limit:
                break
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if any(part in {".git", "node_modules", "__pycache__", ".next", ".worktrees"}
                   for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="strict")
            except (OSError, UnicodeDecodeError):
                continue
            if len(text) < 2000 or text in seen:
                continue
            seen.add(text)
            docs.append(text)
    return docs


def _chunk(ids: List[int], seq_len: int) -> List[List[int]]:
    n = len(ids) // seq_len
    return [ids[i * seq_len:(i + 1) * seq_len] for i in range(n)]


def capture_pair(
    source_id: str,
    target_id: str,
    out_dir: Path,
    *,
    corpus_roots: Sequence[Path],
    seq_len: int = 256,
    train_seqs: int = 64,
    val_seqs: int = 24,
    batch_size: int = 4,
    device: str = "cpu",
    dtype: str = "float32",
    max_docs: int = 400,
) -> CaptureManifest:
    """Run both models over one aligned corpus and persist their KV."""
    import torch  # local: this module is importable without torch for the gate
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = getattr(torch, dtype)

    tok_src = AutoTokenizer.from_pretrained(source_id)
    tok_tgt = AutoTokenizer.from_pretrained(target_id)
    cfg_src = AutoConfig.from_pretrained(source_id)
    cfg_tgt = AutoConfig.from_pretrained(target_id)
    geom_src = KVGeometry.from_hf(source_id, cfg_src, tok_src, dtype)
    geom_tgt = KVGeometry.from_hf(target_id, cfg_tgt, tok_tgt, dtype)

    verdict = check_pair(geom_src, geom_tgt)
    verdict.raise_if_blocked()
    for flag in verdict.regime_flags:
        logger.warning("[regime] %s", flag)

    docs = gather_corpus(corpus_roots, (".md", ".py", ".txt"), max_docs)
    if not docs:
        raise RuntimeError(f"no corpus documents found under {[str(r) for r in corpus_roots]}")

    # Document-level split BEFORE chunking — see the module docstring.
    n_val_docs = max(1, len(docs) // 4)
    val_docs, train_docs = docs[:n_val_docs], docs[n_val_docs:]

    def build(doc_list: List[str], want: int) -> List[List[int]]:
        seqs: List[List[int]] = []
        for doc in doc_list:
            if len(seqs) >= want:
                break
            ids_src = tok_src(doc, add_special_tokens=False)["input_ids"]
            ids_tgt = tok_tgt(doc, add_special_tokens=False)["input_ids"]
            if ids_src != ids_tgt:
                raise RuntimeError(
                    "tokenizers disagree on a corpus document despite matching vocab "
                    f"digests ({len(ids_src)} vs {len(ids_tgt)} ids). Row i of the "
                    "source cache would not be row i of the target cache."
                )
            seqs.extend(_chunk(ids_src, seq_len))
        return seqs[:want]

    train_ids = build(train_docs, train_seqs)
    val_ids = build(val_docs, val_seqs)
    if len(train_ids) < train_seqs or len(val_ids) < val_seqs:
        raise RuntimeError(
            f"corpus too small: got {len(train_ids)}/{train_seqs} train and "
            f"{len(val_ids)}/{val_seqs} val sequences of {seq_len} tokens"
        )

    corpus_hash = hashlib.sha256()
    for seq in train_ids + val_ids:
        corpus_hash.update(np.asarray(seq, dtype=np.int32).tobytes())

    positions = np.arange(seq_len, dtype=np.int64)
    # Role-driven rather than a hardcoded k/v pair: a latent model's cache is
    # ("c", "kr") of different widths, so the store layout comes from geometry.
    widths = {
        (which, role): geom.n_layers * geom.role_width(role)
        for which, geom in (("src", geom_src), ("tgt", geom_tgt))
        for role in geom.roles()
    }

    for split, seqs in (("train", train_ids), ("val", val_ids)):
        n_tok = len(seqs) * seq_len
        stores = {
            key: np.lib.format.open_memmap(
                out_dir / f"{split}_{key[0]}_{key[1]}.npy", mode="w+",
                dtype=np.float16, shape=(n_tok, width),
            )
            for key, width in widths.items()
        }
        for which, model_id, geom in (
            ("src", source_id, geom_src), ("tgt", target_id, geom_tgt),
        ):
            t0 = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch_dtype, attn_implementation="eager",
            ).to(device).eval()
            # A latent model's cache is not in past_key_values (HF materialises
            # per-head K/V before caching); it is hooked at the compression
            # projection. See LatentCapturer.
            capturer = LatentCapturer(model, geom) if geom.is_latent else None
            cursor = 0
            for start in range(0, len(seqs), batch_size):
                batch = seqs[start:start + batch_size]
                ids = torch.tensor(batch, dtype=torch.long, device=device)
                if capturer is not None:
                    capturer.reset()
                with torch.no_grad():
                    out = model(ids, use_cache=not geom.is_latent)
                if capturer is not None:
                    blocks = capturer.collect()
                else:
                    k_flat, v_flat = _stack_layers(_extract_kv(out.past_key_values),
                                                   geom, positions)
                    blocks = {"k": k_flat, "v": v_flat}
                rows = next(iter(blocks.values())).shape[0]
                for role, arr in blocks.items():
                    stores[(which, role)][cursor:cursor + rows] = arr
                cursor += rows
                logger.info("[%s/%s] %d/%d tokens", split, which, cursor, n_tok)
            if cursor != n_tok:
                raise RuntimeError(f"{which}: wrote {cursor} rows, expected {n_tok}")
            if capturer is not None:
                capturer.close()
            del model
            logger.info("[%s/%s] done in %.1fs", split, which, time.time() - t0)
        for store in stores.values():
            store.flush()
        del stores

    manifest = CaptureManifest(
        source=geom_src.to_dict(),
        target=geom_tgt.to_dict(),
        seq_len=seq_len,
        n_train_tokens=len(train_ids) * seq_len,
        n_val_tokens=len(val_ids) * seq_len,
        n_train_docs=len(train_docs),
        n_val_docs=len(val_docs),
        corpus_sha256=corpus_hash.hexdigest(),
        regime_flags=verdict.regime_flags,
        created_at=time.time(),
    )
    manifest.write(out_dir)
    return manifest


def _self_test() -> int:
    """Exercise the parts that do not need weights on disk."""
    failures: List[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(f"{name}: {detail}")

    class _FakeTensor:
        def __init__(self, arr: np.ndarray) -> None:
            self._a = arr

        def detach(self) -> "_FakeTensor":
            return self

        def to(self, *_a: Any, **_k: Any) -> "_FakeTensor":
            return self

        def float(self) -> "_FakeTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self._a

    geom = KVGeometry(
        model_id="fake", family="llama", n_layers=3, n_kv_heads=2, head_dim=4,
        hidden_size=8, rope_theta=10000.0, rope_type="default",
        rope_scaling_factor=1.0, tokenizer_sha256="c" * 64, torch_dtype="float32",
    )
    rng = np.random.default_rng(7)
    pairs = [(_FakeTensor(rng.standard_normal((2, 2, 5, 4))),
              _FakeTensor(rng.standard_normal((2, 2, 5, 4)))) for _ in range(3)]
    k_flat, v_flat = _stack_layers(pairs, geom, np.arange(5))
    check("stack_shape", k_flat.shape == (10, 24), f"got {k_flat.shape}")

    # Column layout must be layer-major: layer 1's block starts at column 8.
    # If this is wrong, every source-layer selection points at the wrong data
    # and the only symptom is a mediocre R^2.
    v_layer1 = pairs[1][1].numpy().transpose(0, 2, 1, 3).reshape(10, 8)
    check("layer_major_columns",
          np.allclose(v_flat[:, 8:16].astype(np.float32), v_layer1, atol=1e-2),
          "layer 1 is not at columns 8:16 — block layout disagrees with mapper")

    # Layer-count disagreement must raise, not silently truncate.
    try:
        _stack_layers(pairs[:2], geom, np.arange(5))
        check("layer_count_checked", False, "accepted 2 layers for a 3-layer config")
    except RuntimeError as exc:
        check("layer_count_reason", "config says" in str(exc),
              f"refused, but not for the layer count: {exc}")

    # Head geometry disagreement must raise.
    bad = [(_FakeTensor(rng.standard_normal((2, 5, 5, 4))),
            _FakeTensor(rng.standard_normal((2, 5, 5, 4)))) for _ in range(3)]
    try:
        _stack_layers(bad, geom, np.arange(5))
        check("head_count_checked", False, "accepted 5 KV heads for a 2-head config")
    except RuntimeError as exc:
        check("head_count_reason", "contradicts geometry" in str(exc),
              f"refused, but not for the head geometry: {exc}")

    # An unrecognised cache object must raise rather than yield nothing.
    try:
        _extract_kv(object())
        check("empty_cache_refused", False, "returned from an unknown cache type")
    except RuntimeError as exc:
        check("empty_cache_reason", "cannot extract KV" in str(exc),
              f"refused, but not for an unreadable cache: {exc}")

    check("chunking", _chunk(list(range(10)), 4) == [[0, 1, 2, 3], [4, 5, 6, 7]],
          "chunker kept a short trailing window")

    # ── latent (MLA) capture ────────────────────────────────────────────────
    # Exercised against a synthetic module tree with DeepSeek's names, because
    # the real model is 304B and cannot run here. What this pins is the part
    # that would silently misbehave: WHICH modules are hooked, that every layer
    # is covered, and that the rope key is sliced off after the latent rank.
    check("layer_index", _layer_index("model.layers.7.self_attn.kv_a_layernorm") == 7,
          f"got {_layer_index('model.layers.7.self_attn.kv_a_layernorm')}")
    check("layer_index_none", _layer_index("model.embed_tokens") is None,
          "found a layer index where there is none")

    try:
        import torch
        from torch import nn
    except ImportError:
        torch = None  # type: ignore[assignment]

    if torch is not None:
        rank, rope_dim, n_layers = 8, 4, 3
        lat_geom = KVGeometry(
            model_id="fake-mla", family="deepseek_v4", n_layers=n_layers,
            n_kv_heads=1, head_dim=rank, hidden_size=16, rope_theta=10000.0,
            rope_type="default", rope_scaling_factor=1.0,
            tokenizer_sha256="d" * 64, torch_dtype="float32",
            latent_attention="MLA", rope_key_dim=rope_dim,
        )

        class FakeAttn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.kv_a_proj_with_mqa = nn.Linear(16, rank + rope_dim)
                self.kv_a_layernorm = nn.LayerNorm(rank)

            def forward(self, x):
                proj = self.kv_a_proj_with_mqa(x)
                return self.kv_a_layernorm(proj[..., :rank])

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([FakeAttn() for _ in range(n_layers)])

            def forward(self, x):
                for layer in self.layers:
                    layer(x)
                return x

        model = FakeModel().eval()
        cap = LatentCapturer(model, lat_geom)
        with torch.no_grad():
            model(torch.randn(2, 5, 16))
        blocks = cap.collect()
        check("latent_roles_captured", set(blocks) == {"c", "kr"}, f"got {set(blocks)}")
        check("latent_c_shape", blocks["c"].shape == (10, n_layers * rank),
              f"got {blocks['c'].shape}")
        check("latent_kr_shape", blocks["kr"].shape == (10, n_layers * rope_dim),
              f"got {blocks['kr'].shape}")

        # The rope key must be the TAIL of the projection, not the head. Getting
        # this slice backwards yields correctly-shaped tensors of the wrong
        # content — the latent would be fitted as the rope key and vice versa.
        with torch.no_grad():
            x = torch.randn(1, 3, 16)
            expect_kr = model.layers[0].kv_a_proj_with_mqa(x)[..., rank:]
            cap.reset()
            model(x)
            got = cap.collect()["kr"][:, :rope_dim]
        check("latent_kr_is_tail",
              np.allclose(got.astype(np.float32),
                          expect_kr.reshape(-1, rope_dim).numpy(), atol=1e-2),
              "kr is not the tail of kv_a_proj_with_mqa — the slice is wrong")
        cap.close()

        # Fail closed when the module tree does not match on every layer.
        partial = FakeModel()
        del partial.layers[2].kv_a_layernorm
        try:
            LatentCapturer(partial, lat_geom)
            check("partial_tree_refused", False,
                  "accepted a model with latent modules on only some layers")
        except RuntimeError as exc:
            check("partial_tree_reason", "latent capture found" in str(exc),
                  f"refused for the wrong reason: {exc}")

    for f in failures:
        print(f"FAIL {f}")
    print(f"capture self-test: {'PASS' if not failures else 'FAIL'} ({len(failures)} failures)")
    return 1 if failures else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Capture aligned KV from a model pair")
    ap.add_argument("--source")
    ap.add_argument("--target")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--corpus", type=Path, nargs="*", default=[Path(".")])
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--train-seqs", type=int, default=64)
    ap.add_argument("--val-seqs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.self_test:
        return _self_test()
    if not (args.source and args.target and args.out):
        ap.error("--source, --target and --out are required")
    manifest = capture_pair(
        args.source, args.target, args.out, corpus_roots=args.corpus,
        seq_len=args.seq_len, train_seqs=args.train_seqs, val_seqs=args.val_seqs,
        batch_size=args.batch_size, device=args.device, dtype=args.dtype,
    )
    print(json.dumps({
        "train_tokens": manifest.n_train_tokens,
        "val_tokens": manifest.n_val_tokens,
        "regime_flags": manifest.regime_flags,
    }, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
