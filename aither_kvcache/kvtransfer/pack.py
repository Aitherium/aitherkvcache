"""The mapper pack artifact — and the fail-closed rules for loading one.

A pack is weights plus a plaintext sidecar manifest — the same shape a
content-addressed KV blob store uses, and for the same reason: **a blob whose
provenance cannot be verified is unsafe, not merely unknown.**

The failure this guards is worse than the one the KV connector guards. Restoring
a mismatched KV blob at least produces obvious garbage fairly quickly. A mapper
fitted on the wrong pair, or on too little data, or against a different revision
of the target, produces a cache that is *plausible*: the target model decodes
fluent, on-topic, confidently wrong text. There is no exception, no 500, and no
degraded-looking output. So every one of these is a refusal, never a warning:

* **KVT001** no manifest, or unparseable -> unverifiable -> refuse.
* **KVT002** no held-out reconstruction metrics -> an unvalidated mapper. The
  fit always "succeeds"; R^2 is the only thing that distinguishes a map from a
  random projection.
* **KVT003** no DOWNSTREAM acceptance evidence, or below the declared floor.
  Reconstruction R^2 is necessary and not sufficient — 79% of key variance is
  compatible with a badly degraded model, so the pack must carry a measurement
  of what the receiving model actually does with the translated cache.
* **KVT007** acceptance measured over too few positions. A mean over 24 samples
  passes KVT003 exactly as well as a mean over 24,000, and is noise.
* **KVT004** geometry disagreement with the live models (layers, heads, head_dim,
  rope) -> the weights are shaped for a different model.
* **KVT005** tokenizer digest disagreement -> positions do not correspond.
* **KVT006** weight digest disagreement -> the blob was edited or truncated.

``load_pack`` requires the caller to pass the live source and target geometry.
There is deliberately no "just load it" path: a pack loaded without anything to
check it against is the exact state KVT004/005 exist to prevent, and an optional
argument would become the default within a week.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .geometry import KVGeometry
from .mapper import LayerMap

logger = logging.getLogger("kvtransfer.pack")

PACK_FORMAT_VERSION = "1"

# Defaults for the two downstream gates. Deliberately conservative: the cost of
# a false refusal is one prefill, the cost of a false accept is a confidently
# wrong answer with no error anywhere.
DEFAULT_MIN_TOP1_AGREEMENT = 0.80
DEFAULT_MAX_NLL_DELTA = 0.25   # nats/token above the target's own prefill

# KVT007. A measurement taken over too few positions is not weak evidence, it is
# NOISE WEARING THE COSTUME OF EVIDENCE — and it passes KVT003 exactly as well as
# a real one, because KVT003 only reads the mean.
#
# This is not hypothetical: the first pack fitted in this repo recorded its
# acceptance over **24 positions**. Top-1 agreement is a proportion, so its
# standard error there is sqrt(0.5*0.5/24) ~= 0.10 — ten percentage points. A
# pack that truly sits at 0.65 clears a 0.80 floor by luck often enough to
# matter, and once recorded the number is indistinguishable from one taken over
# 10,000 positions. 384 keeps the standard error near 2.5pp.
DEFAULT_MIN_ACCEPTANCE_POSITIONS = 384


class PackRefusedError(Exception):
    """A pack was rejected. The message always names the rule and the reason."""


@dataclass
class AcceptanceMetrics:
    """What the receiving model actually does with a translated cache.

    ``top1_agreement`` — fraction of held-out positions where the target model,
    decoding from the TRANSLATED cache, predicts the same next token as it does
    from its OWN prefill. Agreement with the target's own behaviour, not with
    ground truth: the mapper's job is to reproduce the target model, and a
    mapper that "improves" accuracy is a mapper that changed the model.

    ``nll_delta`` — mean next-token NLL from the translated cache minus the same
    from a real prefill, in nats. Positive is worse. This catches the case
    top-1 cannot: unchanged argmax with a badly flattened distribution.

    ``top1_control`` / ``top1_nocontext`` are the arms that make the headline
    interpretable. A mapper that emits pure noise still scores well above zero
    on agreement, because a language model's next token is often forced by the
    last few tokens alone regardless of what the cache holds. Without a floor to
    compare against, "68% agreement" is unreadable — it could be excellent or it
    could be exactly what a dead mapper scores. ``control`` translates from a
    DIFFERENT sequence (so the map runs, on the wrong content) and
    ``nocontext`` gives the target no cache at all.
    """

    top1_agreement: float
    nll_translated: float
    nll_reference: float
    nll_delta: float
    n_positions: int
    eval_sequences: int
    top1_control: Optional[float] = None
    nll_control: Optional[float] = None
    top1_nocontext: Optional[float] = None
    nll_nocontext: Optional[float] = None
    notes: str = ""


@dataclass
class PackManifest:
    """Everything needed to decide whether these weights may be used."""

    format_version: str
    source: Dict[str, Any]
    target: Dict[str, Any]
    top_k: int
    weight_dtype: str
    weights_sha256: str
    val_r2_k: float
    val_r2_v: float
    per_layer: Dict[str, Any]
    regime_flags: List[str]
    train_tokens: int
    val_tokens: int
    corpus_sha256: str
    created_at: float
    min_top1_agreement: float = DEFAULT_MIN_TOP1_AGREEMENT
    max_nll_delta: float = DEFAULT_MAX_NLL_DELTA
    min_acceptance_positions: int = DEFAULT_MIN_ACCEPTANCE_POSITIONS
    acceptance: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _weights_path(pack_dir: Path) -> Path:
    return pack_dir / "mapper.npz"


def _manifest_path(pack_dir: Path) -> Path:
    return pack_dir / "mapper.manifest.json"


def _digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_pack(
    pack_dir: Path,
    maps: Dict[str, List[LayerMap]],
    *,
    source: KVGeometry,
    target: KVGeometry,
    top_k: int,
    regime_flags: List[str],
    train_tokens: int,
    val_tokens: int,
    corpus_sha256: str,
    created_at: float,
    weight_dtype: str = "float16",
) -> PackManifest:
    """Persist fitted maps plus the manifest that makes them loadable."""
    pack_dir.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    per_layer: Dict[str, Any] = {}
    np_dtype = np.dtype(weight_dtype)

    for role, layer_maps in maps.items():
        for lm in layer_maps:
            arrays[f"{role}.{lm.target_layer}.W"] = lm.weight.astype(np_dtype)
            arrays[f"{role}.{lm.target_layer}.b"] = lm.bias.astype(np_dtype)
            arrays[f"{role}.{lm.target_layer}.src"] = np.asarray(
                lm.source_layers, dtype=np.int32)
            per_layer[f"{role}.{lm.target_layer}"] = {
                "source_layers": lm.source_layers,
                "lam_rel": lm.lam_rel,
                "val_r2_mean": lm.val_r2_mean,
                "val_r2_per_head": lm.val_r2_per_head,
                "best_single_source_r2": (
                    max(lm.candidate_r2.values()) if lm.candidate_r2 else None),
            }

    np.savez(_weights_path(pack_dir), **arrays)

    def mean_r2(role: str) -> float:
        vals = [lm.val_r2_mean for lm in maps.get(role, [])]
        return float(np.mean(vals)) if vals else float("nan")

    manifest = PackManifest(
        format_version=PACK_FORMAT_VERSION,
        source=source.to_dict(),
        target=target.to_dict(),
        top_k=top_k,
        weight_dtype=weight_dtype,
        weights_sha256=_digest_file(_weights_path(pack_dir)),
        val_r2_k=mean_r2("k"),
        val_r2_v=mean_r2("v"),
        per_layer=per_layer,
        regime_flags=regime_flags,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        corpus_sha256=corpus_sha256,
        created_at=created_at,
    )
    _manifest_path(pack_dir).write_text(manifest.to_json(), encoding="utf-8")
    return manifest


def record_acceptance(pack_dir: Path, metrics: AcceptanceMetrics) -> PackManifest:
    """Attach downstream evidence to an existing pack and re-digest it."""
    manifest = read_manifest(pack_dir)
    manifest.acceptance = asdict(metrics)
    _manifest_path(pack_dir).write_text(manifest.to_json(), encoding="utf-8")
    return manifest


def read_manifest(pack_dir: Path) -> PackManifest:
    path = _manifest_path(pack_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackRefusedError(f"KVT001 manifest unreadable at {path}: {exc}") from exc
    known = set(PackManifest.__annotations__)
    unknown = set(data) - known
    if unknown:
        raise PackRefusedError(
            f"KVT001 manifest has unknown fields {sorted(unknown)} — written by a "
            "different format version"
        )
    try:
        return PackManifest(**data)
    except TypeError as exc:
        raise PackRefusedError(f"KVT001 manifest incomplete: {exc}") from exc


@dataclass
class LoadedPack:
    """Verified weights, ready to apply."""

    manifest: PackManifest
    source: KVGeometry
    target: KVGeometry
    weights: Dict[str, np.ndarray] = field(repr=False, default_factory=dict)

    def layer_map(self, role: str, layer: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        try:
            w = self.weights[f"{role}.{layer}.W"]
            b = self.weights[f"{role}.{layer}.b"]
            src = self.weights[f"{role}.{layer}.src"]
        except KeyError as exc:
            raise PackRefusedError(
                f"KVT004 pack has no map for role={role!r} layer={layer}"
            ) from exc
        return w, b, [int(x) for x in src]


def _geometry_conflict(stored: KVGeometry, live: KVGeometry, side: str) -> Optional[str]:
    """First field that makes the weights wrong for this model, or None.

    ``model_id`` and ``torch_dtype`` are deliberately NOT compared: the same
    weights are served from different paths and at different precisions, and
    refusing on those produces false misses that push people to disable the
    check. What must match is the KV geometry and the rotation schedule.
    """
    for field_name in (
        "n_layers", "n_kv_heads", "head_dim", "hidden_size",
        "rope_theta", "rope_type", "rope_scaling_factor",
    ):
        a, b = getattr(stored, field_name), getattr(live, field_name)
        if a != b:
            return f"KVT004 {side} {field_name}: pack={a!r} live={b!r}"
    return None


def load_pack(
    pack_dir: Path,
    *,
    live_source: KVGeometry,
    live_target: KVGeometry,
    min_top1_agreement: Optional[float] = None,
    max_nll_delta: Optional[float] = None,
    min_acceptance_positions: Optional[int] = None,
    require_acceptance: bool = True,
) -> LoadedPack:
    """Load a pack, refusing on any rule. Never returns a partially-checked pack."""
    manifest = read_manifest(pack_dir)

    if manifest.format_version != PACK_FORMAT_VERSION:
        raise PackRefusedError(
            f"KVT001 pack format {manifest.format_version!r}, this build reads "
            f"{PACK_FORMAT_VERSION!r}"
        )

    stored_src = KVGeometry.from_dict(manifest.source)
    stored_tgt = KVGeometry.from_dict(manifest.target)

    if stored_src.tokenizer_sha256 != live_source.tokenizer_sha256:
        raise PackRefusedError("KVT005 source tokenizer digest differs from the pack's")
    if stored_tgt.tokenizer_sha256 != live_target.tokenizer_sha256:
        raise PackRefusedError("KVT005 target tokenizer digest differs from the pack's")
    for conflict in (
        _geometry_conflict(stored_src, live_source, "source"),
        _geometry_conflict(stored_tgt, live_target, "target"),
    ):
        if conflict:
            raise PackRefusedError(conflict)

    for role, value in (("k", manifest.val_r2_k), ("v", manifest.val_r2_v)):
        if value is None or not np.isfinite(value):
            raise PackRefusedError(
                f"KVT002 pack carries no finite held-out R^2 for role {role!r} — "
                "an unvalidated mapper is indistinguishable from a random projection"
            )

    if require_acceptance:
        acc = manifest.acceptance
        if not acc:
            raise PackRefusedError(
                "KVT003 pack carries no downstream acceptance evidence. Held-out R^2 "
                "measures reconstruction, not whether the receiving model still works."
            )
        floor = (min_top1_agreement if min_top1_agreement is not None
                 else manifest.min_top1_agreement)
        ceiling = (max_nll_delta if max_nll_delta is not None
                   else manifest.max_nll_delta)
        # KVT007 BEFORE the thresholds, deliberately: checking a mean first and
        # its sample size second would admit a lucky small-n pack whenever the
        # mean happens to clear, which is the whole failure this rule closes.
        min_positions = min_acceptance_positions or manifest.min_acceptance_positions
        got_positions = int(acc.get("n_positions", 0) or 0)
        if got_positions < min_positions:
            raise PackRefusedError(
                f"KVT007 acceptance measured over {got_positions} positions, floor is "
                f"{min_positions}. Top-1 agreement is a proportion; at n={got_positions} "
                f"its standard error is ~{(0.25 / max(got_positions, 1)) ** 0.5:.3f}, so "
                "the mean cannot distinguish a passing pack from a failing one."
            )

        got_top1 = float(acc.get("top1_agreement", float("nan")))
        got_delta = float(acc.get("nll_delta", float("inf")))
        if not np.isfinite(got_top1) or got_top1 < floor:
            raise PackRefusedError(
                f"KVT003 top-1 agreement {got_top1:.4f} below floor {floor:.4f}"
            )
        if not np.isfinite(got_delta) or got_delta > ceiling:
            raise PackRefusedError(
                f"KVT003 NLL delta {got_delta:+.4f} nats exceeds ceiling {ceiling:.4f}"
            )

    weights_file = _weights_path(pack_dir)
    if not weights_file.exists():
        raise PackRefusedError(f"KVT006 weights missing at {weights_file}")
    actual = _digest_file(weights_file)
    if actual != manifest.weights_sha256:
        raise PackRefusedError(
            f"KVT006 weight digest mismatch: manifest={manifest.weights_sha256[:16]} "
            f"file={actual[:16]}"
        )

    with np.load(weights_file) as data:
        weights = {k: np.asarray(data[k]) for k in data.files}

    expected = stored_tgt.n_layers * 3 * 2   # W, b, src for each role and layer
    if len(weights) != expected:
        raise PackRefusedError(
            f"KVT006 pack holds {len(weights)} arrays, expected {expected} for a "
            f"{stored_tgt.n_layers}-layer target"
        )
    return LoadedPack(manifest=manifest, source=stored_src, target=stored_tgt,
                      weights=weights)


# ── self-test ───────────────────────────────────────────────────────────────


def _self_test() -> int:
    import shutil
    import tempfile
    import time

    failures: List[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(f"{name}: {detail}")

    def refuses(name: str, rule: str, fn: Any) -> None:
        try:
            fn()
            failures.append(f"{name}: accepted a pack that violates {rule}")
        except PackRefusedError as exc:
            if rule not in str(exc):
                failures.append(f"{name}: refused with {exc} (expected {rule})")

    def geom(model_id: str, layers: int, kv: int, tok: str) -> KVGeometry:
        return KVGeometry(
            model_id=model_id, family="llama", n_layers=layers, n_kv_heads=kv,
            head_dim=4, hidden_size=16, rope_theta=10000.0, rope_type="default",
            rope_scaling_factor=1.0, tokenizer_sha256=tok, torch_dtype="float32",
        )

    src = geom("small", 3, 2, "d" * 64)
    tgt = geom("big", 2, 4, "d" * 64)
    tmp = Path(tempfile.mkdtemp(prefix="kvtpack"))
    try:
        maps = {
            role: [
                LayerMap(target_layer=t, role=role, source_layers=[0, 1], lam_rel=1e-3,
                         weight=np.full((2 * 2 * 4, 4 * 4), 0.01, dtype=np.float32),
                         bias=np.zeros(16, dtype=np.float32), val_r2_mean=0.8,
                         val_r2_per_head=[0.8] * 4)
                for t in range(2)
            ]
            for role in ("k", "v")
        }
        write_pack(tmp, maps, source=src, target=tgt, top_k=2, regime_flags=[],
                   train_tokens=1000, val_tokens=200, corpus_sha256="e" * 64,
                   created_at=time.time())

        # No acceptance evidence yet -> KVT003.
        refuses("no_acceptance", "KVT003",
                lambda: load_pack(tmp, live_source=src, live_target=tgt))

        # ...but loadable when acceptance is explicitly not required.
        loaded = load_pack(tmp, live_source=src, live_target=tgt,
                           require_acceptance=False)
        check("loads_without_gate", loaded.manifest.top_k == 2)
        w, b, s = loaded.layer_map("k", 0)
        check("layer_map_shapes", w.shape == (16, 16) and b.shape == (16,) and s == [0, 1],
              f"{w.shape} {b.shape} {s}")

        record_acceptance(tmp, AcceptanceMetrics(
            top1_agreement=0.91, nll_translated=2.10, nll_reference=2.00,
            nll_delta=0.10, n_positions=500, eval_sequences=20))
        check("loads_with_acceptance",
              load_pack(tmp, live_source=src, live_target=tgt) is not None)

        # Below the floor -> refuse.
        record_acceptance(tmp, AcceptanceMetrics(
            top1_agreement=0.42, nll_translated=3.9, nll_reference=2.0,
            nll_delta=1.9, n_positions=500, eval_sequences=20))
        refuses("below_floor", "KVT003",
                lambda: load_pack(tmp, live_source=src, live_target=tgt))
        record_acceptance(tmp, AcceptanceMetrics(
            top1_agreement=0.91, nll_translated=2.10, nll_reference=2.00,
            nll_delta=0.10, n_positions=500, eval_sequences=20))

        # KVT007: a mean over too few positions is not evidence.
        record_acceptance(tmp, AcceptanceMetrics(
            top1_agreement=0.99, nll_translated=2.0, nll_reference=1.99,
            nll_delta=0.01, n_positions=24, eval_sequences=2))
        refuses("small_n_acceptance", "KVT007",
                lambda: load_pack(tmp, live_source=src, live_target=tgt))

        # MUTATION GUARD: the SAME excellent metrics with enough positions must
        # load. Without this, a KVT007 that refuses on any value at all passes
        # the check above while being a blanket denial.
        record_acceptance(tmp, AcceptanceMetrics(
            top1_agreement=0.99, nll_translated=2.0, nll_reference=1.99,
            nll_delta=0.01, n_positions=500, eval_sequences=20))
        check("large_n_acceptance_admits",
              load_pack(tmp, live_source=src, live_target=tgt) is not None,
              "identical metrics with 500 positions were still refused")

        # And KVT007 must be checked BEFORE the thresholds, so a lucky small-n
        # pack cannot slip through on its mean.
        record_acceptance(tmp, AcceptanceMetrics(
            top1_agreement=0.99, nll_translated=2.0, nll_reference=1.99,
            nll_delta=0.01, n_positions=10, eval_sequences=1))
        refuses("small_n_beats_thresholds", "KVT007",
                lambda: load_pack(tmp, live_source=src, live_target=tgt))
        record_acceptance(tmp, AcceptanceMetrics(
            top1_agreement=0.91, nll_translated=2.10, nll_reference=2.00,
            nll_delta=0.10, n_positions=500, eval_sequences=20))

        # Tokenizer drift -> KVT005.
        refuses("tokenizer_drift", "KVT005",
                lambda: load_pack(tmp, live_source=src,
                                  live_target=geom("big", 2, 4, "f" * 64)))
        # Geometry drift -> KVT004.
        refuses("geometry_drift", "KVT004",
                lambda: load_pack(tmp, live_source=src,
                                  live_target=geom("big", 2, 8, "d" * 64)))
        # Tampered weights -> KVT006.
        blob = _weights_path(tmp)
        original = blob.read_bytes()
        blob.write_bytes(original[:-8] + b"\x00" * 8)
        refuses("tampered_weights", "KVT006",
                lambda: load_pack(tmp, live_source=src, live_target=tgt))
        blob.write_bytes(original)

        # MUTATION GUARD: after restoring the blob the pack must load again.
        # Without this, a load_pack that refuses EVERYTHING passes every
        # refusal check above and is indistinguishable from a working gate.
        check("mutation_guard_restores",
              load_pack(tmp, live_source=src, live_target=tgt) is not None,
              "pack did not load after restoring the original bytes — the gate "
              "may be refusing unconditionally")

        # Missing manifest -> KVT001.
        _manifest_path(tmp).unlink()
        refuses("no_manifest", "KVT001",
                lambda: load_pack(tmp, live_source=src, live_target=tgt))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for f in failures:
        print(f"FAIL {f}")
    print(f"pack self-test: {'PASS' if not failures else 'FAIL'} ({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
