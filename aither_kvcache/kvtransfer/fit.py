"""Fit a mapper pack from a capture directory.

One data pass per (role, split) builds the shared source Gram ``X'X``; each
target layer then needs only its own ``X'Y``. Ranking 30 candidate source layers
and sweeping 7 ridge strengths after that is submatrix algebra, so the expensive
part happens once instead of ``n_source_layers * n_lambdas`` times.

The headline the run prints is deliberately three numbers, not one:

* **best single source layer** — the paper's 56% baseline. One layer, one ridge.
* **top-k combined** — the paper's 79%. If this is not meaningfully above the
  single-layer number, cross-layer selection is buying nothing on this pair and
  the extra ``k``-fold weight cost is not justified.
* **per-role split** — K and V transfer differently (V has no rotation to strip
  and is generally easier). A single averaged figure hides which half is weak.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .capture import CaptureManifest
from .geometry import KVGeometry, check_pair
from .mapper import GramStats, LayerMap, fit_layer_map
from .pack import write_pack

logger = logging.getLogger("kvtransfer.fit")


def _load_source(cap_dir: Path, split: str, role: str) -> np.ndarray:
    """Source activations as float32. Held fully in RAM — it is the small side."""
    arr = np.load(cap_dir / f"{split}_src_{role}.npy", mmap_mode="r")
    return np.asarray(arr, dtype=np.float32)


def _shared_gram(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(X'X, sum_x) in float64 — computed once per (role, split)."""
    xd = x.astype(np.float64, copy=False)
    return xd.T @ xd, xd.sum(axis=0)


def _layer_gram(
    x: np.ndarray,
    xtx: np.ndarray,
    sum_x: np.ndarray,
    y: np.ndarray,
) -> GramStats:
    """Build GramStats for one target layer, reusing the shared source Gram."""
    yd = y.astype(np.float64, copy=False)
    return GramStats(
        n=int(x.shape[0]),
        sum_x=sum_x,
        sum_y=yd.sum(axis=0),
        xtx=xtx,
        xty=x.astype(np.float64, copy=False).T @ yd,
        yty_diag=np.einsum("ij,ij->j", yd, yd),
    )


def fit_pack(
    cap_dir: Path,
    out_dir: Path,
    *,
    top_k: int = 8,
    roles: Sequence[str] = ("k", "v"),
    layers: Optional[Sequence[int]] = None,
    weight_dtype: str = "float16",
) -> Dict[str, object]:
    """Fit every target layer and write the pack. Returns a summary dict."""
    cap = CaptureManifest.read(cap_dir)
    geom_src = KVGeometry.from_dict(cap.source)
    geom_tgt = KVGeometry.from_dict(cap.target)
    check_pair(geom_src, geom_tgt).raise_if_blocked()

    src_block = geom_src.per_layer_width
    tgt_block = geom_tgt.per_layer_width
    target_layers = list(layers) if layers is not None else list(range(geom_tgt.n_layers))
    if top_k > geom_src.n_layers:
        raise ValueError(
            f"top_k={top_k} exceeds the source's {geom_src.n_layers} layers"
        )

    all_maps: Dict[str, List[LayerMap]] = {}
    summary: Dict[str, object] = {}

    for role in roles:
        t_role = time.time()
        x_tr = _load_source(cap_dir, "train", role)
        x_va = _load_source(cap_dir, "val", role)
        logger.info("[%s] source Gram over %d train / %d val tokens (D=%d)",
                    role, x_tr.shape[0], x_va.shape[0], x_tr.shape[1])
        xtx_tr, sx_tr = _shared_gram(x_tr)
        xtx_va, sx_va = _shared_gram(x_va)

        y_tr_all = np.load(cap_dir / f"train_tgt_{role}.npy", mmap_mode="r")
        y_va_all = np.load(cap_dir / f"val_tgt_{role}.npy", mmap_mode="r")

        maps: List[LayerMap] = []
        for t in target_layers:
            sl = slice(t * tgt_block, (t + 1) * tgt_block)
            tr = _layer_gram(x_tr, xtx_tr, sx_tr, np.asarray(y_tr_all[:, sl]))
            va = _layer_gram(x_va, xtx_va, sx_va, np.asarray(y_va_all[:, sl]))
            lm = fit_layer_map(
                tr, va, target_layer=t, role=role,
                n_source_layers=geom_src.n_layers, source_block=src_block,
                top_k=top_k, n_target_heads=geom_tgt.n_kv_heads,
                head_dim=geom_tgt.head_dim,
            )
            maps.append(lm)
            best_single = max(lm.candidate_r2.values()) if lm.candidate_r2 else float("nan")
            logger.info(
                "[%s] layer %2d  single=%.4f  top%d=%.4f  src=%s  lam=%g",
                role, t, best_single, top_k, lm.val_r2_mean, lm.source_layers, lm.lam_rel,
            )
        all_maps[role] = maps

        singles = [max(m.candidate_r2.values()) for m in maps if m.candidate_r2]
        summary[role] = {
            "val_r2_topk": float(np.mean([m.val_r2_mean for m in maps])),
            "val_r2_best_single_source": float(np.mean(singles)) if singles else None,
            "layers_fitted": len(maps),
            "seconds": round(time.time() - t_role, 1),
        }
        del x_tr, x_va, xtx_tr, xtx_va

    manifest = write_pack(
        out_dir, all_maps, source=geom_src, target=geom_tgt, top_k=top_k,
        regime_flags=cap.regime_flags, train_tokens=cap.n_train_tokens,
        val_tokens=cap.n_val_tokens, corpus_sha256=cap.corpus_sha256,
        created_at=time.time(), weight_dtype=weight_dtype,
    )
    weight_bytes = (out_dir / "mapper.npz").stat().st_size
    summary["pack"] = {
        "dir": str(out_dir),
        "weight_bytes": weight_bytes,
        "weight_mib": round(weight_bytes / 2 ** 20, 1),
        "val_r2_k": manifest.val_r2_k,
        "val_r2_v": manifest.val_r2_v,
        "regime_flags": manifest.regime_flags,
    }
    return summary


def _self_test() -> int:
    """Fit end to end on a synthetic capture directory."""
    import shutil
    import tempfile

    failures: List[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(f"{name}: {detail}")

    rng = np.random.default_rng(31337)
    tmp = Path(tempfile.mkdtemp(prefix="kvtfit"))
    try:
        n_src_layers, src_kv, dh = 6, 2, 4
        n_tgt_layers, tgt_kv = 3, 3
        src_w, tgt_w = n_src_layers * src_kv * dh, n_tgt_layers * tgt_kv * dh
        n_tr, n_va = 3000, 900

        # Plant: every target layer is a linear function of source layers 1 and 4.
        planted = [1, 4]
        cols = np.concatenate([np.arange(s * src_kv * dh, (s + 1) * src_kv * dh)
                               for s in planted])
        w_true = rng.standard_normal((len(cols), tgt_w))

        for split, n in (("train", n_tr), ("val", n_va)):
            for role in ("k", "v"):
                xs = rng.standard_normal((n, src_w)).astype(np.float16)
                ys = (xs[:, cols].astype(np.float32) @ w_true
                      + 0.03 * rng.standard_normal((n, tgt_w))).astype(np.float16)
                np.save(tmp / f"{split}_src_{role}.npy", xs)
                np.save(tmp / f"{split}_tgt_{role}.npy", ys)

        def geom(mid: str, layers: int, kv: int) -> Dict[str, object]:
            return KVGeometry(
                model_id=mid, family="llama", n_layers=layers, n_kv_heads=kv,
                head_dim=dh, hidden_size=64, rope_theta=10000.0, rope_type="default",
                rope_scaling_factor=1.0, tokenizer_sha256="a" * 64,
                torch_dtype="float32").to_dict()

        CaptureManifest(
            source=geom("src", n_src_layers, src_kv),
            target=geom("tgt", n_tgt_layers, tgt_kv),
            seq_len=64, n_train_tokens=n_tr, n_val_tokens=n_va,
            n_train_docs=10, n_val_docs=3, corpus_sha256="b" * 64,
            regime_flags=[], created_at=0.0,
        ).write(tmp)

        out = tmp / "pack"
        summary = fit_pack(tmp, out, top_k=2)
        check("recovers_r2", summary["k"]["val_r2_topk"] > 0.98,
              f"val R2 {summary['k']['val_r2_topk']:.4f} on planted-linear data")
        check("pack_written", (out / "mapper.npz").exists() and
              (out / "mapper.manifest.json").exists())

        man = json.loads((out / "mapper.manifest.json").read_text(encoding="utf-8"))
        picked = man["per_layer"]["k.0"]["source_layers"]
        check("selects_planted_layers", picked == planted,
              f"selected {picked}, planted {planted}")

        # MUTATION GUARD: top-k must beat a single source layer on data whose
        # signal is genuinely split across two layers. If these are equal, the
        # cross-layer machinery is inert and the headline comparison is a lie.
        single = fit_pack(tmp, tmp / "pack1", top_k=1)
        check("topk_beats_single",
              summary["k"]["val_r2_topk"] > single["k"]["val_r2_topk"] + 0.05,
              f"top2={summary['k']['val_r2_topk']:.4f} vs "
              f"top1={single['k']['val_r2_topk']:.4f} — selection adds nothing")

        try:
            fit_pack(tmp, tmp / "pack2", top_k=99)
            check("rejects_big_topk", False, "accepted top_k > n_source_layers")
        except ValueError as exc:
            check("rejects_big_topk_reason", "exceeds" in str(exc),
                  f"refused, but not for top_k: {exc}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for f in failures:
        print(f"FAIL {f}")
    print(f"fit self-test: {'PASS' if not failures else 'FAIL'} ({len(failures)} failures)")
    return 1 if failures else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fit a cross-model KV mapper pack")
    ap.add_argument("--capture", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--roles", nargs="*", default=["k", "v"])
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--weight-dtype", default="float16")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.self_test:
        return _self_test()
    if not (args.capture and args.out):
        ap.error("--capture and --out are required")
    summary = fit_pack(args.capture, args.out, top_k=args.top_k, roles=args.roles,
                       layers=args.layers, weight_dtype=args.weight_dtype)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
